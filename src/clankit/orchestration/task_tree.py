"""Task Tree: recursive decomposition into a tree of agents.

The root agent talks to the user. Any agent may split its task into subtasks with the
`delegate` tool; each subtask gets its own agent (possibly on a different model), which may
split further, down to the depth limit. Leaves build. Each agent that delegated reviews
what came back, resolves conflicts and gaps, glues the parts together and verifies the
whole before it reports upward; the root answers the user.

What the evidence says, and how it shapes this module:

* Splitting pays only when parts are genuinely independent, or the work is too large for one
  context; otherwise one agent working step by step is faster, cheaper and more reliable.
  So every node is a leaf by default and the prompts make decomposition earn its place.
* Parallel writers on shared files are the main failure mode. Each subtask owns a write
  scope, enforced by the file tools; subtasks whose scopes may overlap never run at the
  same time, and shared decisions are made by the parent before it delegates.
* Self-reported success is unreliable. Acceptance commands named in a brief are run by the
  harness after the child reports, file changes are taken from the harness's own records,
  and a failed check earns one fix attempt, then one fresh attempt, then an honest failure.
* Depth alone does not bound a tree (three children per node, ten levels deep, is 88,573
  agents). Node count per prompt, children per call and parallelism are limited too, the
  run budget applies to all agents, and at the limits the `delegate` tool is simply absent.
* Context flows down as a self-contained brief plus the user's goal verbatim, and up as a
  short structured report. Agents never share transcripts.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.events import NodeInfo
from clankit.agent.loop import Agent
from clankit.config.schema import Setting
from clankit.llm.provider import ProviderError
from clankit.llm.types import ModelRef
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import (
    CheckResult,
    FinishTool,
    changed_paths,
    diff_stat,
    models_guide,
    resolve_model,
    run_check,
    run_diff,
    run_to_finish,
    subtree_ids,
)
from clankit.orchestration.review import findings_message, review
from clankit.tools import standard_tools
from clankit.tools.base import Tool, ToolContext, ToolOutput, scopes_overlap

if TYPE_CHECKING:
    from clankit.agent.session import Session

REPORT_STATUSES = ("done", "partial", "blocked", "failed")
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")

# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------

SPLITTING = {
    "adaptive": """\
## Doing the work yourself or splitting it

Do the work yourself unless splitting clearly helps. One agent working through a task step \
by step is usually faster, cheaper and more reliable than several agents, and coordinating \
agents costs more than it looks.

Use `delegate` only when one of these holds:
- The task has parts that are genuinely independent: they can be built at the same time, \
each touches files the others do not, and each can be checked on its own.
- The task is too large to finish well in one conversation, even working step by step.""",
    "eager": """\
## Splitting the work

Split the work up front with `delegate` whenever it has parts that can be built \
independently: several components, files, features or fixes that can proceed at the same \
time. Hand each part to a focused agent so the parts run in parallel - that parallelism is \
the point. Keep for yourself only what is small, tightly coupled or strictly sequential, \
and the final integration.""",
    "minimal": """\
## Splitting the work

Do the work yourself. Use `delegate` only when the task clearly cannot be finished in one \
conversation, and then split it into as few parts as possible.""",
}

HOW_TO_SPLIT = """\
When you do split:
1. First settle everything the parts share: interfaces, data formats, names, file layout. \
Create shared files, types or stubs yourself before delegating, so the parts fit together.
2. Give each subtask a brief another engineer could act on without asking you anything: \
the goal and why it matters, the context and decisions it needs, what it must provide to or \
use from other parts, and what it must not do. Subtask agents see only their brief and the \
user's original request, not this conversation.
3. Give each subtask the files it owns (`write_scope`); parallel subtasks must not share \
files. Give a command that proves the part works (`checks`) wherever one exists.
4. Use `depends_on` when one part needs another's result; independent parts run in parallel.
5. When the results come back, treat each report as a claim to verify: read the changes \
that matter, run the checks on the combined result, and fix conflicts and gaps yourself.

Never delegate your whole task as a single subtask, and do not split work that is small or \
sequential."""

MODEL_CHOICE = """\
## Choosing models for subtasks

Each subtask can run on a different model (`model`); without one it runs on {default}. \
Put the strongest models on work that needs judgment (design, subtle logic, debugging) and \
cheaper ones on well-specified, mechanical work. Available models:
{guide}"""

ROOT_ROLE = """\
# Working as a task tree

You are the root of a tree of agents. You can split work into subtasks with `delegate`; \
each subtask gets its own agent, which can split further. You stay responsible for the \
whole result: the user sees only what you say."""

CHILD_ROLE = """\
# Your role

You are one agent in a tree of agents working on a task for a user. The agent that \
assigned you this task wrote the first message; you report back to it with `report`. The \
user does not see your messages, and nobody can answer questions while you work: make \
reasonable decisions within your brief and record them.

{splitting}

## Reporting

When you are finished, or cannot go further, call `report` once:
- "done" only when the task is complete and a command you ran showed that it works;
- "partial" when some of it is done: say exactly what remains;
- "blocked" when you need a decision or information you cannot get: ask it in open_issues;
- "failed" when it cannot be done as specified: say why.
Keep the summary short and factual; do not paste code, the files are on disk. Mention \
anything you changed outside your brief, every assumption you made, and anything you \
noticed that the agent above you should know."""

LEAF_ONLY = """\
## Doing the work

Do this task yourself; this branch of the tree cannot be split any further. If it turns \
out to be too large, finish the most important coherent part, then report "partial" and \
describe what remains."""

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(REPORT_STATUSES)},
        "summary": {
            "type": "string",
            "description": "What you did and what the result is, in a few sentences.",
        },
        "verification": {
            "type": "string",
            "description": "The commands you ran to check the work and what they showed.",
        },
        "open_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Remaining work, assumptions, changes outside the brief, questions.",
        },
    },
    "required": ["status", "summary"],
    "additionalProperties": False,
}

NO_REPORT = "The agent stopped without calling `report`."


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Subtask:
    """One entry of a `delegate` call, validated."""

    key: str
    title: str
    brief: str
    write_scope: tuple[str, ...] | None
    checks: tuple[str, ...]
    depends_on: tuple[str, ...]
    model: ModelRef


@dataclass(slots=True)
class Report:
    status: str
    summary: str
    verification: str = ""
    open_issues: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, payload: dict[str, Any] | None, fallback: str) -> Report:
        if payload is None:
            return cls("failed", fallback)
        status = payload.get("status")
        issues = payload.get("open_issues") or []
        return cls(
            status=status if status in REPORT_STATUSES else "failed",
            summary=str(payload.get("summary") or "").strip() or "(no summary given)",
            verification=str(payload.get("verification") or "").strip(),
            open_issues=[str(i).strip() for i in issues if str(i).strip()],
        )


@dataclass(slots=True)
class Outcome:
    """How a subtask ended, as the harness sees it."""

    subtask: Subtask
    node: NodeInfo
    report: Report
    checks: list[CheckResult] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    attempts: int = 1
    error: str = ""

    @property
    def checks_passed(self) -> bool:
        return all(c.passed or c.denied for c in self.checks)

    @property
    def accepted(self) -> bool:
        return not self.error and self.report.status in ("done", "partial") and self.checks_passed

    @property
    def label(self) -> str:
        if self.error:
            return "error"
        if self.report.status in ("done", "partial") and not self.checks_passed:
            return "checks failed"
        return self.report.status

    def render(self) -> str:
        lines = [
            f"## [{self.label}] {self.subtask.title}  ({self.subtask.model}, node {self.node.id})"
        ]
        if self.attempts > 1:
            lines.append(f"Attempts: {self.attempts}")
        if self.error:
            lines.append(f"Error: {self.error}")
        lines.append(f"Summary: {self.report.summary}")
        if self.report.verification:
            lines.append(f"Verification reported by the agent: {self.report.verification}")
        if self.checks:
            lines.append("Acceptance checks run by the harness:")
            lines.extend(_indent(c.render(tail=1500)) for c in self.checks)
        elif self.report.status in ("done", "partial") and not self.error:
            lines.append("No acceptance checks were given, so the harness did not verify this.")
        if self.files:
            lines.append("Files changed (harness records):")
            lines.extend(f"  {f}" for f in self.files)
        else:
            lines.append("Files changed: none recorded")
        if self.report.open_issues:
            lines.append("Open issues:")
            lines.extend(f"- {issue}" for issue in self.report.open_issues)
        return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _one_line(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------


class DelegateTool(Tool):
    """Split the current task into subtasks run by other agents, and wait for their reports."""

    name = "delegate"
    read_only = False

    #: The closing line of the tool description, per decomposition mode. The model reads this
    #: with the tool definition, so it must agree with the guidance in the system prompt.
    _GUIDANCE = {
        "adaptive": "Read the guidance in your instructions before using this: most tasks are "
        "better done yourself.",
        "eager": "Splitting is expected: use this whenever your task has parts that can be "
        "built independently.",
        "minimal": "Use only when the task clearly cannot be finished in one conversation; "
        "split it into as few parts as possible.",
    }

    def __init__(self, tree: TaskTreeOrchestrator, node: NodeInfo, *, cross_model: bool) -> None:
        self._tree = tree
        self._node = node
        max_children = tree.setting("max-children")
        mode = tree.setting("decomposition")
        self.description = (
            "Split your task into subtasks, each done by its own agent, and wait for their "
            "reports. Independent subtasks with separate write scopes run in parallel. "
            f"{self._GUIDANCE.get(mode, self._GUIDANCE['adaptive'])}"
        )
        item: dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": 'Short key, unique in this call, e.g. "parser".',
                },
                "title": {"type": "string", "description": "A few words naming the subtask."},
                "brief": {
                    "type": "string",
                    "description": "Self-contained instructions: goal, context, decisions, "
                    "interfaces to provide or use, constraints.",
                },
                "write_scope": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files or directories (relative paths, globs allowed) this "
                    "subtask may change. Omit to allow any file; that subtask then never runs "
                    "in parallel with another. An empty list means read-only.",
                },
                "checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Shell commands that must succeed for the work to be "
                    "accepted, e.g. a test command. Run by the harness after the subtask reports.",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ids of subtasks in this call that must finish first.",
                },
            },
            "required": ["id", "title", "brief"],
            "additionalProperties": False,
        }
        if cross_model:
            item["properties"]["model"] = {
                "type": "string",
                "description": "Model to run this subtask, as provider/model. Omit for the default.",
            }
        self.input_schema = {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": item,
                    "description": f"Between 2 and {max_children} subtasks.",
                }
            },
            "required": ["subtasks"],
            "additionalProperties": False,
        }

    def summarize(self, tool_input: dict[str, Any]) -> str:
        subtasks = tool_input.get("subtasks") or []
        titles = ", ".join(str(s.get("title", "?")) for s in subtasks[:4] if isinstance(s, dict))
        more = f" and {len(subtasks) - 4} more" if len(subtasks) > 4 else ""
        return f"delegate {len(subtasks)}: {titles}{more}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._tree.delegate(self._node, ctx, tool_input)


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class TaskTreeOrchestrator(Orchestrator):
    name = "task-tree"
    title = "Task Tree"
    summary = "Agents split work into subtasks recursively; parents review and integrate."
    guidance = (
        "Best for large tasks with genuinely independent parts. How eagerly agents split is "
        "set by decomposition: minimal keeps them together, eager splits multi-part work up "
        "front, adaptive is in between. Subtasks own separate files, are accepted by "
        "commands the harness runs, and can use other models. Deep or wide trees multiply "
        "cost: the node and run budgets cap it."
    )
    settings = (
        Setting(
            "max-depth",
            "int",
            10,
            "How many levels below the root the tree may grow.",
            minimum=0,
            maximum=20,
        ),
        Setting(
            "max-nodes",
            "int",
            50,
            "Subtasks one prompt may create in total.",
            minimum=1,
            maximum=1000,
        ),
        Setting(
            "max-children",
            "int",
            6,
            "Subtasks one agent may create in one call.",
            minimum=2,
            maximum=20,
        ),
        Setting(
            "max-parallel",
            "int",
            4,
            "Subtasks of one call that may run at the same time.",
            minimum=1,
            maximum=32,
        ),
        Setting(
            "decomposition",
            "choice",
            "adaptive",
            "adaptive: split only when it clearly helps. eager: split multi-part work up front. "
            "minimal: split only work too large for one agent.",
            choices=("adaptive", "eager", "minimal"),
        ),
        Setting(
            "cross-model", "bool", True, "Let agents choose a different model for each subtask."
        ),
        Setting(
            "fix-attempts",
            "int",
            1,
            "When a subtask's acceptance checks fail, how many times its agent is asked to fix it.",
            minimum=0,
            maximum=5,
        ),
        Setting(
            "redispatch",
            "bool",
            True,
            "If a subtask still fails, try once more with a fresh agent (on the parent's model).",
        ),
        Setting(
            "review",
            "choice",
            "final",
            "Independent review of changes by a fresh agent. final: before answering the user. "
            "branches: also before each agent that delegated reports back. off: never.",
            choices=("off", "final", "branches"),
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")
        self._goal = ""
        self._nodes_created = 0

    # -- root -----------------------------------------------------------------------------------

    def _ensure_root(self) -> Agent:
        if self.root_agent is None:
            session = self.session
            tools: list[Tool] = [*standard_tools(), AskUserTool(session)]
            role = prompts.USER_FACING
            if self.setting("max-depth") >= 1:
                tools.append(
                    DelegateTool(self, self._root, cross_model=self.setting("cross-model"))
                )
                role = f"{role}\n\n{self._splitting_text(self._root, root=True)}"
            ctx = session.tool_context(self._root)
            self.root_agent = session.agent(
                self._root, system=prompts.system_prompt(session, role), tools=tools, ctx=ctx
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        fresh = self.root_agent is None
        agent = self._ensure_root()
        self._goal = prompt
        self._nodes_created = 0
        if fresh and (recap := session.recap()):
            prompt = f"{recap}\n\nThe user now says:\n{prompt}"

        session.update_node(self._root, status="running")
        result = await agent.run(prompt)
        if result.stop == "refusal":
            session.update_node(self._root, status="failed")
            return f"The model declined this request. {result.detail}".strip()

        if result.stop == "end_turn" and self.setting("review") != "off":
            diff = run_diff(session)
            if diff:
                session.update_node(self._root, status="integrating")
                outcome = await review(session, parent=self._root, task=self._goal, diff=diff)
                if outcome.blocking:
                    result = await agent.run(findings_message(outcome))
        session.update_node(self._root, status="done")
        return result.text

    # -- prompts --------------------------------------------------------------------------------

    def _splitting_text(self, node: NodeInfo, *, root: bool) -> str:
        parts = [ROOT_ROLE] if root else []
        parts.append(SPLITTING[self.setting("decomposition")])
        parts.append(HOW_TO_SPLIT)
        if not root and node.depth >= 3:
            parts.append(
                f"You are {node.depth} levels deep (the limit is {self.setting('max-depth')}). "
                "At this depth, split only if the work truly cannot be done by one agent."
            )
        if self.setting("cross-model"):
            guide = models_guide(self.session)
            if guide:
                parts.append(
                    MODEL_CHOICE.format(default=self._default_child_model(node), guide=guide)
                )
        return "\n\n".join(parts)

    def _child_role(self, node: NodeInfo, can_delegate: bool) -> str:
        splitting = self._splitting_text(node, root=False) if can_delegate else LEAF_ONLY
        return CHILD_ROLE.format(splitting=splitting)

    def _default_child_model(self, parent: NodeInfo) -> ModelRef:
        if self.session.config.models.roles.get("subagent"):
            return self.session.model_for("subagent")
        return ModelRef.parse(parent.model) if parent.model else self.session.model

    # -- delegation -----------------------------------------------------------------------------

    def _validate(self, parent: NodeInfo, tool_input: dict[str, Any]) -> list[Subtask] | str:
        raw = tool_input.get("subtasks") or []
        max_children = self.setting("max-children")
        if len(raw) < 2:
            return (
                "A delegation needs at least two subtasks. To do a single piece of work, do it "
                "yourself: handing your whole task to one other agent only adds cost."
            )
        if len(raw) > max_children:
            return f"At most {max_children} subtasks per call; group related work or delegate in waves."
        remaining = self.setting("max-nodes") - self._nodes_created
        if len(raw) > remaining:
            return (
                f"The subtask budget for this request is nearly used up ({remaining} left). "
                "Do the remaining work yourself, or split it into fewer, larger parts."
            )

        subtasks: list[Subtask] = []
        keys: set[str] = set()
        for index, item in enumerate(raw, 1):
            key = str(item.get("id", "")).strip()
            if not _KEY.match(key):
                return f"Subtask {index}: id must be a short word of letters, digits, '-' or '_'."
            if key in keys:
                return f"Subtask id {key!r} is used twice."
            keys.add(key)
            title = _one_line(str(item.get("title", "")), 80)
            brief = str(item.get("brief", "")).strip()
            if not title or len(brief) < 20:
                return (
                    f"Subtask {key!r}: give it a title and a brief that another agent can act on."
                )
            scope = self._scope(item.get("write_scope"))
            if isinstance(scope, str):
                return f"Subtask {key!r}: {scope}"
            checks = tuple(c.strip() for c in item.get("checks") or [] if c.strip())
            depends = tuple(str(d).strip() for d in item.get("depends_on") or [] if str(d).strip())
            try:
                model = (
                    resolve_model(self.session, item["model"])
                    if item.get("model")
                    else self._default_child_model(parent)
                )
            except ValueError as exc:
                return f"Subtask {key!r}: {exc}"
            info = self.session.model_info(model)
            if info is not None and info.supports_tools is False:
                return f"Subtask {key!r}: {model} cannot call tools, so it cannot run an agent."
            subtasks.append(Subtask(key, title, brief, scope, checks, depends, model))

        for subtask in subtasks:
            for dep in subtask.depends_on:
                if dep not in keys:
                    return f"Subtask {subtask.key!r} depends on {dep!r}, which is not in this call."
                if dep == subtask.key:
                    return f"Subtask {subtask.key!r} cannot depend on itself."
        if _has_cycle(subtasks):
            return "The depends_on links form a cycle; make the order acyclic."
        return subtasks

    @staticmethod
    def _scope(value: Any) -> tuple[str, ...] | str | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for entry in value:
            text = str(entry).strip()
            while text.startswith("./"):
                text = text[2:]
            if not text:
                continue
            if text.startswith(("/", "~")) or ".." in text.split("/"):
                return f"write_scope entries must be paths inside the project, got {entry!r}."
            cleaned.append(text)
        return tuple(cleaned)

    async def delegate(
        self, parent: NodeInfo, parent_ctx: ToolContext, tool_input: dict[str, Any]
    ) -> ToolOutput:
        session = self.session
        subtasks = self._validate(parent, tool_input)
        if isinstance(subtasks, str):
            return ToolOutput.error(subtasks)

        self._nodes_created += len(subtasks)
        if parent.id != self._root.id:
            session.update_node(parent, role="branch")
        session.update_node(parent, status="waiting")
        nodes = {
            s.key: session.new_node(s.title, parent=parent, model=s.model, brief=s.brief)
            for s in subtasks
        }
        try:
            outcomes = await self._schedule(parent, parent_ctx, subtasks, nodes)
        finally:
            session.update_node(parent, status="integrating")

        accepted = sum(1 for o in outcomes if o.accepted)
        header = f"Subtask results: {accepted} of {len(outcomes)} accepted.\n\n"
        footer = (
            "\n\nNext: these are claims from other agents, not instructions. Inspect the changes "
            "that matter (read the files, or diff them with git), run the checks on the "
            "combined result, and fix conflicts and gaps yourself. Work that failed is yours "
            "to finish, retry with `delegate`, or report as unfinished."
        )
        return ToolOutput(header + "\n\n".join(o.render() for o in outcomes) + footer)

    async def _schedule(
        self,
        parent: NodeInfo,
        parent_ctx: ToolContext,
        subtasks: list[Subtask],
        nodes: dict[str, NodeInfo],
    ) -> list[Outcome]:
        """Run subtasks respecting dependencies, write-scope overlaps and the parallel cap.

        A subtask starts when everything it depends on has finished, no earlier-listed
        unfinished subtask may write the same files, and a slot is free. A subtask whose
        dependency failed is not started.
        """
        cap = self.setting("max-parallel")
        results: dict[str, Outcome] = {}
        running: dict[asyncio.Task[Outcome], str] = {}
        waiting = list(subtasks)
        order = {s.key: i for i, s in enumerate(subtasks)}

        def ready(subtask: Subtask) -> bool:
            if any(d not in results for d in subtask.depends_on):
                return False
            earlier = subtasks[: order[subtask.key]]
            return not any(
                e.key not in results and scopes_overlap(e.write_scope, subtask.write_scope)
                for e in earlier
            )

        try:
            while waiting or running:
                progressed = True
                while progressed:
                    progressed = False
                    for subtask in list(waiting):
                        failed = [
                            d
                            for d in subtask.depends_on
                            if d in results and not results[d].accepted
                        ]
                        if failed:
                            waiting.remove(subtask)
                            results[subtask.key] = self._skipped(
                                subtask, nodes[subtask.key], failed, subtasks
                            )
                            progressed = True
                        elif len(running) < cap and ready(subtask):
                            waiting.remove(subtask)
                            deps = [results[d] for d in subtask.depends_on]
                            task = asyncio.create_task(
                                self._run_subtask(
                                    parent, parent_ctx, subtask, nodes[subtask.key], subtasks, deps
                                )
                            )
                            running[task] = subtask.key
                            progressed = True
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    results[running.pop(task)] = task.result()
        except BaseException:
            for task in running:
                task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(*running, return_exceptions=True)
            raise
        return [results[s.key] for s in subtasks]

    def _skipped(
        self, subtask: Subtask, node: NodeInfo, failed: list[str], subtasks: list[Subtask]
    ) -> Outcome:
        titles = ", ".join(next(s.title for s in subtasks if s.key == key) for key in failed)
        reason = f"Not started: it depends on {titles}, which did not succeed."
        self.session.update_node(node, status="cancelled", summary=reason)
        return Outcome(subtask, node, Report("failed", reason))

    # -- one subtask ----------------------------------------------------------------------------

    def _brief(
        self, node: NodeInfo, subtask: Subtask, subtasks: list[Subtask], deps: list[Outcome]
    ) -> str:
        session = self.session
        path = []
        current: NodeInfo | None = session.node(node.parent_id) if node.parent_id else None
        while current is not None and current.parent_id is not None:
            path.append(current.title)
            current = session.node(current.parent_id)
        path.reverse()
        parent = session.node(node.parent_id) if node.parent_id else None

        sections = [
            "You are working on one part of a larger task.",
            f"<user_request>\n{self._goal.strip()}\n</user_request>",
        ]
        if path:
            sections.append(
                "Where this task sits: " + " > ".join(["the user's request", *path, subtask.title])
            )
        if parent is not None and parent.brief:
            sections.append(f"<parent_task>\n{parent.brief.strip()[:3000]}\n</parent_task>")
        sections.append(f"<your_task>\n# {subtask.title}\n\n{subtask.brief}\n</your_task>")
        if deps:
            rendered = "\n\n".join(
                f"## {d.subtask.title}\n{d.report.summary}\nFiles: {', '.join(f.split('  ')[0] for f in d.files) or 'none'}"
                for d in deps
            )
            sections.append(f"<results_you_build_on>\n{rendered}\n</results_you_build_on>")
        siblings = [s for s in subtasks if s.key != subtask.key]
        if siblings:
            lines = []
            for s in siblings:
                owns = (
                    ", ".join(s.write_scope)
                    if s.write_scope
                    else ("nothing" if s.write_scope == () else "unspecified files")
                )
                lines.append(f"- {s.title} (owns {owns})")
            sections.append(
                "Other agents are working on these parts; do not change their files:\n"
                + "\n".join(lines)
            )

        if subtask.write_scope is None:
            scope = "Any file in the project that your task needs."
        elif not subtask.write_scope:
            scope = "None: this is a read-only task."
        else:
            scope = "\n".join(f"- {p}" for p in subtask.write_scope)
        sections.append(
            f"## Files you may change\n{scope}\nIf your task needs a change elsewhere, do not make it: "
            "describe it in your report's open_issues."
        )
        if subtask.checks:
            commands = "\n".join(f"- `{c}`" for c in subtask.checks)
            sections.append(
                f"## How your work will be accepted\nAfter you report, the harness runs these commands; "
                f"they must succeed:\n{commands}"
            )
        else:
            sections.append(
                "## How your work will be accepted\nNo acceptance commands were given. Verify the work "
                "yourself with the project's tests or checks, and say in your report exactly what you ran."
            )
        sections.append("When you are finished, call `report`.")
        return "\n\n".join(sections)

    def _child_agent(self, node: NodeInfo, ctx: ToolContext, model: ModelRef) -> Agent:
        can_delegate = node.depth < self.setting("max-depth")
        tools: list[Tool] = [*standard_tools()]
        if can_delegate:
            tools.append(DelegateTool(self, node, cross_model=self.setting("cross-model")))
        tools.append(
            FinishTool(
                "report",
                "Report your result to the agent that assigned your task. Call once, at the end.",
                REPORT_SCHEMA,
            )
        )
        return self.session.agent(
            node,
            system=prompts.system_prompt(self.session, self._child_role(node, can_delegate)),
            tools=tools,
            ctx=ctx,
            model=model,
        )

    async def _attempt(self, agent: Agent, prompt: str) -> Report:
        _, payload = await run_to_finish(agent, prompt, "report")
        return Report.parse(payload, NO_REPORT)

    async def _verify(self, ctx: ToolContext, subtask: Subtask) -> list[CheckResult]:
        return [await run_check(self.session, ctx, command) for command in subtask.checks]

    async def _run_subtask(
        self,
        parent: NodeInfo,
        parent_ctx: ToolContext,
        subtask: Subtask,
        node: NodeInfo,
        subtasks: list[Subtask],
        deps: list[Outcome],
    ) -> Outcome:
        session = self.session
        ctx = session.tool_context(node, write_scope=subtask.write_scope, parent=parent_ctx)
        session.update_node(node, status="running")
        brief = self._brief(node, subtask, subtasks, deps)
        outcome = Outcome(subtask, node, Report("failed", NO_REPORT))
        try:
            agent = self._child_agent(node, ctx, subtask.model)
            outcome = await self._work(agent, ctx, subtask, node, brief, outcome)
            if not outcome.accepted and self.setting("redispatch"):
                outcome = await self._redispatch(
                    parent, parent_ctx, subtask, node, subtasks, deps, outcome
                )
        except ProviderError as exc:
            outcome.error = str(exc)
            outcome.report = Report("failed", f"The agent could not reach its model: {exc}")
        outcome.files = diff_stat(
            session, changed_paths(session, subtree_ids(session, outcome.node.id))
        )
        final = "done" if outcome.accepted else "failed"
        session.update_node(
            outcome.node,
            status=final,
            summary=_one_line(f"{outcome.label}: {outcome.report.summary}", 200),
        )
        if outcome.node.id != node.id and node.status not in ("done", "failed", "cancelled"):
            session.update_node(node, status="failed")
        return outcome

    async def _work(
        self,
        agent: Agent,
        ctx: ToolContext,
        subtask: Subtask,
        node: NodeInfo,
        brief: str,
        outcome: Outcome,
    ) -> Outcome:
        """Run an agent on its brief, then hold it to its acceptance checks."""
        report = await self._attempt(agent, brief)
        report = await self._maybe_review_branch(agent, node, subtask, report)
        checks = await self._verify(ctx, subtask) if report.status in ("done", "partial") else []
        fixes = self.setting("fix-attempts")
        while checks and not all(c.passed or c.denied for c in checks) and fixes > 0:
            fixes -= 1
            failures = "\n\n".join(c.render() for c in checks if not (c.passed or c.denied))
            report = await self._attempt(
                agent,
                "[Harness] Your work was not accepted: these acceptance checks failed.\n\n"
                f"{failures}\n\nFind the cause, fix it, run the checks yourself, and call `report` again.",
            )
            checks = (
                await self._verify(ctx, subtask) if report.status in ("done", "partial") else []
            )
        return Outcome(subtask, node, report, checks=checks, attempts=outcome.attempts)

    async def _maybe_review_branch(
        self, agent: Agent, node: NodeInfo, subtask: Subtask, report: Report
    ) -> Report:
        """With review set to "branches", an agent that delegated gets an independent review
        of its integrated result before its report is accepted."""
        if self.setting("review") != "branches" or report.status not in ("done", "partial"):
            return report
        if not self.session.children_of(node.id):
            return report
        paths = changed_paths(self.session, subtree_ids(self.session, node.id))
        diff = run_diff(self.session, paths)
        if not diff:
            return report
        task = f"{subtask.title}\n\n{subtask.brief}"
        outcome = await review(self.session, parent=node, task=task, diff=diff, model=None)
        if not outcome.blocking:
            return report
        return await self._attempt(
            agent, findings_message(outcome) + "\n\nThen call `report` again."
        )

    async def _redispatch(
        self,
        parent: NodeInfo,
        parent_ctx: ToolContext,
        subtask: Subtask,
        node: NodeInfo,
        subtasks: list[Subtask],
        deps: list[Outcome],
        failed: Outcome,
    ) -> Outcome:
        """One fresh attempt by a new agent, on the parent's model when the first attempt used
        a different one. Skipped when the agent said the task cannot be done as specified."""
        if failed.report.status in ("blocked", "failed") and failed.report.summary != NO_REPORT:
            return failed
        if self._nodes_created >= self.setting("max-nodes"):
            return failed
        self._nodes_created += 1
        session = self.session
        parent_model = ModelRef.parse(parent.model) if parent.model else subtask.model
        session.update_node(
            node,
            status="failed",
            summary=_one_line(f"{failed.label}: {failed.report.summary}", 200),
        )
        retry = session.new_node(
            f"{subtask.title} (second attempt)",
            parent=parent,
            model=parent_model,
            brief=subtask.brief,
        )
        ctx = session.tool_context(retry, write_scope=subtask.write_scope, parent=parent_ctx)
        session.update_node(retry, status="running")
        failures = "\n\n".join(c.render() for c in failed.checks if not (c.passed or c.denied))
        history = (
            "\n\n<previous_attempt>\nAnother agent tried this task before you and its work was not "
            f"accepted.\nIts report: {failed.report.summary}\n"
            + (
                f"Open issues it listed: {'; '.join(failed.report.open_issues)}\n"
                if failed.report.open_issues
                else ""
            )
            + (f"Failing checks:\n{failures}\n" if failures else "")
            + "Its changes are still on disk. Build on what is right and fix what is wrong.\n</previous_attempt>"
        )
        brief = self._brief(retry, subtask, subtasks, deps) + history
        agent = self._child_agent(retry, ctx, parent_model)
        retried = Subtask(
            subtask.key,
            subtask.title,
            subtask.brief,
            subtask.write_scope,
            subtask.checks,
            subtask.depends_on,
            parent_model,
        )
        outcome = await self._work(
            agent,
            ctx,
            retried,
            retry,
            brief,
            Outcome(retried, retry, Report("failed", NO_REPORT), attempts=2),
        )
        outcome.attempts = failed.attempts + 1
        return outcome


def _has_cycle(subtasks: Sequence[Subtask]) -> bool:
    graph = {s.key: set(s.depends_on) for s in subtasks}
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(key: str) -> bool:
        if key in done:
            return False
        if key in visiting:
            return True
        visiting.add(key)
        if any(visit(dep) for dep in graph.get(key, ())):
            return True
        visiting.discard(key)
        done.add(key)
        return False

    return any(visit(key) for key in graph)

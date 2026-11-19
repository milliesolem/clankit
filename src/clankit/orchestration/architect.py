"""Architect-Editor: a strong model plans and advises; an editor model makes the changes.

For each request an architect agent, which can read but not change anything, explores the
code and hands in a concrete plan: the files to touch, the approach, the order of work and
the checks that show it works. The editor, the agent that talks to the user, carries the
plan out with every tool and may put a few questions to the architect along the way. A
request that asks for no change is answered by the architect directly.

What the evidence says, and how it shapes this module:

* Splitting planning from editing helped weak executors (several points on edit benchmarks
  with a reasoning model planning for a cheaper editor) and cost strong ones up to about
  three points: a weak primary sets the ceiling, and a strong one gains little from a plan it
  could have made itself. Hence the guidance, and the editor is the one that talks to the
  user and decides.
* Advice is short, typically a few hundred tokens, so questions are capped per request and
  answered briefly. The architect never edits: one writer at a time.
* An optional final check against the plan is an independent review like any other (a fresh
  context, evidence for every finding), so it reuses `clankit.orchestration.review`, on the
  architect's model. It is off by default: its value for this method is unmeasured.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.events import AssistantMessage, NodeInfo, UserMessage
from clankit.agent.loop import Agent, AgentResult
from clankit.config.schema import Setting
from clankit.llm.provider import ProviderError
from clankit.llm.types import ModelRef
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import FinishTool, run_diff, run_to_finish
from clankit.orchestration.review import Review, findings_message, review
from clankit.orchestration.review_loop import configured_model
from clankit.tools import read_only_tools, standard_tools
from clankit.tools.base import Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.session import Session

#: Earlier answers the editor missed (because the architect gave them) that it is told about.
UNSEEN_LIMIT = 6

ARCHITECT_ROLE = """\
# Your role

You are the architect in a two-agent setup. You explore the project and decide how a \
request should be carried out; a separate editor agent then makes the changes by following \
your plan. You cannot change files.

1. If the request asks a question, or asks for nothing to be changed, answer it yourself: \
investigate as needed and submit kind "answer" with the complete answer for the user.
2. Otherwise read the code the change involves and settle what the editor should not have \
to work out: which files change and how, the interfaces and names, the order of the work, \
and the commands that will show it works. Check facts in the code rather than assuming \
them; the editor relies on your plan.
3. Keep the plan concrete and short: name files and functions, and say what changes in each \
and why. Do not write out the full code; the editor writes it.
4. The editor may later ask you questions. Answer them briefly and specifically, from the \
code.

Finish by calling `submit_plan` once."""

EDITOR_ROLE = """\
# Working from an architect's plan

For requests that change code, an architect agent explores the project first and writes a \
plan, which arrives with the request. Follow it; it was made from the code. Where the code \
shows that a step is wrong, deviate and say so in your summary. When you are stuck, or the \
plan does not fit what you find, ask the architect with `consult_architect` instead of \
guessing; it answers briefly and does not change files."""

CONSULT = """\
[Question from the editor]
{question}

Answer briefly and specifically, from the code: a few sentences or a short list. Do not \
write the implementation, and do not call `submit_plan` unless the plan itself has to \
change."""

INCOMPLETE = """\
[Harness] That cannot be used: a plan needs an approach, files or steps, and an answer \
needs its text. Call `submit_plan` again with a complete plan, or with kind "answer" and \
the full answer."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["change", "answer"],
            "description": "answer when the request asks a question or asks for no change.",
        },
        "answer": {
            "type": "string",
            "description": "For kind answer: the complete answer for the user.",
        },
        "approach": {
            "type": "string",
            "description": "For kind change: the approach in a few sentences, with the "
            "decisions you made.",
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "change": {
                        "type": "string",
                        "description": "What changes in this file, and why.",
                    },
                },
                "required": ["path", "change"],
                "additionalProperties": False,
            },
            "description": "Files to create or change.",
        },
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The work, in order.",
        },
        "checks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "How to show it works: commands to run and what they should show.",
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Constraints, pitfalls, and what must not change.",
        },
    },
    "required": ["kind"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Plan:
    approach: str
    files: list[tuple[str, str]]
    steps: list[str]
    checks: list[str]
    notes: list[str]

    @property
    def headline(self) -> str:
        return (
            f"Plan: {len(self.files)} file(s), {len(self.steps)} step(s), "
            f"{len(self.checks)} check(s)."
        )

    def render(self) -> str:
        lines = [f"Approach: {self.approach}"] if self.approach else []
        if self.files:
            lines.append("Files:")
            lines.extend(f"- {path}: {change}" for path, change in self.files)
        if self.steps:
            lines.append("Steps:")
            lines.extend(f"{number}. {step}" for number, step in enumerate(self.steps, 1))
        if self.checks:
            lines.append("Acceptance checks:")
            lines.extend(f"- {check}" for check in self.checks)
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


def parse_plan(payload: dict[str, Any] | None) -> Plan | str | None:
    """A `submit_plan` payload as a plan, as an answer (str), or None when it is unusable."""
    if payload is None:
        return None
    if payload.get("kind") == "answer":
        return str(payload.get("answer") or "").strip() or None
    files = []
    for item in payload.get("files") or []:
        if isinstance(item, dict) and str(item.get("path", "")).strip():
            files.append((str(item["path"]).strip(), str(item.get("change", "")).strip()))
    plan = Plan(
        approach=str(payload.get("approach") or "").strip(),
        files=files,
        steps=_strings(payload.get("steps")),
        checks=_strings(payload.get("checks")),
        notes=_strings(payload.get("notes")),
    )
    if not plan.approach and not plan.files and not plan.steps:
        return None
    return plan


def _strings(value: Any) -> list[str]:
    return [str(v).strip() for v in value or [] if str(v).strip()]


# --------------------------------------------------------------------------------------------
# The editor's line to the architect
# --------------------------------------------------------------------------------------------


class ConsultArchitectTool(Tool):
    """Lets the editor ask the architect a question mid-task and wait for a short answer."""

    name = "consult_architect"
    description = (
        "Ask the architect who planned this request a question, and wait for a short answer. "
        "Use it when you are stuck, or when the plan does not fit what you find in the code; "
        "include what you found (paths, errors). The architect reads code but does not "
        "change files. Only a few questions are allowed per request."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A self-contained question with the facts it depends on.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    # Not read-only: two questions in one turn must not run the architect concurrently.
    read_only = False

    def __init__(self, owner: ArchitectEditorOrchestrator) -> None:
        self._owner = owner

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"consult: {str(tool_input.get('question', ''))[:80]}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._owner.consult(tool_input["question"])


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class ArchitectEditorOrchestrator(Orchestrator):
    name = "architect-editor"
    title = "Architect-Editor"
    summary = "A strong model plans and answers questions; an editor model makes the changes."
    guidance = (
        "Helps weak, free or local models with work beyond them: a strong model explores and "
        "plans, and the editor can ask it questions. Costs about 0.8 to 1.2 times running the "
        "strong model alone. With a strong editor it adds cost and can lower accuracy (by up "
        "to about 3 points where measured), so prefer Solo there."
    )
    settings = (
        Setting(
            "architect-model",
            "model",
            "",
            "Model for the architect, which explores and plans but never edits. Empty uses "
            "the planner role's model, or the session's model. Pick your strongest reasoning "
            "model.",
        ),
        Setting(
            "editor-model",
            "model",
            "",
            "Model for the editor, which makes the changes and talks to you. Empty uses the "
            "session's model. Takes effect when a conversation starts.",
        ),
        Setting(
            "max-consults",
            "int",
            3,
            "Questions the editor may put to the architect per request.",
            minimum=0,
            maximum=10,
        ),
        Setting(
            "final-check",
            "bool",
            False,
            "After the editor finishes, review the changes against the plan in a fresh "
            "context on the architect's model, and hand blocking problems back once.",
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")
        #: The agent the user's mid-run messages go to while it, not the editor, is working.
        self._working: Agent | None = None
        self._architect: Agent | None = None
        self._architect_node: NodeInfo | None = None
        self._plan: Plan | None = None
        self._consults = 0
        #: Set when the run was stopped while the architect answered a question.
        self._stopped: ProviderError | None = None
        #: Requests the architect answered without the editor: (prompt, answer).
        self._unseen: list[tuple[str, str]] = []
        self._answer_ids = itertools.count(1)

    def steer(self, text: str) -> None:
        """Mid-run messages go to whichever agent is working on the request: while the plan is
        being made that is the architect, so the plan can take them into account."""
        target = self._working or self.root_agent
        if target is not None:
            target.steer(text)

    def _ensure_editor(self) -> Agent:
        if self.root_agent is None:
            session = self.session
            model = configured_model(session, "editor-model", self.setting("editor-model"), None)
            self.root_agent = session.agent(
                self._root,
                system=prompts.system_prompt(session, f"{prompts.USER_FACING}\n\n{EDITOR_ROLE}"),
                tools=[*standard_tools(), AskUserTool(session), ConsultArchitectTool(self)],
                model=model,
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        editor = self._ensure_editor()
        self._architect = self._architect_node = self._plan = self._stopped = None
        self._consults = 0
        architect_model = configured_model(
            session, "architect-model", self.setting("architect-model"), "planner"
        ) or session.model_for("planner")

        session.update_node(self._root, status="waiting")
        plan = await self._make_plan(prompt, architect_model)
        if isinstance(plan, str):
            return self._answer_directly(prompt, plan)

        session.update_node(self._root, status="running")
        message = self._editor_message(prompt, plan, architect_model, fresh=not editor.messages)
        result = await self._edit(editor, message)
        if result.stop == "refusal":
            return self._declined(result)
        footer = ""
        if self._plan is not None and self.setting("final-check") and run_diff(session):
            result, footer = await self._final_check(
                editor, prompt, self._plan, architect_model, result
            )
            if result.stop == "refusal":
                return self._declined(result)
        session.update_node(self._root, status="done", summary=self._summary(footer))
        return f"{result.text.rstrip()}\n\n{footer}" if footer else result.text

    # -- planning -------------------------------------------------------------------------------

    async def _make_plan(self, prompt: str, model: ModelRef) -> Plan | str | None:
        """Run the architect. Returns its plan, its answer to a question, or None when it
        produced neither (the editor then works without a plan)."""
        session = self.session
        node = session.new_node(
            "Plan", parent=self._root, role="planner", model=model, status="running"
        )
        agent = session.agent(
            node,
            system=prompts.system_prompt(session, ARCHITECT_ROLE),
            tools=[
                *read_only_tools(),
                FinishTool(
                    "submit_plan",
                    "Hand in your plan, or your answer when nothing is to be changed. Call "
                    "once, at the end.",
                    PLAN_SCHEMA,
                ),
            ],
            ctx=session.tool_context(node, write_scope=()),
            model=model,
        )
        self._working = agent
        try:
            result, payload = await run_to_finish(
                agent, self._architect_brief(prompt), "submit_plan"
            )
            plan = parse_plan(payload)
            if plan is None and payload is not None:
                result, payload = await run_to_finish(agent, INCOMPLETE, "submit_plan", reminders=1)
                plan = parse_plan(payload)
        except ProviderError as exc:
            session.update_node(
                node, status="failed", summary=_one_line(f"Could not reach its model: {exc}", 200)
            )
            raise
        finally:
            self._working = None

        if plan is None:
            why = (
                "The architect declined to plan this request"
                if result.stop == "refusal"
                else "The architect handed in no usable plan"
            )
            session.update_node(node, status="failed", summary=f"{why}.")
            session.notice("warning", f"{why}, so the editor works without one.", self._root.id)
            return None
        if isinstance(plan, str):
            session.update_node(node, status="done", summary="Answered; nothing to change.")
            return plan
        self._architect, self._architect_node, self._plan = agent, node, plan
        session.update_node(node, status="done", summary=plan.headline)
        return plan

    def _architect_brief(self, prompt: str) -> str:
        sections = []
        if recap := self.session.recap():
            sections.append(recap)
        sections.append(f"<request>\n{prompt.strip()}\n</request>")
        sections.append(
            "Explore the code this request involves, then call `submit_plan`: a plan for the "
            "editor, or the answer if the request asks for no change."
        )
        return "\n\n".join(sections)

    def _answer_directly(self, prompt: str, answer: str) -> str:
        """The architect answered a question. The exchange is shown on the conversation, and
        the editor, which did not take part, hears about it before its next request."""
        session = self.session
        session.emit(UserMessage(node_id=self._root.id, text=prompt))
        session.emit(
            AssistantMessage(
                node_id=self._root.id,
                message_id=f"architect-answer{next(self._answer_ids)}",
                text=answer,
            )
        )
        self._unseen.append((prompt, answer))
        session.update_node(
            self._root, status="done", summary="Answered by the architect; nothing to change."
        )
        return answer

    # -- editing --------------------------------------------------------------------------------

    def _editor_message(
        self, prompt: str, plan: Plan | None, architect: ModelRef, *, fresh: bool
    ) -> str:
        """The editor's message: the request, what it missed of the conversation, the plan.

        An editor that has never spoken gets the whole session's recap; otherwise it hears
        only about the requests the architect answered without it.
        """
        context = self.session.recap() if fresh else _unseen_recap(self._unseen)
        self._unseen.clear()
        parts = [f"{context}\n\nThe user now says:\n{prompt}" if context else prompt]
        if plan is not None:
            intro = (
                f"[Plan from the architect]\nAn architect agent ({architect}) explored the code "
                "and wrote the plan below for this request. Follow it, checking it against the "
                "code as you go; where a step turns out to be wrong, deviate and say why in "
                "your summary."
            )
            if self.setting("max-consults"):
                intro += (
                    " If you get stuck, or the plan does not fit what you find, ask the "
                    "architect with `consult_architect` rather than guess."
                )
            parts.append(f"{intro}\n\n<plan>\n{plan.render()}\n</plan>")
        return "\n\n".join(parts)

    async def _edit(self, editor: Agent, message: str) -> AgentResult:
        """Run the editor on `message`. A run stopped while the architect was answering one
        of its questions ends here, with the error that stopped it."""
        result = await editor.run(message)
        if self._stopped is not None:
            stopped, self._stopped = self._stopped, None
            raise stopped
        return result

    async def consult(self, question: str) -> ToolOutput:
        """Put the editor's question to the architect, within the per-request allowance."""
        architect, node, limit = self._architect, self._architect_node, self.setting("max-consults")
        if architect is None or node is None:
            return ToolOutput.error(
                "There is no architect for this request, because no plan was made. Continue "
                "with your own judgment."
            )
        if self._consults >= limit:
            return ToolOutput.error(
                f"No questions to the architect are left for this request ({limit} allowed). "
                "Continue with your own judgment, and mention open questions in your summary."
            )
        self._consults += 1
        session = self.session
        session.update_node(self._root, status="waiting")
        session.update_node(node, status="running")
        try:
            result = await architect.run(CONSULT.format(question=question.strip()))
        except ProviderError as exc:
            if not exc.provider:
                # Raised by the session, not by a model: the run reached its output budget
                # and the user chose to stop (or nobody could be asked). That ends the run,
                # not only this question. The agent loop treats an exception from a tool as
                # a failed call and carries on, which would put the budget question to the
                # user again, so the tool ends the editor's turn and `_edit` re-raises.
                self._stopped = exc
                return ToolOutput(f"[Not answered: {exc}]", is_error=True, finish={})
            session.notice("warning", f"The architect could not answer a question: {exc}", node.id)
            output = ToolOutput.error(
                f"The architect could not answer ({exc}). Continue with your own judgment."
            )
        else:
            output = self._consult_answer(result)
        # After the answer is read (it may revise the plan the summary describes), and never
        # on cancellation, so the session can mark the node cancelled.
        session.update_node(node, status="done", summary=self._architect_summary())
        session.update_node(self._root, status="running")
        return output

    def _consult_answer(self, result: AgentResult) -> ToolOutput:
        if result.stop == "refusal":
            return ToolOutput.error(
                "The architect declined to answer this question. Continue with your own judgment."
            )
        revised = parse_plan(result.payload) if result.stop == "finished" else None
        if isinstance(revised, Plan):
            self._plan = revised
            return ToolOutput(f"The architect revised the plan:\n{revised.render()}")
        text = revised or result.text.strip()
        if not text:
            return ToolOutput.error(
                "The architect gave no answer. Continue with your own judgment."
            )
        return ToolOutput(text)

    async def _final_check(
        self, editor: Agent, prompt: str, plan: Plan, model: ModelRef, result: AgentResult
    ) -> tuple[AgentResult, str]:
        """Review the changes against the request and the plan; hand blocking findings back
        to the editor once. Returns the editor's latest result and a line for the user."""
        session = self.session
        session.update_node(self._root, status="waiting")
        task = (
            f"{prompt.strip()}\n\nThe changes were meant to follow this plan. Check them "
            f"against it as well as against the request:\n{plan.render()}"
        )
        outcome = await review(
            session,
            parent=self._root,
            task=task,
            diff=run_diff(session),
            model=model,
            title="Check against the plan",
        )
        label = f"Check against the plan ({model})"
        if not outcome.completed:
            return result, f"{label}: no result was handed in, so the changes are unchecked."
        if not outcome.blocking:
            return result, f"{label}: no blocking problems."
        session.update_node(self._root, status="integrating")
        result = await self._edit(
            editor, findings_message(Review(True, outcome.blocking, outcome.node_id))
        )
        return result, (
            f"{label}: {len(outcome.blocking)} blocking finding(s) handed back to the editor; "
            "its fixes were not checked again."
        )

    # -- summaries ------------------------------------------------------------------------------

    def _architect_summary(self) -> str:
        headline = self._plan.headline if self._plan is not None else "Plan made."
        return f"{headline} Answered {self._consults} question(s) from the editor."

    def _summary(self, footer: str) -> str:
        if self._plan is None:
            text = "Worked without a plan."
        else:
            text = f"Worked from the architect's plan; {self._consults} question(s) asked."
        return _one_line(f"{text} {footer}".strip(), 200)

    def _declined(self, result: AgentResult) -> str:
        self.session.update_node(
            self._root, status="failed", summary="The model declined the request."
        )
        return f"The model declined this request. {result.detail}".strip()


def _unseen_recap(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return ""
    lines = ["Earlier in this session, the architect answered these requests without you:"]
    for prompt, answer in turns[-UNSEEN_LIMIT:]:
        shown = answer if len(answer) <= 1500 else answer[:1500] + " [...]"
        lines.append(f"\nThe user asked:\n{prompt}\n\nThe answer given:\n{shown}")
    return "\n".join(lines)


def _one_line(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

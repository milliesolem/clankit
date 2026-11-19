"""Best of N: several independent attempts at a change, and the best-evidenced one is applied.

The agent that talks to the user answers questions itself. For a change it calls
`build_candidates`: N fresh agents each build the change in their own git worktree, the
harness tests the results, a selector picks one, and only that one reaches the workspace.

What the evidence says, and how it shapes this module:

* Sampling several attempts and selecting well gains roughly 5 to 8 points at N = 3 to 5 on
  SWE-bench-style tasks, at about N x 1.25-1.4 the cost of one attempt. The gain per extra
  attempt falls fast, so N is capped at 8 and the projected cost is shown before each round.
* Candidates must not see each other's work, so each gets a worktree made from the current
  working tree, uncommitted changes included. They are diversified by an approach hint in
  the prompt, not by sampling parameters, which current models reject or ignore.
* Execution evidence beats opinion. A regression check prunes candidates (all are kept when
  all fail it), duplicate diffs are merged (about 29% of candidates in one study), and a
  separate agent writes tests that tell a correct change from a wrong one, which the harness
  runs against every candidate. Generated tests are often wrong, so they rank and inform
  rather than prune.
* Selection is the weak point of the method: a judge reading diffs alone is the weakest
  selector, and judges favour their own model family and are swayed by surface features.
  The selector sees each candidate's diff with the harness's results, never transcripts or
  self-assessments (claims, not evidence); it runs on another model family when one is
  configured; candidates are shown in a different order to each voter, and with several
  votes the majority decides, stopping as soon as one candidate has it.
* The chosen change is applied with git, every file checkpointed first, so undo reverts it,
  and all worktrees are removed afterwards, also after a failure or a cancellation. Each
  file it touches goes through the permission gate first, as an edit would: a candidate may
  have changed its copy with shell commands, which the edit rules never saw.

Without git the candidates run one after another in the workspace itself, the harness
restoring what they changed with the file tools in between (shell side effects are not
restored, and generated tests are skipped). When the workspace is in a repository but its
state cannot be captured, the round is refused instead: something is wrong with the
repository, and running candidates in the user's files would only add to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from collections import Counter
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.events import TERMINAL_STATUSES, NodeInfo
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
    run_check,
    run_diff,
    run_to_finish,
    subtree_ids,
)
from clankit.orchestration.worktree import (
    Snapshot,
    Worktree,
    WorktreeError,
    WorktreeSet,
    apply_patch,
    capture_files,
    change_key,
    diff,
    patch_paths,
    patch_stat,
    readable,
    repository_root,
    restore_files,
    run_check_in,
    system_prompt,
    take_snapshot,
    tool_context,
    write_files,
)
from clankit.paths import state_dir
from clankit.tools import standard_tools
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput

if TYPE_CHECKING:
    from clankit.agent.session import Session

#: Rough cost of one round relative to one attempt, per candidate (tests and selection).
COST_FACTOR = 1.3
#: Distinct candidates the selector compares; the rest are ranked out by execution evidence.
SHORTLIST = 3
#: Characters of diff shown to the selector, shared among the shortlisted candidates.
DIFF_BUDGET = 60_000
#: Characters of the chosen diff shown with a request for permission to apply it.
PERMISSION_DIFF = 16_000
LABELS = "ABCDEFGH"

#: Approach hints for candidates B onwards; candidate A works the way it judges best.
DEFAULT_HINTS = (
    "Make the smallest change that fully satisfies the request, and keep the diff easy to review.",
    "Start by writing or finding a test that pins down the required behaviour, then make it pass.",
    "Before changing anything, study how the codebase already solves similar problems, and "
    "follow that pattern closely.",
    "Look for the root cause rather than the symptom: trace the data flow end to end before "
    "deciding where to change it.",
    "Think through edge cases and failure paths first (empty input, errors, limits), and make "
    "sure the change handles them.",
    "Decide how the change should look to its callers first, then implement it from the "
    "interface inward.",
    "Re-read the request for anything easy to miss, and verify each requirement separately "
    "before you finish.",
)

# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------

ROOT_ROLE = """\
# Your role

You are working directly with the user. Their message is the task. You do not change files \
yourself in this mode: changes are built by several independent candidates, and the best one \
is applied.

- If the message is a question, or the user is describing a problem or thinking aloud rather \
than asking for a change, answer it yourself: investigate with your tools and report what \
you found.
- If it asks for a change to the project, call `build_candidates` once, with a description \
of the change that stands on its own. Each candidate is a fresh agent that sees only that \
description and the user's message, so include what it needs from this conversation: \
decisions made, constraints, files involved. Investigate first only as far as you need to \
describe the change well.
- When `build_candidates` returns, tell the user what happened from its report: what was \
applied and why that candidate was chosen, how it was checked and tested, which files \
changed, and anything that failed or is left open. Say plainly if nothing was applied.
- Use `ask_user` only for a decision that is genuinely the user's and that would lead to \
materially different work."""

CANDIDATE_ROLE = """\
# Your role

You are building a change to a software project on your own. Several engineers build the \
same change independently and the harness applies the best result to the user's project, \
judged by running it and by reading the diff; nobody reads your summary to decide.

{where}
- Build the complete change and verify it with the project's own tests or checks. Nobody \
can answer questions while you work: make reasonable decisions and mention them in your \
summary.
- Leave git alone: do not commit, stash, switch branches or change git settings. The \
harness reads your changes from the files themselves."""

IN_WORKTREE = """\
- The working directory named above is a private copy of the user's project, made for you \
from their current files. Work only there: relative paths resolve against it and shell \
commands start in it."""

IN_PLACE = """\
- You are working in the user's project itself. Candidates take turns, and after each one \
the harness restores the files it changed. Make every file change with `edit` and `write`: \
changes made by shell commands cannot be restored and would leak into the next candidate's \
work."""

TEST_WRITER_ROLE = """\
# Your role

Several engineers are each building the change described in the first message, in their own \
copies of the project. Your job is different: write tests that tell a correct implementation \
apart from an incorrect or missing one. The harness runs your tests against every candidate, \
and the results help decide which one is applied.

- Test the behaviour the request asks for through the project's public interfaces, with its \
existing test framework. Do not implement the change itself.
- Put the tests in new files with distinctive names (include "distinguishing" in the file \
name), so they do not clash with files the candidates add, and do not edit existing files.
- The change does not exist yet in your copy, so your tests should fail there. Run them and \
make sure they fail for the expected reason (missing behaviour), not because of a mistake in \
the test.
- Keep them few and decisive: a handful of tests on the core of the request and its most \
likely mistakes.
- Finish by calling `submit_tests` with the shell command that runs only your new tests."""

SELECTOR_ROLE = """\
# Your role

Several candidates independently built the same change, each in its own copy of the \
project. Choose the one to apply to the user's project.

Judge by evidence, in this order:
1. The harness's execution results: the regression check and the generated tests. A \
recorded result outranks an impression from reading code. Generated tests can be wrong \
themselves, so read a failing test's output before holding it against a candidate.
2. Whether the diff does everything the request asks, correctly and including edge cases, \
and nothing it does not ask for.
3. Risk: a change that is harder to review or likelier to break something else counts \
against a candidate.

The order in which candidates are listed means nothing, and neither does the length of a \
diff. You can read the project's current files to understand the context; do not change \
anything. Finish by calling `choose` exactly once."""

BUILD_DESCRIPTION = (
    "Have several independent agents build a change to the project, each in its own copy; "
    "the harness tests the results and applies the best one to the workspace, then returns a "
    "report of what was applied and why. For changes only (answer questions yourself), and "
    "at most once per request."
)

TESTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command that runs only your new tests, from the project root.",
        },
        "covers": {
            "type": "string",
            "description": "What behaviour the tests check, in one or two sentences.",
        },
    },
    "required": ["command", "covers"],
    "additionalProperties": False,
}

NO_GIT = (
    "This workspace is not in a git repository, so the candidates cannot work side by side in "
    "isolated copies. They will run one after another in the workspace itself, and before each "
    "next one starts the harness restores the files the last one changed with the edit and "
    "write tools; changes made by shell commands are not undone."
)


def _choice_schema(labels: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate": {"type": "string", "enum": list(labels)},
            "justification": {
                "type": "string",
                "description": "The evidence that decided it, in a few sentences.",
            },
        },
        "required": ["candidate", "justification"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class GeneratedTests:
    """Tests written to tell candidates apart, and how they did on the unchanged code."""

    command: str
    covers: str
    patch: str
    baseline: CheckResult
    files: list[str] = field(default_factory=list)

    @property
    def discriminating(self) -> bool:
        """Tests of new behaviour fail before the change; ones that pass cannot show it."""
        return not self.baseline.passed

    def describe(self) -> str:
        before = (
            "failed, as tests of new behaviour should"
            if self.discriminating
            else "passed, so they cannot show whether the change was made"
        )
        return (
            f"A separate agent wrote tests to tell a correct change from a wrong one "
            f"({self.covers.strip().rstrip('.')}). The harness added them to each candidate's "
            f"copy and ran `{self.command}`. On the unchanged code they {before}."
        )


@dataclass(frozen=True, slots=True)
class Slot:
    """A candidate before it starts: its label, approach hint and node on the tree."""

    label: str
    hint: str
    node: NodeInfo


@dataclass(slots=True)
class Candidate:
    """One attempt, as the harness recorded it. `tree` is its worktree (None without git).
    `problem` says why it produced nothing usable; `twins` are the labels of later
    candidates that made the same change."""

    label: str
    hint: str
    node: NodeInfo
    ctx: ToolContext
    tree: Worktree | None = None
    patch: str = ""
    key: str = ""
    files: list[str] = field(default_factory=list)
    contents: dict[Path, bytes | None] = field(default_factory=dict)
    check: CheckResult | None = None
    tests: CheckResult | None = None
    tests_problem: str = ""
    problem: str = ""
    twins: list[str] = field(default_factory=list)
    twin_of: str = ""

    @property
    def passed_check(self) -> bool:
        return self.check is None or self.check.passed or self.check.denied

    @property
    def tests_rank(self) -> int:
        """0 when the generated tests passed, 1 when there were none, 2 when they failed."""
        if self.tests is not None:
            return 0 if self.tests.passed else 2
        return 2 if self.tests_problem else 1

    def describe(self) -> str:
        """The harness's results in a few words, for the tree and the report."""
        if self.problem:
            return self.problem
        count = len(self.files)
        parts = [f"{count} file{'s' if count != 1 else ''} changed"]
        if self.check is not None:
            parts.append(_check_words("check", self.check))
        if self.tests is not None:
            parts.append(_check_words("generated tests", self.tests))
        elif self.tests_problem:
            parts.append("generated tests could not be added")
        return "; ".join(parts)

    def evidence(self, budget: int, total: int) -> str:
        """Everything the selector sees about this candidate: results, then the diff."""
        lines = [f'<candidate name="{self.label}">']
        if self.twins:
            makers = ", ".join([self.label, *self.twins])
            lines.append(
                f"Candidates {makers} made this same change ({len(self.twins) + 1} of {total})."
            )
        lines.append("Files changed: " + ("; ".join(self.files) or "none"))
        if self.check is not None:
            lines.append("Regression check:\n" + self.check.render(tail=1500))
        if self.tests is not None:
            lines.append("Generated tests:\n" + self.tests.render(tail=1500))
        elif self.tests_problem:
            lines.append(f"Generated tests: {self.tests_problem}")
        shown = readable(self.patch)
        if len(shown) > budget:
            shown = shown[:budget] + f"\n[diff truncated: {len(shown) - budget:,} more characters]"
        lines.append(f"<diff>\n{shown.rstrip()}\n</diff>\n</candidate>")
        return "\n".join(lines)


def _check_words(name: str, result: CheckResult) -> str:
    if result.denied:
        return f"{name} not run (permission denied)"
    if result.passed:
        return f"{name} passed"
    if result.exit_code is None:
        return f"{name} timed out"
    return f"{name} failed (exit code {result.exit_code})"


def _family(ref: ModelRef) -> str:
    """A rough model family from the model id: "claude-opus-5-5" and "anthropic/claude-x"
    give "claude", "gpt-5.1" gives "gpt", "qwen/qwen3-coder" gives "qwen"."""
    name = ref.model.rsplit("/", 1)[-1].lower()
    match = re.match(r"[a-z]+", name)
    return match.group(0) if match else name


def _binary_digest(contents: dict[Path, bytes | None]) -> str:
    """A digest of the files that a text diff shows only as "Binary file changed"."""
    digest = hashlib.sha256()
    for path in sorted(contents):
        data = contents[path]
        if data is None:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            digest.update(str(path).encode("utf-8", "surrogateescape") + b"\0" + data)
    return digest.hexdigest()


def _one_line(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _together(jobs: Sequence[Awaitable[Any]]) -> list[Any]:
    """Run `jobs` concurrently. If one fails or the caller is cancelled, the others are
    cancelled and waited for before the exception propagates."""
    tasks = [asyncio.ensure_future(job) for job in jobs]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# --------------------------------------------------------------------------------------------
# The tool the user-facing agent calls
# --------------------------------------------------------------------------------------------


class BuildCandidatesTool(Tool):
    name = "build_candidates"
    description = BUILD_DESCRIPTION
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The change to make, self-contained: what the user asked for, "
                "plus every decision, constraint and file from this conversation that a fresh "
                "engineer needs.",
            }
        },
        "required": ["task"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(self, orchestrator: BestOfNOrchestrator) -> None:
        self._orchestrator = orchestrator

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"build candidates: {_one_line(str(tool_input.get('task', '')), 70)}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._orchestrator.build(tool_input["task"])


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class BestOfNOrchestrator(Orchestrator):
    name = "best-of-n"
    title = "Best of N"
    summary = "Several agents build the change independently; the best-evidenced one is applied."
    guidance = (
        "For hard, valuable changes where a wrong result is expensive. Each change is built by "
        "N agents in isolated git worktrees, tested, and the one a selector picks from the "
        "execution evidence is applied. It costs about N x 1.3 the cost of one attempt (3.9x "
        "at the default of 3), so it is a poor fit for routine edits; questions are answered "
        "by one agent. Outside a git repository the candidates run one after another."
    )
    settings = (
        Setting(
            "n",
            "int",
            3,
            "Candidates built for each change. Each is a full attempt, so the cost grows with it.",
            minimum=2,
            maximum=8,
        ),
        Setting(
            "check",
            "string",
            "",
            "Shell command run in each candidate's copy, such as the test suite. Candidates "
            "that fail it are set aside, unless all of them do. Empty skips this step.",
        ),
        Setting(
            "write-tests",
            "bool",
            True,
            "Have one more agent write tests that tell a correct change from a wrong one, and "
            "run them against every candidate.",
        ),
        Setting(
            "selector-votes",
            "int",
            3,
            "Independent selections; the majority decides, and voting stops as soon as one "
            "candidate has it. An odd number avoids ties.",
            minimum=1,
            maximum=7,
        ),
        Setting(
            "approach-hints",
            "string",
            "",
            "Approaches handed to the candidates in turn so that they differ, separated by "
            "commas, e.g. 'smallest possible change, test first, follow existing patterns'. "
            "Empty uses built-in hints.",
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")
        self._prompt = ""
        self._built = False

    # -- the conversation -----------------------------------------------------------------------

    def _ensure_root(self) -> Agent:
        if self.root_agent is None:
            session = self.session
            tools: list[Tool] = [
                *standard_tools(("read", "glob", "grep", "bash", "web_fetch", "skill")),
                AskUserTool(session),
                BuildCandidatesTool(self),
            ]
            self.root_agent = session.agent(
                self._root,
                system=prompts.system_prompt(session, ROOT_ROLE),
                tools=tools,
                ctx=session.tool_context(self._root, write_scope=()),
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        fresh = self.root_agent is None
        agent = self._ensure_root()
        self._prompt, self._built = prompt, False
        if fresh and (recap := session.recap()):
            prompt = f"{recap}\n\nThe user now says:\n{prompt}"
        session.update_node(self._root, status="running")
        result = await agent.run(prompt)
        if result.stop == "refusal":
            session.update_node(self._root, status="failed")
            return f"The model declined this request. {result.detail}".strip()
        session.update_node(self._root, status="done")
        return result.text

    # -- a round of candidates ------------------------------------------------------------------

    async def build(self, task: str) -> ToolOutput:
        """Build the change N times, choose one, apply it; the report goes to the root agent."""
        task = task.strip()
        if self._built:
            return ToolOutput.error(
                "Candidates were already built for this request. Report that outcome to the "
                "user; they can ask for another round."
            )
        if len(task) < 15:
            return ToolOutput.error("Describe the change in enough detail to act on it.")
        self._built = True
        session = self.session
        n = self.setting("n")
        session.notice(
            "info",
            f"Best of N: building {n} candidates, about {n} x {COST_FACTOR} the cost of one "
            "attempt.",
            self._root.id,
        )
        try:
            return await self._round(task)
        except (ProviderError, WorktreeError) as exc:
            # Like a failed subtask in the task tree: the agent that talks to the user hears
            # of it. A lasting cause (budget, key) fails the run at that agent's next call.
            self._abandon_round()
            cause = "a model call failed" if isinstance(exc, ProviderError) else "git failed"
            return ToolOutput.error(
                f"The round stopped because {cause}: {exc} Nothing in the workspace was changed."
            )
        finally:
            session.update_node(self._root, status="running")

    async def _round(self, task: str) -> ToolOutput:
        session = self.session
        if await repository_root(session.workspace) is None:
            tests_note = " Generated tests are skipped." if self.setting("write-tests") else ""
            session.notice("warning", NO_GIT + tests_note, self._root.id)
            return ToolOutput(await self._in_place(task))
        try:
            snapshot = await take_snapshot(session.workspace)
        except WorktreeError as exc:
            return ToolOutput.error(
                "Nothing was built or changed: the current state of the repository could not "
                f"be copied for the candidates. {exc}"
            )
        return ToolOutput(await self._isolated(task, snapshot))

    def _abandon_round(self) -> None:
        """Close the round's nodes that a failure left open."""
        for node in self.session.children_of(self._root.id):
            if node.status not in TERMINAL_STATUSES:
                self.session.update_node(node, status="cancelled")

    def _slots(self) -> list[Slot]:
        session = self.session
        n = self.setting("n")
        configured = [h.strip() for h in self.setting("approach-hints").split(",") if h.strip()]
        if configured:
            hints = [configured[i % len(configured)] for i in range(n)]
        else:
            hints = ["", *DEFAULT_HINTS][:n]
        model = session.model
        return [
            Slot(
                label,
                hint,
                session.new_node(
                    f"Candidate {label}",
                    parent=self._root,
                    role="candidate",
                    model=model,
                    brief=hint,
                ),
            )
            for label, hint in zip(LABELS, hints, strict=False)
        ]

    async def _isolated(self, task: str, snapshot: Snapshot) -> str:
        session = self.session
        slots = self._slots()
        tests_node = (
            session.new_node(
                "Distinguishing tests", parent=self._root, role="worker", model=session.model
            )
            if self.setting("write-tests")
            else None
        )
        session.update_node(self._root, status="waiting")
        async with WorktreeSet(snapshot) as trees:
            jobs: list[Awaitable[Any]] = [
                self._build_in_worktree(slot, task, trees) for slot in slots
            ]
            if tests_node is not None:
                jobs.append(self._write_tests(tests_node, task, trees))
            outcomes = await _together(jobs)
            candidates: list[Candidate] = outcomes[: len(slots)]
            tests: GeneratedTests | None = outcomes[-1] if tests_node is not None else None
            if tests is not None:
                await _together(
                    [
                        self._run_tests(c, c.tree, tests)
                        for c in candidates
                        if c.tree is not None and not c.problem
                    ]
                )
            session.update_node(self._root, status="integrating")
            chosen, how = await self._decide(task, candidates, tests)
            applied, failure = await self._apply(chosen, snapshot) if chosen else ([], "")
        self._conclude(candidates, chosen, failure)
        return self._report(candidates, tests, chosen, how, applied, failure)

    async def _in_place(self, task: str) -> str:
        session = self.session
        slots = self._slots()
        session.update_node(self._root, status="waiting")
        candidates = [await self._build_in_place(slot, task) for slot in slots]
        session.update_node(self._root, status="integrating")
        chosen, how = await self._decide(task, candidates, None)
        applied, failure = await self._write(chosen) if chosen else ([], "")
        self._conclude(candidates, chosen, failure)
        return self._report(candidates, None, chosen, how, applied, failure)

    # -- one candidate --------------------------------------------------------------------------

    def _candidate_brief(self, slot: Slot, task: str, where: str) -> str:
        line = f"You are candidate {slot.label}; others are building the same change."
        if slot.hint:
            line += f" Approach to take: {slot.hint}"
        sections = [
            f"<user_request>\n{self._prompt.strip()}\n</user_request>",
            f"<task>\n{task}\n</task>",
            line,
        ]
        check = self.setting("check").strip()
        if check:
            sections.append(
                f"After you finish, the harness runs `{check}` in {where}; a candidate that "
                "fails it is set aside."
            )
        sections.append(
            "When the change is complete and verified, end your turn with a short summary of "
            "what you changed."
        )
        return "\n\n".join(sections)

    async def _work(self, candidate: Candidate, agent: Agent, brief: str) -> bool:
        """Run the candidate's agent. False when it declined the task."""
        session = self.session
        try:
            result = await agent.run(brief)
        except ProviderError as exc:
            session.update_node(
                candidate.node, status="failed", summary=_one_line(f"model error: {exc}")
            )
            raise
        if result.stop == "refusal":
            candidate.problem = "the model declined the task"
            return False
        session.update_node(candidate.node, status="integrating")
        return True

    async def _build_in_worktree(self, slot: Slot, task: str, trees: WorktreeSet) -> Candidate:
        session = self.session
        session.update_node(slot.node, status="running")
        tree = await trees.add(f"candidate-{slot.label}")
        candidate = Candidate(
            slot.label, slot.hint, slot.node, tool_context(session, slot.node, tree), tree
        )
        agent = session.agent(
            slot.node,
            system=system_prompt(session, tree, CANDIDATE_ROLE.format(where=IN_WORKTREE)),
            tools=standard_tools(),
            ctx=candidate.ctx,
            model=session.model,
        )
        if await self._work(candidate, agent, self._candidate_brief(slot, task, "your copy")):
            candidate.patch = await diff(tree)
            if candidate.patch.strip():
                candidate.key = change_key(candidate.patch)
                candidate.files = await patch_stat(candidate.patch, tree.snapshot)
                await self._check(candidate)
            else:
                candidate.problem = "it made no changes"
        self._settle(candidate)
        return candidate

    async def _build_in_place(self, slot: Slot, task: str) -> Candidate:
        session = self.session
        session.update_node(slot.node, status="running")
        candidate = Candidate(slot.label, slot.hint, slot.node, session.tool_context(slot.node))
        agent = session.agent(
            slot.node,
            system=prompts.system_prompt(session, CANDIDATE_ROLE.format(where=IN_PLACE)),
            tools=standard_tools(),
            ctx=candidate.ctx,
            model=session.model,
        )
        try:
            brief = self._candidate_brief(slot, task, "the project")
            if await self._work(candidate, agent, brief):
                paths = changed_paths(session, subtree_ids(session, slot.node.id))
                candidate.patch = run_diff(session, paths)
                if candidate.patch.strip():
                    candidate.files = diff_stat(session, paths)
                    candidate.contents = capture_files(paths)
                    candidate.key = change_key(candidate.patch) + _binary_digest(candidate.contents)
                    await self._check(candidate)
                else:
                    candidate.problem = "it made no changes"
        finally:
            self._restore(candidate)
        self._settle(candidate)
        return candidate

    def _restore(self, candidate: Candidate) -> None:
        """Without git: put back what the candidate changed, so the next one starts clean."""
        session = self.session
        paths = changed_paths(session, subtree_ids(session, candidate.node.id))
        _, lost = restore_files(session.files, paths)
        if lost:
            names = ", ".join(candidate.ctx.relative(path) for path in lost)
            session.notice(
                "warning",
                f"Could not restore {names} after candidate {candidate.label}; later "
                "candidates see it as that one left it.",
                candidate.node.id,
            )

    async def _execute(self, candidate: Candidate, command: str) -> CheckResult:
        """Run a command in the candidate's copy (or the workspace, without git)."""
        if candidate.tree is not None:
            return await run_check_in(self.session, candidate.ctx, candidate.tree, command)
        return await run_check(self.session, candidate.ctx, command)

    async def _check(self, candidate: Candidate) -> None:
        check = self.setting("check").strip()
        if check:
            candidate.check = await self._execute(candidate, check)

    def _settle(self, candidate: Candidate) -> None:
        """Mark the candidate by its own results; selection comes later."""
        ok = not candidate.problem and candidate.passed_check
        self.session.update_node(
            candidate.node,
            status="done" if ok else "failed",
            summary=_one_line(candidate.describe()),
        )

    # -- generated tests ------------------------------------------------------------------------

    async def _write_tests(
        self, node: NodeInfo, task: str, trees: WorktreeSet
    ) -> GeneratedTests | None:
        session = self.session
        session.update_node(node, status="running")
        tree = await trees.add("tests")
        ctx = tool_context(session, node, tree)
        tools = [
            *standard_tools(),
            FinishTool(
                "submit_tests", "Submit the tests you wrote. Call once, at the end.", TESTS_SCHEMA
            ),
        ]
        agent = session.agent(
            node,
            system=system_prompt(session, tree, TEST_WRITER_ROLE),
            tools=tools,
            ctx=ctx,
            model=session.model,
        )
        brief = (
            f"<user_request>\n{self._prompt.strip()}\n</user_request>\n\n<task>\n{task}\n</task>"
            "\n\nWrite tests that tell a correct implementation of this task apart from an "
            "incorrect or missing one, then call `submit_tests`."
        )
        try:
            _, payload = await run_to_finish(agent, brief, "submit_tests")
        except ProviderError as exc:
            session.update_node(node, status="failed", summary=_one_line(f"model error: {exc}"))
            raise
        submitted = payload or {}
        command = str(submitted.get("command") or "").strip()
        if not command:
            session.update_node(node, status="failed", summary="No tests were submitted.")
            return None
        session.update_node(node, status="integrating")
        patch = await diff(tree)
        if not patch.strip():
            session.update_node(node, status="failed", summary="No test files were written.")
            return None
        tests = GeneratedTests(
            command=command,
            covers=str(submitted.get("covers") or ""),
            patch=patch,
            baseline=await run_check_in(session, ctx, tree, command),
            files=await patch_stat(patch, tree.snapshot),
        )
        before = "fail" if tests.discriminating else "pass"
        session.update_node(
            node,
            status="done",
            summary=_one_line(f"{', '.join(tests.files)}; they {before} on the unchanged code"),
        )
        return tests

    async def _run_tests(self, candidate: Candidate, tree: Worktree, tests: GeneratedTests) -> None:
        """Add the generated tests to the candidate's copy and run them there. The candidate's
        own diff was taken before, so the tests never become part of its change."""
        try:
            await apply_patch(tests.patch, tree.root)
        except WorktreeError as exc:
            candidate.tests_problem = f"the generated tests could not be added to it ({exc})"
        else:
            candidate.tests = await self._execute(candidate, tests.command)
        self._settle(candidate)

    # -- selection ------------------------------------------------------------------------------

    async def _decide(
        self, task: str, candidates: list[Candidate], tests: GeneratedTests | None
    ) -> tuple[Candidate | None, str]:
        """Prune, merge duplicates, rank, and select. Returns the choice and how it was made."""
        viable = [c for c in candidates if not c.problem]
        if not viable:
            return None, "no candidate produced a change."
        passing = [c for c in viable if c.passed_check]
        pool = passing or viable
        distinct: dict[str, Candidate] = {}
        for candidate in pool:
            first = distinct.setdefault(candidate.key, candidate)
            if first is not candidate:
                first.twins.append(candidate.label)
                candidate.twin_of = first.label
        ranked = sorted(
            distinct.values(),
            key=lambda c: (not c.passed_check, c.tests_rank, -len(c.twins), c.label),
        )
        caveat = (
            " Every candidate failed the check, so all of them were considered."
            if not passing
            else ""
        )
        if len(ranked) == 1:
            return ranked[0], f"it was the only distinct change left.{caveat}"
        chosen, how = await self._vote(task, ranked[:SHORTLIST], tests, len(candidates))
        return chosen, how + caveat

    async def _vote(
        self,
        task: str,
        shortlist: list[Candidate],
        tests: GeneratedTests | None,
        total: int,
    ) -> tuple[Candidate, str]:
        votes = self.setting("selector-votes")
        majority = votes // 2 + 1
        model = self._selector_model()
        tally: Counter[str] = Counter()
        reasons: dict[str, str] = {}
        for number in range(1, votes + 1):
            turn = (number - 1) % len(shortlist)
            order = shortlist[turn:] + shortlist[:turn]  # every voter sees a different order
            choice = await self._select_once(task, order, tests, total, model, number, votes)
            if choice is None:
                continue
            label, why = choice
            tally[label] += 1
            reasons.setdefault(label, why)
            if tally[label] >= majority:
                break
        if not tally:
            return (
                shortlist[0],
                "no selector made a choice, so the candidate with the strongest execution "
                "evidence was taken.",
            )
        best = max(shortlist, key=lambda c: (tally[c.label], -shortlist.index(c)))
        cast = sum(tally.values())
        return (
            best,
            f"{tally[best.label]} of {cast} selector vote{'s' if cast > 1 else ''} ({model}) "
            f"chose it: {reasons[best.label]}",
        )

    def _selector_model(self) -> ModelRef:
        """The judge model, unless it is the candidates' family and another configured role
        model is not: judges favour output from their own family."""
        session = self.session
        roles = session.config.models.roles
        generator = _family(session.model)
        configured = [
            session.model_for(r) for r in ("judge", "reviewer", "planner") if roles.get(r)
        ]
        other = [ref for ref in configured if _family(ref) != generator]
        return other[0] if other else session.model_for("judge")

    async def _select_once(
        self,
        task: str,
        order: list[Candidate],
        tests: GeneratedTests | None,
        total: int,
        model: ModelRef,
        number: int,
        votes: int,
    ) -> tuple[str, str] | None:
        session = self.session
        title = "Selection" if votes == 1 else f"Selection vote {number} of {votes}"
        node = session.new_node(
            title, parent=self._root, role="judge", model=model, status="running"
        )
        labels = [c.label for c in order]
        tools = [
            *standard_tools(("read", "glob", "grep", "skill")),
            FinishTool(
                "choose",
                "Choose the candidate to apply. Call once, at the end.",
                _choice_schema(labels),
            ),
        ]
        agent = session.agent(
            node,
            system=prompts.system_prompt(session, SELECTOR_ROLE),
            tools=tools,
            ctx=session.tool_context(node, write_scope=()),
            model=model,
        )
        try:
            _, payload = await run_to_finish(
                agent, self._selection_brief(task, order, tests, total), "choose"
            )
        except ProviderError as exc:
            session.update_node(node, status="failed", summary=_one_line(f"model error: {exc}"))
            raise
        label = str((payload or {}).get("candidate") or "")
        if label not in labels:
            session.update_node(node, status="failed", summary="No candidate was chosen.")
            return None
        why = str((payload or {}).get("justification") or "").strip() or "(no reason given)"
        session.update_node(node, status="done", summary=_one_line(f"chose {label}: {why}"))
        return label, why

    def _selection_brief(
        self, task: str, order: list[Candidate], tests: GeneratedTests | None, total: int
    ) -> str:
        sections = [
            f"<user_request>\n{self._prompt.strip()}\n</user_request>",
            f"<task>\n{task}\n</task>",
        ]
        facts = []
        check = self.setting("check").strip()
        if check:
            facts.append(f"The regression check `{check}` was run by the harness for each one.")
        if tests is not None:
            facts.append(tests.describe())
        if facts:
            sections.append(" ".join(facts))
        budget = DIFF_BUDGET // len(order)
        sections.extend(c.evidence(budget, total) for c in order)
        labels = ", ".join(c.label for c in order)
        sections.append(f"Choose the candidate to apply ({labels}), then call `choose`.")
        return "\n\n".join(sections)

    # -- applying the choice --------------------------------------------------------------------

    async def _apply(self, chosen: Candidate, snapshot: Snapshot) -> tuple[list[str], str]:
        """Apply the chosen patch to the workspace. Returns (files, failure)."""
        session = self.session
        refused = await self._refused(chosen, await patch_paths(chosen.patch, snapshot.repo))
        if refused:
            return [], self._unapplied(chosen, f"permission to change {refused} was denied.")
        try:
            await apply_patch(
                chosen.patch, snapshot.repo, files=session.files, node_id=chosen.node.id
            )
        except WorktreeError as exc:
            reason = f"{exc} Files in the workspace probably changed while the candidates worked."
            return [], self._unapplied(chosen, reason)
        return chosen.files, ""

    async def _write(self, chosen: Candidate) -> tuple[list[str], str]:
        """Without git: write the chosen candidate's files back. Returns (files, failure)."""
        refused = await self._refused(chosen, list(chosen.contents))
        if refused:
            return [], f"permission to change {refused} was denied, so nothing was written."
        try:
            write_files(self.session.files, chosen.contents, chosen.node.id)
        except OSError as exc:
            return [], f"writing its files failed ({exc}); /undo restores what was written."
        return chosen.files, ""

    async def _refused(self, chosen: Candidate, paths: list[Path]) -> str:
        """Ask the permission gate about every file the chosen change touches, as the file
        tools would for an edit. A candidate can change files in its copy in ways the edit
        rules never saw (shell commands), so this is where those rules are enforced. Returns
        the first file refused, or ""."""
        session = self.session
        detail = readable(chosen.patch)[:PERMISSION_DIFF]
        for path in paths:
            try:
                target, inside = path.relative_to(session.workspace).as_posix(), True
            except ValueError:
                target, inside = str(path), False
            ask = PermissionAsk(
                "edit",
                f"Apply candidate {chosen.label}'s change to {target}",
                target,
                detail=detail,
                session_rule="*" if inside else target,
            )
            decision = await session.permissions.check(ask, node_id=chosen.node.id, tool="apply")
            if not decision.allowed:
                return target
        return ""

    def _unapplied(self, chosen: Candidate, reason: str) -> str:
        saved = self._save(chosen)
        return (
            f"{reason} Nothing in the workspace was changed. The change is saved in {saved}, "
            f"and `git apply {saved}` applies it."
        )

    def _save(self, chosen: Candidate) -> Path:
        directory = state_dir() / "patches"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.session.id}-candidate-{chosen.label.lower()}-{secrets.token_hex(3)}.patch"
        path = directory / name
        path.write_bytes(chosen.patch.encode("utf-8", "surrogateescape"))
        return path

    # -- reporting ------------------------------------------------------------------------------

    def _conclude(
        self, candidates: list[Candidate], chosen: Candidate | None, failure: str
    ) -> None:
        """Add the selection to each candidate's summary in the tree."""
        for candidate in candidates:
            if candidate.problem:
                continue
            if candidate is chosen:
                verdict = (
                    "selected and applied" if not failure else "selected; could not be applied"
                )
            elif candidate.twin_of:
                verdict = f"same change as {candidate.twin_of}"
            else:
                verdict = "not selected"
            self.session.update_node(
                candidate.node, summary=_one_line(f"{candidate.describe()}; {verdict}")
            )

    def _report(
        self,
        candidates: list[Candidate],
        tests: GeneratedTests | None,
        chosen: Candidate | None,
        how: str,
        applied: list[str],
        failure: str,
    ) -> str:
        model = candidates[0].node.model if candidates else ""
        if chosen is None:
            lines = [f"Nothing was applied: {how}"]
        elif failure:
            lines = [f"Candidate {chosen.label} was chosen, but it could not be applied: {failure}"]
        else:
            lines = [f"Candidate {chosen.label} was applied to the workspace."]
        lines.append("")
        lines.append(f"Candidates ({len(candidates)}, all on {model}):")
        for candidate in candidates:
            hint = f" [approach: {candidate.hint}]" if candidate.hint else ""
            lines.append(f"- {candidate.label}{hint}: {candidate.node.summary}")
        if tests is not None:
            lines.append("")
            lines.append(tests.describe())
        if chosen is not None:
            lines.append("")
            lines.append(f"Why {chosen.label}: {how}")
        if applied:
            lines.append("")
            lines.append("Files changed in the workspace: " + "; ".join(applied))
            lines.append("The user can revert this with /undo.")
        return "\n".join(lines)

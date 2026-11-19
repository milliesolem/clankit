"""Test Loop: one agent writes failing tests first; the agent that talks to the user then
implements the request until the harness sees them pass.

For a request that changes behaviour, a test-writing agent (on the strongest model) writes
acceptance tests without implementing anything, and may only write under the test paths.
The harness runs them and requires them to fail on the untouched code for the right reason:
tests that already pass, or that fail because they are broken, go back to their author a
bounded number of times and are discarded if they never come right. The implementer then
works on the request; after each of its turns the harness runs the tests, and the work is
done when the harness sees them pass or the iteration limit is reached. After every few
failed runs a debugging agent, which cannot change files, diagnoses one failing test for the
implementer. Questions, and requests that tests cannot sensibly check, are handled as in Solo.

What the evidence says, and how it shapes this module:

* The bottleneck is test validity, not fixing: few generated tests tell correct patches from
  incorrect ones, and self-written tests passed wrong code in about one case in six. So the
  strongest model writes the tests, and the harness insists that they fail first.
* Agents game tests, most often by editing them. Read-only tests stop that where prompts
  ("do not cheat") do not, so the implementer cannot change them (see `ProtectedTests`), and
  it is told to report a test that looks wrong rather than work around it.
* The harness decides when the work is done. A debugging sub-agent lifted a test-driven
  pipeline from 87% to 94%; returns diminish quickly, so iterations are capped.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clankit.agent import prompts
from clankit.agent.events import NodeInfo
from clankit.agent.loop import Agent, AgentResult
from clankit.config.schema import Setting
from clankit.llm.provider import ProviderError
from clankit.llm.types import ModelRef
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.orchestration.common import CheckResult, FinishTool, run_diff, run_to_finish
from clankit.orchestration.review_loop import configured_model, visible_check
from clankit.orchestration.solo import SELF_REVIEW
from clankit.tools import standard_tools
from clankit.tools.base import FileLedger, ToolContext, scope_matches
from clankit.tools.search import SKIP_DIRS

if TYPE_CHECKING:
    from clankit.agent.session import Session

DEFAULT_TEST_PATHS = (
    "**/tests/**, **/test/**, **/__tests__/**, **/test_*.py, **/*_test.*, **/*.test.*, "
    "**/*.spec.*, **/conftest.py"
)
#: Times the test author may hand in tests, including the first; then they are discarded.
AUTHOR_ATTEMPTS = 3
#: Characters of test output passed on to agents and to the user.
OUTPUT_TAIL = 2500
#: Test files larger than this are not snapshotted, so changes to them cannot be undone.
MAX_TEST_FILE_BYTES = FileLedger.CHECKPOINT_LIMIT

# --------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------

AUTHOR_ROLE = """\
# Your role

You write the tests that decide whether a request has been carried out. Another agent \
implements the request after you; it cannot change your tests, and its work is accepted \
only when the harness sees them pass. Write tests only: do not implement the request, not \
even part of it.

1. First decide whether tests fit. If the request is a question, asks for no change, is \
about the tests themselves (adding, fixing or removing tests), or is about something tests \
cannot check (documentation wording, visual design, open-ended exploration), write nothing \
and submit `no_tests` with the reason; the request is then handled directly.
2. Read the code involved and the project's existing tests, and follow their framework, \
layout and style. You can only create or change files under the test paths given with the \
request.
3. Test the behaviour the request asks for through the public interface, including the edge \
cases it implies. Leave open what the request leaves open, so that any correct \
implementation passes.
4. Run the tests. On the current code they must fail because the requested behaviour is \
missing or wrong (a failed assertion, a missing function), not because of a mistake in the \
test (a syntax error, a wrong import path, a missing fixture). The harness runs them again \
to check this.

Finish by calling `submit_tests` once."""

IMPLEMENTER_ROLE = """\
# Tests written by another agent

When a request changes behaviour, another agent first writes tests for it and the harness \
checks that they fail on the current code. They arrive with the request and are its \
acceptance criteria: make them pass by implementing the request properly, never by \
special-casing what the tests look for. You cannot change test files; the harness undoes any \
change to them. If a test looks wrong or contradicts the request, do not work around it: say \
which test and why."""

DEBUGGER_ROLE = """\
# Your role

You diagnose one failing test. Another agent is implementing a change and the tests keep \
failing; your diagnosis goes to that agent so it can fix the cause. Do not fix anything \
yourself: do not change files, and use bash only to run the tests, small experiments and \
other read-only commands.

Choose the failing test that looks most informative, reproduce the failure, and trace it to \
its cause in the code under test. Establish the cause by running things rather than by \
guessing. If the test itself looks wrong, say so and show why.

Finish by calling `submit_diagnosis` once."""

REJECTED = """\
[Harness] Your tests were not accepted: {reason}.

{detail}

Fix this and call `submit_tests` again. If you conclude that tests cannot sensibly check \
this request, submit `no_tests` instead."""

DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "test": {"type": "string", "description": "The failing test you examined."},
        "cause": {
            "type": "string",
            "description": "What makes it fail, as precisely as you could establish it.",
        },
        "evidence": {
            "type": "string",
            "description": "Commands and their output, or code locations, that show the cause.",
        },
        "suggestion": {
            "type": "string",
            "description": "The change to the implementation that would fix it, or why the "
            "test itself looks wrong.",
        },
    },
    "required": ["test", "cause", "evidence"],
    "additionalProperties": False,
}


def tests_schema(command: str) -> dict[str, Any]:
    """The `submit_tests` schema. The command field is only asked for when no test command
    is configured, because the harness then runs whatever the author names."""
    properties: dict[str, Any] = {
        "outcome": {
            "type": "string",
            "enum": ["tests_written", "no_tests"],
            "description": "no_tests when the request is a question or tests cannot check it.",
        },
        "tests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One line per test: where it is and what behaviour it checks.",
        },
        "expected_failure": {
            "type": "string",
            "description": "How the tests fail on the current code, and why that shows the "
            "requested behaviour is missing.",
        },
        "why_no_tests": {
            "type": "string",
            "description": "For no_tests: why tests do not fit this request.",
        },
    }
    if not command:
        properties["command"] = {
            "type": "string",
            "description": "The shell command, run from the project root, that runs the new "
            "tests. The harness runs exactly this.",
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["outcome"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------------------------
# Test paths and the files under them
# --------------------------------------------------------------------------------------------


def parse_test_paths(text: str) -> tuple[str, ...]:
    """Globs from a comma- or space-separated list, without duplicates or paths that point
    outside the project (which no write scope could honour)."""
    found: list[str] = []
    for raw in re.split(r"[,\s]+", text):
        pattern = raw.strip()
        while pattern.startswith("./"):
            pattern = pattern[2:]
        if not pattern or pattern.startswith(("/", "~")) or ".." in pattern.split("/"):
            continue
        if pattern not in found:
            found.append(pattern)
    return tuple(found)


def is_test_path(rel: str, patterns: Sequence[str]) -> bool:
    """Same matching as write scopes, so "may the author write it" and "is it protected"
    can never disagree."""
    return any(scope_matches(pattern, rel) for pattern in patterns)


def snapshot_tests(root: Path, patterns: Sequence[str]) -> dict[str, bytes]:
    """Contents of every file under `root` that matches a test path, by relative path.

    Walks like the search tools do: no symbolic links, and none of the directories that hold
    dependencies, caches and build output.
    """
    found: dict[str, bytes] = {}
    pending = [""]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(root / folder) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            rel = f"{folder}{entry.name}"
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIP_DIRS:
                        pending.append(f"{rel}/")
                elif (
                    entry.is_file(follow_symlinks=False)
                    and is_test_path(rel, patterns)
                    and entry.stat(follow_symlinks=False).st_size <= MAX_TEST_FILE_BYTES
                ):
                    found[rel] = Path(entry.path).read_bytes()
            except OSError:
                continue
    return found


def compare_tests(before: dict[str, bytes], after: dict[str, bytes]) -> list[tuple[str, str]]:
    """(path, "created" | "changed" | "deleted") for every difference, sorted by path."""
    changes = []
    for rel in sorted(before.keys() | after.keys()):
        if rel not in after:
            changes.append((rel, "deleted"))
        elif rel not in before:
            changes.append((rel, "created"))
        elif before[rel] != after[rel]:
            changes.append((rel, "changed"))
    return changes


def restore_tests(root: Path, before: dict[str, bytes], changes: list[tuple[str, str]]) -> None:
    """Put the test files back as they were in `before`.

    Written directly, not through the file tools: this is the harness undoing an agent's
    change, so it must not become the "original" that undo would restore.
    """
    for rel, kind in changes:
        path = root / rel
        try:
            if kind == "created":
                path.unlink(missing_ok=True)
            else:
                _unlink_links(root, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(before[rel])
        except OSError:
            continue


def _unlink_links(root: Path, rel: str) -> None:
    """Remove symbolic links on the way to `rel`. The snapshot never follows links, so a
    test file or folder replaced by one reads as deleted; writing the original back through
    the link would land wherever the agent pointed it, possibly outside the project."""
    current = root
    for part in Path(rel).parts:
        current = current / part
        if current.is_symlink():
            current.unlink()


@dataclass(slots=True)
class ProtectedTests:
    """The test files as the harness last accepted them, and the means to put them back.

    Why a snapshot rather than a write scope: scopes can only list what an agent may change,
    not carve exceptions out of the whole project. A scope made of the project's non-test
    directories would miss tests that live next to the code (``src/pkg/test_x.py``,
    ``app.test.ts``), refuse new top-level files, and do nothing about shell commands, which
    no write scope binds. Comparing the test files with a snapshot after every agent turn,
    restoring what changed and deleting what appeared, catches edits made by any means, and
    the implementer is told what was undone.
    """

    root: Path
    patterns: tuple[str, ...]
    baseline: dict[str, bytes] = field(default_factory=dict)

    async def capture(self) -> None:
        self.baseline = await asyncio.to_thread(snapshot_tests, self.root, self.patterns)

    async def changes(self) -> list[tuple[str, str]]:
        current = await asyncio.to_thread(snapshot_tests, self.root, self.patterns)
        return compare_tests(self.baseline, current)

    async def restore(self) -> list[tuple[str, str]]:
        """Undo every change since the last capture; returns what was undone."""
        changes = await self.changes()
        if changes:
            await asyncio.to_thread(restore_tests, self.root, self.baseline, changes)
        return changes


# --------------------------------------------------------------------------------------------
# Judging a test run
# --------------------------------------------------------------------------------------------

_BROKEN_TESTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"ERROR collecting[\s\S]*?\b(?:SyntaxError|IndentationError|TabError)\b"),
        "the tests could not be loaded because of a syntax error",
    ),
    (
        re.compile(r"Test suite failed to run[\s\S]*?\bSyntaxError\b"),
        "the tests could not be loaded because of a syntax error",
    ),
    (re.compile(r"fixture '[^'\n]+' not found"), "a test uses a fixture that does not exist"),
    (
        # pytest prints "collected 0 items / 1 error" when a test module cannot import what
        # it tests; that is the usual first failure of a function not written yet, not an
        # empty run, so a count followed by errors does not count.
        re.compile(
            r"\bno tests ran\b|\bcollected 0 items\b(?![ \t]*/)|\bNo tests found\b"
            r"|\bNo test files found\b|\bno tests to run\b",
            re.IGNORECASE,
        ),
        "the command ran no tests",
    ),
)


def unrelated_failure(check: CheckResult) -> str:
    """Why a failing run says nothing about the requested behaviour, or "" when it looks
    like the behaviour itself is missing.

    Only unambiguous signs count: a run that did not finish, a command that could not start,
    tests that could not even be loaded, or no tests at all. An ImportError of a function
    that does not exist yet is a legitimate first failure, so it is not on the list.
    """
    if check.exit_code is None:
        return "the tests did not finish in time"
    if check.exit_code in (126, 127):
        return "the test command could not be started"
    for pattern, reason in _BROKEN_TESTS:
        if pattern.search(check.output):
            return reason
    return ""


def _tail(check: CheckResult) -> str:
    return check.render(tail=OUTPUT_TAIL)


# --------------------------------------------------------------------------------------------
# Results handed between the steps
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Suite:
    """Tests accepted from the author: the gate the implementation has to pass."""

    command: str
    files: list[tuple[str, str]]
    described: list[str]
    #: The harness's run on the untouched implementation; None when it was not run.
    baseline: CheckResult | None

    @property
    def validated(self) -> bool:
        return self.baseline is not None and not self.baseline.denied

    def file_list(self) -> str:
        return ", ".join(f"`{path}` ({kind})" for path, kind in self.files)


@dataclass(slots=True)
class Diagnosis:
    test: str
    cause: str
    evidence: str
    suggestion: str

    @classmethod
    def parse(cls, payload: dict[str, Any] | None) -> Diagnosis | None:
        if payload is None:
            return None
        values = {k: str(payload.get(k) or "").strip() for k in ("test", "cause", "evidence")}
        if not values["cause"]:
            return None
        return cls(suggestion=str(payload.get("suggestion") or "").strip(), **values)

    def render(self) -> str:
        lines = [
            "[Diagnosis from a debugging agent]",
            "Another agent examined one failing test. Check its conclusions against the code "
            "before you act on them.",
            f"Test: {self.test or '(not named)'}",
            f"Cause: {self.cause}",
            f"Evidence: {self.evidence or '(none given)'}",
        ]
        if self.suggestion:
            lines.append(f"Suggested fix: {self.suggestion}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------------------------


class TestLoopOrchestrator(Orchestrator):
    #: Not a pytest test class, whatever its name suggests.
    __test__ = False

    name = "test-loop"
    title = "Test Loop"
    summary = "One agent writes failing tests first; another implements until the tests pass."
    guidance = (
        "Best for bug fixes and well-specified features in a project with a runnable test "
        "command: tests written first, and checked to fail, give the implementer an objective "
        "target it cannot edit. Costs roughly 2 to 4 times Solo, mostly for writing and "
        "checking the tests. Not suited to visual or exploratory work or projects without "
        "runnable tests; questions are answered as in Solo, after a quick check by the "
        "test-writing agent."
    )
    settings = (
        Setting(
            "test-command",
            "string",
            "",
            "The command that runs the tests, e.g. 'pytest -q'. Empty lets the test-writing "
            "agent name a command that runs its new tests.",
        ),
        Setting(
            "test-paths",
            "string",
            DEFAULT_TEST_PATHS,
            "Where tests live, as comma-separated globs. The test-writing agent may only "
            "write here, and changes the implementing agent makes here are undone.",
        ),
        Setting(
            "max-iterations",
            "int",
            5,
            "Rounds of implementation, each followed by a test run, before giving up.",
            minimum=1,
            maximum=10,
        ),
        Setting(
            "require-failing-first",
            "bool",
            True,
            "Check that the new tests fail before the change. Tests that already pass, or "
            "fail because they are broken, go back to their author.",
        ),
        Setting(
            "debugger-after",
            "int",
            2,
            "After every this many failed test runs, a separate agent diagnoses one failing "
            "test for the implementer. 0 turns this off.",
            minimum=0,
            maximum=10,
        ),
        Setting(
            "author-model",
            "model",
            "",
            "Model for the test-writing and debugging agents. Empty uses the planner role's "
            "model, or the session's model. Pick the strongest you have.",
        ),
    )

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self._root = session.new_node("Conversation", role="root")
        #: The test author while it works; the user's mid-run messages reach it as well.
        self._working: Agent | None = None

    def steer(self, text: str) -> None:
        """Mid-run messages always reach the agent that talks to the user and implements the
        request (it reads them at its next turn); much of what a user adds, such as how to
        build something, is not testable. While the tests are being written their author gets
        the message too, so the tests can take it into account."""
        if self._working is not None:
            self._working.steer(text)
        super().steer(text)

    def _ensure_agent(self) -> Agent:
        if self.root_agent is None:
            role = f"{prompts.USER_FACING}\n\n{IMPLEMENTER_ROLE}"
            self.root_agent = self.session.agent(
                self._root,
                system=prompts.system_prompt(self.session, role),
                tools=[*standard_tools(), AskUserTool(self.session)],
            )
        return self.root_agent

    async def run(self, prompt: str) -> str:
        session = self.session
        agent = self._ensure_agent()
        message = prompt
        if not agent.messages and (recap := session.recap()):
            message = f"{recap}\n\nThe user now says:\n{prompt}"
        patterns = parse_test_paths(self.setting("test-paths")) or parse_test_paths(
            DEFAULT_TEST_PATHS
        )
        author = configured_model(
            session, "author-model", self.setting("author-model"), "planner"
        ) or session.model_for("planner")

        session.update_node(self._root, status="waiting")
        suite = await self._write_tests(prompt, patterns, author)
        if not isinstance(suite, Suite):
            return await self._solo(agent, message, suite)
        return await self._implement(agent, prompt, message, suite, patterns, author)

    # -- step 1: tests --------------------------------------------------------------------------

    async def _write_tests(
        self, prompt: str, patterns: tuple[str, ...], model: ModelRef
    ) -> Suite | str:
        """Have the tests written and validated. Returns the accepted suite, or the reason the
        request goes ahead without tests ("" when tests simply do not fit it)."""
        session = self.session
        command = self.setting("test-command").strip()
        node = session.new_node(
            "Write failing tests", parent=self._root, role="tester", model=model, status="running"
        )
        agent = session.agent(
            node,
            system=prompts.system_prompt(session, AUTHOR_ROLE),
            tools=[
                *standard_tools(),
                FinishTool(
                    "submit_tests",
                    "Hand in your tests, or say that tests do not fit this request. Call once, "
                    "at the end.",
                    tests_schema(command),
                ),
            ],
            ctx=session.tool_context(node, write_scope=patterns),
            model=model,
        )
        tests = ProtectedTests(session.workspace, patterns)
        await tests.capture()
        message = self._author_brief(prompt, patterns, command)
        self._working = agent
        try:
            reason = ""
            for _ in range(AUTHOR_ATTEMPTS):
                result, payload = await run_to_finish(agent, message, "submit_tests")
                if payload is None:
                    stopped = (
                        "declined to write tests"
                        if result.stop == "refusal"
                        else "stopped without handing in tests"
                    )
                    return await self._abandon(node, tests, f"The test-writing agent {stopped}.")
                if payload.get("outcome") == "no_tests":
                    await tests.restore()
                    why = str(payload.get("why_no_tests") or "").strip() or "no reason given"
                    session.update_node(
                        node, status="done", summary=_one_line(f"No tests: {why}", 200)
                    )
                    return ""
                suite_command = command or str(payload.get("command") or "").strip()
                changes = await tests.changes()
                reason, detail, baseline = await self._validate(
                    agent.ctx, suite_command, changes, patterns
                )
                if not reason:
                    suite = Suite(
                        suite_command,
                        changes,
                        [str(t).strip() for t in payload.get("tests") or [] if str(t).strip()],
                        baseline,
                    )
                    session.update_node(node, status="done", summary=_author_summary(suite))
                    return suite
                message = REJECTED.format(reason=reason, detail=detail)
            return await self._abandon(
                node,
                tests,
                f"The tests were not usable after {AUTHOR_ATTEMPTS} attempts ({reason}).",
            )
        except ProviderError as exc:
            session.update_node(
                node, status="failed", summary=_one_line(f"Could not reach its model: {exc}", 200)
            )
            raise
        finally:
            self._working = None

    def _author_brief(self, prompt: str, patterns: tuple[str, ...], command: str) -> str:
        sections = []
        if recap := self.session.recap():
            sections.append(recap)
        sections.append(f"<request>\n{prompt.strip()}\n</request>")
        sections.append(
            "Test paths (you may only create or change files matching these):\n"
            + "\n".join(f"- {pattern}" for pattern in patterns)
        )
        if command:
            sections.append(f"The harness runs the tests with `{command}` from the project root.")
        else:
            sections.append(
                "Put the shell command that runs your new tests from the project root in "
                "`command`; the harness runs exactly that."
            )
        sections.append("Write the tests for this request, run them, and call `submit_tests`.")
        return "\n\n".join(sections)

    async def _validate(
        self,
        ctx: ToolContext,
        command: str,
        changes: list[tuple[str, str]],
        patterns: tuple[str, ...],
    ) -> tuple[str, str, CheckResult | None]:
        """Decide whether handed-in tests can serve as the gate.

        Returns (reason, detail, baseline): reason is "" when they are accepted; baseline is
        the harness's run on the untouched implementation, if there was one.
        """
        if not changes:
            return (
                "no file under the test paths was created or changed",
                "Test paths: " + ", ".join(patterns),
                None,
            )
        if not command:
            return "no test command was given", "Name the command in `command`.", None
        if not self.setting("require-failing-first"):
            return "", "", None
        check = await visible_check(self.session, ctx, command)
        if check.denied:
            self.session.notice(
                "warning",
                f"The harness was not allowed to run `{command}`, so it could not confirm that "
                "the new tests fail before the change.",
                ctx.node_id,
            )
            return "", "", check
        if check.passed:
            return (
                "they pass on the current code, so they do not test the requested change",
                _tail(check),
                check,
            )
        if unrelated := unrelated_failure(check):
            return (
                f"they fail, but not because the requested behaviour is missing: {unrelated}",
                _tail(check),
                check,
            )
        return "", "", check

    async def _abandon(self, node: NodeInfo, tests: ProtectedTests, reason: str) -> str:
        """Give up on tests for this request: discard whatever the author wrote and say why."""
        undone = await tests.restore()
        self.session.update_node(node, status="failed", summary=_one_line(reason, 200))
        discarded = " Its test changes were discarded." if undone else ""
        self.session.notice(
            "warning", f"{reason}{discarded} Continuing without tests, as in Solo.", self._root.id
        )
        return reason

    # -- step 2: implementation -----------------------------------------------------------------

    async def _implement(
        self,
        agent: Agent,
        prompt: str,
        message: str,
        suite: Suite,
        patterns: tuple[str, ...],
        debug_model: ModelRef,
    ) -> str:
        """Implementation rounds, each followed by the harness's test run, until it passes or
        the limit is reached."""
        session = self.session
        limit = self.setting("max-iterations")
        every = self.setting("debugger-after")
        guard = ProtectedTests(session.workspace, patterns)
        await guard.capture()
        text = _implementer_brief(message, suite, patterns)
        undone_total: list[str] = []
        rounds = 0
        while True:
            rounds += 1
            session.update_node(self._root, status="running")
            try:
                result = await agent.run(text)
            finally:
                # However the turn ended (an answer, a refusal, a model error, cancellation),
                # the tests on disk stay as the harness accepted them.
                undone = await self._undo_test_edits(guard)
            if result.stop == "refusal":
                return self._declined(result)
            undone_total.extend(undone)
            check = await visible_check(session, agent.ctx, suite.command)
            await guard.capture()
            if check.passed or check.denied or rounds >= limit:
                break
            diagnosis = ""
            if every and rounds % every == 0:
                diagnosis = await self._diagnose(prompt, suite, check, debug_model)
            text = _feedback(check, undone, diagnosis)

        passed = check.passed or check.denied
        session.update_node(
            self._root,
            status="done" if passed else "failed",
            summary=_outcome_summary(check, rounds),
        )
        report = _report(suite, check, rounds, undone_total)
        return f"{result.text.rstrip()}\n\n{report}"

    async def _undo_test_edits(self, guard: ProtectedTests) -> list[str]:
        """Put the protected tests back and tell the user what was undone."""
        undone = [f"`{path}` ({kind})" for path, kind in await guard.restore()]
        if undone:
            self.session.notice(
                "warning",
                "Undid changes to protected test files: " + ", ".join(undone),
                self._root.id,
            )
        return undone

    async def _diagnose(
        self, prompt: str, suite: Suite, check: CheckResult, model: ModelRef
    ) -> str:
        """Have a debugging agent, which cannot change files, explain one failing test.
        Returns the diagnosis for the implementer, or "" when there is none."""
        session = self.session
        session.update_node(self._root, status="waiting")
        node = session.new_node(
            "Diagnose a failing test",
            parent=self._root,
            role="debugger",
            model=model,
            status="running",
        )
        agent = session.agent(
            node,
            system=prompts.system_prompt(session, DEBUGGER_ROLE),
            tools=[
                *standard_tools(("read", "glob", "grep", "bash", "skill")),
                FinishTool(
                    "submit_diagnosis",
                    "Hand in your diagnosis. Call once, at the end.",
                    DIAGNOSIS_SCHEMA,
                ),
            ],
            ctx=session.tool_context(node, write_scope=()),
            model=model,
        )
        brief = "\n\n".join(
            [
                f"<request>\n{prompt.strip()}\n</request>",
                f"Tests written for this request: {suite.file_list()}\n"
                f"The harness runs them with `{suite.command}`.",
                f"Its latest run:\n{_tail(check)}",
                f"<diff>\n{run_diff(session, limit=30_000) or '(no changes yet)'}\n</diff>",
                "Diagnose one failing test, then call `submit_diagnosis`.",
            ]
        )
        try:
            result, payload = await run_to_finish(agent, brief, "submit_diagnosis")
        except ProviderError as exc:
            session.update_node(
                node, status="failed", summary=_one_line(f"Could not reach its model: {exc}", 200)
            )
            raise
        diagnosis = Diagnosis.parse(payload)
        if diagnosis is None:
            missing = (
                "The model declined." if result.stop == "refusal" else "No diagnosis handed in."
            )
            session.update_node(node, status="failed", summary=missing)
            return ""
        session.update_node(
            node, status="done", summary=_one_line(f"{diagnosis.test}: {diagnosis.cause}", 200)
        )
        return diagnosis.render()

    # -- without tests --------------------------------------------------------------------------

    async def _solo(self, agent: Agent, message: str, note: str) -> str:
        """The request as Solo handles it, for questions and requests tests do not fit."""
        session = self.session
        session.update_node(self._root, status="running")
        result = await agent.run(message)
        if result.stop == "end_turn" and (diff := run_diff(session)):
            result = await agent.run(SELF_REVIEW.format(diff=diff))
        if result.stop == "refusal":
            return self._declined(result)
        session.update_node(
            self._root, status="done", summary=_one_line(note or "Handled without tests.", 200)
        )
        if not note:
            return result.text
        return f"{result.text.rstrip()}\n\n{note} The request was handled without tests."

    def _declined(self, result: AgentResult) -> str:
        self.session.update_node(
            self._root, status="failed", summary="The model declined the request."
        )
        return f"The model declined this request. {result.detail}".strip()


# --------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------


def _implementer_brief(message: str, suite: Suite, patterns: tuple[str, ...]) -> str:
    checked = (
        "and the harness checked that they fail on the current code"
        if suite.validated
        else "but the harness did not run them first"
    )
    lines = [
        message,
        "",
        f"[Harness] Another agent wrote tests for this request, {checked}.",
        f"Test files: {suite.file_list()}",
    ]
    if suite.described:
        lines.append("What they check:")
        lines.extend(f"- {line}" for line in suite.described)
    if suite.validated and suite.baseline is not None:
        lines.extend(["Their result on the current code:", _tail(suite.baseline)])
    lines.extend(
        [
            "",
            f"After each of your turns the harness runs `{suite.command}`; the work is accepted "
            "when it passes.",
            f"Files matching the test paths ({', '.join(patterns)}) are protected: read them, "
            "but do not change them; the harness undoes any change to them.",
        ]
    )
    return "\n".join(lines)


def _feedback(check: CheckResult, undone: list[str], diagnosis: str) -> str:
    parts = [f"[Harness] The tests still fail:\n{_tail(check)}"]
    if undone:
        parts.append(
            "These changes to protected test files were undone; make the tests pass by "
            "changing the implementation only: " + ", ".join(undone)
        )
    if diagnosis:
        parts.append(diagnosis)
    parts.append(
        "Find the cause, fix the implementation, and run the tests yourself before you finish."
    )
    return "\n\n".join(parts)


def _report(suite: Suite, check: CheckResult, rounds: int, undone: list[str]) -> str:
    """What the harness established, appended to the implementer's answer. A run that still
    fails here has used every round: the loop stops early only when the tests pass."""
    lines = [f"Tests written for this request: {suite.file_list()}."]
    if not suite.validated:
        lines.append("The harness did not check that they failed before the change.")
    if check.denied:
        lines.append(
            f"The harness was not allowed to run `{check.command}`, so the result is not verified."
        )
    elif check.passed:
        lines.append(
            f"Harness test run `{check.command}`: passed after {rounds} round(s) of changes."
        )
    else:
        tail = check.output if len(check.output) <= 1500 else "[...]\n" + check.output[-1500:]
        lines.append(
            f"Harness test run `{check.command}`: still failing after {rounds} round(s) of "
            f"changes (the limit). Last output:\n\n```\n{tail.strip()}\n```"
        )
    if undone:
        lines.append("Changes to protected test files were undone: " + ", ".join(undone) + ".")
    return "\n".join(lines)


def _author_summary(suite: Suite) -> str:
    count = f"{len(suite.files)} test file(s)"
    if suite.validated:
        return f"{count}; they fail on the current code, as they should."
    return f"{count}; not run by the harness before the change."


def _outcome_summary(check: CheckResult, rounds: int) -> str:
    if check.denied:
        return "Not verified: the harness was not allowed to run the tests."
    if check.passed:
        return f"Tests pass after {rounds} round(s) of changes."
    return f"Tests still fail after {rounds} round(s) of changes."


def _one_line(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

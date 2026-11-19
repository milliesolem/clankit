"""The Test Loop orchestrator: tests written and validated first, then implementation to green."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event, NodeInfo, Notice, RunFinished
from clankit.agent.session import Session
from clankit.agent.state import Item
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ChatResponse, Message, ToolResultBlock, Usage
from clankit.orchestration import test_loop
from clankit.orchestration.common import CheckResult
from clankit.runtime import Runtime

Step = Any

PROMPT = "make the greeting say hello"
AUTHOR = "Write the tests for this request"
DEBUGGER = "Diagnose one failing test"
TEST_FILE = "tests/test_greet.sh"
TEST_BODY = 'test "$(cat greeting.txt 2>/dev/null)" = hello\n'
COMMAND = f"sh {TEST_FILE}"


class Agents:
    """Routes each model call to the script of the agent that made it.

    Agents are told apart by a marker in their first user message; the marker that occurs
    LATEST in the message wins, because briefs quote the user's request near their top. Each
    agent's script advances by one step per model call; unscripted calls fail loudly.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, list[Step]] = {}
        self.calls: dict[str, int] = {}
        self.requests: dict[str, list[ChatRequest]] = {}

    def add(self, marker: str, *steps: Step) -> None:
        self.scripts[marker] = list(steps)

    def __call__(self, request: ChatRequest) -> Step:
        first = request.messages[0].text
        found = [(first.find(m), m) for m in self.scripts if m in first]
        if not found:
            raise AssertionError(f"no script for an agent whose task starts: {first[:200]!r}")
        marker = max(found)[1]
        n = self.calls.get(marker, 0)
        self.calls[marker] = n + 1
        snapshot = ChatRequest(
            request.model, request.system, list(request.messages), list(request.tools)
        )
        self.requests.setdefault(marker, []).append(snapshot)
        if n >= len(self.scripts[marker]):
            raise AssertionError(f"agent {marker!r} made more calls than scripted")
        return self.scripts[marker][n]


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "test-loop"
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


def write_test(body: str = TEST_BODY) -> Step:
    return call("write", file_path=TEST_FILE, content=body)


def submit(**extra: Any) -> Step:
    payload: dict[str, Any] = {
        "outcome": "tests_written",
        "command": COMMAND,
        "tests": [f"{TEST_FILE}: greeting.txt says hello"],
        "expected_failure": "greeting.txt does not exist yet",
    }
    payload.update(extra)
    return call("submit_tests", **payload)


def greet(text: str) -> Step:
    return call("write", file_path="greeting.txt", content=text)


def refusal(detail: str) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
        usage=Usage(),
        detail=detail,
    )


def tool_result(request: ChatRequest) -> ToolResultBlock:
    return request.messages[-1].tool_result_blocks[0]


def nodes(session: Session) -> dict[str, NodeInfo]:
    return {view.info.title: view.info for view in session.view.nodes.values()}


def checks(session: Session, title: str) -> list[Item]:
    """The harness checks shown on a node's transcript."""
    view = next(v for v in session.view.nodes.values() if v.info.title == title)
    return [item for item in view.items if item.kind == "tool" and item.tool == "check"]


def tool_names(request: ChatRequest) -> set[str]:
    return {tool.name for tool in request.tools}


# --------------------------------------------------------------------------------------------
# The main path
# --------------------------------------------------------------------------------------------


async def test_tests_fail_first_then_the_implementer_iterates_to_green(
    session: Session, agents: Agents, workspace: Path
) -> None:
    agents.add(
        AUTHOR,
        write_test(),
        call("write", file_path="greeting.txt", content="hello"),  # not a test path: refused
        submit(),
    )
    agents.add(
        PROMPT,
        greet("hullo"),
        "Done.",
        greet("hello"),
        "Fixed: greeting.txt now says hello.",
    )

    answer = await session.run(PROMPT)

    assert answer.startswith("Fixed: greeting.txt now says hello.")
    assert f"Tests written for this request: `{TEST_FILE}` (created)." in answer
    assert f"Harness test run `{COMMAND}`: passed after 2 round(s) of changes." in answer
    assert (workspace / "greeting.txt").read_text() == "hello"

    tree = nodes(session)
    assert set(tree) == {"Conversation", "Write failing tests"}
    root, author = tree["Conversation"], tree["Write failing tests"]
    assert (root.role, root.status, root.summary) == (
        "root",
        "done",
        "Tests pass after 2 round(s) of changes.",
    )
    assert (author.role, author.status, author.parent_id) == ("tester", "done", root.id)
    assert author.summary == "1 test file(s); they fail on the current code, as they should."

    refused = tool_result(agents.requests[AUTHOR][2])
    assert refused.is_error and "outside this task's write scope" in refused.content
    author_tools = tool_names(agents.requests[AUTHOR][0])
    assert {
        "write",
        "edit",
        "bash",
        "submit_tests",
    } <= author_tools and "ask_user" not in author_tools
    brief = agents.requests[AUTHOR][0].messages[0].text
    assert f"<request>\n{PROMPT}\n</request>" in brief and "- **/tests/**" in brief

    first = agents.requests[PROMPT][0].messages[0].text
    assert first.startswith(PROMPT)
    assert "the harness checked that they fail on the current code" in first
    assert f"Test files: `{TEST_FILE}` (created)" in first
    assert f"$ {COMMAND}\nFAILED (exit code 1)" in first
    assert "the harness undoes any change to them" in first
    feedback = agents.requests[PROMPT][2].messages[-1].text
    assert feedback.startswith("[Harness] The tests still fail:") and "exit code 1" in feedback
    assert "ask_user" in tool_names(agents.requests[PROMPT][0])

    assert [c.is_error for c in checks(session, "Write failing tests")] == [True]
    assert [c.is_error for c in checks(session, "Conversation")] == [True, False]


async def test_changes_to_protected_tests_are_undone_and_reported(
    session: Session, agents: Agents, workspace: Path, events: list[Event]
) -> None:
    agents.add(AUTHOR, write_test(), submit())
    agents.add(
        PROMPT,
        call("bash", command=f"printf 'exit 0\\n' > {TEST_FILE}"),
        call("write", file_path="tests/test_extra.sh", content="exit 0\n"),
        "The tests pass now.",
        greet("hello"),
        "Implemented it properly.",
    )

    answer = await session.run(PROMPT)

    assert (workspace / TEST_FILE).read_text() == TEST_BODY
    assert not (workspace / "tests" / "test_extra.sh").exists()
    feedback = agents.requests[PROMPT][3].messages[-1].text
    assert "These changes to protected test files were undone" in feedback
    assert f"`{TEST_FILE}` (changed)" in feedback and "`tests/test_extra.sh` (created)" in feedback
    assert "passed after 2 round(s)" in answer
    assert "Changes to protected test files were undone: " in answer
    warnings = [e.text for e in events if isinstance(e, Notice) and e.level == "warning"]
    assert any(w.startswith("Undid changes to protected test files") for w in warnings)


async def test_test_edits_are_undone_even_when_the_implementer_then_refuses(
    session: Session, agents: Agents, workspace: Path, events: list[Event]
) -> None:
    agents.add(AUTHOR, write_test(), submit())
    agents.add(
        PROMPT,
        call("bash", command=f"printf 'exit 0\\n' > {TEST_FILE}"),
        refusal("Not something I can help with."),
    )

    answer = await session.run(PROMPT)

    assert answer == "The model declined this request. Not something I can help with."
    assert (workspace / TEST_FILE).read_text() == TEST_BODY
    warnings = [e.text for e in events if isinstance(e, Notice) and e.level == "warning"]
    assert f"Undid changes to protected test files: `{TEST_FILE}` (changed)" in warnings


async def test_test_edits_are_undone_when_the_run_is_cancelled_mid_turn(
    session: Session, agents: Agents, workspace: Path, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if request.messages[0].text.startswith(PROMPT) and len(request.messages) > 1:
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.add(AUTHOR, write_test(), submit())
    agents.add(PROMPT, call("bash", command=f"printf 'exit 0\\n' > {TEST_FILE}"), "Never reached.")

    task = session.start(PROMPT)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if (workspace / TEST_FILE).exists() and (workspace / TEST_FILE).read_text() != TEST_BODY:
            break
    await asyncio.sleep(0.02)
    assert session.cancel()
    assert await task == ""

    assert (workspace / TEST_FILE).read_text() == TEST_BODY
    assert {info.status for info in nodes(session).values()} == {"done", "cancelled"}
    assert nodes(session)["Conversation"].status == "cancelled"


# --------------------------------------------------------------------------------------------
# Validating the tests
# --------------------------------------------------------------------------------------------


PYTEST_IMPORT_ERROR = """\
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.2.0, pluggy-1.7.0
rootdir: /work
collected 0 items / 1 error

==================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_slug.py ______________________
ImportError while importing test module '/work/tests/test_slug.py'.
tests/test_slug.py:1: in <module>
    from mypkg import slugify
E   ImportError: cannot import name 'slugify' from 'mypkg' (/work/mypkg/__init__.py)
=========================== short test summary info ============================
ERROR tests/test_slug.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.06s ===============================
"""


def test_a_function_that_does_not_exist_yet_is_a_legitimate_first_failure() -> None:
    check = CheckResult("pytest tests", passed=False, exit_code=2, output=PYTEST_IMPORT_ERROR)
    assert test_loop.unrelated_failure(check) == ""
    empty = CheckResult("pytest", passed=False, exit_code=5, output="collected 0 items\n\n")
    assert "ran no tests" in test_loop.unrelated_failure(empty)


@pytest.mark.parametrize(
    "output",
    [
        "No test files found, exiting with code 1\n\ninclude: **/*.{test,spec}.?(c|m)[jt]s?(x)",
        'Error: No test files found: "test/greet.spec.js"',
    ],
)
def test_runners_that_find_no_test_files_ran_no_tests(output: str) -> None:
    check = CheckResult("npx vitest run", passed=False, exit_code=1, output=output)
    assert "ran no tests" in test_loop.unrelated_failure(check)


async def test_pytest_tests_of_a_missing_function_are_accepted(
    session: Session, agents: Agents, workspace: Path
) -> None:
    (workspace / "mypkg").mkdir()
    (workspace / "mypkg" / "__init__.py").write_text("")
    body = 'from mypkg import slugify\n\n\ndef test_slug():\n    assert slugify("A B") == "a-b"\n'
    command = f"{sys.executable} -m pytest -p no:cacheprovider tests/test_slug.py"
    agents.add(
        AUTHOR,
        call("write", file_path="tests/test_slug.py", content=body),
        submit(command=command, tests=["tests/test_slug.py: slugify lowercases and joins"]),
    )
    agents.add(
        PROMPT,
        call("read", file_path="mypkg/__init__.py"),
        call("write", file_path="mypkg/__init__.py", content=SLUGIFY),
        "Added slugify.",
    )

    answer = await session.run(PROMPT)

    assert agents.calls[AUTHOR] == 2  # accepted at the first attempt
    assert nodes(session)["Write failing tests"].summary == (
        "1 test file(s); they fail on the current code, as they should."
    )
    assert f"Harness test run `{command}`: passed after 1 round(s)" in answer


SLUGIFY = 'def slugify(text):\n    return "-".join(text.lower().split())\n'


@pytest.mark.parametrize(
    ("first_attempt", "reason"),
    [
        ([submit()], "no file under the test paths was created or changed"),
        ([write_test(), call("submit_tests", outcome="tests_written")], "no test command"),
        ([write_test("exit 0\n"), submit()], "they pass on the current code"),
        (
            [write_test("echo 'no tests ran in 0.01s'; exit 5\n"), submit()],
            "not because the requested behaviour is missing: the command ran no tests",
        ),
    ],
)
async def test_unusable_tests_go_back_to_their_author(
    session: Session, agents: Agents, first_attempt: list[Step], reason: str
) -> None:
    agents.add(AUTHOR, *first_attempt, write_test(), submit())
    agents.add(PROMPT, greet("hello"), "Done.")

    answer = await session.run(PROMPT)

    rejected = agents.requests[AUTHOR][len(first_attempt)].messages[-1].text
    assert rejected.startswith("[Harness] Your tests were not accepted:") and reason in rejected
    assert "submit `no_tests` instead" in rejected
    assert nodes(session)["Write failing tests"].status == "done"
    assert "passed after 1 round(s)" in answer


async def test_tests_that_never_become_usable_are_discarded(
    session: Session, agents: Agents, workspace: Path, events: list[Event]
) -> None:
    agents.add(AUTHOR, write_test("exit 0\n"), submit(), submit(), submit())
    agents.add(PROMPT, greet("hello"), "Done.", "Checked the diff; it is right.")

    answer = await session.run(PROMPT)

    assert not (workspace / TEST_FILE).exists()
    author = nodes(session)["Write failing tests"]
    assert author.status == "failed"
    assert author.summary.startswith("The tests were not usable after 3 attempts (they pass")
    assert answer.startswith("Checked the diff; it is right.")
    assert answer.endswith("The request was handled without tests.")
    first = agents.requests[PROMPT][0].messages[0].text
    assert first == PROMPT  # no test gate is described to the implementer
    assert "[Harness check before your answer" in agents.requests[PROMPT][2].messages[-1].text
    assert checks(session, "Conversation") == []
    assert any(isinstance(e, Notice) and "Continuing without tests" in e.text for e in events)


async def test_a_question_is_answered_without_tests(session: Session, agents: Agents) -> None:
    agents.add(AUTHOR, call("submit_tests", outcome="no_tests", why_no_tests="It is a question."))
    agents.add("what does greet do", "It prints a greeting.")

    assert await session.run("what does greet do") == "It prints a greeting."

    tree = nodes(session)
    assert (tree["Write failing tests"].status, tree["Write failing tests"].summary) == (
        "done",
        "No tests: It is a question.",
    )
    assert tree["Conversation"].status == "done"
    assert checks(session, "Conversation") == []


async def test_without_failing_first_the_tests_are_not_run_before_the_change(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("test-loop", "require-failing-first", False)
    agents.add(AUTHOR, write_test("exit 0\n"), submit())
    agents.add(PROMPT, greet("hello"), "Done.")

    answer = await session.run(PROMPT)

    assert checks(session, "Write failing tests") == []
    assert "but the harness did not run them first" in agents.requests[PROMPT][0].messages[0].text
    assert "The harness did not check that they failed before the change." in answer
    assert nodes(session)["Write failing tests"].summary.endswith(
        "not run by the harness before the change."
    )


async def test_a_configured_test_command_is_used_and_not_asked_for(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("test-loop", "test-command", COMMAND)
    agents.add(AUTHOR, write_test(), call("submit_tests", outcome="tests_written"))
    agents.add(PROMPT, greet("hello"), "Done.")

    answer = await session.run(PROMPT)

    submit_tool = next(t for t in agents.requests[AUTHOR][0].tools if t.name == "submit_tests")
    assert "command" not in submit_tool.input_schema["properties"]
    assert (
        f"The harness runs the tests with `{COMMAND}`"
        in agents.requests[AUTHOR][0].messages[0].text
    )
    assert f"Harness test run `{COMMAND}`: passed after 1 round(s)" in answer


# --------------------------------------------------------------------------------------------
# Iterating
# --------------------------------------------------------------------------------------------


async def test_a_debugger_diagnoses_after_repeated_failures(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("test-loop", "max-iterations", 3)
    agents.add(AUTHOR, write_test(), submit())
    agents.add(
        PROMPT,
        greet("hullo"),
        "Try one.",
        greet("hallo"),
        "Try two.",
        greet("hello"),
        "Fixed after the diagnosis.",
    )
    agents.add(
        DEBUGGER,
        call("bash", command="cat greeting.txt"),
        call(
            "submit_diagnosis",
            test=TEST_FILE,
            cause="greeting.txt says hallo, not hello",
            evidence="cat greeting.txt printed hallo",
            suggestion="write hello",
        ),
    )

    answer = await session.run(PROMPT)

    assert "passed after 3 round(s)" in answer
    assert agents.calls[DEBUGGER] == 2  # one debugger, after the second failure only
    debugger = nodes(session)["Diagnose a failing test"]
    assert (debugger.role, debugger.status) == ("debugger", "done")
    assert debugger.summary == f"{TEST_FILE}: greeting.txt says hallo, not hello"
    brief = agents.requests[DEBUGGER][0]
    assert "Its latest run:" in brief.messages[0].text and "+hallo" in brief.messages[0].text
    assert {"write", "edit"}.isdisjoint(tool_names(brief))
    assert {"bash", "submit_diagnosis"} <= tool_names(brief)
    handed = agents.requests[PROMPT][4].messages[-1].text
    assert "[Diagnosis from a debugging agent]" in handed
    assert "Cause: greeting.txt says hallo, not hello" in handed
    assert "Suggested fix: write hello" in handed


async def test_the_iteration_limit_ends_with_an_honest_failure(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("test-loop", "max-iterations", 2)
    config.orchestration.set_option("test-loop", "debugger-after", 0)
    agents.add(AUTHOR, write_test(), submit())
    agents.add(PROMPT, greet("hullo"), "All done.", "Still done.")

    answer = await session.run(PROMPT)

    assert answer.startswith("Still done.")
    assert "still failing after 2 round(s) of changes (the limit). Last output:" in answer
    root = nodes(session)["Conversation"]
    assert (root.status, root.summary) == (
        "failed",
        "Tests still fail after 2 round(s) of changes.",
    )
    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "completed"
    assert "Diagnose a failing test" not in nodes(session)


# --------------------------------------------------------------------------------------------
# Models, steering, failures
# --------------------------------------------------------------------------------------------


async def test_the_author_and_debugger_use_the_planner_model(
    session: Session, agents: Agents, config: Config
) -> None:
    other = ScriptedProvider(provider_id="other", router=agents)
    session.runtime.providers.register(other)
    config.models.roles["planner"] = "other/strong-model"
    config.orchestration.set_option("test-loop", "max-iterations", 2)
    config.orchestration.set_option("test-loop", "debugger-after", 1)
    agents.add(AUTHOR, write_test(), submit())
    agents.add(
        DEBUGGER, call("submit_diagnosis", test=TEST_FILE, cause="missing", evidence="cat failed")
    )
    agents.add(PROMPT, "Nothing yet.", greet("hello"), "Done.")

    await session.run(PROMPT)

    tree = nodes(session)
    assert tree["Write failing tests"].model == "other/strong-model"
    assert tree["Diagnose a failing test"].model == "other/strong-model"
    assert tree["Conversation"].model == "scripted/test-model"
    assert {r.model for r in other.requests} == {"strong-model"}


async def test_an_unusable_author_model_setting_is_reported_and_ignored(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("test-loop", "author-model", "nowhere/model-x")
    agents.add(AUTHOR, call("submit_tests", outcome="no_tests", why_no_tests="Nothing to test."))
    agents.add(PROMPT, "Answered.")

    await session.run(PROMPT)

    assert nodes(session)["Write failing tests"].model == "scripted/test-model"
    assert any(
        isinstance(e, Notice) and e.text.startswith("Ignoring the author-model setting")
        for e in events
    )


async def test_steering_while_tests_are_written_reaches_the_author(
    session: Session, agents: Agents
) -> None:
    def steer_then_write(request: ChatRequest) -> Step:
        assert session.steer("also cover an empty file")
        return write_test()

    agents.add(AUTHOR, steer_then_write, submit())
    agents.add(PROMPT, greet("hello"), "Done.")

    await session.run(PROMPT)

    note = agents.requests[AUTHOR][1].messages[-1].text
    assert "[Message from the user while you were working]" in note
    assert "also cover an empty file" in note
    assert all("empty file" not in m.text for m in agents.requests[PROMPT][0].messages)


async def test_steering_while_tests_are_written_also_reaches_the_implementer(
    session: Session, agents: Agents
) -> None:
    def steer_then_write(request: ChatRequest) -> Step:
        assert session.steer("keep the greeting on one line")
        return write_test()

    agents.add(AUTHOR, steer_then_write, submit())
    agents.add(PROMPT, greet("hello"), "Done.")

    await session.run(PROMPT)

    assert "keep the greeting on one line" in agents.requests[AUTHOR][1].messages[-1].text
    note = agents.requests[PROMPT][1].messages[-1].text
    assert "[Message from the user while you were working]" in note
    assert "keep the greeting on one line" in note


async def test_a_refusing_implementer_ends_the_request(session: Session, agents: Agents) -> None:
    agents.add(AUTHOR, write_test(), submit())
    agents.add(PROMPT, refusal("Not something I can help with."))

    answer = await session.run(PROMPT)

    assert answer == "The model declined this request. Not something I can help with."
    assert nodes(session)["Conversation"].status == "failed"


async def test_a_refusing_author_falls_back_to_working_without_tests(
    session: Session, agents: Agents
) -> None:
    agents.add(AUTHOR, refusal("No."))
    agents.add(PROMPT, "Answered directly.")

    answer = await session.run(PROMPT)

    assert answer == (
        "Answered directly.\n\nThe test-writing agent declined to write tests. "
        "The request was handled without tests."
    )
    assert nodes(session)["Write failing tests"].status == "failed"


async def test_a_provider_failure_ends_the_run_as_failed(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(AUTHOR, ProviderError("invalid api key", kind="auth", provider="scripted"))

    assert await session.run(PROMPT) == ""

    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "invalid api key" in finished.error
    author = nodes(session)["Write failing tests"]
    assert author.status == "failed"
    assert author.summary.startswith("Could not reach its model:")
    assert "invalid api key" in author.summary
    assert nodes(session)["Conversation"].status == "failed"


async def test_the_run_budget_stops_the_loop(
    runtime: Runtime, workspace: Path, scripted: ScriptedProvider
) -> None:
    session = runtime.create_session(workspace, interactive=False)
    session.config.agent.run_budget = 12  # each scripted call generates 5 output tokens
    agents = Agents()
    scripted._router = agents
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    agents.add(AUTHOR, write_test(), submit())
    agents.add(PROMPT, greet("hullo"), "Done.")

    assert await session.run(PROMPT) == ""

    finished = seen[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "budget" in finished.error
    assert agents.calls[PROMPT] == 1


async def test_a_test_run_the_user_does_not_allow_leaves_the_result_unverified(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.permissions.deny = [f"shell:{COMMAND}"]
    agents.add(AUTHOR, write_test(), submit())
    agents.add(PROMPT, greet("hello"), "Done.")

    answer = await session.run(PROMPT)

    assert "but the harness did not run them first" in agents.requests[PROMPT][0].messages[0].text
    assert (
        f"The harness was not allowed to run `{COMMAND}`, so the result is not verified." in answer
    )
    root = nodes(session)["Conversation"]
    assert (root.status, root.summary) == (
        "done",
        "Not verified: the harness was not allowed to run the tests.",
    )
    assert agents.calls[PROMPT] == 2  # no further rounds without a verdict
    assert any(isinstance(e, Notice) and "could not confirm" in e.text for e in events)


async def test_cancelling_during_a_harness_check_cancels_everything(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("test-loop", "test-command", "sleep 30")
    agents.add(AUTHOR, write_test(), call("submit_tests", outcome="tests_written"))

    task = session.start(PROMPT)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if checks(session, "Write failing tests"):
            break
    await asyncio.sleep(0.05)
    assert session.cancel()
    assert await task == ""

    assert {info.status for info in nodes(session).values()} == {"cancelled"}
    (check,) = checks(session, "Write failing tests")
    assert check.done and check.is_error and check.output == "Interrupted."


async def test_the_conversation_continues_across_requests(session: Session, agents: Agents) -> None:
    agents.add(
        AUTHOR,
        call("submit_tests", outcome="no_tests", why_no_tests="A question."),
        write_test(),
        submit(),
    )
    agents.add("what is greeting.txt", "It holds the greeting.", greet("hello"), "Done.")

    await session.run("what is greeting.txt")
    answer = await session.run(PROMPT)

    assert "passed after 1 round(s)" in answer
    second_author = agents.requests[AUTHOR][1].messages[0].text
    assert "Earlier in this session:" in second_author and "what is greeting.txt" in second_author
    later = agents.requests["what is greeting.txt"][1].messages[-1].text
    assert later.startswith(PROMPT) and "[Harness] Another agent wrote tests" in later


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def test_test_paths_are_parsed_and_matched_like_write_scopes() -> None:
    patterns = test_loop.parse_test_paths(
        " tests/**, ./spec/**  tests/** /etc/** ../up/** **/*_test.go "
    )
    assert patterns == ("tests/**", "spec/**", "**/*_test.go")

    defaults = test_loop.parse_test_paths(test_loop.DEFAULT_TEST_PATHS)
    for rel in (
        "tests/test_api.py",
        "pkg/tests/fixtures/data.json",
        "src/pkg/test_util.py",
        "web/app.test.ts",
        "web/__tests__/app.js",
        "cmd/main_test.go",
        "conftest.py",
    ):
        assert test_loop.is_test_path(rel, defaults), rel
    for rel in ("src/app.py", "README.md", "src/contest.py", "latest/notes.txt"):
        assert not test_loop.is_test_path(rel, defaults), rel


async def test_protected_tests_skip_dependency_folders_and_restore_what_changed(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("a")
    (tmp_path / "node_modules" / "lib").mkdir(parents=True)
    (tmp_path / "node_modules" / "lib" / "x.test.js").write_text("vendored")
    guard = test_loop.ProtectedTests(
        tmp_path, test_loop.parse_test_paths(test_loop.DEFAULT_TEST_PATHS)
    )
    await guard.capture()
    assert set(guard.baseline) == {"tests/test_a.py"}

    (tmp_path / "tests" / "test_a.py").write_text("weakened")
    (tmp_path / "tests" / "test_b.py").write_text("new")
    (tmp_path / "node_modules" / "lib" / "x.test.js").write_text("changed, but not ours")
    assert await guard.restore() == [("tests/test_a.py", "changed"), ("tests/test_b.py", "created")]
    assert (tmp_path / "tests" / "test_a.py").read_text() == "a"
    assert not (tmp_path / "tests" / "test_b.py").exists()

    (tmp_path / "tests" / "test_a.py").unlink()
    assert await guard.restore() == [("tests/test_a.py", "deleted")]
    assert (tmp_path / "tests" / "test_a.py").read_text() == "a"


async def test_restoring_tests_never_writes_through_a_symbolic_link(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "tests" / "unit").mkdir(parents=True)
    (root / "tests" / "test_a.py").write_text("a")
    (root / "tests" / "unit" / "test_b.py").write_text("b")
    outside = tmp_path / "outside"
    outside.mkdir()
    precious = outside / "notes.txt"
    precious.write_text("precious")
    guard = test_loop.ProtectedTests(root, ("tests/**",))
    await guard.capture()

    (root / "tests" / "test_a.py").unlink()
    os.symlink(precious, root / "tests" / "test_a.py")
    (root / "tests" / "unit" / "test_b.py").unlink()
    (root / "tests" / "unit").rmdir()
    os.symlink(outside, root / "tests" / "unit")

    assert await guard.restore() == [
        ("tests/test_a.py", "deleted"),
        ("tests/unit/test_b.py", "deleted"),
    ]
    assert precious.read_text() == "precious"
    assert sorted(p.name for p in outside.iterdir()) == ["notes.txt"]
    assert not (root / "tests" / "test_a.py").is_symlink()
    assert (root / "tests" / "test_a.py").read_text() == "a"
    assert not (root / "tests" / "unit").is_symlink()
    assert (root / "tests" / "unit" / "test_b.py").read_text() == "b"


@pytest.mark.parametrize(
    ("exit_code", "output", "reason"),
    [
        (1, "E   AssertionError: assert 'hullo' == 'hello'\n1 failed", ""),
        (2, "ERROR collecting tests/test_x.py\nE   ImportError: cannot import name 'slug'", ""),
        (
            2,
            "ERROR collecting tests/test_x.py\nE     def broken(\nE   SyntaxError: '(' was never closed",
            "syntax error",
        ),
        (
            1,
            "FAIL src/a.test.js\n  Test suite failed to run\n    SyntaxError: Unexpected token",
            "syntax error",
        ),
        (1, "E       fixture 'tmp_dir' not found", "fixture that does not exist"),
        (5, "no tests ran in 0.01s", "ran no tests"),
        (127, "sh: 1: pytset: not found", "could not be started"),
        (None, "Timed out after 120s.", "did not finish in time"),
        (1, "Failed: DID NOT RAISE <class 'SyntaxError'>", ""),
    ],
)
def test_failures_unrelated_to_the_behaviour_are_recognised(
    exit_code: int | None, output: str, reason: str
) -> None:
    check = CheckResult("pytest -q", passed=False, exit_code=exit_code, output=output)
    found = test_loop.unrelated_failure(check)
    assert (reason in found) if reason else found == ""

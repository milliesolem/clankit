"""Best of N: isolated candidates in git worktrees, pruning, generated tests, selection, apply."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from clankit import paths
from clankit.agent.events import Event, Notice, RunFinished
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ChatResponse, Message, Usage
from clankit.orchestration import get_orchestrator
from clankit.orchestration.best_of_n import DEFAULT_HINTS, BestOfNOrchestrator
from clankit.runtime import Runtime

Step = Any

PROMPT = "make the greeting friendlier"
TASK = "Change greeting.txt so that it greets with the word hello."
CHECK = "grep -q hello greeting.txt"
CHOOSE = "Choose the candidate to apply"
TESTS = "Write tests that tell"


class Agents:
    """Routes each model call to the script of the agent that made it.

    Agents are told apart by a marker in their first user message: the user's prompt for the
    agent that talks to the user, "You are candidate X" for candidates, and the closing line
    of the test writer's and the selector's briefs. Because those briefs quote the user's
    request near their top, the marker that occurs LATEST in the message wins. Each agent's
    script advances by one step per model call; unscripted calls fail loudly.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, list[Step]] = {}
        self.calls: dict[str, int] = {}
        self.requests: dict[str, list[ChatRequest]] = {}

    def add(self, marker: str, *steps: Step) -> None:
        self.scripts[marker] = list(steps)

    def candidate(self, label: str, *steps: Step) -> None:
        self.add(f"You are candidate {label}", *steps)

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

    def first(self, marker: str) -> ChatRequest:
        return self.requests[marker][0]


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def rewrite(content: str, summary: str = "Changed the greeting.") -> list[Step]:
    """A candidate that reads greeting.txt, replaces it and says so."""
    return [
        call("read", file_path="greeting.txt"),
        call("write", file_path="greeting.txt", content=content),
        summary,
    ]


def build(task: str = TASK) -> Any:
    return call("build_candidates", task=task)


def choose(label: str, why: str = "It passes the check and does what was asked.") -> Any:
    return call("choose", candidate=label, justification=why)


def report(agents: Agents) -> str:
    """What `build_candidates` returned to the agent that talks to the user."""
    return agents.requests[PROMPT][1].messages[-1].tool_result_blocks[0].content


def worktrees_left(repo: Path) -> tuple[list[Path], int]:
    directory = paths.state_dir() / "worktrees"
    left = list(directory.iterdir()) if directory.exists() else []
    return left, len(git(repo, "worktree", "list").splitlines())


def nodes(session: Session) -> dict[str, tuple[str, str, str]]:
    """title -> (role, status, summary)"""
    return {
        n.info.title: (n.info.role, n.info.status, n.info.summary)
        for n in session.view.nodes.values()
    }


@pytest.fixture(autouse=True)
def plain_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own git configuration (signing, hooks, identity) out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "best-of-n"
    config.orchestration.set_option("best-of-n", "check", CHECK)
    config.orchestration.set_option("best-of-n", "write-tests", False)
    config.orchestration.set_option("best-of-n", "selector-votes", 1)
    return config


@pytest.fixture
def repo(workspace: Path) -> Path:
    git(workspace, "init", "-q")
    git(workspace, "config", "user.email", "test@example.com")
    git(workspace, "config", "user.name", "Test")
    (workspace / "greeting.txt").write_text("hi\n")
    git(workspace, "add", "-A")
    git(workspace, "commit", "-qm", "initial")
    return workspace


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


async def test_candidates_build_in_isolation_and_the_chosen_one_is_applied(
    session: Session, agents: Agents, repo: Path, events: list[Event]
) -> None:
    seen: dict[str, str] = {}

    def look_then_choose(request: ChatRequest) -> Any:
        seen["workspace"] = (repo / "greeting.txt").read_text()
        return choose("B")

    agents.add(PROMPT, build(), "Applied candidate B: the greeting now says hello there.")
    agents.candidate("A", *rewrite("hi there\n"))
    agents.candidate("B", *rewrite("hello there\n", "PRIVATE SUMMARY OF B"))
    agents.candidate(
        "C", call("write", file_path="notes/C.md", content="notes\n"), *rewrite("hello world\n")
    )
    agents.add(CHOOSE, look_then_choose)

    answer = await session.run(PROMPT)

    assert answer == "Applied candidate B: the greeting now says hello there."
    assert seen["workspace"] == "hi\n"  # untouched until a candidate was chosen
    assert (repo / "greeting.txt").read_text() == "hello there\n"
    assert not (repo / "notes").exists()
    assert git(repo, "status", "--porcelain") == " M greeting.txt\n"
    assert git(repo, "stash", "list") == "" and git(repo, "branch").strip() in (
        "* main",
        "* master",
    )
    assert worktrees_left(repo) == ([], 1)

    selector = agents.first(CHOOSE)
    brief = selector.messages[0].text
    assert "+hello there" in brief and "+hello world" in brief and "hi there" not in brief
    assert "Files changed: greeting.txt  +1 -1; notes/C.md  +1 -0" in brief
    assert brief.count("Regression check:\n$ grep -q hello greeting.txt\npassed") == 2
    assert "PRIVATE SUMMARY OF B" not in brief and "Changed the greeting." not in brief
    choose_tool = next(t for t in selector.tools if t.name == "choose")
    assert choose_tool.input_schema["properties"]["candidate"]["enum"] == ["B", "C"]
    assert {t.name for t in selector.tools} == {"read", "glob", "grep", "skill", "choose"}

    text = report(agents)
    assert text.startswith("Candidate B was applied to the workspace.")
    assert "- A: 1 file changed; check failed (exit code 1); not selected" in text
    assert "- B [approach: Make the smallest change" in text
    assert "Files changed in the workspace: greeting.txt  +1 -1" in text and "/undo" in text
    assert "Why B: 1 of 1 selector vote (scripted/test-model) chose it" in text

    assert nodes(session) == {
        "Conversation": ("root", "done", ""),
        "Candidate A": (
            "candidate",
            "failed",
            "1 file changed; check failed (exit code 1); not selected",
        ),
        "Candidate B": ("candidate", "done", "1 file changed; check passed; selected and applied"),
        "Candidate C": ("candidate", "done", "2 files changed; check passed; not selected"),
        "Selection": ("judge", "done", "chose B: It passes the check and does what was asked."),
    }
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert "Best of N: building 3 candidates, about 3 x 1.3 the cost of one attempt." in notices

    assert session.undo() == ["greeting.txt"]
    assert (repo / "greeting.txt").read_text() == "hi\n"


async def test_each_candidate_is_told_its_own_copy_and_approach(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "Done.")
    for label in "ABC":
        agents.candidate(label, *rewrite("hello\n"))
    await session.run(PROMPT)

    root_tools = {t.name for t in agents.first(PROMPT).tools}
    assert {"build_candidates", "ask_user", "bash"} <= root_tools
    assert not {"write", "edit"} & root_tools
    for label, hint in zip("ABC", ["", DEFAULT_HINTS[0], DEFAULT_HINTS[1]], strict=True):
        request = agents.first(f"You are candidate {label}")
        brief = request.messages[0].text
        assert brief.startswith(f"<user_request>\n{PROMPT}\n</user_request>\n\n<task>\n{TASK}")
        assert ("Approach to take:" in brief) == bool(hint) and hint in brief
        assert f"the harness runs `{CHECK}` in your copy" in brief
        assert f"- Working directory: {repo}\n" not in request.system
        assert "worktrees/candidate-" in request.system and "private copy" in request.system
        assert {t.name for t in request.tools}.isdisjoint({"build_candidates", "ask_user"})


async def test_identical_changes_are_merged_and_counted(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hello world\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hello   there  \n"))  # the same change, spaced differently
    agents.add(CHOOSE, choose("B"))
    await session.run(PROMPT)

    brief = agents.first(CHOOSE).messages[0].text
    assert brief.count('<candidate name="') == 2
    assert brief.index('<candidate name="B">') < brief.index('<candidate name="A">')
    assert "Candidates B, C made this same change (2 of 3)." in brief
    assert nodes(session)["Candidate C"][2].endswith("; same change as B")


async def test_a_single_survivor_is_applied_without_a_selector(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hi there\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hey\n"))
    await session.run(PROMPT)
    assert (repo / "greeting.txt").read_text() == "hello there\n"
    assert CHOOSE not in agents.requests
    assert "Why B: it was the only distinct change left." in report(agents)


async def test_when_every_candidate_fails_the_check_all_are_considered(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hi there\n"))
    agents.candidate("B", *rewrite("hey there\n"))
    agents.candidate("C", *rewrite("howdy\n"))
    agents.add(CHOOSE, choose("C"))
    await session.run(PROMPT)
    assert agents.first(CHOOSE).messages[0].text.count('<candidate name="') == 3
    assert "Every candidate failed the check, so all of them were considered." in report(agents)
    assert (repo / "greeting.txt").read_text() == "howdy\n"


async def test_generated_tests_run_against_every_candidate(
    session: Session, agents: Agents, repo: Path, config: Config
) -> None:
    config.orchestration.set_option("best-of-n", "write-tests", True)
    agents.add(PROMPT, build(), "Done.")
    agents.add(
        TESTS,
        call(
            "write",
            file_path="test_distinguishing.sh",
            content='grep -qx "hello there" greeting.txt\n',
        ),
        call(
            "submit_tests",
            command="sh test_distinguishing.sh",
            covers="The greeting is exactly 'hello there'.",
        ),
    )
    agents.candidate("A", *rewrite("hello world\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hello there\n"))
    agents.add(CHOOSE, choose("B"))

    await session.run(PROMPT)

    brief = agents.first(CHOOSE).messages[0].text
    assert "added them to each candidate's copy and ran `sh test_distinguishing.sh`" in brief
    assert "On the unchanged code they failed, as tests of new behaviour should." in brief
    assert brief.index('<candidate name="B">') < brief.index('<candidate name="A">')
    a_section = brief[brief.index('<candidate name="A">') :]
    assert "Generated tests:\n$ sh test_distinguishing.sh\nFAILED (exit code 1)" in a_section
    assert "test_distinguishing.sh" not in a_section.split("<diff>")[1]
    assert not (repo / "test_distinguishing.sh").exists()  # the tests are never applied
    assert nodes(session)["Distinguishing tests"] == (
        "worker",
        "done",
        "test_distinguishing.sh +1 -0; they fail on the unchanged code",
    )
    assert nodes(session)["Candidate A"][2] == (
        "1 file changed; check passed; generated tests failed (exit code 1); not selected"
    )
    writer = agents.first(TESTS)
    assert "submit_tests" in {t.name for t in writer.tools}
    assert "distinguishing" in writer.system and "worktrees/tests-" in writer.system


async def test_a_question_is_answered_without_candidates(
    session: Session, agents: Agents, repo: Path, events: list[Event]
) -> None:
    agents.add("what does the greeting say", call("read", file_path="greeting.txt"), "It says hi.")
    assert await session.run("what does the greeting say") == "It says hi."
    assert list(nodes(session)) == ["Conversation"]
    assert not any(isinstance(e, Notice) for e in events)


async def test_candidates_are_built_only_once_per_request(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), build(), "Done.")
    for label in "ABC":
        agents.candidate(label, *rewrite("hello\n"))
    await session.run(PROMPT)
    second = agents.requests[PROMPT][2].messages[-1].tool_result_blocks[0]
    assert second.is_error and "already built for this request" in second.content


async def test_declining_and_idle_candidates_drop_out(
    session: Session, agents: Agents, repo: Path
) -> None:
    refused = ChatResponse(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
        usage=Usage(),
        detail="no",
    )
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", refused)
    agents.candidate("B", "Nothing needs to change.")
    agents.candidate("C", *rewrite("hello\n"))
    await session.run(PROMPT)
    assert nodes(session)["Candidate A"][1:] == ("failed", "the model declined the task")
    assert nodes(session)["Candidate B"][1:] == ("failed", "it made no changes")
    assert (repo / "greeting.txt").read_text() == "hello\n"


async def test_nothing_is_applied_when_no_candidate_changed_anything(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "Nothing was applied.")
    for label in "ABC":
        agents.candidate(label, "Nothing to do.")
    await session.run(PROMPT)
    assert report(agents).startswith("Nothing was applied: no candidate produced a change.")
    assert (repo / "greeting.txt").read_text() == "hi\n"
    assert worktrees_left(repo) == ([], 1)


async def test_selector_votes_stop_at_a_majority(
    session: Session, agents: Agents, repo: Path, config: Config
) -> None:
    config.orchestration.set_option("best-of-n", "selector-votes", 3)
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hi there\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hello world\n"))
    agents.add(CHOOSE, choose("C", "Shorter."), choose("C", "Clearer."), choose("B"))
    await session.run(PROMPT)

    briefs = [r.messages[0].text for r in agents.requests[CHOOSE]]
    assert len(briefs) == 2  # the third vote was not needed
    assert briefs[0].index('name="B"') < briefs[0].index('name="C"')
    assert briefs[1].index('name="C"') < briefs[1].index('name="B"')
    assert "Why C: 2 of 2 selector votes (scripted/test-model) chose it: Shorter." in report(agents)
    titles = [t for t, (role, _, _) in nodes(session).items() if role == "judge"]
    assert titles == ["Selection vote 1 of 3", "Selection vote 2 of 3"]
    assert (repo / "greeting.txt").read_text() == "hello world\n"


async def test_a_split_vote_goes_to_the_plurality(
    session: Session, agents: Agents, repo: Path, config: Config
) -> None:
    config.orchestration.set_option("best-of-n", "selector-votes", 3)
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hello world\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hi\n"))  # unchanged: no diff, so it drops out
    agents.add(CHOOSE, choose("A"), "I cannot decide.", "Still undecided.", "No.", choose("A"))
    await session.run(PROMPT)
    assert "Why A: 2 of 2 selector votes" in report(agents)
    assert nodes(session)["Selection vote 2 of 3"][1:] == ("failed", "No candidate was chosen.")


async def test_the_selector_prefers_another_model_family(
    session: Session, agents: Agents, repo: Path, config: Config, runtime: Runtime
) -> None:
    other = ScriptedProvider(provider_id="other", router=agents)
    runtime.providers.register(other)
    config.models.roles = {"reviewer": "other/gpt-judge"}
    agents.add(PROMPT, build(), "Done.")
    agents.candidate("A", *rewrite("hello world\n"))
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", *rewrite("hello there\n"))
    agents.add(CHOOSE, choose("B"))
    await session.run(PROMPT)
    assert [r.model for r in other.requests] == ["gpt-judge"]
    selection = next(n for n in session.view.nodes.values() if n.info.role == "judge")
    assert selection.info.model == "other/gpt-judge"


async def test_a_change_that_no_longer_applies_is_saved_not_forced(
    session: Session, agents: Agents, repo: Path
) -> None:
    def user_edits_meanwhile(request: ChatRequest) -> Any:
        (repo / "greeting.txt").write_text("edited by the user\n")
        return "Nothing to do."

    agents.add(PROMPT, build(), "It could not be applied.")
    agents.candidate("A", user_edits_meanwhile)
    agents.candidate("B", *rewrite("hello there\n"))
    agents.candidate("C", "Nothing to do.")
    await session.run(PROMPT)

    text = report(agents)
    assert text.startswith("Candidate B was chosen, but it could not be applied: git apply")
    assert "Nothing in the workspace was changed." in text
    assert (repo / "greeting.txt").read_text() == "edited by the user\n"
    saved = list((paths.state_dir() / "patches").iterdir())
    assert len(saved) == 1 and str(saved[0]) in text and "+hello there" in saved[0].read_text()
    assert nodes(session)["Candidate B"][2].endswith("selected; could not be applied")
    assert worktrees_left(repo) == ([], 1)


async def test_edit_rules_are_enforced_when_the_change_is_applied(
    session: Session, agents: Agents, repo: Path, config: Config
) -> None:
    config.permissions.deny = ["edit:greeting.txt"]
    agents.add(PROMPT, build(), "Permission was refused.")
    agents.candidate("A", call("bash", command="printf 'hello\\n' > greeting.txt"), "Done.")
    for label in "BC":
        agents.candidate(label, *rewrite("hello\n"))  # refused in the copy too
    await session.run(PROMPT)
    text = report(agents)
    assert text.startswith(
        "Candidate A was chosen, but it could not be applied: permission to change "
        "greeting.txt was denied. Nothing in the workspace was changed."
    )
    assert (repo / "greeting.txt").read_text() == "hi\n"
    assert nodes(session)["Candidate B"][1:] == ("failed", "it made no changes")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read unreadable files")
async def test_a_working_tree_that_cannot_be_captured_is_refused(
    session: Session, agents: Agents, repo: Path
) -> None:
    secret = repo / "unreadable.txt"
    secret.write_text("x")
    secret.chmod(0)
    try:
        agents.add(PROMPT, build(), "Nothing was changed.")
        await session.run(PROMPT)
    finally:
        secret.chmod(0o600)
    result = agents.requests[PROMPT][1].messages[-1].tool_result_blocks[0]
    assert result.is_error and result.content.startswith("Nothing was built or changed")
    assert list(nodes(session)) == ["Conversation"]


async def test_a_failed_model_call_stops_the_round_and_removes_the_worktrees(
    session: Session, agents: Agents, repo: Path
) -> None:
    agents.add(PROMPT, build(), "The round failed: the model key was rejected.")
    agents.candidate("A", *rewrite("hello\n"))
    agents.candidate("B", ProviderError("invalid API key", kind="auth"))
    agents.candidate("C", *rewrite("hello\n"))

    answer = await session.run(PROMPT)

    assert answer == "The round failed: the model key was rejected."
    result = agents.requests[PROMPT][1].messages[-1].tool_result_blocks[0]
    assert result.is_error and result.content == (
        "The round stopped because a model call failed: invalid API key Nothing in the "
        "workspace was changed."
    )
    assert nodes(session)["Candidate B"][1:] == ("failed", "model error: invalid API key")
    assert {status for _, status, _ in nodes(session).values()} <= {"done", "failed", "cancelled"}
    assert worktrees_left(repo) == ([], 1)
    assert (repo / "greeting.txt").read_text() == "hi\n"


async def test_worktrees_are_removed_when_the_run_is_cancelled(
    session: Session, agents: Agents, repo: Path, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest) -> AsyncIterator[Any]:
        if "You are candidate" in request.messages[0].text:
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.add(PROMPT, build())
    task = session.start(PROMPT)
    directory = paths.state_dir() / "worktrees"
    for _ in range(400):
        await asyncio.sleep(0.01)
        if directory.exists() and len(list(directory.iterdir())) == 3:
            break
    assert len(list(directory.iterdir())) == 3
    session.cancel()
    await task
    assert worktrees_left(repo) == ([], 1)
    assert {status for _, status, _ in nodes(session).values()} == {"cancelled"}


async def test_the_run_budget_stops_all_candidates(
    runtime: Runtime, agents: Agents, repo: Path, config: Config
) -> None:
    session = runtime.create_session(repo, interactive=False)
    config.agent.run_budget = 8  # every scripted call produces 5 output tokens
    agents.add(PROMPT, build(), "never reached: the budget stops this call too")
    for label in "ABC":
        agents.candidate(label, *rewrite("hello\n"))
    finished: list[RunFinished] = []
    session.bus.subscribe(lambda e: finished.append(e) if isinstance(e, RunFinished) else None)
    await session.run(PROMPT)
    assert finished[-1].status == "failed" and "budget" in finished[-1].error
    assert worktrees_left(repo) == ([], 1)
    assert (repo / "greeting.txt").read_text() == "hi\n"


async def test_without_git_candidates_take_turns_in_the_workspace(
    session: Session, agents: Agents, workspace: Path, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("best-of-n", "write-tests", True)
    (workspace / "greeting.txt").write_text("hi\n")
    seen: list[tuple[str, bool]] = []

    def look_then_read(request: ChatRequest) -> Any:
        seen.append(
            ((workspace / "greeting.txt").read_text(), (workspace / "scratch.txt").exists())
        )
        return call("read", file_path="greeting.txt")

    agents.add(PROMPT, build(), "Applied B.")
    agents.candidate(
        "A",
        call("write", file_path="scratch.txt", content="notes\n"),
        *rewrite("hi there\n"),
    )
    agents.candidate("B", look_then_read, *rewrite("hello there\n")[1:])
    agents.candidate("C", look_then_read, *rewrite("hello world\n")[1:])
    agents.add(CHOOSE, choose("B"))

    await session.run(PROMPT)

    assert seen == [("hi\n", False), ("hi\n", False)]
    assert (workspace / "greeting.txt").read_text() == "hello there\n"
    assert not (workspace / "scratch.txt").exists()
    warning = next(e.text for e in events if isinstance(e, Notice) and e.level == "warning")
    assert "not in a git repository" in warning and "Generated tests are skipped." in warning
    assert TESTS not in agents.requests
    candidate = agents.first("You are candidate B")
    assert "Make every file change with `edit` and `write`" in candidate.system
    brief = agents.first(CHOOSE).messages[0].text
    assert "+hello there" in brief and "+hello world" in brief
    assert session.undo() == ["greeting.txt", "scratch.txt"]
    assert (workspace / "greeting.txt").read_text() == "hi\n"


async def test_configured_approach_hints_are_handed_out_in_turn(
    session: Session, agents: Agents, repo: Path, config: Config
) -> None:
    config.orchestration.set_option("best-of-n", "approach-hints", "smallest change, test first")
    agents.add(PROMPT, build(), "Done.")
    for label in "ABC":
        agents.candidate(label, *rewrite("hello\n"))
    await session.run(PROMPT)
    hints = [
        agents.first(f"You are candidate {label}").messages[0].text.split("Approach to take: ")[1]
        for label in "ABC"
    ]
    assert [h.split("\n")[0] for h in hints] == ["smallest change", "test first", "smallest change"]


def test_best_of_n_is_registered_with_bounded_settings() -> None:
    cls = get_orchestrator("best-of-n")
    assert cls is BestOfNOrchestrator and cls.title == "Best of N"
    settings = {s.key: s for s in cls.settings}
    assert (settings["n"].default, settings["n"].minimum, settings["n"].maximum) == (3, 2, 8)
    assert settings["selector-votes"].maximum == 7 and settings["write-tests"].default is True
    assert "about N x 1.3 the cost of one attempt" in cls.guidance

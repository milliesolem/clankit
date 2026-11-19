"""The Task Tree orchestrator: delegation, scheduling, acceptance checks, retries, review."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ToolResultBlock

Step = Any


class Agents:
    """Routes each model call to the script of the agent that made it.

    Agents are told apart by a marker in their first user message: the user's prompt for
    the root, the subtask title for children. Because every brief quotes the user's request
    near its top, the marker that occurs LATEST in the message wins. Each agent's script
    advances by one step per model call that agent makes; unscripted calls fail loudly.
    """

    def __init__(self) -> None:
        self.scripts: dict[str, list[Step]] = {}
        self.calls: dict[str, int] = {}
        self.requests: dict[str, list[ChatRequest]] = {}

    def add(self, marker: str, *steps: Step) -> None:
        self.scripts[marker] = list(steps)

    def child(self, title: str, *steps: Step) -> None:
        """Script the agent working on the subtask titled `title`."""
        self.add(task_marker(title), *steps)

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


def task_marker(title: str) -> str:
    return f"<your_task>\n# {title}\n"


def sub(
    key: str,
    title: str,
    *,
    scope: list[str] | None = None,
    checks: list[str] | None = None,
    depends: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": key,
        "title": title,
        "brief": f"Build the {title.lower()} as specified here.",
    }
    if scope is not None:
        item["write_scope"] = scope
    if checks is not None:
        item["checks"] = checks
    if depends is not None:
        item["depends_on"] = depends
    if model is not None:
        item["model"] = model
    return item


def report(status: str = "done", summary: str = "Built it.", **extra: Any) -> Any:
    return call("report", status=status, summary=summary, **extra)


def tool_result(request: ChatRequest) -> ToolResultBlock:
    return request.messages[-1].tool_result_blocks[0]


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "task-tree"
    config.orchestration.set_option("task-tree", "review", "off")
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


def statuses(session: Session) -> dict[str, str]:
    return {n.info.title: n.info.status for n in session.view.nodes.values()}


async def test_simple_task_is_done_by_the_root_alone(session: Session, agents: Agents) -> None:
    agents.add("what is 2+2", "4")
    assert await session.run("what is 2+2") == "4"
    assert [n.info.title for n in session.view.nodes.values()] == ["Conversation"]
    tools = [t.name for t in agents.requests["what is 2+2"][0].tools]
    assert "delegate" in tools and "ask_user" in tools and "report" not in tools


async def test_eager_mode_pushes_splitting_in_the_prompt_and_the_tool(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "decomposition", "eager")
    agents.add("wide", "done")
    await session.run("wide")
    request = agents.requests["wide"][0]
    delegate = next(t for t in request.tools if t.name == "delegate")
    assert "Splitting is expected" in delegate.description
    assert "most tasks are better done yourself" not in delegate.description
    assert "Example call:" in delegate.description
    assert '"depends_on": ["parser"]' in delegate.description
    assert "Split the work up front" in request.system


async def test_adaptive_mode_keeps_the_cautious_guidance(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "decomposition", "adaptive")
    agents.add("narrow", "done")
    await session.run("narrow")
    delegate = next(t for t in agents.requests["narrow"][0].tools if t.name == "delegate")
    assert "most tasks are better done yourself" in delegate.description


async def test_parallel_subtasks_are_built_checked_and_reported(
    session: Session, agents: Agents, workspace: Path
) -> None:
    agents.add(
        "build two parts",
        call(
            "delegate",
            subtasks=[
                sub("a", "Part A", scope=["a.txt"], checks=["test -f a.txt"]),
                sub("b", "Part B", scope=["b.txt"], checks=["grep -q bee b.txt"]),
            ],
        ),
        "Both parts are in place.",
    )
    agents.child("Part A", call("write", file_path="a.txt", content="ay\n"), report())
    agents.child(
        "Part B",
        call("write", file_path="b.txt", content="bee\n"),
        report(verification="cat b.txt"),
    )

    answer = await session.run("build two parts")
    assert answer == "Both parts are in place."
    assert (workspace / "a.txt").read_text() == "ay\n" and (
        workspace / "b.txt"
    ).read_text() == "bee\n"
    assert statuses(session) == {"Conversation": "done", "Part A": "done", "Part B": "done"}

    result = tool_result(agents.requests["build two parts"][1])
    assert not result.is_error
    text = result.content
    assert "2 of 2 accepted" in text
    assert "  $ test -f a.txt\n  passed" in text and "  $ grep -q bee b.txt\n  passed" in text
    assert "a.txt  +1 -0 (new)" in text and "b.txt  +1 -0 (new)" in text
    assert "Verification reported by the agent: cat b.txt" in text

    brief = agents.requests[task_marker("Part A")][0].messages[0].text
    assert "<user_request>\nbuild two parts\n</user_request>" in brief
    assert "Part B (owns b.txt)" in brief and "- `test -f a.txt`" in brief
    child_tools = [t.name for t in agents.requests[task_marker("Part A")][0].tools]
    assert "report" in child_tools and "delegate" in child_tools and "ask_user" not in child_tools


async def test_children_cannot_write_outside_their_scope(
    session: Session, agents: Agents, workspace: Path
) -> None:
    agents.add(
        "scoped work",
        call(
            "delegate",
            subtasks=[sub("a", "Part A", scope=["a/"]), sub("b", "Part B", scope=["b/"])],
        ),
        "done",
    )
    agents.child(
        "Part A",
        call("write", file_path="b/stolen.txt", content="x"),
        report("partial", "Could not write."),
    )
    agents.child("Part B", report())
    await session.run("scoped work")
    refused = tool_result(agents.requests[task_marker("Part A")][1])
    assert refused.is_error
    assert not (workspace / "b" / "stolen.txt").exists()


@pytest.mark.parametrize(
    ("subtasks", "message"),
    [
        ([sub("a", "Only one")], "at least two"),
        ([sub("a", "A"), sub("a", "B")], "used twice"),
        ([sub("a", "A", depends=["zzz"]), sub("b", "B")], "not in this call"),
        ([sub("a", "A", depends=["b"]), sub("b", "B", depends=["a"])], "cycle"),
        ([sub("a", "A", scope=["../outside"]), sub("b", "B")], "inside the project"),
        ([sub("a", "A", scope=["/etc"]), sub("b", "B")], "inside the project"),
        ([sub("a", "A", model="nowhere/model-x"), sub("b", "B")], "not configured"),
        ([sub(f"s{i}", f"S{i}") for i in range(7)], "At most 6"),
        ([{"id": "a", "title": "A", "brief": "short"}, sub("b", "B")], "brief"),
    ],
)
async def test_invalid_delegations_are_refused_without_creating_nodes(
    session: Session, agents: Agents, subtasks: list[dict[str, Any]], message: str
) -> None:
    agents.add("plan it", call("delegate", subtasks=subtasks), "ok, doing it myself")
    await session.run("plan it")
    result = tool_result(agents.requests["plan it"][1])
    assert result.is_error and message in result.content
    assert len(session.view.nodes) == 1


async def test_failed_check_gets_a_fix_attempt(
    session: Session, agents: Agents, workspace: Path
) -> None:
    agents.add(
        "make it pass",
        call(
            "delegate",
            subtasks=[
                sub("a", "Part A", scope=["a.txt"], checks=["grep -q fixed a.txt"]),
                sub("b", "Part B", scope=["b.txt"]),
            ],
        ),
        "done",
    )
    agents.child(
        "Part A",
        call("write", file_path="a.txt", content="broken\n"),
        report(),
        call("write", file_path="a.txt", content="fixed\n"),
        report(),
    )
    agents.child("Part B", report())
    await session.run("make it pass")
    fix_prompt = agents.requests[task_marker("Part A")][2].messages[-1].text
    assert "acceptance checks failed" in fix_prompt and "grep -q fixed a.txt" in fix_prompt
    assert statuses(session)["Part A"] == "done"
    assert "2 of 2 accepted" in tool_result(agents.requests["make it pass"][1]).content


async def test_persistent_failure_is_redispatched_to_a_fresh_agent(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "fix-attempts", 0)
    agents.add(
        "hard part",
        call(
            "delegate",
            subtasks=[
                sub("a", "Part A", scope=["a.txt"], checks=["test -f a.txt"]),
                sub("b", "Part B", scope=["b.txt"]),
            ],
        ),
        "done",
    )
    agents.add(
        "Another agent tried this task", call("write", file_path="a.txt", content="ok"), report()
    )
    agents.child("Part A", report(summary="I think it works."))  # but writes nothing
    agents.child("Part B", report())
    await session.run("hard part")
    titles = statuses(session)
    assert titles["Part A"] == "failed" and titles["Part A (second attempt)"] == "done"
    retry_brief = agents.requests["Another agent tried this task"][0].messages[0].text
    assert "<previous_attempt>" in retry_brief and "I think it works." in retry_brief
    text = tool_result(agents.requests["hard part"][1]).content
    assert "2 of 2 accepted" in text and "Attempts: 2" in text


async def test_blocked_subtask_is_not_retried(session: Session, agents: Agents) -> None:
    agents.add(
        "needs info", call("delegate", subtasks=[sub("a", "Part A"), sub("b", "Part B")]), "asked"
    )
    agents.child(
        "Part A", report("blocked", "Which database?", open_issues=["Postgres or SQLite?"])
    )
    agents.child("Part B", report())
    await session.run("needs info")
    text = tool_result(agents.requests["needs info"][1]).content
    assert "[blocked] Part A" in text and "Postgres or SQLite?" in text
    assert "second attempt" not in " ".join(statuses(session))


async def test_dependencies_run_in_order_and_failures_cascade(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "redispatch", False)
    agents.add(
        "chain",
        call(
            "delegate",
            subtasks=[
                sub("first", "First", scope=["one/"]),
                sub("second", "Second", scope=["two/"], depends=["first"]),
                sub("third", "Third", scope=["three/"], depends=["second"]),
                sub("other", "Other", scope=["four/"]),
            ],
        ),
        "done",
    )
    agents.child("First", report(summary="Interface is in one/api.py."))
    agents.child("Second", report("failed", "Could not do it."))
    agents.child("Other", report())
    await session.run("chain")
    second_brief = agents.requests[task_marker("Second")][0].messages[0].text
    assert (
        "<results_you_build_on>" in second_brief and "Interface is in one/api.py." in second_brief
    )
    assert statuses(session)["Third"] == "cancelled"
    assert task_marker("Third") not in agents.calls
    assert "2 of 4 accepted" in tool_result(agents.requests["chain"][1]).content


async def test_overlapping_scopes_never_run_together(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    active = peak = 0
    original = scripted.stream

    async def tracking(request: ChatRequest):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        child = "# " in request.messages[0].text and "report" in [t.name for t in request.tools]
        if child:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
        try:
            async for event in original(request):
                yield event
        finally:
            if child:
                active -= 1

    scripted.stream = tracking  # type: ignore[method-assign]
    agents.add(
        "overlap",
        call(
            "delegate",
            subtasks=[
                sub("a", "Part A", scope=["src/"]),
                sub("b", "Part B", scope=["src/b.py"]),
                sub("c", "Part C"),
            ],
        ),
        "done",
    )
    for title in ("Part A", "Part B", "Part C"):
        agents.child(title, report())
    await session.run("overlap")
    assert peak == 1  # src/ contains src/b.py, and an unscoped subtask overlaps everything

    peak = 0
    agents.scripts["overlap"] += [  # the root agent keeps its conversation across prompts
        call(
            "delegate",
            subtasks=[
                sub("a", "Part D", scope=["d/"]),
                sub("b", "Part E", scope=["e/"]),
                sub("c", "Part F", scope=["f/"]),
            ],
        ),
        "done",
    ]
    for title in ("Part D", "Part E", "Part F"):
        agents.child(title, report())
    await session.run("now disjoint parts")
    assert peak == 3


async def test_depth_limit_removes_the_delegate_tool(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "max-depth", 1)
    agents.add(
        "shallow", call("delegate", subtasks=[sub("a", "Part A"), sub("b", "Part B")]), "done"
    )
    agents.child("Part A", report())
    agents.child("Part B", report())
    await session.run("shallow")
    names = [t.name for t in agents.requests[task_marker("Part A")][0].tools]
    assert "delegate" not in names and "report" in names
    assert "cannot be split any further" in agents.requests[task_marker("Part A")][0].system


async def test_depth_zero_means_no_delegation_at_all(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "max-depth", 0)
    agents.add("flat", "done")
    await session.run("flat")
    assert "delegate" not in [t.name for t in agents.requests["flat"][0].tools]


async def test_nested_delegation_and_node_budget(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "max-nodes", 4)
    agents.add(
        "deep",
        call(
            "delegate",
            subtasks=[sub("a", "Part A", scope=["a/"]), sub("b", "Part B", scope=["b/"])],
        ),
        "done",
    )
    agents.child(
        "Part A",
        call(
            "delegate",
            subtasks=[sub("x", "Piece X", scope=["a/x"]), sub("y", "Piece Y", scope=["a/y"])],
        ),
        call(
            "delegate",
            subtasks=[sub("z", "Piece Z", scope=["a/z"]), sub("w", "Piece W", scope=["a/w"])],
        ),
        report(),
    )
    agents.child("Part B", report())
    agents.child("Piece X", report())
    agents.child("Piece Y", report())
    await session.run("deep")
    view = session.view
    part_a = next(n for n in view.nodes.values() if n.info.title == "Part A")
    assert part_a.info.role == "branch" and len(part_a.children) == 2
    assert {view.nodes[c].info.depth for c in part_a.children} == {2}
    budget = tool_result(agents.requests[task_marker("Part A")][2])
    assert budget.is_error and "budget" in budget.content
    piece_brief = agents.requests[task_marker("Piece X")][0].messages[0].text
    assert "the user's request > Part A > Piece X" in piece_brief and "<parent_task>" in piece_brief


async def test_subtasks_can_run_on_other_models(
    session: Session, agents: Agents, scripted: ScriptedProvider, config: Config
) -> None:
    other = ScriptedProvider(provider_id="other", router=agents)
    session.runtime.providers.register(other)
    agents.add(
        "mixed",
        call(
            "delegate",
            subtasks=[
                sub("a", "Part A", model="other/cheap-model"),
                sub("b", "Part B", scope=["b/"]),
            ],
        ),
        "done",
    )
    agents.child("Part A", report())
    agents.child("Part B", report())
    await session.run("mixed")
    nodes = {n.info.title: n.info.model for n in session.view.nodes.values()}
    assert nodes["Part A"] == "other/cheap-model" and nodes["Part B"] == "scripted/test-model"
    assert [r.model for r in other.requests] == ["cheap-model"]
    assert "model" in str(agents.requests["mixed"][0].tools[-1].input_schema)


async def test_cross_model_can_be_switched_off(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "cross-model", False)
    agents.add("single model", "done")
    await session.run("single model")
    delegate = next(t for t in agents.requests["single model"][0].tools if t.name == "delegate")
    assert "model" not in delegate.input_schema["properties"]["subtasks"]["items"]["properties"]


async def test_final_review_feeds_findings_back_to_the_root(
    session: Session, agents: Agents, config: Config, workspace: Path
) -> None:
    config.orchestration.set_option("task-tree", "review", "final")
    agents.add(
        "add a file",
        call("write", file_path="f.py", content="x = 1\n"),
        "Added f.py.",
        "Fixed the finding.",
    )
    agents.add(
        "Review the result against the task",
        call(
            "submit_review",
            verdict="changes_needed",
            findings=[
                {
                    "severity": "major",
                    "location": "f.py:1",
                    "problem": "Wrong value.",
                    "evidence": "x = 1 but the task needs 2",
                },
                {
                    "severity": "minor",
                    "location": "f.py:1",
                    "problem": "No evidence.",
                    "evidence": "",
                },
            ],
        ),
    )
    answer = await session.run("add a file")
    assert answer == "Fixed the finding."
    reviewer = next(n for n in session.view.nodes.values() if n.info.role == "reviewer")
    assert reviewer.info.status == "done" and reviewer.info.summary == "1 finding(s), 1 blocking."
    handed_back = agents.requests["add a file"][2].messages[-1].text
    assert "Wrong value." in handed_back and "No evidence." not in handed_back
    review_request = agents.requests["Review the result against the task"][0]
    assert "+x = 1" in review_request.messages[0].text
    assert {"write", "edit", "delegate"}.isdisjoint(t.name for t in review_request.tools)


async def test_no_review_when_nothing_changed(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "review", "final")
    agents.add("just answer", "An answer.")
    assert await session.run("just answer") == "An answer."
    assert len(session.view.nodes) == 1


async def test_agent_that_never_reports_is_reminded_then_failed(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("task-tree", "redispatch", False)
    agents.add(
        "forgetful", call("delegate", subtasks=[sub("a", "Part A"), sub("b", "Part B")]), "done"
    )
    agents.child("Part A", "I did it.", "Really, I did.", "Promise.")
    agents.child("Part B", report())
    await session.run("forgetful")
    reminders = [r.messages[-1].text for r in agents.requests[task_marker("Part A")][1:]]
    assert all("without calling `report`" in text for text in reminders)
    assert statuses(session)["Part A"] == "failed"
    assert "1 of 2 accepted" in tool_result(agents.requests["forgetful"][1]).content


async def test_cancelling_during_delegation_cancels_every_node(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if "# Part" in request.messages[0].text:
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.add(
        "slow",
        call(
            "delegate", subtasks=[sub("a", "Part A", scope=["a"]), sub("b", "Part B", scope=["b"])]
        ),
    )
    task = session.start("slow")
    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(session.view.nodes) == 3:
            break
    await asyncio.sleep(0.02)
    session.cancel()
    await task
    assert set(statuses(session).values()) == {"cancelled"}


async def test_events_tell_the_tree_story(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(
        "story",
        call(
            "delegate", subtasks=[sub("a", "Part A", scope=["a"]), sub("b", "Part B", scope=["b"])]
        ),
        "done",
    )
    agents.child("Part A", report(summary="A is built."))
    agents.child("Part B", report())
    await session.run("story")
    root_changes = [
        e.status for e in events if e.kind == "node_changed" and e.node_id == "n1" and e.status
    ]  # type: ignore[attr-defined]
    assert root_changes == ["running", "waiting", "integrating", "done"]
    part_a = next(n for n in session.view.nodes.values() if n.info.title == "Part A")
    assert part_a.info.summary == "done: A is built." and part_a.info.brief.startswith(
        "Build the part a"
    )


def test_report_parsing_is_defensive() -> None:
    from clankit.orchestration.task_tree import NO_REPORT, Report

    assert Report.parse(None, NO_REPORT).status == "failed"
    parsed = Report.parse(
        {"status": "weird", "summary": "  ", "open_issues": ["", " x "]}, NO_REPORT
    )
    assert (parsed.status, parsed.summary, parsed.open_issues) == (
        "failed",
        "(no summary given)",
        ["x"],
    )

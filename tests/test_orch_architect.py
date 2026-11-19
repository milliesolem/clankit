"""The Architect-Editor orchestrator: planning, consultations, direct answers, final check."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clankit.agent.events import Event, NodeChanged, NodeInfo, Notice, QuestionAsked, RunFinished
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ChatResponse, Message, ToolResultBlock, Usage
from clankit.orchestration.architect import Plan, parse_plan
from clankit.runtime import Runtime

Step = Any

PROMPT = "add a constant to f.py"
ARCHITECT = "Explore the code this request involves"
REVIEWER = "Review the result against the task"


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
    config.orchestration.method = "architect-editor"
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


def plan(**extra: Any) -> Step:
    payload: dict[str, Any] = {
        "kind": "change",
        "approach": "Add X next to the other constants.",
        "files": [{"path": "f.py", "change": "add X = 2"}],
        "steps": ["Add the constant.", "Run the check."],
        "checks": ["python -c 'import f; assert f.X == 2'"],
        "notes": ["Do not touch g.py."],
    }
    payload.update(extra)
    return call("submit_plan", **payload)


def write_f(value: int = 2) -> Step:
    return call("write", file_path="f.py", content=f"X = {value}\n")


def refusal(detail: str) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
        usage=Usage(),
        detail=detail,
    )


def nodes(session: Session) -> dict[str, NodeInfo]:
    return {view.info.title: view.info for view in session.view.nodes.values()}


def tool_result(request: ChatRequest) -> ToolResultBlock:
    return request.messages[-1].tool_result_blocks[0]


def tool_names(request: ChatRequest) -> set[str]:
    return {tool.name for tool in request.tools}


def statuses(events: list[Event], node_id: str) -> list[str]:
    return [
        e.status for e in events if isinstance(e, NodeChanged) and e.node_id == node_id and e.status
    ]


# --------------------------------------------------------------------------------------------
# Plan, then build
# --------------------------------------------------------------------------------------------


async def test_the_architect_plans_and_the_editor_builds_from_the_plan(
    session: Session, agents: Agents, workspace: Path, events: list[Event]
) -> None:
    (workspace / "f.py").write_text("Y = 1\n")
    agents.add(ARCHITECT, call("read", file_path="f.py"), plan())
    agents.add(PROMPT, call("read", file_path="f.py"), write_f(), "Added X = 2 to f.py.")

    answer = await session.run(PROMPT)

    assert answer == "Added X = 2 to f.py."
    assert (workspace / "f.py").read_text() == "X = 2\n"
    tree = nodes(session)
    assert set(tree) == {"Conversation", "Plan"}
    root, architect = tree["Conversation"], tree["Plan"]
    assert (architect.role, architect.status, architect.parent_id) == ("planner", "done", root.id)
    assert architect.summary == "Plan: 1 file(s), 2 step(s), 1 check(s)."
    assert (root.status, root.summary) == (
        "done",
        "Worked from the architect's plan; 0 question(s) asked.",
    )
    assert statuses(events, root.id) == ["waiting", "running", "done"]

    architect_request = agents.requests[ARCHITECT][0]
    assert f"<request>\n{PROMPT}\n</request>" in architect_request.messages[0].text
    assert tool_names(architect_request) == {
        "read",
        "glob",
        "grep",
        "web_fetch",
        "skill",
        "submit_plan",
    }

    message = agents.requests[PROMPT][0].messages[0].text
    assert message.startswith(PROMPT)
    assert "[Plan from the architect]\nAn architect agent (scripted/test-model)" in message
    assert (
        "<plan>\nApproach: Add X next to the other constants.\nFiles:\n- f.py: add X = 2" in message
    )
    assert "1. Add the constant.\n2. Run the check." in message
    assert (
        "Acceptance checks:\n- python -c" in message and "Notes:\n- Do not touch g.py." in message
    )
    assert "`consult_architect`" in message
    editor_tools = tool_names(agents.requests[PROMPT][0])
    assert {"write", "edit", "bash", "ask_user", "consult_architect"} <= editor_tools
    assert "Working from an architect's plan" in agents.requests[PROMPT][0].system


async def test_the_editor_can_consult_the_architect_within_a_limit(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("architect-editor", "max-consults", 1)
    agents.add(ARCHITECT, plan(), "Put X after Y; Y is imported elsewhere.")
    agents.add(
        PROMPT,
        call("consult_architect", question="Where does X go? f.py has Y."),
        call("consult_architect", question="And the tests?"),
        write_f(),
        "Done.",
    )

    await session.run(PROMPT)

    consult = agents.requests[ARCHITECT][1].messages[-1].text
    assert consult.startswith("[Question from the editor]\nWhere does X go? f.py has Y.")
    answer = tool_result(agents.requests[PROMPT][1])
    assert not answer.is_error and answer.content == "Put X after Y; Y is imported elsewhere."
    refused = tool_result(agents.requests[PROMPT][2])
    assert refused.is_error and "No questions to the architect are left" in refused.content
    tree = nodes(session)
    assert (
        tree["Plan"].summary
        == "Plan: 1 file(s), 2 step(s), 1 check(s). Answered 1 question(s) from the editor."
    )
    assert statuses(events, tree["Plan"].id) == ["done", "running", "done"]  # created running
    assert tree["Conversation"].summary == "Worked from the architect's plan; 1 question(s) asked."


async def test_a_revised_plan_from_a_consultation_replaces_the_old_one(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("architect-editor", "final-check", True)
    agents.add(ARCHITECT, plan(), plan(approach="Put X in a new module instead."))
    agents.add(
        PROMPT,
        call("consult_architect", question="f.py is generated; where should X go?"),
        call("write", file_path="constants.py", content="X = 2\n"),
        "Done.",
    )
    agents.add(REVIEWER, call("submit_review", verdict="approve", findings=[]))

    await session.run(PROMPT)

    revised = tool_result(agents.requests[PROMPT][1]).content
    assert revised.startswith("The architect revised the plan:\nApproach: Put X in a new module")
    assert (
        "Approach: Put X in a new module instead." in agents.requests[REVIEWER][0].messages[0].text
    )


async def test_without_consultations_the_plan_does_not_offer_them(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("architect-editor", "max-consults", 0)
    agents.add(ARCHITECT, plan())
    agents.add(PROMPT, call("consult_architect", question="Where?"), "Done.")

    await session.run(PROMPT)

    assert "`consult_architect`" not in agents.requests[PROMPT][0].messages[0].text
    refused = tool_result(agents.requests[PROMPT][1])
    assert refused.is_error and "(0 allowed)" in refused.content
    assert agents.calls[ARCHITECT] == 1


# --------------------------------------------------------------------------------------------
# Questions and missing plans
# --------------------------------------------------------------------------------------------


async def test_a_question_is_answered_by_the_architect_and_the_editor_hears_of_it(
    session: Session, agents: Agents
) -> None:
    agents.add(
        ARCHITECT,
        call("submit_plan", kind="answer", answer="f.py defines Y, used by g.py."),
        plan(),
    )
    agents.add(PROMPT, write_f(), "Done.")

    assert await session.run("what is in f.py") == "f.py defines Y, used by g.py."
    assert PROMPT not in agents.calls  # the editor did not run
    tree = nodes(session)
    assert tree["Plan"].summary == "Answered; nothing to change."
    assert tree["Conversation"].summary == "Answered by the architect; nothing to change."
    conversation = [(i.kind, i.text) for i in session.view.conversation() if i.kind != "notice"]
    assert conversation == [
        ("user", "what is in f.py"),
        ("assistant", "f.py defines Y, used by g.py."),
    ]

    await session.run(PROMPT)

    message = agents.requests[PROMPT][0].messages[0].text
    assert message.startswith("Earlier in this session:")
    assert "what is in f.py" in message and "f.py defines Y, used by g.py." in message
    assert f"The user now says:\n{PROMPT}" in message


async def test_answers_the_editor_missed_are_passed_on_once(
    session: Session, agents: Agents
) -> None:
    agents.add(
        ARCHITECT,
        plan(),
        call("submit_plan", kind="answer", answer="It is 2 now."),
        plan(),
    )
    agents.add(PROMPT, write_f(), "First change.", write_f(3), "Second change.")

    await session.run(PROMPT)
    await session.run("what is X now")
    await session.run("set X to 3")

    second = agents.requests[PROMPT][2].messages[-1].text
    assert second.startswith("Earlier in this session, the architect answered these requests")
    assert "what is X now" in second and "It is 2 now." in second
    assert "The user now says:\nset X to 3" in second


async def test_an_unusable_plan_is_asked_for_again_then_the_editor_works_alone(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(
        ARCHITECT,
        call("submit_plan", kind="change"),
        call("submit_plan", kind="answer", answer="  "),
    )
    agents.add(PROMPT, call("consult_architect", question="Any advice?"), write_f(), "Done alone.")

    answer = await session.run(PROMPT)

    assert answer == "Done alone."
    assert (
        agents.requests[ARCHITECT][1].messages[-1].text.startswith("[Harness] That cannot be used")
    )
    assert (nodes(session)["Plan"].status, nodes(session)["Plan"].summary) == (
        "failed",
        "The architect handed in no usable plan.",
    )
    assert agents.requests[PROMPT][0].messages[0].text == PROMPT
    assert "no plan was made" in tool_result(agents.requests[PROMPT][1]).content
    assert any(isinstance(e, Notice) and "the editor works without one" in e.text for e in events)
    assert nodes(session)["Conversation"].summary == "Worked without a plan."


async def test_a_refusing_architect_leaves_the_request_to_the_editor(
    session: Session, agents: Agents
) -> None:
    agents.add(ARCHITECT, refusal("No."))
    agents.add(PROMPT, "I can do this one.")

    assert await session.run(PROMPT) == "I can do this one."
    assert nodes(session)["Plan"].summary == "The architect declined to plan this request."


async def test_a_refusing_editor_ends_the_request(session: Session, agents: Agents) -> None:
    agents.add(ARCHITECT, plan())
    agents.add(PROMPT, refusal("Not able to help."))

    assert await session.run(PROMPT) == "The model declined this request. Not able to help."
    assert nodes(session)["Conversation"].status == "failed"


# --------------------------------------------------------------------------------------------
# Final check
# --------------------------------------------------------------------------------------------


async def test_the_final_check_hands_blocking_findings_back_once(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("architect-editor", "final-check", True)
    agents.add(ARCHITECT, plan())
    agents.add(PROMPT, write_f(1), "Added X.", write_f(2), "Fixed X.")
    agents.add(
        REVIEWER,
        call(
            "submit_review",
            verdict="changes_needed",
            findings=[
                {
                    "severity": "major",
                    "location": "f.py:1",
                    "problem": "The plan says X = 2.",
                    "evidence": "f.py:1 reads X = 1",
                }
            ],
        ),
    )

    answer = await session.run(PROMPT)

    assert answer == (
        "Fixed X.\n\nCheck against the plan (scripted/test-model): 1 blocking finding(s) handed "
        "back to the editor; its fixes were not checked again."
    )
    check = nodes(session)["Check against the plan"]
    assert (check.role, check.status) == ("reviewer", "done")
    brief = agents.requests[REVIEWER][0].messages[0].text
    assert "The changes were meant to follow this plan" in brief and "Approach: Add X" in brief
    assert "The plan says X = 2." in agents.requests[PROMPT][2].messages[-1].text


async def test_the_final_check_is_skipped_when_nothing_changed(
    session: Session, agents: Agents, config: Config
) -> None:
    config.orchestration.set_option("architect-editor", "final-check", True)
    agents.add(ARCHITECT, plan())
    agents.add(PROMPT, "Nothing needed changing.")

    assert await session.run(PROMPT) == "Nothing needed changing."
    assert "Check against the plan" not in nodes(session)


# --------------------------------------------------------------------------------------------
# Models, steering, failures
# --------------------------------------------------------------------------------------------


async def test_architect_and_editor_run_on_their_configured_models(
    session: Session, agents: Agents, config: Config
) -> None:
    strong = ScriptedProvider(provider_id="strong", router=agents)
    cheap = ScriptedProvider(provider_id="cheap", router=agents)
    session.runtime.providers.register(strong)
    session.runtime.providers.register(cheap)
    config.models.roles["planner"] = "strong/role-model"
    config.orchestration.set_option("architect-editor", "editor-model", "cheap/small-model")
    agents.add(ARCHITECT, plan(), plan())
    agents.add(PROMPT, write_f(), "Done.", "Done again.")

    await session.run(PROMPT)
    assert nodes(session)["Plan"].model == "strong/role-model"
    assert nodes(session)["Conversation"].model == "cheap/small-model"

    config.orchestration.set_option("architect-editor", "architect-model", "strong/best-model")
    await session.run("again")
    plans = [n.info for n in session.view.nodes.values() if n.info.role == "planner"]
    assert plans[-1].model == "strong/best-model"
    assert [r.model for r in strong.requests] == ["role-model", "best-model"]
    assert {r.model for r in cheap.requests} == {"small-model"}


async def test_steering_while_planning_reaches_the_architect(
    session: Session, agents: Agents
) -> None:
    def steer_then_read(request: ChatRequest) -> Step:
        assert session.steer("keep g.py untouched")
        return call("glob", pattern="*.py")

    agents.add(ARCHITECT, steer_then_read, plan())
    agents.add(PROMPT, "Done.")

    await session.run(PROMPT)

    assert "keep g.py untouched" in agents.requests[ARCHITECT][1].messages[-1].text
    assert all("keep g.py" not in m.text for m in agents.requests[PROMPT][0].messages)


async def test_an_unreachable_architect_during_a_consultation_does_not_stop_the_editor(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(ARCHITECT, plan(), ProviderError("overloaded", kind="bad_request", provider="x"))
    agents.add(PROMPT, call("consult_architect", question="Where?"), write_f(), "Done anyway.")

    assert await session.run(PROMPT) == "Done anyway."
    result = tool_result(agents.requests[PROMPT][1])
    assert result.is_error and "The architect could not answer (" in result.content
    assert any(isinstance(e, Notice) and "could not answer a question" in e.text for e in events)
    assert nodes(session)["Plan"].status == "done"


async def test_the_run_budget_stops_the_run_even_inside_a_consultation(
    runtime: Runtime, workspace: Path, scripted: ScriptedProvider
) -> None:
    session = runtime.create_session(workspace, interactive=False)
    session.config.agent.run_budget = 10  # each scripted call generates 5 output tokens
    agents = Agents()
    scripted._router = agents
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    agents.add(ARCHITECT, plan(), "Never reached.")
    agents.add(PROMPT, call("consult_architect", question="Where?"), "Never reached.")

    assert await session.run(PROMPT) == ""

    finished = seen[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "budget" in finished.error
    assert agents.calls == {ARCHITECT: 1, PROMPT: 1}


async def test_stopping_at_the_budget_during_a_consultation_is_not_asked_again(
    runtime: Runtime, workspace: Path, scripted: ScriptedProvider
) -> None:
    session = runtime.create_session(workspace, interactive=True)
    session.config.agent.run_budget = 10  # each scripted call generates 5 output tokens
    agents = Agents()
    scripted._router = agents
    seen: list[Event] = []

    def answer_stop(event: Event) -> None:
        seen.append(event)
        if isinstance(event, QuestionAsked):
            asyncio.get_running_loop().call_soon(session.answer_question, event.request_id, "Stop")

    session.bus.subscribe(answer_stop)
    agents.add(ARCHITECT, plan(), "Never reached.")
    agents.add(PROMPT, call("consult_architect", question="Where?"), "Never reached.")

    assert await session.run(PROMPT) == ""

    asked = [e for e in seen if isinstance(e, QuestionAsked)]
    assert len(asked) == 1 and asked[0].node_id == nodes(session)["Plan"].id
    finished = seen[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    assert "budget" in finished.error
    assert agents.calls == {ARCHITECT: 1, PROMPT: 1}  # the editor did not carry on
    assert not any(isinstance(e, Notice) and "could not answer" in e.text for e in seen)


async def test_cancelling_during_a_consultation_cancels_the_architect_too(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if request.messages[-1].text.startswith("[Question from the editor]"):
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.add(ARCHITECT, plan(), "Never reached.")
    agents.add(PROMPT, call("consult_architect", question="Where?"), "Never reached.")

    task = session.start(PROMPT)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if agents.calls.get(PROMPT) and nodes(session)["Plan"].status == "running":
            break
    await asyncio.sleep(0.02)
    assert session.cancel()
    assert await task == ""

    assert {n.status for n in nodes(session).values()} == {"cancelled"}


async def test_the_architect_node_summarises_the_plan_a_consultation_revised(
    session: Session, agents: Agents
) -> None:
    revised = plan(
        approach="Split the constants into two modules.",
        files=[{"path": "a.py", "change": "add X"}, {"path": "b.py", "change": "add Y"}],
        steps=["Create a.py.", "Create b.py.", "Import both."],
    )
    agents.add(ARCHITECT, plan(), revised)
    agents.add(PROMPT, call("consult_architect", question="f.py is generated; where?"), "Done.")

    await session.run(PROMPT)

    assert nodes(session)["Plan"].summary == (
        "Plan: 2 file(s), 3 step(s), 1 check(s). Answered 1 question(s) from the editor."
    )


async def test_an_unreachable_architect_while_planning_fails_the_run(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.add(ARCHITECT, ProviderError("invalid api key", kind="auth", provider="scripted"))

    assert await session.run(PROMPT) == ""
    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "failed"
    architect = nodes(session)["Plan"]
    assert architect.status == "failed" and "invalid api key" in architect.summary


async def test_cancelling_while_planning_cancels_every_node(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    original = scripted.stream

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if ARCHITECT in request.messages[0].text:
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    task = session.start(PROMPT)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if "Plan" in nodes(session):
            break
    await asyncio.sleep(0.02)
    assert session.cancel()
    assert await task == ""
    assert {n.status for n in nodes(session).values()} == {"cancelled"}


def test_plans_are_parsed_defensively() -> None:
    assert parse_plan(None) is None
    assert parse_plan({"kind": "answer", "answer": " It is 2. "}) == "It is 2."
    assert parse_plan({"kind": "answer"}) is None
    assert parse_plan({"kind": "change", "files": [{"path": " "}], "steps": ["", " "]}) is None
    parsed = parse_plan(
        {"kind": "change", "files": [{"path": "a.py", "change": "x"}, "junk"], "steps": [" s "]}
    )
    assert parsed == Plan(approach="", files=[("a.py", "x")], steps=["s"], checks=[], notes=[])
    assert parsed.render() == "Files:\n- a.py: x\nSteps:\n1. s"

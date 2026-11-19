"""The Ultra orchestrator, end to end through Session.run."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from clankit import paths
from clankit.agent.events import Event, NodeChanged, Notice, ToolStarted
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, ToolResultBlock, Usage
from clankit.orchestration import get_orchestrator
from clankit.orchestration.ultra import UltraOrchestrator
from clankit.orchestration.workflow import Limits, WorkflowRun, compile_script, load_saved

Step = Any

SCAN = """\
meta = {"name": "scan-files", "description": "Scan files.", "phases": ["Scan"]}
phase("Scan")
found = await parallel([agent(f"scan {path}", label=f"scan {path}") for path in args["files"]])
log(f"scanned {len(found)} files")
result = {"issues": found}
"""


class Agents:
    """Routes the user-facing agent to its script of steps and workflow agents by marker."""

    def __init__(self) -> None:
        self.root: list[Step] = []
        self.root_requests: list[ChatRequest] = []
        self.rules: list[tuple[str, Step]] = []
        self.worker_tasks: list[str] = []

    def on(self, marker: str, step: Step) -> None:
        self.rules.append((marker, step))

    def __call__(self, request: ChatRequest) -> Step:
        if "# Working with workflows" in request.system:
            self.root_requests.append(
                ChatRequest(request.model, request.system, list(request.messages), request.tools)
            )
            if len(self.root_requests) > len(self.root):
                raise AssertionError("the user-facing agent made more calls than scripted")
            return self.root[len(self.root_requests) - 1]
        task = request.messages[0].text
        self.worker_tasks.append(task)
        for marker, step in self.rules:
            if marker in task:
                return step
        raise AssertionError(f"no script for a workflow agent whose task is {task[:160]!r}")


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "ultra"
    return config


@pytest.fixture
def agents(scripted: ScriptedProvider) -> Agents:
    routed = Agents()
    scripted._router = routed
    return routed


def tool_result(request: ChatRequest) -> ToolResultBlock:
    return request.messages[-1].tool_result_blocks[0]


def echo_result(request: ChatRequest) -> str:
    """The user-facing agent's last turn: report the workflow's tool result verbatim."""
    return f"Here is what the workflow found:\n{tool_result(request).content}"


def roles(session: Session) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for node in session.view.nodes.values():
        grouped.setdefault(node.info.role, []).append(node.info)
    return grouped


def test_ultra_is_a_registered_method_with_its_settings(session: Session, config: Config) -> None:
    cls = get_orchestrator("ultra")
    assert cls is UltraOrchestrator and (cls.name, cls.title) == ("ultra", "Ultra")
    defaults = {s.key: s.default for s in cls.settings}
    assert defaults == {
        "activation": "always",
        "max-concurrent": 6,
        "max-agents": 60,
        "workflow-budget": 1_500_000,
        "verify-votes": 3,
        "large-workflow": 25,
    }
    config.orchestration.set_option("ultra", "max-agents", 12)
    config.orchestration.set_option("ultra", "verify-votes", 5)
    config.orchestration.set_option("ultra", "max-concurrent", 0)  # invalid: default applies
    limits = UltraOrchestrator(session).limits()
    assert (limits.max_agents, limits.verify_votes, limits.max_concurrent) == (12, 5, 6)


async def test_the_role_prompt_explains_workflows_for_the_chosen_activation(
    session: Session, agents: Agents, config: Config
) -> None:
    agents.root = ["Hello!"]
    assert await session.run("hi") == "Hello!"
    request = agents.root_requests[0]
    assert "Plan a workflow for every substantive task" in request.system
    assert "await agent(prompt, *, schema=None" in request.system
    assert "adversarial_verify(claims, votes=3" in request.system
    assert "- `review`: Review code changes" in request.system
    assert "one workflow may start 60 agents and use 1,500,000 tokens" in request.system
    assert {"workflow", "ask_user", "bash", "edit"} <= {t.name for t in request.tools}
    assert "Models for workflow agents" not in request.system

    config.orchestration.set_option("ultra", "activation", "auto")
    other = Session(session.runtime, session.workspace)
    agents.root.append("Hi again.")
    await other.run("hi")
    assert "Use a workflow when it clearly helps" in agents.root_requests[1].system


async def test_the_example_in_the_instructions_is_a_working_script(
    session: Session, agents: Agents
) -> None:
    text = UltraOrchestrator(session).role_text(load_saved())
    example = text.split("```python\n", 1)[1].split("```", 1)[0]
    script = compile_script(example)
    agents.on("list every SQL query", call("return_result", issues=["a.py:3 formats SQL"]))
    agents.on("try to refute it", call("return_result", verdict="upheld", reason="seen"))
    root = session.new_node("Conversation", role="root")
    outcome = await WorkflowRun(session, parent=root, limits=Limits()).execute(
        script, {"files": ["a.py"]}
    )
    assert outcome.status == "completed" and outcome.result == ["a.py:3 formats SQL"]


async def test_the_role_prompt_lists_models_when_there_is_a_choice(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    session.runtime.providers.register(
        ScriptedProvider(provider_id="other", router=agents, models=("cheap-model",))
    )
    await session.runtime.providers.discover()
    agents.root = ["Hello!"]
    await session.run("hi")
    system = agents.root_requests[0].system
    assert "## Models for workflow agents" in system and "`other/cheap-model`" in system


async def test_a_workflow_script_runs_and_its_result_reaches_the_user(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    agents.root = [
        call("workflow", script=SCAN, args={"files": ["a.py", "b.py"]}, estimated_agents=2),
        echo_result,
    ]
    agents.on("scan a.py", "a.py:3 builds SQL from input")
    agents.on("scan b.py", "nothing in b.py")
    answer = await session.run("find unsafe SQL in a.py and b.py")

    assert "a.py:3 builds SQL from input" in answer and "nothing in b.py" in answer
    assert 'Workflow "scan-files" completed: 2 agents (2 done)' in answer
    assert "- scanned 2 files" in answer
    result = tool_result(agents.root_requests[1])
    assert not result.is_error and "<result>" in result.content

    by_role = roles(session)
    (root,) = by_role["root"]
    (workflow,) = by_role["workflow"]
    (scan,) = by_role["phase"]
    assert (workflow.parent_id, scan.parent_id) == (root.id, workflow.id)
    assert [w.title for w in by_role["worker"]] == ["scan a.py", "scan b.py"]
    assert all(w.parent_id == scan.id and w.status == "done" for w in by_role["worker"])
    root_states = [
        e.status for e in events if isinstance(e, NodeChanged) and e.node_id == root.id and e.status
    ]
    assert root_states == ["running", "waiting", "running", "done"]
    started = next(e for e in events if isinstance(e, ToolStarted) and e.tool == "workflow")
    assert started.summary == "workflow: scan-files"


async def test_a_saved_workflow_runs_by_name(session: Session, agents: Agents) -> None:
    agents.root = [
        call("workflow", name="understand", args={"question": "How is config saved?"}),
        echo_result,
    ]
    agents.on("Investigate this question", call("return_result", summary="s", facts=[]))
    agents.on("Look for what is missing", call("return_result", gaps=[]))
    agents.on("Answer this question about the codebase", "Config is saved atomically.")
    answer = await session.run("explain config saving")
    assert "Config is saved atomically." in answer
    assert [p.title for p in roles(session)["phase"]] == ["Explore", "Check", "Explain"]


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [
        (
            {"script": 'meta = {"name": "x", "description": "d"}\nimport os\nresult = 1\n'},
            "line 2: imports are not available",
        ),
        ({"script": "x = 1\n", "name": "review"}, "not both and not neither"),
        ({}, "not both and not neither"),
        ({"name": "no-such-thing"}, "Saved: implement, research, review, understand."),
        ({"name": "review", "estimated_agents": 500}, "at most 60 (setting max-agents)"),
    ],
)
async def test_unusable_requests_are_refused_before_anything_runs(
    session: Session, agents: Agents, tool_input: dict[str, Any], expected: str
) -> None:
    agents.root = [call("workflow", **tool_input), "I will fix that."]
    assert await session.run("do it") == "I will fix that."
    result = tool_result(agents.root_requests[1])
    assert result.is_error and expected in result.content
    assert set(roles(session)) == {"root"} and not agents.worker_tasks


async def test_script_problems_are_all_listed_for_the_agent_to_fix(
    session: Session, agents: Agents
) -> None:
    script = 'meta = {"name": "x", "description": "d"}\nimport os\nx = time.time()\n'
    agents.root = [call("workflow", script=script), "Fixing."]
    await session.run("do it")
    content = tool_result(agents.root_requests[1]).content
    assert content.startswith("The script was not run: 3 problems found.")
    assert "line 2:" in content and "line 3:" in content and "never assigns `result`" in content


async def test_a_large_workflow_is_announced_to_the_user(
    session: Session, agents: Agents, config: Config, events: list[Event]
) -> None:
    config.orchestration.set_option("ultra", "large-workflow", 3)
    agents.root = [
        call("workflow", script=SCAN, args={"files": ["a.py"]}, estimated_agents=5),
        "done",
    ]
    agents.on("scan", "ok")
    await session.run("scan")
    notices = [e for e in events if isinstance(e, Notice)]
    assert any('Large workflow "scan-files": about 5 agents' in n.text for n in notices)


async def test_the_workflow_budget_stops_it_and_returns_what_finished(
    session: Session, agents: Agents, config: Config, scripted: ScriptedProvider
) -> None:
    config.orchestration.set_option("ultra", "workflow-budget", 10_000)
    config.orchestration.set_option("ultra", "max-concurrent", 1)
    scripted._usage = Usage(input_tokens=4_000, output_tokens=2_000)
    agents.root = [
        call("workflow", script=SCAN, args={"files": ["a.py", "b.py", "c.py"]}),
        "Partial results only.",
    ]
    agents.on("scan", lambda request: f"scanned {request.messages[0].text[5:]}")
    await session.run("scan three files")
    result = tool_result(agents.root_requests[1])
    assert result.is_error
    assert 'Workflow "scan-files" stopped before it finished' in result.content
    assert "12,000 tokens of the 10,000-token budget" in result.content
    assert "- scan a.py [Scan]: scanned a.py" in result.content
    assert "workflow-budget" in result.content
    statuses = [w.status for w in roles(session)["worker"]]
    assert statuses == ["done", "done"]


async def test_a_failed_script_can_be_fixed_and_resumed(session: Session, agents: Agents) -> None:
    broken = SCAN.replace('result = {"issues": found}', 'result = found["issues"]')
    agents.root = [
        call("workflow", script=broken, args={"files": ["a.py", "b.py"]}),
        call("workflow", script=SCAN, args={"files": ["a.py", "b.py"]}, resume=True),
        echo_result,
    ]
    agents.on("scan", "fine")
    answer = await session.run("scan")
    failed = tool_result(agents.root_requests[1])
    assert failed.is_error and "line 5: TypeError" in failed.content
    assert "resume: true" in failed.content
    assert len(agents.worker_tasks) == 2  # the resumed run reused both agents
    assert "2 agents (2 reused)" in answer and '"fine"' in answer


async def test_cancelling_mid_workflow_cancels_every_agent_and_leaves_no_tasks(
    session: Session, agents: Agents, scripted: ScriptedProvider
) -> None:
    original = scripted.stream
    running = asyncio.Event()

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        if "# Working with workflows" not in request.system:
            running.set()
            await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    agents.root = [call("workflow", script=SCAN, args={"files": ["a.py", "b.py", "c.py"]})]
    agents.on("scan", "never")
    before = asyncio.all_tasks()
    task = session.start("scan")
    await asyncio.wait_for(running.wait(), timeout=5)
    await asyncio.sleep(0.01)
    assert session.cancel()
    assert await task == ""
    by_role = roles(session)
    assert len(by_role["worker"]) == 3
    assert {n.status for nodes in by_role.values() for n in nodes} == {"cancelled"}
    assert session.view.last_run is not None and session.view.last_run.status == "cancelled"
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before - {asyncio.current_task()} == set()


async def test_broken_user_workflows_are_reported_when_the_agent_starts(
    session: Session, agents: Agents, events: list[Event]
) -> None:
    folder = paths.config_dir() / "workflows"
    folder.mkdir(parents=True)
    (folder / "mine.py").write_text('meta = {"name": "mine"}\nresult = 1\n')
    agents.root = ["Hello!"]
    await session.run("hi")
    warnings = [e.text for e in events if isinstance(e, Notice) and e.level == "warning"]
    assert len(warnings) == 1 and "mine.py" in warnings[0] and "description" in warnings[0]
    assert "- `mine`" not in agents.root_requests[0].system

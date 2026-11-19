"""The workflow engine: primitives, limits, structured results, the node tree, resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clankit import paths
from clankit.agent.events import Event, NodeInfo, Notice
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider, call
from clankit.llm.types import ChatRequest, Usage
from clankit.orchestration.workflow import Journal, Limits, Outcome, WorkflowRun, compile_script
from clankit.orchestration.workflow import engine as engine_module

Step = Any
META = 'meta = {"name": "t", "description": "A test workflow."}\n'


class Workers:
    """Answers each workflow agent by the first marker found in its task.

    Each marker has a list of steps used in order across all calls that match it; the last
    step repeats. Steps may be callables taking the request (see `ScriptedProvider`).
    """

    def __init__(self) -> None:
        self.rules: list[tuple[str, list[Step]]] = []
        self.requests: dict[str, list[ChatRequest]] = {}

    def on(self, marker: str, *steps: Step) -> None:
        self.rules.append((marker, list(steps)))

    def __call__(self, request: ChatRequest) -> Step:
        task = request.messages[0].text
        for marker, steps in self.rules:
            if marker in task:
                seen = self.requests.setdefault(marker, [])
                # The agent keeps appending to its message list; keep this call's view.
                seen.append(
                    ChatRequest(
                        request.model, request.system, list(request.messages), request.tools
                    )
                )
                return steps[min(len(seen) - 1, len(steps) - 1)]
        raise AssertionError(f"no script for a workflow agent whose task is {task[:160]!r}")

    def tasks(self) -> list[str]:
        return [r.messages[0].text for calls in self.requests.values() for r in calls]


@pytest.fixture
def workers(scripted: ScriptedProvider) -> Workers:
    routed = Workers()
    scripted._router = routed
    return routed


@pytest.fixture
def root(session: Session) -> NodeInfo:
    return session.new_node("Conversation", role="root")


async def run(
    session: Session,
    root: NodeInfo,
    source: str,
    args: Any = None,
    *,
    limits: Limits | None = None,
    journal: Journal | None = None,
    resume: bool = False,
    estimated: int | None = None,
) -> Outcome:
    script = compile_script(source)
    workflow = WorkflowRun(
        session,
        parent=root,
        limits=limits or Limits(),
        journal=journal.begin(script.meta.name, resume=resume) if journal else None,
    )
    return await workflow.execute(script, args, estimated_agents=estimated)


def nodes(session: Session, role: str) -> list[NodeInfo]:
    return [n.info for n in session.view.nodes.values() if n.info.role == role]


def fails(message: str = "the model is gone") -> ProviderError:
    return ProviderError(message, kind="bad_request")


ANSWER_SCHEMA = (
    'schema = {"type": "object", "properties": {"answer": {"type": "string"}}, '
    '"required": ["answer"], "additionalProperties": False}\n'
)


# --------------------------------------------------------------------------------------------
# agent()
# --------------------------------------------------------------------------------------------


async def test_agent_returns_the_final_text_and_appears_as_a_worker(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("say hello", "Hello there.")
    outcome = await run(session, root, META + 'result = await agent("say hello", label="greet")\n')
    assert outcome.status == "completed" and outcome.result == "Hello there."
    (worker,) = nodes(session, "worker")
    assert (worker.title, worker.model, worker.status) == ("greet", "scripted/test-model", "done")
    assert worker.summary == "Hello there." and worker.brief == "say hello"
    request = workers.requests["say hello"][0]
    assert "return value that the program reads" in request.system
    assert "ask_user" not in [t.name for t in request.tools]


async def test_read_only_agents_get_no_write_tools(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("look", "Seen.")
    await run(session, root, META + 'result = await agent("look", tools="read-only")\n')
    names = {t.name for t in workers.requests["look"][0].tools}
    assert names == {"read", "glob", "grep", "web_fetch", "skill"}


async def test_write_scope_limits_what_an_agent_may_change(
    session: Session, root: NodeInfo, workers: Workers, workspace: Path
) -> None:
    workers.on("write two files", call("write", file_path="b.txt", content="b"), "tried")
    await run(
        session, root, META + 'result = await agent("write two files", write_scope=["a.txt"])\n'
    )
    assert not (workspace / "b.txt").exists()
    refused = workers.requests["write two files"][1].messages[-1].tool_result_blocks[0]
    assert refused.is_error


async def test_a_failed_agent_returns_none_and_is_accounted_for(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("doomed", fails())
    outcome = await run(session, root, META + 'x = await agent("doomed")\nresult = [x]\n')
    assert outcome.status == "completed" and outcome.result == [None]
    assert nodes(session, "worker")[0].status == "failed"
    assert outcome.failures and outcome.failures[0].startswith('line 2: agent "doomed"')
    assert "could not be reached" in outcome.failures[0]


@pytest.mark.parametrize(
    ("call_text", "expected"),
    [
        ('agent("")', "agent prompt must be a non-empty string"),
        ('agent("x", tools="everything")', 'tools must be "all" or "read-only"'),
        ('agent("x", effort="huge")', "effort must be one of"),
        ('agent("x", model="nowhere/model")', "not configured"),
        ('agent("x", write_scope=["../outside"])', "inside the project"),
        ('agent("x", write_scope="a.txt")', "list of paths"),
        ('agent("x", schema={"type": "nonsense"})', "not a valid JSON Schema"),
        ('agent("x", schema={"type": "array"})', "root must be"),
        ('agent("x", schema={"type": "object", "required": ["a"]})', "does not define it"),
        ('agent("x", isolation="worktree")', "unexpected keyword argument 'isolation'"),
    ],
)
async def test_agent_argument_mistakes_fail_on_the_calling_line(
    session: Session, root: NodeInfo, workers: Workers, call_text: str, expected: str
) -> None:
    outcome = await run(session, root, META + f"x = 1\nresult = await {call_text}\n")
    assert outcome.status == "failed"
    assert outcome.error.startswith("line 3:") and expected in outcome.error
    assert not nodes(session, "worker") and not workers.requests


async def test_model_roles_and_explicit_models_choose_where_agents_run(
    session: Session, root: NodeInfo, workers: Workers, config: Config
) -> None:
    other = ScriptedProvider(provider_id="other", router=workers)
    session.runtime.providers.register(other)
    config.models.roles["reviewer"] = "other/strict-model"
    workers.on("check", "ok")
    await run(
        session,
        root,
        META + 'a = await agent("check a", model="reviewer")\n'
        'b = await agent("check b", model="other/cheap-model")\n'
        'c = await agent("check c")\nresult = 1\n',
    )
    assert [w.model for w in nodes(session, "worker")] == [
        "other/strict-model",
        "other/cheap-model",
        "scripted/test-model",
    ]
    assert [r.model for r in other.requests] == ["strict-model", "cheap-model"]


# --------------------------------------------------------------------------------------------
# Structured results
# --------------------------------------------------------------------------------------------


async def test_schema_results_come_back_as_validated_dicts(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("answer it", call("return_result", answer="42"))
    outcome = await run(
        session, root, META + ANSWER_SCHEMA + 'result = await agent("answer it", schema=schema)\n'
    )
    assert outcome.result == {"answer": "42"}
    request = workers.requests["answer it"][0]
    tool = next(t for t in request.tools if t.name == "return_result")
    assert tool.input_schema["required"] == ["answer"]  # the model sees the real schema
    assert "call `return_result` once" in request.messages[0].text


async def test_invalid_results_get_corrective_feedback(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on(
        "answer it",
        call("return_result", answer=42),
        call("return_result", answer="42"),
    )
    outcome = await run(
        session, root, META + ANSWER_SCHEMA + 'result = await agent("answer it", schema=schema)\n'
    )
    assert outcome.result == {"answer": "42"}
    feedback = workers.requests["answer it"][1].messages[-1].tool_result_blocks[0]
    assert feedback.is_error and "attempt 1 of 5" in feedback.content
    assert "answer: 42 is not of type 'string'" in feedback.content


async def test_results_that_stay_invalid_fail_after_five_attempts(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("answer it", call("return_result", answer=1))
    outcome = await run(
        session,
        root,
        META + ANSWER_SCHEMA + 'x = await agent("answer it", schema=schema)\nresult = [x]\n',
    )
    assert outcome.result == [None]
    assert len(workers.requests["answer it"]) == 5
    assert "no valid result after 5 attempts" in outcome.failures[0]
    assert nodes(session, "worker")[0].status == "failed"


async def test_an_agent_that_forgets_the_result_tool_is_reminded(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("answer it", "It is 42.", call("return_result", answer="42"))
    outcome = await run(
        session, root, META + ANSWER_SCHEMA + 'result = await agent("answer it", schema=schema)\n'
    )
    assert outcome.result == {"answer": "42"}
    assert "without calling `return_result`" in workers.requests["answer it"][1].messages[-1].text


# --------------------------------------------------------------------------------------------
# parallel() and pipeline()
# --------------------------------------------------------------------------------------------


async def test_parallel_is_a_barrier_whose_failed_slots_become_none(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("fine", "ok")
    workers.on("broken", fails())
    outcome = await run(
        session,
        root,
        META + "result = await parallel([\n"
        '    agent("fine"),\n'
        '    agent("broken"),\n'
        "    lambda: 1 / 0,\n"
        "    lambda: 7,\n"
        "])\n",
    )
    assert outcome.status == "completed"
    assert outcome.result == ["ok", None, None, 7]
    assert any("ZeroDivisionError" in f and "line 5" in f for f in outcome.failures)
    assert outcome.empties == ["line 2: parallel of 4 slots: 2 ended without a result"]


async def test_parallel_rejects_values_it_cannot_run(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(session, root, META + "result = await parallel([1, 2])\n")
    assert outcome.status == "failed" and "parallel() takes awaitable calls" in outcome.error


async def test_coroutines_in_a_comprehension_keep_their_own_loop_values(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("vote", lambda request: request.messages[0].text.upper())
    outcome = await run(
        session,
        root,
        META + "a = await parallel([agent(f'vote {k}') for k in range(3)])\n"
        "b = await parallel([lambda k=k: agent(f'vote {k + 3}') for k in range(3)])\n"
        "result = a + b\n",
    )
    assert outcome.result == [f"VOTE {k}" for k in range(6)]


async def test_pipeline_has_no_barrier_between_stages(
    session: Session, root: NodeInfo, workers: Workers, scripted: ScriptedProvider
) -> None:
    fast_finished = asyncio.Event()
    original = scripted.stream

    async def gated(request: ChatRequest):  # type: ignore[no-untyped-def]
        task = request.messages[0].text
        if task == "stage1 slow":
            # Only a pipeline without barriers lets the fast item reach stage 3 meanwhile.
            await asyncio.wait_for(fast_finished.wait(), timeout=5)
        async for event in original(request):
            yield event
        if task == "stage3 fast":
            fast_finished.set()

    scripted.stream = gated  # type: ignore[method-assign]
    workers.on("stage", lambda request: f"did {request.messages[0].text}")
    outcome = await run(
        session,
        root,
        META + "async def first(previous, item, index):\n"
        "    return await agent(f'stage1 {item}')\n"
        "result = await pipeline(\n"
        "    ['slow', 'fast'],\n"
        "    first,\n"
        "    lambda previous, item, index: agent(f'stage2 {item}'),\n"
        "    lambda previous, item, index: agent(f'stage3 {item}'),\n"
        ")\n",
    )
    assert outcome.status == "completed"
    assert outcome.result == ["did stage3 slow", "did stage3 fast"]
    order = workers.tasks()
    assert order.index("stage3 fast") < order.index("stage1 slow")


async def test_pipeline_stages_get_previous_item_and_index_and_none_ends_an_item(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(
        session,
        root,
        META + "def keep_even(previous, item, index):\n"
        "    return previous if item % 2 == 0 else None\n"
        "def describe(previous, item, index):\n"
        "    return f'{index}:{item}:{previous * 10}'\n"
        "result = await pipeline([0, 1, 2, 3], keep_even, describe)\n",
    )
    assert outcome.result == ["0:0:0", None, "2:2:20", None]
    assert outcome.empties == ["line 6: pipeline of 4 items: 2 ended without a result"]
    assert not outcome.failures


async def test_a_failing_stage_drops_only_its_item(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(
        session,
        root,
        META + "result = await pipeline(['a', 'b'], lambda p, item, i: {'a': 1}[item])\n",
    )
    assert outcome.status == "completed" and outcome.result == [1, None]
    assert "KeyError: 'b'" in outcome.failures[0] and outcome.failures[0].startswith("line 2:")


# --------------------------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------------------------


async def test_concurrency_is_capped(
    session: Session, root: NodeInfo, workers: Workers, scripted: ScriptedProvider
) -> None:
    active = peak = 0
    original = scripted.stream

    async def counting(request: ChatRequest):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        try:
            async for event in original(request):
                yield event
        finally:
            active -= 1

    scripted.stream = counting  # type: ignore[method-assign]
    workers.on("job", "done")
    outcome = await run(
        session,
        root,
        META + "result = await parallel([agent(f'job {k}') for k in range(5)])\n",
        limits=Limits(max_concurrent=2),
    )
    assert outcome.result == ["done"] * 5
    assert peak == 2
    assert {n.status for n in nodes(session, "worker")} == {"done"}


async def test_starting_more_agents_than_allowed_stops_the_workflow(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", lambda request: f"finished {request.messages[0].text}")
    outcome = await run(
        session,
        root,
        META + "done = []\nfor k in range(5):\n    done.append(await agent(f'job {k}'))\n"
        "result = done\n",
        limits=Limits(max_agents=3),
    )
    assert outcome.status == "stopped"
    assert "more than 3 agents" in outcome.error
    assert outcome.result is None and len(outcome.records) == 3
    text = outcome.render()
    assert "stopped before it finished" in text and "- job 2: finished job 2" in text
    assert nodes(session, "workflow")[0].status == "failed"


async def test_more_than_4096_items_is_an_error_not_a_silent_cap(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(
        session, root, META + "result = await parallel([agent(f'x {k}') for k in range(4097)])\n"
    )
    assert outcome.status == "failed"
    assert "4,097 items; the limit is 4,096" in outcome.error and outcome.error.startswith("line 2")
    assert not workers.requests and not outcome.warnings


async def test_runaway_loops_and_huge_ranges_are_stopped(
    session: Session, root: NodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "MAX_STEPS", 500)
    looping = await run(session, root, META + "n = 0\nwhile True:\n    n += 1\nresult = n\n")
    assert looping.status == "failed" and "more than 500 loop iterations" in looping.error
    comprehension = await run(session, root, META + "result = [k for k in range(1000)]\n")
    assert "loop iterations" in comprehension.error
    huge = await run(session, root, META + "result = list(range(10 ** 8))\n")
    assert "too large" in huge.error


async def test_the_token_budget_stops_the_workflow_with_partial_results(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", lambda request: f"finished {request.messages[0].text}")
    outcome = await run(
        session,
        root,
        META + "done = []\nfor k in range(5):\n    done.append(await agent(f'job {k}'))\n"
        "result = done\n",
        limits=Limits(budget=40),  # each scripted call costs 15 tokens
    )
    assert outcome.status == "stopped"
    assert "used 45 tokens of the 40-token budget" in outcome.error
    assert [r.status for r in outcome.records] == ["done", "done", "done"]
    assert outcome.tokens == 45


async def test_the_budget_stops_agents_mid_flight_and_cancels_their_nodes(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("long job", call("glob", pattern="*"), call("glob", pattern="*.py"), "done")
    outcome = await run(
        session,
        root,
        META + "result = await parallel([agent(f'long job {k}') for k in range(3)])\n",
        limits=Limits(budget=40),
    )
    assert outcome.status == "stopped" and "budget" in outcome.error
    statuses = {n.status for n in nodes(session, "worker")}
    assert "cancelled" in statuses and not statuses & {"pending", "running"}


async def test_the_session_run_budget_also_stops_workflow_agents(
    runtime: Any, workspace: Path, workers: Workers, config: Config
) -> None:
    config.agent.run_budget = 8  # output tokens; each scripted call produces 5
    session = runtime.create_session(workspace, interactive=False)
    root = session.new_node("Conversation", role="root")
    workers.on("job", "done")
    outcome = await run(
        session, root, META + "result = [await agent(f'job {k}') for k in range(3)]\n"
    )
    assert outcome.status == "stopped" and "run reached its budget" in outcome.error


async def test_scripts_can_read_the_budget(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", "done")
    outcome = await run(
        session,
        root,
        META + "before = budget.remaining()\nawait agent('job')\n"
        "result = [budget.total, before, budget.spent(), budget.remaining()]\n",
        limits=Limits(budget=1000),
    )
    assert outcome.result == [1000, 1000, 15, 985]


async def test_cache_reads_do_not_count_against_the_budget(
    session: Session, root: NodeInfo, workers: Workers, scripted: ScriptedProvider
) -> None:
    scripted._usage = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=10_000)
    workers.on("job", "done")
    outcome = await run(session, root, META + "result = await agent('job')\n")
    assert outcome.tokens == 15


# --------------------------------------------------------------------------------------------
# Progress: phases, log, notices
# --------------------------------------------------------------------------------------------


async def test_the_node_tree_has_workflow_phase_and_worker_levels(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", "done")
    await run(
        session,
        root,
        'meta = {"name": "tree", "description": "d", "phases": ["Find", "Check"]}\n'
        "loose = await agent('job loose')\n"
        "phase('Find')\n"
        "found = await parallel([agent(f'job {k}', label=f'finder {k}') for k in range(2)])\n"
        "phase('Check')\n"
        "checked = await agent('job check', label='checker')\n"
        "late = await agent('job late', phase='Find', label='late finder')\n"
        "result = 1\n",
    )
    view = session.view
    (workflow,) = nodes(session, "workflow")
    assert (workflow.parent_id, workflow.title, workflow.status) == (root.id, "tree", "done")
    phases = {p.title: p for p in nodes(session, "phase")}
    assert set(phases) == {"Find", "Check"}
    assert all(p.parent_id == workflow.id and p.status == "done" for p in phases.values())

    def titles(node_id: str) -> list[str]:
        return [view.nodes[c].info.title for c in view.nodes[node_id].children]

    assert titles(workflow.id) == ["job loose", "Find", "Check"]
    assert titles(phases["Find"].id) == ["finder 0", "finder 1", "late finder"]
    assert titles(phases["Check"].id) == ["checker"]
    assert phases["Find"].summary == "3 agents"
    assert workflow.summary == "5 agents (5 done)"


async def test_log_and_print_reach_the_user_as_notices_on_the_workflow_node(
    session: Session, root: NodeInfo, events: list[Event]
) -> None:
    outcome = await run(
        session, root, META + "log('dropped 3 duplicates')\nprint('items:', 4)\nresult = 1\n"
    )
    notices = [e for e in events if isinstance(e, Notice)]
    workflow = nodes(session, "workflow")[0]
    assert [(n.text, n.node_id) for n in notices] == [
        ("t: dropped 3 duplicates", workflow.id),
        ("t: items: 4", workflow.id),
    ]
    assert outcome.logs == ["dropped 3 duplicates", "items: 4"]


async def test_large_workflows_are_announced_up_front_and_when_they_grow(
    session: Session, root: NodeInfo, workers: Workers, events: list[Event]
) -> None:
    workers.on("job", "done")
    await run(session, root, META + "result = 1\n", limits=Limits(announce_above=2), estimated=5)
    await run(
        session,
        root,
        META + "result = [await agent(f'job {k}') for k in args['ks']]\n",
        {"ks": [1, 2, 3]},
        limits=Limits(announce_above=2),
    )
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert notices[0].startswith('Large workflow "t": about 5 agents, at most 6 at a time')
    assert notices[1].startswith('Large workflow "t": 3 agents started so far')
    assert len(notices) == 2


# --------------------------------------------------------------------------------------------
# Script errors and mistakes
# --------------------------------------------------------------------------------------------


async def test_a_script_error_names_its_line_and_keeps_what_finished(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", "found three things")
    outcome = await run(
        session, root, META + "found = await agent('job')\nresult = found['missing']\n"
    )
    assert outcome.status == "failed"
    assert outcome.error.startswith("line 3: TypeError: string indices must be integers")
    assert "result = found['missing']" in outcome.error
    assert "- job: found three things" in outcome.render()
    assert nodes(session, "workflow")[0].status == "failed"


async def test_long_failure_lists_are_cut_and_say_so(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("doomed", fails("x" * 2000))
    outcome = await run(
        session, root, META + "result = await parallel([agent(f'doomed {k}') for k in range(40)])\n"
    )
    text = outcome.render()
    assert len(outcome.failures) == 40
    assert "- (10 more not shown)" in text
    assert "[truncated: 500 of" in text


async def test_calls_that_were_never_awaited_are_reported(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(session, root, META + "forgotten = agent('job')\nresult = 1\n")
    assert outcome.status == "completed"
    assert outcome.warnings == ["line 2: agent(...) was never awaited, so it never ran"]
    assert not workers.requests


async def test_awaiting_a_call_twice_is_explained(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("job", "done")
    outcome = await run(
        session, root, META + "call = agent('job')\nfirst = await call\nresult = await call\n"
    )
    assert outcome.status == "failed" and "was awaited twice" in outcome.error


async def test_a_script_without_result_at_run_time_gets_a_warning(
    session: Session, root: NodeInfo
) -> None:
    outcome = await run(session, root, META + "if args.get('x'):\n    result = 1\n")
    assert outcome.result is None
    assert outcome.warnings == ['workflow "t" ended without assigning `result`']


# --------------------------------------------------------------------------------------------
# Nested workflows
# --------------------------------------------------------------------------------------------


async def test_a_script_can_run_a_saved_workflow_one_level_deep(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    folder = paths.config_dir() / "workflows"
    folder.mkdir(parents=True)
    (folder / "inner.py").write_text(
        'meta = {"name": "inner", "description": "Echo.", "phases": ["Echo"]}\n'
        "phase('Echo')\n"
        "result = await agent(f\"echo {args['word']}\")\n"
    )
    (folder / "deeper.py").write_text(
        'meta = {"name": "deeper", "description": "Nests."}\n'
        "result = await workflow('inner', {'word': 'x'})\n"
    )
    workers.on("echo", lambda request: request.messages[0].text.upper())
    outcome = await run(
        session,
        root,
        'meta = {"name": "outer", "description": "d", "phases": ["Run"]}\n'
        "phase('Run')\n"
        "result = await workflow('inner', {'word': 'hi'})\n",
    )
    assert outcome.result == "ECHO HI"
    inner = next(n for n in nodes(session, "workflow") if n.title == "inner")
    assert session.view.nodes[inner.parent_id].info.title == "Run"
    assert [p.title for p in nodes(session, "phase")] == ["Run", "Echo"]

    too_deep = await run(session, root, META + "result = await workflow('deeper')\n")
    assert too_deep.status == "failed" and "one level deep only" in too_deep.error
    missing = await run(session, root, META + "result = await workflow('nope')\n")
    assert "no saved workflow named 'nope'" in missing.error and "inner" in missing.error


# --------------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------------


def resumable(second: str, expression: str) -> str:
    return (
        'meta = {"name": "resumable", "description": "d"}\n'
        "first = await parallel([agent(f'read {k}', tools='read-only') for k in range(2)])\n"
        f"second = await agent('{second}', tools='read-only')\n"
        f"result = {expression}\n"
    )


def with_writer(writer: str) -> str:
    return (
        'meta = {"name": "w", "description": "d"}\n'
        "a = await agent('read before', tools='read-only')\n"
        f"b = await agent('{writer}')\n"
        "c = await agent('read after', tools='read-only')\n"
        "result = [a, b, c]\n"
    )


async def test_resume_reuses_unchanged_agents_after_a_failure(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("read", lambda request: f"text of {request.messages[0].text}")
    workers.on("summarise", lambda r: "short" if "briefly" in r.messages[0].text else "summary")
    journal = Journal()
    broken = resumable("summarise", "second['oops']")
    first = await run(session, root, broken, journal=journal)
    assert first.status == "failed" and len(workers.tasks()) == 3

    fixed = resumable("summarise", "[first, second]")
    second = await run(session, root, fixed, journal=journal, resume=True)
    assert second.status == "completed"
    assert second.result == [["text of read 0", "text of read 1"], "summary"]
    assert len(workers.tasks()) == 3  # nothing ran again
    assert [r.status for r in second.records] == ["reused"] * 3
    reused = [n for n in nodes(session, "worker") if n.summary.startswith("Reused")]
    assert len(reused) == 3 and "3 reused" in second.headline()

    changed = resumable("summarise briefly", "[first, second]")
    third = await run(session, root, changed, journal=journal, resume=True)
    assert third.result[1] == "short"
    assert [r.status for r in third.records] == ["reused", "reused", "done"]


async def test_without_resume_everything_runs_again(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("read", "text")
    workers.on("summarise", "summary")
    journal = Journal()
    script = resumable("summarise", "second")
    await run(session, root, script, journal=journal)
    await run(session, root, script, journal=journal)
    assert len(workers.tasks()) == 6


async def test_a_rerun_writer_invalidates_everything_after_it(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    workers.on("read", "text")
    workers.on("write", "written")
    journal = Journal()
    await run(session, root, with_writer("write v1"), journal=journal)
    outcome = await run(session, root, with_writer("write v2"), journal=journal, resume=True)
    assert [r.status for r in outcome.records] == ["reused", "done", "done"]


async def test_resume_with_nothing_to_resume_says_so(
    session: Session, root: NodeInfo, workers: Workers
) -> None:
    outcome = await run(session, root, META + "result = 1\n", journal=Journal(), resume=True)
    assert outcome.warnings == ["resume was requested, but no earlier run left results"]


# --------------------------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------------------------


async def test_cancelling_a_workflow_cancels_every_agent_and_leaves_no_tasks(
    session: Session, root: NodeInfo, workers: Workers, scripted: ScriptedProvider
) -> None:
    original = scripted.stream
    started = asyncio.Event()

    async def stalling(request: ChatRequest):  # type: ignore[no-untyped-def]
        started.set()
        await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted.stream = stalling  # type: ignore[method-assign]
    workers.on("job", "never")
    before = asyncio.all_tasks()
    task = asyncio.ensure_future(
        run(
            session,
            root,
            'meta = {"name": "slow", "description": "d", "phases": ["Work"]}\n'
            "phase('Work')\n"
            "result = await pipeline(range(4), lambda p, item, i: agent(f'job {item}'))\n",
            limits=Limits(max_concurrent=2),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    workers_after = nodes(session, "worker")
    assert len(workers_after) == 4
    assert {n.status for n in workers_after} == {"cancelled"}
    assert nodes(session, "workflow")[0].status == "cancelled"
    assert nodes(session, "phase")[0].status == "cancelled"
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before - {asyncio.current_task()} == set()

"""Session lifecycle: runs, cancellation, questions, the folded view, and stored sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clankit.agent.events import Event, RunFinished, RunStarted
from clankit.agent.session import Session
from clankit.agent.state import SessionView
from clankit.agent.store import SessionStore
from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.llm.provider import ProviderError
from clankit.llm.scripted import ScriptedProvider
from clankit.orchestration import register, unregister
from clankit.orchestration.base import AskUserTool, Orchestrator
from clankit.runtime import Runtime, RuntimeThread


class OneAgent(Orchestrator):
    """Minimal method for these tests: a persistent root agent with an ask_user tool."""

    name = "test-one"
    title = "Test"

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.root = session.new_node("conversation", role="root")
        self.agent = session.agent(self.root, system="test", tools=[AskUserTool(session)])

    async def run(self, prompt: str) -> str:
        self.session.update_node(self.root, status="running")
        child = self.session.new_node("subtask", parent=self.root, status="running")
        result = await self.agent.run(prompt)
        self.session.update_node(child, status="done", summary="child done")
        self.session.update_node(self.root, status="done")
        return result.text

    def steer(self, text: str) -> None:
        self.agent.steer(text)


@pytest.fixture(scope="module", autouse=True)
def _test_one_method() -> None:
    """Register the module's method for its tests only: the global registry must stay
    clean for other test files that enumerate orchestration methods."""
    register(OneAgent)
    try:
        yield
    finally:
        unregister(OneAgent.name)


@pytest.fixture
def config(config: Config) -> Config:
    config.orchestration.method = "test-one"
    return config


async def test_run_publishes_lifecycle_and_records_turn(
    session: Session, scripted: ScriptedProvider, events: list[Event]
) -> None:
    scripted.push("the answer")
    assert await session.run("question") == "the answer"
    kinds = [e.kind for e in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    finished = events[-1]
    assert (
        isinstance(finished, RunFinished)
        and finished.status == "completed"
        and finished.answer == "the answer"
    )
    assert [(t.prompt, t.answer, t.status) for t in session.turns] == [
        ("question", "the answer", "completed")
    ]
    assert not session.busy
    assert [seq for seq in (e.seq for e in events)] == sorted(e.seq for e in events)


async def test_conversation_continues_across_runs(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push("one", "two")
    await session.run("first")
    await session.run("second")
    assert [m.role for m in scripted.requests[1].messages] == ["user", "assistant", "user"]
    texts = [i.text for i in session.view.conversation() if i.kind in ("user", "assistant")]
    assert texts == ["first", "one", "second", "two"]
    assert "first" in session.recap() and "two" in session.recap()


async def test_view_tracks_tree_and_usage(session: Session, scripted: ScriptedProvider) -> None:
    scripted.push("ok")
    await session.run("go")
    view = session.view
    (root_id,) = view.roots
    root = view.nodes[root_id]
    assert root.info.status == "done" and len(root.children) == 1
    child = view.nodes[root.children[0]]
    assert (child.info.depth, child.info.status, child.info.summary) == (1, "done", "child done")
    assert root.info.model == "scripted/test-model"
    assert root.tokens == 15 and view.total_usage["output_tokens"] == 5
    assert [depth for depth, _ in view.walk()] == [0, 1]
    snap = view.snapshot()
    assert snap["running"] is False and set(snap["nodes"]) == set(view.nodes)


async def test_provider_failure_ends_run_as_failed(
    session: Session, scripted: ScriptedProvider, events: list[Event]
) -> None:
    scripted.push(ProviderError("invalid api key", kind="auth", provider="scripted"))
    assert await session.run("go") == ""
    finished = events[-1]
    assert (
        isinstance(finished, RunFinished)
        and finished.status == "failed"
        and "invalid api key" in finished.error
    )
    assert all(n.info.status in ("failed", "done") for n in session.view.nodes.values())
    assert not session.busy


async def test_cancel_marks_nodes_cancelled_and_session_stays_usable(
    session: Session, scripted: ScriptedProvider, events: list[Event]
) -> None:
    async def stall(request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(60)

    scripted._router = lambda request: stall  # a step that never completes
    # a callable step is evaluated synchronously, so use a provider-level stall instead:
    original = scripted.stream

    async def slow_stream(request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(60)
        async for event in original(request):
            yield event

    scripted._router = None
    scripted.stream = slow_stream  # type: ignore[method-assign]
    task = session.start("go")
    await asyncio.sleep(0.05)
    assert session.busy and session.cancel()
    assert await task == ""
    finished = events[-1]
    assert isinstance(finished, RunFinished) and finished.status == "cancelled"
    assert {n.info.status for n in session.view.nodes.values()} == {"cancelled"}
    assert not session.busy and not session.cancel()

    scripted.stream = original  # type: ignore[method-assign]
    scripted.push("back again")
    assert await session.run("again") == "back again"


async def test_second_run_while_busy_is_rejected(
    session: Session, scripted: ScriptedProvider
) -> None:
    scripted.push("x")
    task = session.start("go")
    with pytest.raises(RuntimeError):
        session.start("again")
    await task


async def test_ask_user_round_trip(session: Session, scripted: ScriptedProvider) -> None:
    from clankit.llm.scripted import call

    scripted.push(call("ask_user", question="Tabs or spaces?", options=["tabs", "spaces"]), "noted")
    task = session.start("go")
    for _ in range(100):
        await asyncio.sleep(0.01)
        if session.view.questions:
            break
    (question,) = session.view.questions.values()
    assert question.options == ["tabs", "spaces"]
    assert session.answer_question(question.request_id, "spaces")
    assert await task == "noted"
    assert scripted.requests[1].messages[-1].tool_result_blocks[0].content == "spaces"
    assert not session.view.questions


async def test_non_interactive_session_answers_questions_itself(
    runtime: Runtime, workspace: Path, scripted: ScriptedProvider
) -> None:
    from clankit.llm.scripted import call

    session = runtime.create_session(workspace, interactive=False)
    scripted.push(call("ask_user", question="Sure?"), "went ahead")
    assert await session.run("go") == "went ahead"
    assert "non-interactive" in scripted.requests[1].messages[-1].tool_result_blocks[0].content


async def test_no_usable_model_fails_cleanly(workspace: Path) -> None:
    cfg = Config()
    cfg.orchestration.method = "test-one"
    runtime = Runtime(ConfigStore.in_memory(cfg), persist_sessions=False)
    session = runtime.create_session(workspace)
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    assert await session.run("hi") == ""
    assert isinstance(seen[0], RunStarted)
    assert isinstance(seen[-1], RunFinished) and "API key" in seen[-1].error


async def test_unknown_method_fails_cleanly(session: Session, events: list[Event]) -> None:
    session.method_override = "no-such-method"
    assert await session.run("hi") == ""
    assert isinstance(events[-1], RunFinished) and "no-such-method" in events[-1].error


async def test_stored_session_can_be_resumed(
    config: Config, scripted: ScriptedProvider, workspace: Path, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime = Runtime(ConfigStore.in_memory(config), session_store=store)
    runtime.providers.register(scripted)
    session = runtime.create_session(workspace)
    assert store.list() == []  # nothing on disk until something happens
    scripted.push("stored answer")
    await session.run("remember this")
    session.close()

    (meta,) = store.list()
    assert (meta.id, meta.title, meta.turns) == (session.id, "remember this", 1)
    assert store.list(workspace=tmp_path) == []

    runtime2 = Runtime(ConfigStore.in_memory(config), session_store=store)
    runtime2.providers.register(scripted)
    resumed = runtime2.resume_session(session.id)
    assert resumed.workspace == workspace
    assert [(t.prompt, t.answer) for t in resumed.turns] == [("remember this", "stored answer")]
    texts = [i.text for i in resumed.view.conversation() if i.kind in ("user", "assistant")]
    assert texts == ["remember this", "stored answer"]
    # new nodes do not collide with restored ids
    before = set(resumed.view.nodes)
    scripted.push("more")
    await resumed.run("next")
    assert before < set(resumed.view.nodes) and len(resumed.turns) == 2


def test_view_is_a_pure_fold_of_events(tmp_path: Path) -> None:
    """Replaying the logged (non-transient) events reproduces the live view."""

    async def scenario() -> tuple[SessionView, list[Event]]:
        cfg = Config()
        cfg.models.default = "scripted/test-model"
        cfg.orchestration.method = "test-one"
        cfg.permissions.mode = "auto"
        runtime = Runtime(ConfigStore.in_memory(cfg), persist_sessions=False)
        runtime.providers.register(ScriptedProvider(["alpha", "beta"], provider_id="scripted"))
        session = runtime.create_session(tmp_path)
        seen: list[Event] = []
        session.bus.subscribe(seen.append)
        await session.run("one")
        await session.run("two")
        return session.view, seen

    live, seen = asyncio.run(scenario())
    replayed = SessionView()
    for event in seen:
        if not event.transient:
            replayed.apply(event)
    assert replayed.snapshot() == live.snapshot()


def test_runtime_thread_marshals_calls(
    config: Config, scripted: ScriptedProvider, workspace: Path
) -> None:
    def factory() -> Runtime:
        rt = Runtime(ConfigStore.in_memory(config), persist_sessions=False)
        rt.providers.register(scripted)
        return rt

    thread = RuntimeThread(factory).start()
    try:
        scripted.push("from another thread")
        session = thread.submit(lambda: thread.runtime.create_session(workspace))
        got: list[Event] = []
        session.bus.subscribe(got.append)  # subscribing is thread-safe
        assert thread.call(lambda: session.run("hi")) == "from another thread"
        assert got[-1].kind == "run_finished"
        assert thread.submit(lambda: session.view.snapshot()["running"]) is False
    finally:
        thread.stop()


async def test_budget_pauses_run_and_user_can_continue_or_stop(
    session: Session, scripted: ScriptedProvider, events: list[Event]
) -> None:
    from clankit.llm.scripted import call

    session.config.agent.run_budget = 8  # each scripted call generates 5 output tokens
    scripted.push(call("ask_user", question="q1"), call("ask_user", question="q2"), "finished")

    async def answer_everything() -> None:
        answered: set[str] = set()
        while session.busy:
            for request_id, question in list(session.view.questions.items()):
                if request_id not in answered:
                    answered.add(request_id)
                    session.answer_question(
                        request_id, "Continue" if "budget" in question.question else "fine"
                    )
            await asyncio.sleep(0.005)

    task = session.start("go")
    await asyncio.gather(task, answer_everything())
    assert task.result() == "finished"
    assert any(e.kind == "question_asked" and "budget" in e.question for e in events)  # type: ignore[attr-defined]

    # declining stops the run with a clear error
    scripted.push(call("ask_user", question="q1"), call("ask_user", question="q2"), "never reached")

    async def decline() -> None:
        answered: set[str] = set()
        while session.busy:
            for request_id, question in list(session.view.questions.items()):
                if request_id not in answered:
                    answered.add(request_id)
                    session.answer_question(
                        request_id, "Stop" if "budget" in question.question else "fine"
                    )
            await asyncio.sleep(0.005)

    task = session.start("again")
    await asyncio.gather(task, decline())
    finished = events[-1]
    assert (
        isinstance(finished, RunFinished)
        and finished.status == "failed"
        and "budget" in finished.error
    )


async def test_undo_restores_files_changed_in_the_last_run(
    session: Session, scripted: ScriptedProvider, workspace: Path
) -> None:
    existing = workspace / "keep.txt"
    existing.write_text("original")
    created = workspace / "new.txt"

    # stand in for the file tools: checkpoint before writing
    session.files.begin_run()
    session.files.checkpoint(existing)
    existing.write_text("changed")
    session.files.checkpoint(created)
    created.write_text("brand new")
    session.files.checkpoint(existing)  # second touch in the same run keeps the first snapshot
    existing.write_text("changed again")

    assert session.undo() == ["keep.txt", "new.txt"]
    assert existing.read_text() == "original" and not created.exists()
    assert session.undo() == []

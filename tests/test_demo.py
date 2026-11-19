"""The scripted demo run: works with no provider, touches every kind of event."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clankit.agent.events import Event
from clankit.config.schema import Config
from clankit.config.store import ConfigStore
from clankit.runtime import Runtime


@pytest.fixture
def demo_runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", "0")
    config = Config()
    config.orchestration.method = "demo"
    return Runtime(ConfigStore.in_memory(config), persist_sessions=False)


async def test_demo_runs_without_any_provider(demo_runtime: Runtime, tmp_path: Path) -> None:
    session = demo_runtime.create_session(tmp_path, interactive=False)
    seen: list[Event] = []
    session.bus.subscribe(seen.append)
    answer = await session.run("anything")
    assert "keeps comments" in answer
    kinds = {e.kind for e in seen}
    assert kinds >= {
        "run_started",
        "run_finished",
        "node_added",
        "node_changed",
        "user_message",
        "text_delta",
        "thinking_delta",
        "assistant_message",
        "tool_started",
        "tool_finished",
        "usage_changed",
    }
    view = session.view
    assert len(view.nodes) == 8 and max(n.info.depth for n in view.nodes.values()) == 2
    assert {n.info.status for n in view.nodes.values()} <= {"done", "cancelled"}
    assert any(i.display for n in view.nodes.values() for i in n.items if i.kind == "tool")
    assert any(i.is_error for n in view.nodes.values() for i in n.items if i.kind == "tool")
    assert session.usage.output_tokens > 0
    session.cost()  # prices may be unknown without a provider; it must simply not raise
    assert not list(tmp_path.iterdir())  # noqa: ASYNC240 - the demo never touches the workspace


async def test_demo_waits_for_permission_and_question(
    demo_runtime: Runtime, tmp_path: Path
) -> None:
    demo_runtime.config.permissions.mode = "ask"
    session = demo_runtime.create_session(tmp_path)
    task = session.start("go")

    async def until(predicate) -> None:  # type: ignore[no-untyped-def]
        for _ in range(500):
            if predicate():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("demo did not reach the expected state")

    await until(lambda: session.view.permissions)
    (request,) = session.view.permissions.values()
    assert request.category == "shell" and request.session_rule == "shell:pytest *"
    assert session.view.nodes[request.node_id].info.title == "Round-trip tests"
    session.resolve_permission(request.request_id, "allow")

    await until(lambda: session.view.questions)
    (question,) = session.view.questions.values()
    assert len(question.options) == 2
    session.answer_question(question.request_id, "Update the documentation too")
    await task
    docs = next(n for n in session.view.nodes.values() if n.info.title == "Documentation")
    assert docs.info.status == "done"


async def test_demo_can_be_cancelled(
    demo_runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLANKIT_DEMO_SPEED", "100")
    session = demo_runtime.create_session(tmp_path)
    task = session.start("go")
    await asyncio.sleep(0.1)
    assert session.cancel()
    await task
    assert session.view.last_run is not None and session.view.last_run.status == "cancelled"
    assert all(n.info.status in ("cancelled", "done") for n in session.view.nodes.values())

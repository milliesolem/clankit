"""The main screen, driven headless: runs, requests, steering, cancelling and layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from clankit.agent.events import TERMINAL_STATUSES
from clankit.agent.session import Session
from clankit.llm.scripted import ScriptedProvider
from clankit.runtime import Runtime
from clankit.tui import exit_summary
from clankit.tui.app import ClankitApp
from clankit.tui.screens import TreeScreen
from clankit.tui.transcript import AssistantBlock, ToolBlock
from test_tui_support import demo_runtime, demo_workspace, screen_text, wait_for

WIDE = (140, 40)


def make_app(runtime: Runtime, workspace: Path) -> tuple[ClankitApp, Session]:
    session = runtime.create_session(workspace)
    return ClankitApp(runtime, session, flush_interval=0.01), session


async def submit(pilot, app: ClankitApp, text: str) -> None:  # type: ignore[no-untyped-def]
    app.main.composer.load(text)
    app.main.composer.focus()
    await pilot.press("enter")


def block_kinds(app: ClankitApp) -> list[str]:
    return [block.kind for block in app.main.transcript.blocks]


async def test_demo_runs_to_completion_through_permission_and_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, session = make_app(demo_runtime(monkeypatch), demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "Add a KDL config loader")
        await wait_for(pilot, lambda: app.main.requests.has_focus)
        text = screen_text(app)
        assert "permission" in text and "1 of 1" in text
        assert "Round-trip tests" in text and "deepseek-v4-pro" in text
        assert "pytest tests/config/test_roundtrip.py -q --hypothesis-seed=0" in text
        assert "shell:pytest *" in text
        tree = {row.title: row for row in app.main.panel.task_tree.rows}
        assert tree["Round-trip tests"].state == "needs-input"

        await pilot.press("y")
        await wait_for(pilot, lambda: session.view.questions and app.main.requests.has_focus)
        text = screen_text(app)
        assert "question" in text and "Update the documentation too" in text

        await pilot.press("2")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))

        kinds = block_kinds(app)
        assert kinds[0] == "prompt"
        assert {"thinking", "assistant", "tool", "resolution", "run-end"} <= set(kinds)
        answer = [b for b in app.main.transcript.blocks if isinstance(b, AssistantBlock)][-1]
        assert "keeps comments" in answer.prose.source
        tools = [b for b in app.main.transcript.blocks if isinstance(b, ToolBlock)]
        assert all("done" in screen_text(app) or b.header.item.done for b in tools)

        rows = app.main.panel.task_tree.rows
        assert rows[0].title == "Conversation" and rows[0].state == "done"
        by_title = {row.title: row for row in rows}
        assert by_title["Documentation"].state == "done"
        writer = by_title["Writer that preserves comments"]
        assert writer.hidden == 3 and writer.caret == "▸"
        assert all(n.info.status in TERMINAL_STATUSES for n in session.view.nodes.values())
        assert "tasks 8" in screen_text(app)


async def test_typed_y_in_a_non_empty_composer_never_approves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, session = make_app(demo_runtime(monkeypatch), demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await pilot.press(*"draft")
        session.start("go")
        await wait_for(pilot, lambda: app.main.requests.request is not None)
        await pilot.pause(0.05)
        assert app.focused is app.main.composer
        assert "ctrl+y" in screen_text(app)
        await pilot.press("y")
        await pilot.pause(0.05)
        assert app.main.composer.text == "drafty"
        assert session.view.permissions, "a typed y must not approve the request"

        await pilot.press("ctrl+y")
        assert app.focused is app.main.requests
        await pilot.press("n")
        await wait_for(pilot, lambda: not session.view.permissions)
        await wait_for(pilot, lambda: session.view.questions)
        assert app.main.composer.text == "drafty"


async def test_question_is_answered_by_the_next_composer_submission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.permissions.mode = "auto"
    app, session = make_app(runtime, demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "go")
        await wait_for(pilot, lambda: app.main.requests.has_focus)
        await pilot.press(*"update")
        assert app.focused is app.main.composer and app.main.composer.text == "update"
        await pilot.press("enter")
        await wait_for(pilot, lambda: not session.busy)
        docs = next(n for n in session.view.nodes.values() if n.info.title == "Documentation")
        assert docs.info.status == "done"


async def test_steering_and_cancelling_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, session = make_app(demo_runtime(monkeypatch, speed="100"), demo_workspace(tmp_path))
    steered: list[str] = []
    original = session.steer
    monkeypatch.setattr(session, "steer", lambda text: steered.append(text) or original(text))
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "go")
        await wait_for(pilot, lambda: session.busy)
        assert "esc" in screen_text(app)
        await submit(pilot, app, "please keep it small")
        await wait_for(pilot, lambda: "steer" in block_kinds(app))
        assert steered == ["please keep it small"]
        assert "sent while the agent was working" in screen_text(app)

        await pilot.press("escape")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))
        assert session.view.last_run is not None and session.view.last_run.status == "cancelled"
        assert "cancelled" in screen_text(app)


async def test_prompt_runs_through_a_scripted_model(
    runtime: Runtime, scripted: ScriptedProvider, workspace: Path
) -> None:
    scripted.push("Hello from the **model**.")
    app, session = make_app(runtime, workspace)
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "hi there")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))
        assert scripted.requests, "the prompt reached the model"
        prose = [b for b in app.main.transcript.blocks if isinstance(b, AssistantBlock)]
        assert prose and prose[-1].prose.source == "Hello from the **model**."
        text = screen_text(app)
        assert "hi there" in text and "Hello from the model." in text
        assert app.main.composer.text == ""
        await pilot.press("up")
        assert app.main.composer.text == "hi there"


async def test_failed_run_shows_the_error(
    runtime: Runtime, scripted: ScriptedProvider, workspace: Path
) -> None:
    app, session = make_app(runtime, workspace)
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "this has no scripted reply")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))
        assert "notice" in block_kinds(app)
        assert "ran out of steps" in screen_text(app)


async def test_first_run_explains_how_to_add_a_key(tmp_path: Path) -> None:
    from clankit.config.schema import Config
    from clankit.config.store import ConfigStore

    runtime = Runtime(ConfigStore.in_memory(Config()), persist_sessions=False)
    app, _ = make_app(runtime, demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause(0.05)
        text = screen_text(app)
        assert "No model is set up yet" in text and "f2" in text
        assert "/method demo" in text and "clankit --demo" in text
        await pilot.press("f2")
        await wait_for(pilot, lambda: app.screen is not app.main)
        assert getattr(app.screen, "category", None) == "providers"


async def test_narrow_terminal_collapses_the_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.permissions.mode = "auto"
    app, session = make_app(runtime, demo_workspace(tmp_path))
    async with app.run_test(size=(90, 30)) as pilot:
        await submit(pilot, app, "go")
        await wait_for(pilot, lambda: session.view.questions and app.main.requests.has_focus)
        await pilot.press("1")
        await wait_for(pilot, lambda: not session.busy)
        await pilot.pause(0.05)
        assert not app.main.panel.display
        assert app.main.rollup.display
        assert "ctrl+t" in screen_text(app)

        await pilot.press("ctrl+t")
        await wait_for(pilot, lambda: isinstance(app.screen, TreeScreen))
        await pilot.pause(0.05)
        assert "Parser with source spans" in screen_text(app)
        await pilot.press("down", "enter")
        await wait_for(pilot, lambda: app.screen is app.main and app.main.viewed is not None)
        assert "Parser with source spans" in screen_text(app)
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.main.viewed is None)

        await pilot.resize_terminal(70, 24)
        await pilot.pause(0.1)
        assert not app.main.rollup.display
        assert "tasks 8" in screen_text(app)


async def test_tree_keys_focus_and_peek_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.permissions.mode = "auto"
    app, session = make_app(runtime, demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "go")
        await wait_for(pilot, lambda: app.main.requests.has_focus)
        await pilot.press("1")
        await wait_for(pilot, lambda: not session.busy)
        tree = app.main.panel.task_tree
        tree.focus()
        await pilot.press("down", "down")
        assert tree.cursor is not None
        writer = session.view.nodes[tree.cursor]
        assert writer.info.title == "Writer that preserves comments"
        await pilot.press("right")
        await wait_for(pilot, lambda: len(tree.rows) == 8)
        await pilot.press("left")
        await wait_for(pilot, lambda: len(tree.rows) == 5)
        await pilot.press("space")
        await pilot.pause(0.05)
        assert app.main.panel.peeking == writer.info.id
        assert "Writer complete; 42 tests pass." in screen_text(app)
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.main.viewed == writer.info.id)
        assert "Writer that preserves comments" in screen_text(app)
        await pilot.press("escape")
        assert app.main.panel.peeking is None
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.main.viewed is None)


async def test_tool_rows_open_and_close(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.permissions.mode = "auto"
    app, session = make_app(runtime, demo_workspace(tmp_path))
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "go")
        await wait_for(pilot, lambda: app.main.requests.has_focus)
        await pilot.press("1")
        await wait_for(pilot, lambda: not session.busy)
        node = next(n for n in session.view.nodes.values() if n.info.title == "Atomic file replace")
        await app.main.view_node(node.info.id)
        await pilot.pause(0.05)
        text = screen_text(app)
        assert "+2 -1" in text and "os.replace(tmp, path)" in text
        assert "FAILED test_replace_keeps_mode" in text

        await pilot.press("ctrl+o")
        await pilot.pause(0.1)
        assert "6 passed in 0.18s" in screen_text(app).split("$ pytest")[-1]
        app.main.transcript.focus()
        await pilot.press("enter")
        await pilot.pause(0.05)
        tools = [b for b in app.main.transcript.blocks if isinstance(b, ToolBlock)]
        assert tools[-1].opened is False and not tools[-1].has_class("-open")
        await pilot.press("up", "enter")
        await pilot.pause(0.05)
        assert tools[-2].opened is False and tools[-2].has_class("-cursor")


async def test_exit_closes_the_runtime_and_summary_names_the_session(
    runtime: Runtime, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    original = runtime.aclose

    async def aclose() -> None:
        closed.append(True)
        await original()

    monkeypatch.setattr(runtime, "aclose", aclose)
    app, session = make_app(runtime, workspace)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause(0.05)
        assert "ctrl+c again" in screen_text(app)
        await pilot.press("ctrl+c")
        await pilot.pause(0.05)
    assert closed == [True]
    summary = exit_summary(session)
    assert session.id in summary and f"clankit --resume {session.id}" in summary


async def test_composer_grows_with_new_lines_and_submits_on_enter(
    runtime: Runtime, scripted: ScriptedProvider, workspace: Path
) -> None:
    scripted.push("Noted.")
    app, session = make_app(runtime, workspace)
    async with app.run_test(size=WIDE) as pilot:
        composer = app.main.composer
        assert composer.size.height == 1
        await pilot.press("a", "ctrl+j", "b", "ctrl+j", "c")
        await pilot.pause(0.05)
        assert composer.text == "a\nb\nc" and composer.size.height == 3
        await pilot.press("enter")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))
        assert composer.text == "" and composer.size.height == 1
        assert app.main.history.runs[0].prompt == "a\nb\nc"


async def test_scrolling_up_counts_new_blocks_until_jumping_back(
    runtime: Runtime, scripted: ScriptedProvider, workspace: Path
) -> None:
    long_answer = "\n\n".join(f"Paragraph {n} of a long answer." for n in range(40))
    scripted.push(long_answer, "Short follow-up.")
    app, session = make_app(runtime, workspace)
    async with app.run_test(size=WIDE) as pilot:
        await submit(pilot, app, "write a lot")
        await wait_for(pilot, lambda: not session.busy and "run-end" in block_kinds(app))
        transcript = app.main.transcript
        await wait_for(pilot, lambda: transcript.is_anchored)
        transcript.scroll_home(animate=False)
        await pilot.pause(0.05)
        await submit(pilot, app, "and a little more")
        await wait_for(pilot, lambda: not session.busy and transcript.unseen > 0)
        assert app.main.newbar.display
        assert f"{transcript.unseen} new" in screen_text(app) and "ctrl+g" in screen_text(app)
        await pilot.press("ctrl+g")
        await wait_for(pilot, lambda: transcript.unseen == 0)
        assert not app.main.newbar.display
        assert "Short follow-up." in screen_text(app)

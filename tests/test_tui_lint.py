"""Glyph lint: rendered screens contain no frames, hairlines or emoji, in several themes."""

from __future__ import annotations

from pathlib import Path

import pytest

from clankit.llm.scripted import ScriptedProvider
from clankit.tui.app import ClankitApp
from clankit.tui.screens import TreeScreen
from clankit.tui.settings import CATEGORIES, SettingsScreen
from clankit.tui.transcript import DiffView
from test_tui_support import assert_clean, demo_runtime, demo_workspace, screen_text, wait_for

THEMES = ["ayu-dark", "github-light", "dracula"]


async def demo_at_permission(pilot, app: ClankitApp) -> None:  # type: ignore[no-untyped-def]
    app.main.composer.load("Add a KDL config loader")
    app.main.composer.focus()
    await pilot.press("enter")
    await wait_for(pilot, lambda: app.main.requests.has_focus)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("size", [(140, 40), (90, 30)])
async def test_main_screen_mid_demo_is_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, theme: str, size: tuple[int, int]
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.ui.theme = theme
    session = runtime.create_session(demo_workspace(tmp_path))
    app = ClankitApp(runtime, session, flush_interval=0.01)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.05)
        assert_clean(app, "the empty main screen")
        await demo_at_permission(pilot, app)
        assert "permission" in screen_text(app)
        assert_clean(app, "the main screen with a permission request")

        writer = next(
            n for n in session.view.nodes.values() if n.info.title == "Atomic file replace"
        )
        await app.main.view_node(writer.info.id)
        await pilot.pause(0.1)
        diff = app.main.transcript.query_one(DiffView)
        app.main.transcript.scroll_to_widget(diff, animate=False, top=True)
        await pilot.pause(0.05)
        text = screen_text(app)
        assert "tmp.write_text(text)" in text and "permission" in text
        assert_clean(app, "a node with a diff and a pending request")

        app.main.panel.task_tree.focus()
        await pilot.press("space")
        await pilot.pause(0.05)
        assert_clean(app, "the tree with a peek")

        await app.main.view_node(None)
        app.main.requests.focus()
        await pilot.press("y")
        await wait_for(pilot, lambda: session.view.questions and app.main.requests.has_focus)
        assert_clean(app, "the main screen with a question")
        await pilot.press("1")
        await wait_for(pilot, lambda: not session.busy)
        assert_clean(app, "the finished run")

        app.main.composer.focus()
        await pilot.press("slash")
        await pilot.pause(0.05)
        assert_clean(app, "the command completion list")


@pytest.mark.parametrize("theme", THEMES)
async def test_settings_and_pickers_are_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, theme: str
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.config.ui.theme = theme
    runtime.providers.register(ScriptedProvider(provider_id="scripted", models=["fast", "smart"]))
    session = runtime.create_session(demo_workspace(tmp_path))
    app = ClankitApp(runtime, session, flush_interval=0.01)
    async with app.run_test(size=(140, 44)) as pilot:
        for category, _ in CATEGORIES:
            app.push_screen(SettingsScreen(category=category))
            await wait_for(pilot, lambda: isinstance(app.screen, SettingsScreen))
            await pilot.pause(0.1)
            assert_clean(app, f"settings, {category}")
            await pilot.press("tab")
            await pilot.pause(0.05)
            assert_clean(app, f"settings, {category}, with a field focused")
            app.pop_screen()
            await wait_for(pilot, lambda: app.screen is app.main)

        for view in ("models", "methods", "themes", "sessions", "help"):
            app.open_view(view)
            await wait_for(pilot, lambda: app.screen is not app.main)
            await pilot.pause(0.1)
            assert_clean(app, f"the {view} view")
            await pilot.press("escape")
            await wait_for(pilot, lambda: app.screen is app.main)

        app.open_tree()
        await wait_for(pilot, lambda: isinstance(app.screen, TreeScreen))
        assert_clean(app, "the full-screen tree")
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.screen is app.main)

        await pilot.press("ctrl+p")
        await pilot.pause(0.3)
        assert_clean(app, "the command palette")
        await pilot.press(*"set")
        await pilot.pause(0.3)
        assert_clean(app, "the command palette with a query")


async def test_model_picker_lists_models_by_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = demo_runtime(monkeypatch)
    runtime.providers.register(ScriptedProvider(provider_id="scripted", models=["fast", "smart"]))
    app = ClankitApp(runtime, runtime.create_session(demo_workspace(tmp_path)), flush_interval=0.01)
    async with app.run_test(size=(120, 30)) as pilot:
        app.open_view("models")
        await wait_for(pilot, lambda: app.screen is not app.main)
        await pilot.press("ctrl+r")
        await wait_for(pilot, lambda: "smart" in screen_text(app))
        assert "scripted" in screen_text(app) and "2 models known" in screen_text(app)
        await pilot.press(*"sma")
        await pilot.pause(0.05)
        assert "fast" not in screen_text(app).split("type to filter")[-1].split("smart")[0]
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.screen is app.main)
        await wait_for(pilot, lambda: runtime.config.models.default == "scripted/smart")

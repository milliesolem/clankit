"""Slash commands typed into the composer, and the command palette: every kind of outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

from clankit import commands
from clankit.llm.scripted import ScriptedProvider
from clankit.runtime import Runtime
from clankit.skills import forge
from clankit.tui.app import ClankitApp
from clankit.tui.pickers import MethodPicker, ModelPicker, SessionPicker, ThemePicker
from clankit.tui.screens import HelpScreen
from clankit.tui.settings import SettingsScreen
from test_tui_support import screen_text, wait_for

SIZE = (140, 40)


@pytest.fixture
def app(runtime: Runtime, workspace: Path) -> ClankitApp:
    return ClankitApp(runtime, runtime.create_session(workspace), flush_interval=0.01)


async def run_line(pilot, app: ClankitApp, line: str) -> None:  # type: ignore[no-untyped-def]
    app.main.composer.load(line)
    app.main.composer.focus()
    await pilot.press("enter")
    await pilot.pause(0.05)


def messages(app: ClankitApp) -> list[tuple[str, str]]:
    return [(e.level, e.text) for e in app.main.history.local if e.kind == "message"]


@pytest.mark.parametrize(
    ("line", "screen", "category"),
    [
        ("/help", HelpScreen, None),
        ("/settings", SettingsScreen, None),
        ("/skills", SettingsScreen, "skills"),
        ("/model", ModelPicker, None),
        ("/method", MethodPicker, None),
        ("/theme", ThemePicker, None),
        ("/resume", SessionPicker, None),
    ],
)
async def test_view_outcomes_open_their_screen(
    app: ClankitApp, line: str, screen: type, category: str | None
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, line)
        await wait_for(pilot, lambda: isinstance(app.screen, screen))
        if category is not None:
            assert app.screen.category == category  # type: ignore[attr-defined]
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.screen is app.main)


async def test_message_outcomes_are_shown_in_the_transcript(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/effort high")
        await run_line(pilot, app, "/permissions read-only")
        await run_line(pilot, app, "/cost")
        await run_line(pilot, app, "/context")
        await run_line(pilot, app, "/undo")
        await run_line(pilot, app, "/compact")
        await wait_for(pilot, lambda: len(messages(app)) == 6)
        texts = [text for _, text in messages(app)]
        assert texts[0].startswith("Reasoning effort set to high")
        assert app.runtime.config.agent.effort == "high"
        assert app.runtime.config.permissions.mode == "read-only"
        assert texts[2].startswith("This session:")
        assert texts[3] == "No conversation yet."
        assert texts[4].startswith("Nothing to undo")
        assert texts[5] == "Nothing to compact yet."
        assert "Reasoning effort set to high" in screen_text(app)


async def test_error_outcomes_are_marked_as_errors(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/nonsense")
        await run_line(pilot, app, "/effort enormous")
        await wait_for(pilot, lambda: len(messages(app)) == 2)
        assert [level for level, _ in messages(app)] == ["error", "error"]
        assert "Unknown command /nonsense" in screen_text(app)


async def test_setting_commands_with_arguments(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/theme dracula")
        await wait_for(pilot, lambda: app.theme == "dracula")
        assert app.runtime.config.ui.theme == "dracula"
        await run_line(pilot, app, "/model scripted/test-model")
        await run_line(pilot, app, "/method solo")
        await wait_for(pilot, lambda: len(messages(app)) == 3)
        assert app.runtime.config.orchestration.method == "solo"
        assert "Model set to scripted/test-model" in screen_text(app)


async def test_prompt_outcome_starts_a_run(app: ClankitApp, scripted: ScriptedProvider) -> None:
    scripted.push("AGENTS.md written.")
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/init")
        await wait_for(pilot, lambda: app.main.history.runs and not app.session.busy)
        assert app.main.history.runs[0].prompt == commands.INIT_PROMPT
        assert "AGENTS.md written." in screen_text(app)


async def test_skillify_with_nothing_to_propose_is_quiet(
    app: ClankitApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def nothing(session, focus=""):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(forge, "generate_from_session", nothing)
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/skillify")
        await pilot.pause(0.1)
        assert messages(app) == []


async def test_new_session_action_switches_sessions(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        first = app.session
        first.method_override = "demo"
        await run_line(pilot, app, "/new")
        await wait_for(pilot, lambda: app.session is not first)
        assert app.session.workspace == first.workspace
        assert app.session.method_override == "demo"
        assert app.session.id in app.runtime.sessions


async def test_quit_action_exits(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/quit")
        await pilot.pause(0.1)
        assert app.return_code == 0
        assert not app.is_running


async def test_completion_list_offers_matching_commands(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        app.main.composer.focus()
        await pilot.press("slash", "s", "e")
        await pilot.pause(0.05)
        completion = app.main.completion
        assert completion.is_open and completion.names == ["settings"]
        assert "Open settings" in screen_text(app)
        await pilot.press("escape")
        assert not completion.is_open
        await pilot.press("backspace")
        await pilot.pause(0.05)
        assert completion.names == ["settings", "skills", "skillify"]
        await pilot.press("down", "tab")
        assert app.main.composer.text == "/skills"
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, SettingsScreen))


async def test_command_palette_runs_commands(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause(0.2)
        await pilot.press(*"help")
        await pilot.pause(0.3)
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, HelpScreen))
        assert "keys and commands" in screen_text(app) and "ctrl+j" in screen_text(app)


async def test_resume_picker_swaps_to_a_stored_session(
    tmp_path: Path, scripted: ScriptedProvider
) -> None:
    from clankit.config.schema import Config
    from clankit.config.store import ConfigStore

    config = Config()
    config.models.default = "scripted/test-model"
    config.permissions.mode = "auto"
    runtime = Runtime(ConfigStore.in_memory(config))
    runtime.providers.register(scripted)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    earlier = runtime.create_session(workspace)
    scripted.push("First answer.")
    await earlier.run("first question")
    earlier.close()
    runtime.sessions.clear()

    app = ClankitApp(runtime, runtime.create_session(workspace), flush_interval=0.01)
    async with app.run_test(size=SIZE) as pilot:
        await run_line(pilot, app, "/resume")
        await wait_for(pilot, lambda: isinstance(app.screen, SessionPicker))
        assert "first question" in screen_text(app)
        await pilot.press("down", "enter")
        await wait_for(pilot, lambda: app.session.id == earlier.id)
        await wait_for(pilot, lambda: app.screen is app.main)
        text = screen_text(app)
        assert "first question" in text and "First answer." in text
        assert [run.prompt for run in app.main.history.runs] == ["first question"]

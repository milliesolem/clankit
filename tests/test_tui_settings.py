"""The settings screen: every change goes through runtime.update_config, errors show inline."""

from __future__ import annotations

from pathlib import Path

import pytest

from clankit.config.store import ConfigStore
from clankit.runtime import Runtime
from clankit.themes import all_themes
from clankit.tui.app import ClankitApp
from clankit.tui.editors import ActionButton, FieldRow, Segments, TextEditor
from clankit.tui.settings import CATEGORIES, SettingsScreen, ThemeList, display_path
from test_tui_support import screen_text, wait_for

SIZE = (140, 44)


@pytest.fixture
def app(runtime: Runtime, workspace: Path) -> ClankitApp:
    return ClankitApp(runtime, runtime.create_session(workspace), flush_interval=0.01)


async def open_settings(pilot, app: ClankitApp, category: str) -> SettingsScreen:  # type: ignore[no-untyped-def]
    app.push_screen(SettingsScreen(category=category))
    await wait_for(pilot, lambda: isinstance(app.screen, SettingsScreen))
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    await wait_for(pilot, lambda: screen.category == category and bool(screen.form.children))
    return screen


def control(screen: SettingsScreen, key: str):  # type: ignore[no-untyped-def]
    return next(w for w in screen.query("*") if getattr(w, "key", None) == key)


def has_control(screen: SettingsScreen, key: str) -> bool:
    return any(getattr(w, "key", None) == key for w in screen.query("*"))


def row(screen: SettingsScreen, key: str) -> FieldRow:
    return next(r for r in screen.query(FieldRow) if r.field_key == key)


async def type_into(pilot, editor: TextEditor, text: str) -> None:  # type: ignore[no-untyped-def]
    editor.focus()
    editor.value = ""
    await pilot.press(*text, "enter")
    await pilot.pause(0.05)


async def test_every_category_renders(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        for category, title in CATEGORIES:
            await open_settings(pilot, app, category)
            assert title in screen_text(app)
            await pilot.press("escape")
            await wait_for(pilot, lambda: app.screen is app.main)


async def test_number_field_saves_and_rejects_bad_input(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "agents")
        editor = control(screen, "agent.max_turns")
        await type_into(pilot, editor, "50")
        assert app.runtime.config.agent.max_turns == 50
        assert "saved" in screen_text(app)

        await type_into(pilot, editor, "lots")
        assert app.runtime.config.agent.max_turns == 50
        assert row(screen, "agent.max_turns").error.display
        assert "expected a whole number" in screen_text(app)
        assert "not saved" in screen_text(app)

        await type_into(pilot, editor, "0")
        assert "must be at least 1" in screen_text(app)
        await type_into(pilot, editor, "75")
        assert app.runtime.config.agent.max_turns == 75
        assert not row(screen, "agent.max_turns").error.display


async def test_switches_and_choices_save_immediately(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "permissions")
        mode = control(screen, "permissions.mode")
        assert isinstance(mode, Segments)
        mode.focus()
        await pilot.press("right")
        await pilot.pause(0.05)
        assert app.runtime.config.permissions.mode == "read-only"
        await pilot.press("left", "left")
        await pilot.pause(0.05)
        assert app.runtime.config.permissions.mode == "accept-edits"

        screen.categories.highlighted = [c for c, _ in CATEGORIES].index("appearance")
        await wait_for(pilot, lambda: screen.category == "appearance")
        thinking = control(screen, "ui.show_thinking")
        thinking.focus()
        await pilot.press("space")
        await pilot.pause(0.05)
        assert app.runtime.config.ui.show_thinking is False


async def test_model_field_validates_references(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "models")
        editor = control(screen, "models.roles.subagent")
        await type_into(pilot, editor, "nonsense")
        assert "provider/model" in screen_text(app)
        assert "subagent" not in app.runtime.config.models.roles
        await type_into(pilot, editor, "scripted/test-model")
        assert app.runtime.config.models.roles["subagent"] == "scripted/test-model"


async def test_saving_writes_the_configuration_file(
    tmp_path: Path, workspace: Path, scripted
) -> None:  # type: ignore[no-untyped-def]
    store = ConfigStore.open(tmp_path / "config.kdl")
    runtime = Runtime(store, persist_sessions=False)
    runtime.providers.register(scripted)
    app = ClankitApp(runtime, runtime.create_session(workspace), flush_interval=0.01)
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "agents")
        await type_into(pilot, control(screen, "agent.shell_timeout"), "45")
        assert "shell-timeout 45" in (tmp_path / "config.kdl").read_text()
        assert display_path(store.path) in screen_text(app)


async def test_orchestration_method_and_its_settings(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "orchestration")
        methods = screen.query_one("#method-list")
        methods.focus()
        names = [o.id for o in methods._options]  # type: ignore[attr-defined]
        methods.highlighted = names.index("demo")  # type: ignore[attr-defined]
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.runtime.config.orchestration.method == "demo")
        await wait_for(pilot, lambda: "demo settings" in screen_text(app))

        speed = control(screen, "method:demo:speed")
        await type_into(pilot, speed, "250")
        assert app.runtime.config.orchestration.option("demo", "speed") == 250
        await type_into(pilot, speed, "5000")
        assert "at most 1000" in screen_text(app)
        assert app.runtime.config.orchestration.option("demo", "speed") == 250


async def test_api_key_is_stored_but_only_ever_shown_masked(app: ClankitApp) -> None:
    secret = "sk-ant-api03-SECRETSECRETSECRET-a91f"
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        await type_into(pilot, control(screen, "provider:anthropic:key"), secret)
        await wait_for(pilot, lambda: "sk-ant-…a91f" in screen_text(app))
        assert app.runtime.config.providers["anthropic"].api_key == secret
        text = screen_text(app)
        assert secret not in text and "SECRETSECRET" not in text
        assert "key in config" in text
        assert all(
            secret not in str(getattr(widget, "value", "")) for widget in screen.query(TextEditor)
        )

        remove = control(screen, "provider:anthropic:remove-key")
        assert isinstance(remove, ActionButton)
        remove.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: "anthropic" not in app.runtime.config.providers)


async def test_api_key_with_spaces_is_refused(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        editor = control(screen, "provider:anthropic:key")
        editor.focus()
        editor.value = "two words"
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert "cannot contain spaces" in screen_text(app)
        assert "anthropic" not in app.runtime.config.providers


async def test_local_provider_can_be_switched_on(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        providers = screen.query_one("#provider-list")
        ids = [o.id for o in providers._options]  # type: ignore[attr-defined]
        providers.highlighted = ids.index("ollama")  # type: ignore[attr-defined]
        await wait_for(pilot, lambda: has_control(screen, "provider:ollama:enabled"))
        switch = control(screen, "provider:ollama:enabled")
        switch.focus()
        await pilot.press("space")
        await wait_for(pilot, lambda: "ollama" in app.runtime.config.providers)
        assert app.runtime.config.providers["ollama"].enabled is True


async def test_custom_provider_can_be_added_and_removed(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        await type_into(pilot, control(screen, "new:id"), "homelab")
        await type_into(pilot, control(screen, "new:base-url"), "http://localhost:9000/v1")
        keyless = control(screen, "new:keyless")
        keyless.focus()
        await pilot.press("space")
        add = control(screen, "new:add")
        add.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: "homelab" in app.runtime.config.providers)
        assert app.runtime.config.providers["homelab"].base_url == "http://localhost:9000/v1"
        assert app.runtime.config.providers["homelab"].keyless is True

        await wait_for(pilot, lambda: has_control(screen, "provider:homelab:remove"))
        remove = control(screen, "provider:homelab:remove")
        remove.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: "homelab" not in app.runtime.config.providers)


async def test_bad_custom_provider_shows_why(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        await type_into(pilot, control(screen, "new:id"), "Bad Id")
        add = control(screen, "new:add")
        add.focus()
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert "provider id starts with a letter" in screen_text(app)


async def test_theme_list_previews_keeps_and_reverts(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "appearance")
        themes = screen.query_one(ThemeList)
        themes.focus()
        await pilot.press("down")
        await pilot.pause(0.05)
        assert app.theme == "ayu-mirage"
        assert app.runtime.config.ui.theme == "ayu-dark"
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert app.theme == "ayu-dark" and isinstance(app.screen, SettingsScreen)

        await pilot.press("down", "down", "enter")
        await wait_for(pilot, lambda: app.runtime.config.ui.theme == "ayu-light")
        assert app.theme == "ayu-light"
        assert "saved" in screen_text(app)


async def test_skill_proposals_can_be_approved_rejected_and_deleted(app: ClankitApp) -> None:
    library = app.runtime.skills
    body = "---\nname: {name}\ndescription: {description}\nmetadata:\n  clankit.reason: {reason}\n---\n\n# Steps\n\nRun the tests first.\n"
    for name in ("run-tests", "lint-first"):
        library.save(
            name,
            {
                "SKILL.md": body.format(
                    name=name, description=f"How to {name}.", reason=f"{name} failed twice"
                )
            },
            pending=True,
        )
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "skills")
        text = screen_text(app)
        assert "proposed" in text and "lint-first failed twice" in text
        assert "Run the tests first." in text

        approve = control(screen, "skill:approve:lint-first")
        approve.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: library.get("lint-first") is not None)
        assert "Approved lint-first" in screen_text(app)

        skills = screen.query_one("#skill-list")
        ids = [option.id for option in skills._options]  # type: ignore[attr-defined]
        skills.highlighted = ids.index("pending:run-tests")  # type: ignore[attr-defined]
        await wait_for(pilot, lambda: has_control(screen, "skill:reject:run-tests"))
        reject = control(screen, "skill:reject:run-tests")
        reject.focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: not library.pending())

        await wait_for(pilot, lambda: has_control(screen, "skill:delete:lint-first"))
        delete = control(screen, "skill:delete:lint-first")
        delete.focus()
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert library.get("lint-first") is not None, "the first press only asks to confirm"
        assert "Press again to delete" in screen_text(app)
        control(screen, "skill:delete:lint-first").focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: library.get("lint-first") is None)


async def test_stale_provider_events_after_a_category_switch_are_ignored(app: ClankitApp) -> None:
    """A highlight or save from the provider list can still be delivered after the form
    switched to another category (its message pump runs alongside the screen's); the queued
    rebuilds must not crash and must leave the other category's form alone."""
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "providers")
        provider = screen._provider
        assert provider is not None
        await screen.show_category("models")
        await screen._rebuild("#provider-detail", screen._provider_detail)
        await screen._refresh_providers(focus="provider:anthropic:key")
        assert screen.category == "models"
        assert screen._provider == provider
        assert has_control(screen, "models.roles.subagent")


async def test_stale_method_highlight_updates_only_the_selection(app: ClankitApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        screen = await open_settings(pilot, app, "orchestration")
        methods = screen.query_one("#method-list")
        option = next(o for o in methods._options if o.id == "demo")  # type: ignore[attr-defined]
        await screen.show_category("agents")
        screen._method_highlighted(
            methods.OptionHighlighted(methods, option, 1)  # type: ignore[attr-defined]
        )
        assert screen._method == "demo"
        assert has_control(screen, "agent.max_turns")


@pytest.mark.parametrize("where", ["main", "settings"])
async def test_every_theme_applies_without_errors(app: ClankitApp, where: str) -> None:
    async with app.run_test(size=SIZE) as pilot:
        if where == "settings":
            await open_settings(pilot, app, "appearance")
        for theme_id in all_themes(app.runtime.config):
            app.runtime.update_config(lambda c, t=theme_id: setattr(c.ui, "theme", t))
            await pilot.pause(0.02)
            assert app.theme == theme_id
            assert screen_text(app).strip()

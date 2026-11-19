"""The Textual application: themes, the event pump, commands, screens and sessions.

The engine runs on the app's own event loop: `Session.start()` creates the run as a task on
the running loop, and the session's `EventBus` calls the pump synchronously from that task.
The pump hands batches to the main screen, which re-reads the session's folded view.

Configuration changes made anywhere (the settings screen, a slash command) go through
`Runtime.update_config`; the app listens to the store and applies what affects it (the
theme, custom themes, display options) as soon as the file is written.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import ClassVar

from textual.app import App
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider

from clankit import commands
from clankit.agent.events import Event, RunStarted
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.orchestration import get_orchestrator
from clankit.runtime import Runtime
from clankit.themes import get_theme
from clankit.themes.textual import textual_themes, variable_defaults
from clankit.tui.entries import History
from clankit.tui.format import model_alias
from clankit.tui.main import EventsArrived, MainScreen
from clankit.tui.pump import EventPump
from clankit.tui.styles import APP_CSS, install_scrollbar

#: Seconds within which a second ctrl+c quits.
QUIT_WINDOW = 2.0


class CommandSource(Provider):
    """Every slash command, offered in the command palette.

    It replaces Textual's own system commands, whose theme switcher would offer Textual's
    stock themes and change the theme without saving it.
    """

    def _hits(self) -> list[tuple[str, str, str]]:
        return [(f"/{c.name}", c.summary, f"/{c.name}") for c in commands.COMMANDS]

    async def discover(self) -> Hits:
        app = self.app
        assert isinstance(app, ClankitApp)
        for text, help_text, line in self._hits():
            yield DiscoveryHit(text, partial(app.run_command_line, line), help=help_text)

    async def search(self, query: str) -> Hits:
        app = self.app
        assert isinstance(app, ClankitApp)
        matcher = self.matcher(query)
        for text, help_text, line in self._hits():
            score = max(matcher.match(text), matcher.match(help_text))
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(text),
                    partial(app.run_command_line, line),
                    help=help_text,
                )


class ClankitApp(App[None]):
    """The terminal interface for one runtime, showing one session at a time."""

    CSS = APP_CSS
    TITLE = "Clankit"
    COMMANDS: ClassVar[set] = {CommandSource}
    COMMAND_PALETTE_BINDING = "ctrl+p"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "interrupt", "Quit", priority=True, show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]

    def __init__(
        self, runtime: Runtime, session: Session, *, flush_interval: float = 1 / 20
    ) -> None:
        super().__init__()
        install_scrollbar()
        self.runtime = runtime
        self.session = session
        self.pump = EventPump(self._deliver, flush_interval)
        self.main = MainScreen()
        self._quit_armed_at = 0.0
        self._unsubscribe_config = runtime.store.subscribe(self._config_changed)
        self._register_themes(runtime.config)
        self.theme = get_theme(runtime.config.ui.theme, runtime.config).id

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return variable_defaults()

    # -- lifecycle ------------------------------------------------------------------------------

    async def on_mount(self) -> None:
        await self.push_screen(self.main)
        self.pump.attach(self.session.bus)
        await self.main.attach(self._history_for(self.session))

    async def on_unmount(self) -> None:
        self.pump.detach()
        self._unsubscribe_config()
        await self.runtime.aclose()
        await asyncio.sleep(0)

    def _deliver(self, events: list[Event]) -> None:
        self.main.post_message(EventsArrived(events))

    def _history_for(self, session: Session) -> History:
        """The run log of `session`: from its bus, or from its stored log when resumed."""
        events = session.bus.since(0) or []
        if session.turns and not any(isinstance(e, RunStarted) for e in events):
            try:
                _, stored = self.runtime.session_store.load(session.id)
            except (OSError, ValueError):
                stored = []
            events = [*stored, *events]
        return History.from_events(events)

    # -- configuration --------------------------------------------------------------------------

    def _register_themes(self, config: Config) -> None:
        for theme in textual_themes(config):
            self.register_theme(theme)

    def _config_changed(self, config: Config) -> None:
        self._register_themes(config)
        wanted = get_theme(config.ui.theme, config).id
        if wanted != self.theme:
            self.theme = wanted
        else:
            self.refresh_css()
        self.main.config_changed(config)

    def model_label(self) -> str:
        """The model the next run's root agent would use, as the status row names it."""
        session = self.session
        try:
            if not get_orchestrator(session.method).needs_model:
                return "no model needed"
        except KeyError:
            pass
        ref = session.model_override or self.runtime.providers.default_model()
        return model_alias(str(ref)) if ref is not None else "no model"

    # -- runs and sessions ----------------------------------------------------------------------

    def start_run(self, prompt: str) -> None:
        try:
            self.session.start(prompt)
        except RuntimeError as exc:
            self.main.say(str(exc).capitalize() + ".", "warning")

    async def switch_session(self, session: Session) -> None:
        """Show `session` instead of the current one."""
        if session is self.session:
            return
        self.session = session
        self.pump.attach(session.bus)
        if self.screen is not self.main:
            await self._return_to_main()
        await self.main.attach(self._history_for(session))

    async def _return_to_main(self) -> None:
        while self.screen is not self.main and len(self.screen_stack) > 1:
            self.pop_screen()
            await asyncio.sleep(0)

    async def new_session(self) -> None:
        old = self.session
        if old.busy:
            self.main.say("Stop the current run first (esc), then start a new session.", "warning")
            return
        session = self.runtime.create_session(old.workspace, interactive=True)
        session.method_override = old.method_override
        session.model_override = old.model_override
        await self.switch_session(session)
        self.main.flash(f"New session {session.id}")

    async def resume(self, session_id: str) -> None:
        if session_id == self.session.id:
            await self._return_to_main()
            return
        if self.session.busy:
            self.main.say(
                "Stop the current run first (esc), then resume another session.", "warning"
            )
            return
        try:
            session = self.runtime.resume_session(session_id, interactive=True)
        except (OSError, ValueError, KeyError) as exc:
            self.main.say(f"Could not resume session {session_id}: {exc}", "error")
            return
        await self.switch_session(session)
        self.main.flash(f"Resumed session {session_id}")

    # -- commands -------------------------------------------------------------------------------

    def run_command_line(self, line: str) -> None:
        parsed = commands.parse(line)
        if parsed is not None:
            self.run_command(*parsed)

    def run_command(self, name: str, argument: str = "") -> None:
        """Run a slash command in the background; its outcome is shown when it finishes."""
        self.run_worker(self._command(name, argument), group="commands", exit_on_error=False)

    async def _command(self, name: str, argument: str) -> None:
        try:
            outcome = await commands.execute(self.runtime, self.session, name, argument)
        except Exception as exc:
            self.main.say(f"/{name} failed: {exc}", "error")
            return
        if outcome.message:
            self.main.say(outcome.message, outcome.level)
        if outcome.view:
            self.open_view(outcome.view)
        if outcome.prompt:
            await self.main.show_conversation()
            self.start_run(outcome.prompt)
        if outcome.action == "new-session":
            await self.new_session()
        elif outcome.action == "quit":
            self.exit()

    def open_view(self, name: str) -> None:
        """Open one of the views a command can ask for."""
        from clankit.tui.pickers import MethodPicker, ModelPicker, SessionPicker, ThemePicker
        from clankit.tui.screens import HelpScreen
        from clankit.tui.settings import SettingsScreen

        factories = {
            "help": HelpScreen,
            "settings": SettingsScreen,
            "skills": partial(SettingsScreen, category="skills"),
            "models": ModelPicker,
            "methods": MethodPicker,
            "themes": ThemePicker,
            "sessions": SessionPicker,
        }
        factory = factories.get(name)
        if factory is not None:
            self.push_screen(factory())

    def action_open_view(self, name: str) -> None:
        self.open_view(name)

    def open_tree(self) -> None:
        from clankit.tui.screens import TreeScreen

        self.push_screen(TreeScreen())

    # -- keys -----------------------------------------------------------------------------------

    def action_interrupt(self) -> None:
        """ctrl+c copies a selection; otherwise pressing it twice quits."""
        selected = self.screen.get_selected_text()
        focused = self.focused
        if focused is not None and getattr(focused, "selected_text", ""):
            self.copy_to_clipboard(focused.selected_text)  # type: ignore[attr-defined]
            return
        if selected:
            self.copy_to_clipboard(selected)
            return
        now = time.monotonic()
        if now - self._quit_armed_at <= QUIT_WINDOW:
            self.exit()
            return
        self._quit_armed_at = now
        self.main.flash("Press ctrl+c again to quit.", QUIT_WINDOW)

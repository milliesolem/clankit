"""Keyboard-first lists for choosing a model, a method, a theme or a stored session.

Each picker is a full-screen list with a filter row: typing narrows the list (every word must
match), up and down move, enter chooses, esc goes back. Choosing runs the matching slash
command, so a choice made here behaves exactly like one typed as ``/model ...``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from clankit.orchestration import all_orchestrators
from clankit.themes import Theme, all_themes
from clankit.tui.format import ellipsize, pad
from clankit.tui.screens import ViewScreen


@dataclass(frozen=True, slots=True)
class Choice:
    """One row of a picker. A choice without an id is a group heading."""

    id: str | None
    prompt: Content
    search: str = ""


def matches(choice: Choice, query: str) -> bool:
    words = query.lower().split()
    haystack = choice.search.lower()
    return all(word in haystack for word in words)


def swatches(theme: Theme) -> Content:
    """Six solid cells: canvas, chrome, raised surface, accent, success and error."""
    colours = (theme.bg, theme.bg_alt, theme.bg_raised, theme.accent, theme.success, theme.error)
    return Content.assemble(*(Content.styled("  ", f"on {colour}") for colour in colours))


def theme_prompt(theme: Theme, saved: str | None) -> Content:
    """A theme as lists show it: swatches, name, whether it is the saved one, dark or light."""
    return Content.assemble(
        swatches(theme),
        "  ",
        (pad(theme.name, 36), "$text"),
        (pad("current" if theme.id == saved else "", 9), "$fg-muted"),
        ("dark" if theme.dark else "light", "$fg-muted"),
    )


class PickerScreen(ViewScreen):
    """A filterable list. Subclasses provide `choices` and act in `choose`."""

    DEFAULT_CSS = """
    PickerScreen > #filter {
        margin: 1 2 0 2;
    }
    PickerScreen > #choices {
        height: 1fr;
        margin: 1 2 1 2;
        background: $background;
    }
    PickerScreen > #choices > .option-list--option {
        padding: 0 1;
    }
    PickerScreen > #empty {
        margin: 1 3;
        height: auto;
        display: none;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "close", "Back", show=False),
        Binding("up", "move('cursor_up')", show=False),
        Binding("down", "move('cursor_down')", show=False),
        Binding("pageup", "move('page_up')", show=False),
        Binding("pagedown", "move('page_down')", show=False),
    ]

    placeholder = "type to filter"
    empty_text = "Nothing to choose from."

    def __init__(self) -> None:
        super().__init__()
        self.filter_input = Input(placeholder=self.placeholder, id="filter", compact=True)
        self.choice_list = OptionList(id="choices", compact=True)
        self.empty_note = Static(Content.styled(self.empty_text, "$fg-muted"), id="empty")
        self._choices: list[Choice] = []

    def compose(self) -> ComposeResult:
        yield self.title_bar
        yield self.filter_input
        yield self.empty_note
        yield self.choice_list
        yield self.status_bar

    def on_mount(self) -> None:
        self.reload()
        self.filter_input.focus()

    def choices(self) -> list[Choice]:
        """The rows this picker offers. Subclasses provide them; the base has none."""
        return []

    def current(self) -> str | None:
        """The id of the choice in effect now, highlighted when the picker opens."""
        return None

    def choose(self, choice_id: str) -> None:
        """Act on a chosen row. Subclasses run a command; the base just closes."""
        self.dismiss()

    def preview(self, choice_id: str) -> None:
        """Called when the highlight moves; most pickers do nothing."""

    def reload(self) -> None:
        self._choices = self.choices()
        self._fill(self.filter_input.value, keep=self.current())

    def _fill(self, query: str, keep: str | None = None) -> None:
        visible: list[Choice] = []
        heading: Choice | None = None
        for choice in self._choices:
            if choice.id is None:
                heading = choice
            elif matches(choice, query):
                if heading is not None:
                    visible.append(heading)
                    heading = None
                visible.append(choice)
        self.choice_list.set_options(
            [Option(c.prompt, id=c.id, disabled=c.id is None) for c in visible]
        )
        self.empty_note.display = not visible
        if not self._choices:
            self.empty_note.update(Content.styled(self.empty_text, "$fg-muted"))
        elif not visible:
            self.empty_note.update(Content.styled("Nothing matches the filter.", "$fg-muted"))
        ids = [c.id for c in visible]
        first = next((i for i, c in enumerate(visible) if c.id is not None), None)
        self.choice_list.highlighted = (
            ids.index(keep) if keep is not None and keep in ids else first
        )

    @on(Input.Changed, "#filter")
    def _filtered(self, event: Input.Changed) -> None:
        self._fill(event.value)

    @on(Input.Submitted, "#filter")
    def _submitted(self) -> None:
        option = self.choice_list.highlighted_option
        if option is not None and option.id is not None and not option.disabled:
            self.choose(option.id)

    @on(OptionList.OptionSelected, "#choices")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self.choose(event.option.id)

    @on(OptionList.OptionHighlighted, "#choices")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id is not None:
            self.preview(event.option.id)

    async def action_move(self, action: str) -> None:
        """Move the list's highlight while the filter keeps the keyboard."""
        await self.choice_list.run_action(action)


class ModelPicker(PickerScreen):
    """Models of every usable provider, grouped by provider."""

    title_text = "model"
    title_keys: ClassVar[list[tuple[str, str]]] = [
        ("enter", "choose"),
        ("ctrl+r", "ask providers"),
        ("esc", "back"),
    ]
    BINDINGS: ClassVar[list[Binding]] = [
        *PickerScreen.BINDINGS,
        Binding("ctrl+r", "discover", "Ask providers for their models", show=False),
    ]
    empty_text = "No usable provider yet. Add an API key in the settings (f2), then come back."

    def current(self) -> str | None:
        session = self.clankit.session
        ref = session.model_override or self.clankit.runtime.providers.default_model()
        return str(ref) if ref is not None else None

    def choices(self) -> list[Choice]:
        current = self.current()
        result: list[Choice] = []
        provider = None
        for info in self.clankit.runtime.providers.known_models():
            if info.provider != provider:
                provider = info.provider
                result.append(Choice(None, Content.styled(provider, "bold $fg-muted")))
            ref = str(info.ref)
            tags = []
            if info.free:
                tags.append("free")
            if ref == current:
                tags.append("current")
            prompt = Content.assemble(
                (pad(info.id, 34), "$text"),
                " ",
                (pad("  ".join(tags), 14), "$fg-muted"),
                (ellipsize(info.note, 90), "$fg-muted"),
            )
            result.append(Choice(ref, prompt, f"{ref} {info.name} {info.note} {' '.join(tags)}"))
        return result

    def choose(self, choice_id: str) -> None:
        self.dismiss()
        self.clankit.run_command("model", choice_id)

    def action_discover(self) -> None:
        self.set_status(Content.styled("Asking providers for their models...", "$fg-muted"))
        self._discover()

    @work(exclusive=True, group="discover", exit_on_error=False)
    async def _discover(self) -> None:
        models = await self.clankit.runtime.providers.discover()
        self.reload()
        self.set_status(Content.styled(f"{len(models)} models known", "$fg-muted"))


class MethodPicker(PickerScreen):
    """Orchestration methods, with what each one does."""

    title_text = "orchestration method"
    title_keys: ClassVar[list[tuple[str, str]]] = [("enter", "choose"), ("esc", "back")]

    def current(self) -> str | None:
        return self.clankit.session.method

    def choices(self) -> list[Choice]:
        current = self.current()
        return [
            Choice(
                method.name,
                Content.assemble(
                    (pad(method.title, 20), "bold $text" if method.name == current else "$text"),
                    (pad("current" if method.name == current else "", 9), "$fg-muted"),
                    (method.summary, "$fg-muted"),
                ),
                f"{method.name} {method.title} {method.summary}",
            )
            for method in all_orchestrators()
        ]

    def preview(self, choice_id: str) -> None:
        method = next((m for m in all_orchestrators() if m.name == choice_id), None)
        if method is not None:
            self.set_status(Content.styled(method.guidance, "$fg-muted"))

    def choose(self, choice_id: str) -> None:
        self.dismiss()
        self.clankit.run_command("method", choice_id)


class ThemePicker(PickerScreen):
    """Colour themes, previewed as the highlight moves; esc puts the saved one back."""

    title_text = "theme"
    title_keys: ClassVar[list[tuple[str, str]]] = [
        ("enter", "keep"),
        ("esc", "put back"),
    ]

    def current(self) -> str | None:
        return self.clankit.runtime.config.ui.theme

    def choices(self) -> list[Choice]:
        current = self.current()
        return [
            Choice(
                theme.id,
                theme_prompt(theme, current),
                f"{theme.id} {theme.name} {'dark' if theme.dark else 'light'}",
            )
            for theme in all_themes(self.clankit.runtime.config).values()
        ]

    def preview(self, choice_id: str) -> None:
        if choice_id in self.app.available_themes:
            self.app.theme = choice_id
            self.set_status(Content.styled(f"previewing {choice_id}", "$fg-muted"))

    def choose(self, choice_id: str) -> None:
        self.dismiss()
        self.clankit.run_command("theme", choice_id)

    def action_close(self) -> None:
        saved = self.current()
        if saved and saved in self.app.available_themes:
            self.app.theme = saved
        self.dismiss()


class SessionPicker(PickerScreen):
    """Stored sessions of this workspace, newest first."""

    title_text = "resume a session"
    title_keys: ClassVar[list[tuple[str, str]]] = [("enter", "resume"), ("esc", "back")]
    empty_text = "No stored sessions for this directory yet."

    def current(self) -> str | None:
        return self.clankit.session.id

    def choices(self) -> list[Choice]:
        session = self.clankit.session
        result = []
        for meta in self.clankit.runtime.session_store.list(session.workspace):
            when = datetime.datetime.fromtimestamp(meta.updated).strftime("%Y-%m-%d %H:%M")
            turns = f"{meta.turns} turn" + ("" if meta.turns == 1 else "s")
            marker = "current" if meta.id == session.id else ""
            prompt = Content.assemble(
                (when, "$fg-muted"),
                "  ",
                (pad(turns, 10, right=True), "$fg-muted"),
                "  ",
                (pad(marker, 8), "$fg-muted"),
                (meta.title or "(untitled)", "$text"),
            )
            result.append(Choice(meta.id, prompt, f"{meta.id} {meta.title} {when}"))
        return result

    def choose(self, choice_id: str) -> None:
        self.dismiss()
        self.call_later(self.clankit.resume, choice_id)

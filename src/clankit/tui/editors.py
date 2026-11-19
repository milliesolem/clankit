"""Form controls for the settings screen.

Every control reports a change with one message, `Edited`, carrying the key of the setting
and the new value; the settings screen owns the saving. Controls are flat: a text field is a
raised strip with an accent bar while it has focus, a switch or a choice is a row of solid
segments with the current one on the selection surface. Nothing draws a frame.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Input, Static


class Edited(Message):
    """A control's value was changed by the user."""

    def __init__(self, key: str, value: Any) -> None:
        super().__init__()
        self.key = key
        self.value = value


class Pressed(Message):
    """An action control was activated."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key


class TextEditor(Input):
    """A one-line text field that saves on enter, and on leaving it if it changed."""

    def __init__(
        self,
        key: str,
        value: str,
        *,
        placeholder: str = "",
        password: bool = False,
        suggestions: list[str] | None = None,
        keep: bool = True,
    ) -> None:
        super().__init__(
            value,
            placeholder=placeholder,
            password=password,
            compact=True,
            suggester=SuggestFromList(suggestions, case_sensitive=False) if suggestions else None,
        )
        self.key = key
        self.saved_value = value
        #: False for fields that are cleared after saving (secrets, one-shot forms).
        self.keep = keep

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._commit()

    def on_blur(self) -> None:
        if self.keep and self.value != self.saved_value:
            self._commit()

    def _commit(self) -> None:
        self.post_message(Edited(self.key, self.value))

    def accepted(self) -> None:
        """The screen saved the value; remember it (or forget it, for a secret)."""
        if self.keep:
            self.saved_value = self.value
        else:
            self.value = ""


class Segments(Widget, can_focus=True):
    """A row of solid segments, one of which is chosen; left and right change it."""

    DEFAULT_CSS = """
    Segments {
        height: auto;
        width: 1fr;
        border-left: blank;
        padding: 0 0 0 1;
    }
    Segments:focus {
        border-left: outer $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "step(-1)", "Previous", show=False),
        Binding("right", "step(1)", "Next", show=False),
        Binding("space,enter", "step(1)", "Next", show=False),
    ]

    def __init__(self, key: str, options: list[tuple[Any, str]], value: Any) -> None:
        super().__init__()
        self.key = key
        self.options = options
        self.value = value

    @property
    def index(self) -> int:
        values = [value for value, _ in self.options]
        return values.index(self.value) if self.value in values else 0

    def render(self) -> Content:
        parts: list[Content | str | tuple[str, str]] = []
        for position, (value, label) in enumerate(self.options):
            chosen = value == self.value
            style = "bold $text on $selection" if chosen else "$fg-muted on $bg-raised"
            if chosen and self.has_focus:
                style = "bold $on-accent on $accent"
            parts.extend([" " if position else "", (f" {label} ", style)])
        return Content.assemble(*parts)

    def action_step(self, step: int) -> None:
        if not self.options:
            return
        index = (self.index + step) % len(self.options)
        self.value = self.options[index][0]
        self.refresh()
        self.post_message(Edited(self.key, self.value))

    def on_click(self, event: events.Click) -> None:
        offset = event.x - 1
        position = 0
        for value, label in self.options:
            width = len(label) + 2
            if position <= offset < position + width:
                if value != self.value:
                    self.value = value
                    self.refresh()
                    self.post_message(Edited(self.key, self.value))
                return
            position += width + 1

    def on_focus(self) -> None:
        self.refresh()

    def on_blur(self) -> None:
        self.refresh()


def toggle(key: str, value: bool) -> Segments:
    return Segments(key, [(True, "on"), (False, "off")], value)


def choice(key: str, choices: tuple[str, ...], value: str, empty: str = "") -> Segments:
    return Segments(key, [(c, c or empty or "none") for c in choices], value)


class ActionButton(Widget, can_focus=True):
    """A flat button: a solid segment that runs its action on enter or a click."""

    DEFAULT_CSS = """
    ActionButton {
        width: auto;
        height: 1;
        padding: 0 2;
        margin: 0 2 0 0;
        background: $selection;
        color: $text;
    }
    ActionButton:focus {
        background: $accent;
        color: $on-accent;
        text-style: bold;
    }
    ActionButton:hover {
        background: $line;
    }
    ActionButton.-danger:focus {
        background: $error;
        color: $on-accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("enter,space", "press", "Press", show=False)]

    def __init__(self, key: str, label: str, *, danger: bool = False) -> None:
        super().__init__(classes="-danger" if danger else "")
        self.key = key
        self.label = label

    def render(self) -> Content:
        return Content(self.label)

    def action_press(self) -> None:
        self.post_message(Pressed(self.key))

    def on_click(self) -> None:
        self.action_press()


class FieldRow(Vertical):
    """One setting: a label and a control side by side, help text and errors beneath."""

    DEFAULT_CSS = """
    FieldRow {
        height: auto;
        margin: 0 0 1 0;
    }
    FieldRow > .field-line {
        height: auto;
    }
    FieldRow .field-label {
        width: 26;
        color: $text;
        padding: 0 2 0 0;
    }
    FieldRow .field-control {
        width: 1fr;
        height: auto;
    }
    FieldRow > .field-help {
        margin: 0 0 0 28;
        color: $fg-muted;
        height: auto;
    }
    FieldRow > .field-error {
        margin: 0 0 0 26;
        padding: 0 1;
        border-left: outer $error;
        color: $text;
        height: auto;
        display: none;
    }
    """

    def __init__(self, key: str, label: str, control: Widget, help_text: str = "") -> None:
        super().__init__()
        self.field_key = key
        self.label = label
        self.control = control
        self.help_text = help_text
        self.error = Static("", classes="field-error", markup=False)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="field-line"):
            yield Static(Content(self.label), classes="field-label")
            with Vertical(classes="field-control"):
                yield self.control
        if self.help_text:
            yield Static(Content(self.help_text), classes="field-help")
        yield self.error

    def show_error(self, text: str) -> None:
        self.error.update(Content(text))
        self.error.display = True

    def clear_error(self) -> None:
        self.error.display = False


class SectionTitle(Static):
    """A bold, lowercase section heading in the form."""

    DEFAULT_CSS = """
    SectionTitle {
        color: $fg-muted;
        text-style: bold;
        margin: 1 0 1 0;
        height: auto;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(Content(text.lower()))


class Note(Static):
    """Explanatory text in the form, muted."""

    DEFAULT_CSS = """
    Note {
        color: $fg-muted;
        height: auto;
        margin: 0 0 1 0;
    }
    """

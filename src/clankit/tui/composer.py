"""The composer: where prompts, steering messages, answers and commands are typed.

Enter sends. Ctrl+J always inserts a newline; Shift+Enter and Alt+Enter do too on terminals
that report them (the kitty keyboard protocol), and on others they arrive as a plain Enter.
Up and Down walk the history when the cursor is on the first or last line. A line that
starts with ``/`` opens a completion list of commands above the composer.
"""

from __future__ import annotations

from typing import ClassVar

from textual.binding import Binding
from textual.content import Content
from textual.message import Message
from textual.widgets import OptionList, TextArea
from textual.widgets.option_list import Option

from clankit import commands
from clankit.tui.format import pad

MAX_HISTORY = 500


class CompletionList(OptionList, can_focus=False):
    """Commands matching what has been typed after ``/``."""

    DEFAULT_CSS = """
    CompletionList {
        height: auto;
        max-height: 10;
        background: $bg-raised;
        margin: 0 2;
        padding: 0;
        display: none;
    }
    CompletionList > .option-list--option-highlighted {
        background: $selection;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, compact=True)
        self.names: list[str] = []

    @property
    def is_open(self) -> bool:
        return bool(self.display) and bool(self.names)

    def show_for(self, text: str) -> None:
        """Offer the commands that complete `text`, or hide when there are none."""
        stripped = text.strip()
        if not stripped.startswith("/") or " " in stripped or "\n" in text:
            self.hide()
            return
        matches = commands.complete(stripped)
        if not matches or (len(matches) == 1 and f"/{matches[0].name}" == stripped):
            self.hide()
            return
        names = [command.name for command in matches]
        if names != self.names:
            self.names = names
            width = max(len(name) for name in names) + 2
            self.set_options(
                [
                    Option(
                        Content.assemble(
                            (pad(f"/{command.name}", width), "bold"),
                            (pad(command.argument, 40), "$fg-muted"),
                            " ",
                            Content.styled(command.summary, "$fg-muted"),
                        ),
                        id=command.name,
                    )
                    for command in matches
                ]
            )
            self.highlighted = 0
        self.display = True

    def hide(self) -> None:
        self.display = False
        self.names = []

    def move(self, step: int) -> None:
        if self.names:
            current = self.highlighted or 0
            self.highlighted = (current + step) % len(self.names)

    def chosen(self) -> str | None:
        if not self.is_open or self.highlighted is None:
            return None
        return self.names[self.highlighted]


class Composer(TextArea):
    """A growing text box that submits on Enter."""

    DEFAULT_CSS = """
    Composer {
        height: auto;
        min-height: 1;
        max-height: 8;
        padding: 0;
        background: $bg-raised;
        border: none;
    }
    Composer:focus {
        background: $bg-raised;
    }
    Composer > .text-area--cursor-line {
        background: $bg-raised;
    }
    Composer > .text-area--placeholder {
        color: $fg-subtle;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", priority=True, show=False),
        Binding("ctrl+j,shift+enter,alt+enter", "newline", "New line", priority=True, show=False),
        Binding("tab", "complete", "Complete", priority=True, show=False),
    ]

    class Submitted(Message):
        """The user pressed Enter on some text."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, *, id: str | None = None, placeholder: str = "") -> None:
        super().__init__(
            id=id,
            compact=True,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            highlight_cursor_line=False,
            placeholder=placeholder,
        )
        self.completion: CompletionList | None = None
        self.prompt_history: list[str] = []
        self._history_index: int | None = None
        self._draft = ""

    # -- history --------------------------------------------------------------------------------

    def remember(self, text: str) -> None:
        if text.strip() and (not self.prompt_history or self.prompt_history[-1] != text):
            self.prompt_history.append(text)
            del self.prompt_history[:-MAX_HISTORY]
        self._history_index = None

    def _recall(self, step: int) -> bool:
        if not self.prompt_history:
            return False
        if self._history_index is None:
            if step > 0:
                return False
            self._draft = self.text
            index = len(self.prompt_history) - 1
        else:
            index = self._history_index + step
        if index >= len(self.prompt_history):
            self._history_index = None
            self.load(self._draft)
            return True
        index = max(0, index)
        self._history_index = index
        self.load(self.prompt_history[index])
        return True

    def load(self, text: str) -> None:
        """Replace the text and put the cursor at its end."""
        self.text = text
        self.move_cursor(self.document.end)

    # -- actions --------------------------------------------------------------------------------

    def _completion_visible(self) -> bool:
        return self.completion is not None and self.completion.is_open

    def action_submit(self) -> None:
        if self._completion_visible():
            assert self.completion is not None
            name = self.completion.chosen()
            if name is not None:
                self.load(f"/{name}")
                self.completion.hide()
        text = self.text
        if not text.strip():
            return
        self.remember(text)
        self.load("")
        self.post_message(self.Submitted(text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        if self._completion_visible():
            assert self.completion is not None
            name = self.completion.chosen()
            if name is not None:
                command = commands.find(name)
                self.load(f"/{name}" + (" " if command and command.argument else ""))
                self.completion.hide()
            return
        self.screen.focus_next()

    def action_cursor_up(self, select: bool = False) -> None:
        if self._completion_visible() and not select:
            assert self.completion is not None
            self.completion.move(-1)
            return
        if not select and self.cursor_location[0] == 0 and self._recall(-1):
            return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if self._completion_visible() and not select:
            assert self.completion is not None
            self.completion.move(1)
            return
        last_row = self.document.line_count - 1
        if not select and self.cursor_location[0] == last_row and self._recall(1):
            return
        super().action_cursor_down(select)

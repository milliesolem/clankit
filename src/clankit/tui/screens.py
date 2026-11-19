"""Full-screen views that are not the main screen: keys and commands, and the task tree.

Every secondary view shares one frame: a chrome title row naming the view with its keys on
the right, the view's content on the canvas, and a chrome status row at the bottom. None of
them floats over the session as a dialog.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Static

from clankit import commands
from clankit.tui.main import Bar
from clankit.tui.requests import pending_requests
from clankit.tui.tasktree import TaskTree, TreePanel

if TYPE_CHECKING:
    from clankit.tui.app import ClankitApp


def keys_content(pairs: list[tuple[str, str]]) -> Content:
    """``key  word   key  word`` for a title or status row."""
    parts: list[Content | str | tuple[str, str]] = []
    for index, (key, word) in enumerate(pairs):
        parts.extend(["   " if index else "", (key, "bold $text"), (f" {word}", "$fg-muted")])
    return Content.assemble(*parts)


class ViewScreen(Screen[None]):
    """A full-screen view with a title row, a canvas and a status row."""

    DEFAULT_CSS = """
    ViewScreen {
        layout: vertical;
    }
    ViewScreen > #view-title {
        background: $bg-alt;
    }
    ViewScreen > #view-status {
        background: $bg-alt;
        dock: bottom;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "close", "Back", show=False)]

    title_text = ""
    title_keys: ClassVar[list[tuple[str, str]]] = [("esc", "back")]

    def __init__(self) -> None:
        super().__init__()
        self.title_bar = Bar(id="view-title")
        self.status_bar = Bar(id="view-status")

    @property
    def clankit(self) -> ClankitApp:
        from clankit.tui.app import ClankitApp

        app = self.app
        assert isinstance(app, ClankitApp)
        return app

    def on_mount(self) -> None:
        # Textual calls this after any subclass's own on_mount, so it keeps their status text.
        self.title_bar.show(
            Content.styled(self.title_text, "bold $text"), keys_content(self.title_keys)
        )
        self.set_status(self.status_bar.left)

    def set_status(self, content: Content) -> None:
        self.status_bar.show(content, Content.styled(self.app.theme, "$fg-muted"))

    def action_close(self) -> None:
        self.dismiss()


_KEYS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "composer",
        (
            ("enter", "send; while a run works, steer it; answer a waiting question"),
            ("ctrl+j", "new line (shift+enter and alt+enter where the terminal reports them)"),
            ("up, down", "earlier prompts, from the first or last line"),
            ("/", "commands, completed as you type"),
            ("tab", "next region: composer, transcript, task tree"),
        ),
    ),
    (
        "session",
        (
            ("esc", "stop the run; leave a node; close a list or peek"),
            ("ctrl+y", "answer a permission request or a question"),
            ("y, a, n", "allow once, allow for this session, deny"),
            ("1-9", "pick an option of a question"),
            ("ctrl+t", "task tree panel (full screen on narrow terminals)"),
            ("ctrl+o", "open or close every tool row"),
            ("ctrl+g", "jump to the latest output"),
            ("ctrl+p", "command palette"),
            ("f1, f2, f3", "keys, settings, skills"),
            ("ctrl+c twice", "quit (ctrl+q quits at once)"),
        ),
    ),
    (
        "transcript",
        (
            ("up, down", "move between tool calls and reasoning"),
            ("enter, space", "open or close the one under the cursor"),
            ("page keys", "scroll"),
        ),
    ),
    (
        "task tree",
        (
            ("up, down", "move"),
            ("right, left", "open or step in, close or step out"),
            ("enter", "show the node's work in the transcript"),
            ("space", "peek at the node"),
            ("*", "open every sibling"),
        ),
    ),
)


def help_content() -> Content:
    lines: list[Content] = []
    for title, rows in _KEYS:
        lines.append(Content.styled(title, "bold $fg-muted"))
        lines.extend(
            Content.assemble((f"  {key:<14}", "bold $text"), (word, "$text")) for key, word in rows
        )
        lines.append(Content(""))
    lines.append(Content.styled("commands", "bold $fg-muted"))
    for command in commands.COMMANDS:
        lines.append(
            Content.assemble(
                (f"  /{command.name:<13}", "bold $text"),
                (command.summary, "$text"),
                (f"  {command.argument}" if command.argument else "", "$fg-muted"),
            )
        )
    return Content("\n").join(lines)


class HelpScreen(ViewScreen):
    """Keys and commands."""

    DEFAULT_CSS = """
    HelpScreen #help {
        padding: 1 2;
        background: $background;
    }
    HelpScreen #help > Static {
        height: auto;
    }
    """

    title_text = "keys and commands"

    def compose(self) -> ComposeResult:
        yield self.title_bar
        with VerticalScroll(id="help"):
            yield Static(help_content())
        yield self.status_bar

    def on_mount(self) -> None:
        self.query_one("#help").focus()


class TreeScreen(ViewScreen):
    """The task tree, full screen, for terminals too narrow for the side panel."""

    DEFAULT_CSS = """
    TreeScreen > TreePanel {
        height: 1fr;
        width: 1fr;
    }
    """

    title_text = "task tree"
    title_keys: ClassVar[list[tuple[str, str]]] = [
        ("enter", "focus"),
        ("space", "peek"),
        ("esc", "back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.panel = TreePanel()

    def compose(self) -> ComposeResult:
        yield self.title_bar
        yield self.panel
        yield self.status_bar

    async def on_mount(self) -> None:
        await self.refresh_tree()
        self.panel.task_tree.focus()
        self.set_interval(0.25, self.refresh_tree)

    async def refresh_tree(self) -> None:
        session = self.clankit.session
        view = session.view
        waiting = {r.node_id for r in pending_requests(view)}
        main = self.clankit.main
        await self.panel.sync(
            view, waiting, time.time(), viewed=main.viewed, depth_limit=main.depth_limit()
        )

    @on(TaskTree.Focused)
    async def _focused(self, message: TaskTree.Focused) -> None:
        message.stop()
        self.dismiss()
        await self.clankit.main.view_node(message.node_id)

    @on(TaskTree.Peeked)
    def _peeked(self, message: TaskTree.Peeked) -> None:
        message.stop()
        view = self.clankit.session.view
        waiting = {r.node_id for r in pending_requests(view)}
        self.panel.show_peek(message.node_id, view, waiting, time.time())

"""Permission requests and questions, shown inline above the composer.

One request is shown at a time with a ``1 of N`` counter, always naming the node that asks
(its path in the tree) and its model, and showing exactly what would run: the command, or
the diff. The panel answers single keys - y, a, n for a permission, a number for a
question's option - but only while it has keyboard focus, and the main screen never gives
it focus while the composer holds text, so a "y" typed into a message can never approve
anything.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from clankit.agent.events import PermissionChoice, PermissionRequested, QuestionAsked
from clankit.agent.state import SessionView
from clankit.tui.format import ellipsize_left, model_alias
from clankit.tui.transcript import DiffView

Request = PermissionRequested | QuestionAsked

_OWN_KEYS = frozenset("yanv123456789")


def node_path(view: SessionView, node_id: str) -> str:
    """The titles from the root down to `node_id`, joined by `` / ``."""
    titles: list[str] = []
    node = view.nodes.get(node_id)
    while node is not None:
        titles.append(node.info.title)
        node = view.nodes.get(node.info.parent_id or "")
    return " / ".join(reversed(titles))


def pending_requests(view: SessionView) -> list[Request]:
    """Every unanswered permission request and question, oldest first."""
    pending: list[Request] = [*view.permissions.values(), *view.questions.values()]
    pending.sort(key=lambda request: request.seq)
    return pending


def looks_like_diff(text: str) -> bool:
    lines = text.splitlines()
    return any(line.startswith("@@") for line in lines) and any(
        line.startswith(("+", "-")) for line in lines
    )


class RequestTitle(Widget):
    """``permission  1 of 2   root / writer / tests   model``, with the path cut to fit."""

    DEFAULT_CSS = """
    RequestTitle {
        height: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="request-title")
        self.chip = ""
        self.counter = ""
        self.path = ""
        self.model = ""

    def show(self, request: Request, view: SessionView, count: int) -> None:
        node = view.nodes.get(request.node_id)
        self.chip = " permission " if isinstance(request, PermissionRequested) else " question "
        self.counter = f"1 of {count}"
        self.path = node_path(view, request.node_id)
        self.model = model_alias(node.info.model) if node else ""
        self.refresh()

    def render(self) -> Content:
        head = Content.assemble(
            (self.chip, "bold $on-accent on $warning"), "  ", (self.counter, "$fg-muted"), "   "
        )
        tail = Content.styled(f"   {self.model}" if self.model else "", "$fg-muted")
        room = self.content_region.width - head.cell_length - tail.cell_length
        return Content.assemble(head, (ellipsize_left(self.path, max(8, room)), "$text"), tail)


class RequestPanel(Vertical, can_focus=True):
    """The oldest pending request, with the keys that answer it."""

    DEFAULT_CSS = """
    RequestPanel {
        height: auto;
        max-height: 45vh;
        background: $bg-raised;
        border-left: outer $warning;
        padding: 1 1 1 1;
        margin: 1 2 0 2;
        display: none;
    }
    RequestPanel > .request-row {
        height: auto;
    }
    RequestPanel > #request-what {
        margin: 1 0 0 0;
    }
    RequestPanel > #request-detail {
        height: auto;
        max-height: 10;
        margin: 1 0 0 0;
        background: $bg-alt;
        scrollbar-background: $bg-alt;
    }
    RequestPanel > #request-detail > Static {
        padding: 0 1;
        height: auto;
    }
    RequestPanel > #request-keys {
        margin: 1 0 0 0;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "permit('allow')", "Allow once", show=False),
        Binding("a", "permit('allow_session')", "Allow for this session", show=False),
        Binding("n", "permit('deny')", "Deny", show=False),
        Binding("escape", "dismiss", "Deny or return", show=False),
        Binding("v", "view_node", "View node", show=False),
        *(Binding(str(n), f"option({n})", show=False) for n in range(1, 10)),
    ]

    class Permitted(Message):
        def __init__(self, request_id: str, choice: PermissionChoice) -> None:
            super().__init__()
            self.request_id = request_id
            self.choice = choice

    class Answered(Message):
        def __init__(self, request_id: str, answer: str) -> None:
            super().__init__()
            self.request_id = request_id
            self.answer = answer

    class Typed(Message):
        """A key that belongs in the composer was pressed while the panel had focus."""

        def __init__(self, character: str) -> None:
            super().__init__()
            self.character = character

    class ViewNode(Message):
        def __init__(self, node_id: str) -> None:
            super().__init__()
            self.node_id = node_id

    class Released(Message):
        """The panel gave up focus (Esc on a question)."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.request: Request | None = None
        self.count = 0
        self._title = RequestTitle()
        self._what = Static("", classes="request-row", id="request-what")
        self._detail = VerticalScroll(id="request-detail")
        self._keys = Static("", classes="request-row", id="request-keys")
        self._shown: tuple[str, int, bool] | None = None

    def compose(self) -> ComposeResult:
        yield self._title
        yield self._what
        yield self._detail
        yield self._keys

    @property
    def answering(self) -> bool:
        """Does the panel have the keyboard? (Read from the screen, which knows before the
        panel's own focus flag is updated.)"""
        return self.is_mounted and self.screen.focused is self

    @property
    def request_id(self) -> str | None:
        return self.request.request_id if self.request else None

    async def show(self, requests: list[Request], view: SessionView) -> None:
        """Show the oldest of `requests`, or hide the panel when there are none."""
        self.request = requests[0] if requests else None
        self.count = len(requests)
        self.display = self.request is not None
        if self.request is None:
            self._shown = None
            return
        key = (self.request.request_id, self.count, self.answering)
        if key == self._shown:
            return
        new_request = self._shown is None or self._shown[0] != self.request.request_id
        self._shown = key
        self._title.show(self.request, view, self.count)
        self._keys.update(self._keys_content(self.request))
        if new_request:
            await self._show_body(self.request)

    async def _show_body(self, request: Request) -> None:
        await self._detail.remove_children()
        if isinstance(request, PermissionRequested):
            self._what.update(
                Content.assemble((request.tool, "bold"), "   ", Content(request.summary))
            )
            detail = request.detail.strip("\n")
            if detail and detail != request.summary:
                if looks_like_diff(detail):
                    await self._detail.mount(DiffView(detail))
                else:
                    await self._detail.mount(Static(Content(detail)))
            self._detail.display = bool(detail and detail != request.summary)
        else:
            options = [
                Content.assemble((f"{index}", "bold"), " ", Content(option))
                for index, option in enumerate(request.options, start=1)
            ]
            self._what.update(Content(request.question))
            if options:
                await self._detail.mount(Static(Content("\n").join(options)))
            self._detail.display = bool(options)
        self._detail.scroll_home(animate=False)

    def _keys_content(self, request: Request) -> Content:
        if not self.answering:
            return Content.assemble(
                ("ctrl+y", "bold"),
                (" answer this request (the composer has your text)", "$fg-muted"),
            )
        if isinstance(request, QuestionAsked):
            parts: list[Content | str | tuple[str, str]] = []
            if request.options:
                parts.extend(
                    [("1-" + str(len(request.options)), "bold"), (" choose   ", "$fg-muted")]
                )
            parts.extend(
                [
                    ("type", "bold"),
                    (" your own answer and press enter   ", "$fg-muted"),
                    ("v", "bold"),
                    (" view node", "$fg-muted"),
                ]
            )
            return Content.assemble(*parts)
        rule = request.session_rule or f"{request.category}:*"
        return Content.assemble(
            ("y", "bold"),
            (" allow once    ", "$fg-muted"),
            ("a", "bold"),
            (" allow ", "$fg-muted"),
            (rule, "$text"),
            (" for this session    ", "$fg-muted"),
            ("n", "bold"),
            (" deny    ", "$fg-muted"),
            ("v", "bold"),
            (" view node", "$fg-muted"),
        )

    def refresh_keys(self) -> None:
        if self.request is not None:
            self._keys.update(self._keys_content(self.request))
            self._shown = (self.request.request_id, self.count, self.answering)

    def on_focus(self) -> None:
        self.refresh_keys()

    def on_blur(self) -> None:
        self.refresh_keys()

    # -- actions --------------------------------------------------------------------------------

    def action_permit(self, choice: PermissionChoice) -> None:
        if isinstance(self.request, PermissionRequested):
            self.post_message(self.Permitted(self.request.request_id, choice))
        elif isinstance(self.request, QuestionAsked):
            self.post_message(self.Typed({"allow": "y", "allow_session": "a", "deny": "n"}[choice]))

    def action_option(self, number: int) -> None:
        request = self.request
        if isinstance(request, QuestionAsked) and 0 < number <= len(request.options):
            self.post_message(self.Answered(request.request_id, request.options[number - 1]))
        elif isinstance(request, QuestionAsked):
            self.post_message(self.Typed(str(number)))

    def action_dismiss(self) -> None:
        if isinstance(self.request, PermissionRequested):
            self.post_message(self.Permitted(self.request.request_id, "deny"))
        else:
            self.post_message(self.Released())

    def action_view_node(self) -> None:
        if self.request is not None:
            self.post_message(self.ViewNode(self.request.node_id))

    def on_key(self, event: events.Key) -> None:
        if event.character and event.is_printable and event.character not in _OWN_KEYS:
            event.stop()
            event.prevent_default()
            self.post_message(self.Typed(event.character))

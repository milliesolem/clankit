"""The main screen: crumb row, transcript, task tree, requests, composer and status row.

Everything on this screen is drawn from the session's folded `SessionView` plus the
interface's own `History`. Engine events arrive in batches (see `clankit.tui.pump`); each
batch is folded into the history and triggers one refresh of whatever reads the view.

Width breakpoints: from 110 columns the task tree is a panel on the right; below that it
collapses to a one-line rollup under the crumb and opens full screen with ctrl+t; below 80
columns the rollup moves into the status row.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static, TextArea

from clankit import commands
from clankit.agent.events import Event, PermissionRequested, QuestionAsked, RunStarted
from clankit.config.schema import ConfigError
from clankit.orchestration import get_orchestrator
from clankit.tui.composer import CompletionList, Composer
from clankit.tui.entries import History, node_entries, root_entries
from clankit.tui.format import (
    PROMPT_MARK,
    STATE_GLYPHS,
    STATE_STYLES,
    STATE_WORDS,
    ellipsize_left,
    human_elapsed,
    human_tokens,
    node_state,
)
from clankit.tui.requests import RequestPanel, node_path, pending_requests
from clankit.tui.tasktree import TaskTree, TreePanel, counts_content, elapsed_of, tree_counts
from clankit.tui.transcript import Context, Transcript

if TYPE_CHECKING:
    from clankit.agent.session import Session
    from clankit.config.schema import Config
    from clankit.tui.app import ClankitApp

#: From this width the task tree is a side panel; below it, a rollup row.
WIDE = 110
#: Below this width the rollup moves into the status row.
NARROW = 80
#: Below this height the composer and banner give up their padding to the transcript.
SHORT = 36
#: Seconds between clock refreshes (elapsed times) while a run works.
TICK = 0.5
#: The task tree panel takes what the transcript does not need for a comfortable measure.
PANEL_MIN = 46
PANEL_MAX = 64
PROSE_COLUMNS = 78


class EventsArrived(Message):
    """A batch of engine events for the main screen."""

    def __init__(self, events: list[Event]) -> None:
        super().__init__()
        self.events = events


class Refresh(Message):
    """Re-read the session and repaint."""


class Bar(Widget):
    """A one-row bar with content on the left and on the right."""

    DEFAULT_CSS = """
    Bar {
        height: 1;
        padding: 0 2;
        background: $bg-alt;
        color: $fg-muted;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.left = Content("")
        self.right = Content("")

    def show(self, left: Content, right: Content) -> None:
        if left.markup != self.left.markup or right.markup != self.right.markup:
            self.left, self.right = left, right
            self.refresh()

    def render(self) -> Content:
        width = self.content_region.width
        right = self.right
        room = max(0, width - right.cell_length - 3)
        left = (
            self.left.truncate(room, ellipsis=True) if self.left.cell_length > room else self.left
        )
        gap = max(1, width - left.cell_length - right.cell_length)
        return Content.assemble(left, " " * gap, right)


class MainScreen(Screen[None]):
    """The session view."""

    DEFAULT_CSS = """
    MainScreen {
        layout: vertical;
    }
    MainScreen > #rollup {
        display: none;
    }
    MainScreen > #body {
        height: 1fr;
    }
    MainScreen #main-column {
        width: 1fr;
        height: 1fr;
    }
    MainScreen #banner {
        height: auto;
        padding: 1 2 1 2;
        background: $bg-alt;
        display: none;
    }
    MainScreen TreePanel {
        width: 50;
        height: 1fr;
    }
    MainScreen #bottom {
        height: auto;
        max-height: 70%;
    }
    MainScreen #newbar {
        height: 1;
        margin: 0 2;
        padding: 0 1;
        background: $bg-raised;
        display: none;
    }
    MainScreen #composer-box {
        height: auto;
        margin: 1 2 1 2;
        padding: 1 1 1 1;
        background: $bg-raised;
        border-left: outer $fg-subtle;
    }
    MainScreen #composer-box:focus-within {
        border-left: outer $accent;
    }
    MainScreen #composer-row {
        height: auto;
    }
    MainScreen #mark {
        width: 2;
        color: $fg-subtle;
    }
    MainScreen #composer-box:focus-within #mark {
        color: $accent;
    }
    MainScreen #hint {
        height: 1;
        margin: 0 0 0 2;
        color: $fg-muted;
    }
    MainScreen.-short #composer-box {
        margin: 1 2 0 2;
        padding: 0 1;
    }
    MainScreen.-short #banner {
        padding: 0 2;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "escape", "Stop or go back", show=False),
        Binding("ctrl+y", "review", "Answer request", priority=True, show=False),
        Binding("ctrl+t", "tree", "Task tree", show=False),
        Binding("ctrl+o", "expand_all", "Open all tool rows", show=False),
        Binding("ctrl+g", "jump", "Jump to latest", show=False),
        Binding("f1", "app.open_view('help')", "Keys", show=False),
        Binding("f2", "app.open_view('settings')", "Settings", show=False),
        Binding("f3", "app.open_view('skills')", "Review skills", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.history = History()
        self.viewed: str | None = None
        self.tree_open = True
        self.crumb = Bar(id="crumb")
        self.rollup = Bar(id="rollup")
        self.banner = Static("", id="banner")
        self.transcript = Transcript(id="transcript")
        self.panel = TreePanel(id="tree")
        self.newbar = Static("", id="newbar")
        self.requests = RequestPanel(id="requests")
        self.completion = CompletionList(id="completion")
        self.composer = Composer(id="composer")
        self.hint = Static("", id="hint")
        self.status = Bar(id="status")
        self.composer.completion = self.completion
        self._lock = asyncio.Lock()
        self._show_tree = True
        self._notice: tuple[Content, float] | None = None
        self._last_request: str | None = None

    @property
    def clankit(self) -> ClankitApp:
        from clankit.tui.app import ClankitApp

        app = self.app
        assert isinstance(app, ClankitApp)
        return app

    @property
    def session(self) -> Session:
        return self.clankit.session

    def compose(self) -> ComposeResult:
        yield self.crumb
        yield self.rollup
        with Horizontal(id="body"):
            with Vertical(id="main-column"):
                yield self.banner
                yield self.transcript
                with Vertical(id="bottom"):
                    yield self.newbar
                    yield self.requests
                    yield self.completion
                    with Vertical(id="composer-box"):
                        with Horizontal(id="composer-row"):
                            yield Static(PROMPT_MARK, id="mark")
                            yield self.composer
                        yield self.hint
            yield self.panel
        yield self.status

    def on_mount(self) -> None:
        self._show_tree = self.tree_open = self.clankit.runtime.config.ui.show_tree
        self.set_interval(TICK, self._tick)
        self.composer.focus()
        self._layout()

    # -- session --------------------------------------------------------------------------------

    async def attach(self, history: History) -> None:
        """Start showing the app's current session, with its past as `history`."""
        self.history = history
        self.viewed = None
        self.panel.task_tree.overrides.clear()
        self.panel.show_peek(None, self.session.view, set(), time.time())
        self.composer.prompt_history = [turn.prompt for turn in self.session.turns]
        await self.transcript.clear()
        await self.refresh_view()

    @on(EventsArrived)
    async def _events(self, message: EventsArrived) -> None:
        for event in message.events:
            self.history.observe(event)
            if isinstance(event, RunStarted):
                self.transcript.expand_all = None
        await self.refresh_view()

    @on(Refresh)
    async def _refresh(self) -> None:
        await self.refresh_view()

    def config_changed(self, config: Config) -> None:
        """Apply a saved configuration: display options here, the theme in the app."""
        if config.ui.show_tree != self._show_tree:
            self._show_tree = self.tree_open = config.ui.show_tree
            self._layout()
        self.post_message(Refresh())

    def say(self, text: str, level: str = "info") -> None:
        """Show a message from the interface itself in the transcript."""
        self.history.add("message", text, level)
        self.post_message(Refresh())

    def flash(self, text: str, seconds: float = 4.0) -> None:
        """Show a short note on the right of the status row."""
        self._notice = (Content.styled(text, "$text"), time.time() + seconds)
        self._paint_status()

    # -- refresh --------------------------------------------------------------------------------

    async def refresh_view(self) -> None:
        async with self._lock:
            await self._refresh_view()

    async def _refresh_view(self) -> None:
        session = self.session
        view = session.view
        now = time.time()
        if self.viewed is not None and self.viewed not in view.nodes:
            self.viewed = None
        requests = pending_requests(view)
        waiting = {request.node_id for request in requests}
        show_thinking = self.clankit.runtime.config.ui.show_thinking
        if self.viewed is None:
            entries = root_entries(view, self.history, show_thinking=show_thinking)
            self.transcript.set_empty(self._empty_state())
            caption = "sent to the agent"
        else:
            entries = node_entries(view, self.viewed, show_thinking=show_thinking)
            self.transcript.set_empty(Content.styled("Nothing here yet.", "$fg-muted"))
            caption = "brief"
        ctx = Context(now=now, expand_all=self.transcript.expand_all, message_caption=caption)
        await self.transcript.sync(entries, ctx)
        if self.panel.display:
            await self.panel.sync(
                view, waiting, now, viewed=self.viewed, depth_limit=self.depth_limit()
            )
        await self.requests.show(requests, view)
        self._route_request_focus()
        self._paint_crumb()
        self._paint_banner(waiting, now)
        self._paint_rollup(waiting)
        self._paint_hint()
        self._paint_status()

    async def _tick(self) -> None:
        if not self.session.busy and self._notice is None:
            return
        if self._notice is not None and time.time() > self._notice[1]:
            self._notice = None
        await self.refresh_view()

    def depth_limit(self) -> int | None:
        """The method's depth limit, when it has one (the task tree's ``max-depth``)."""
        try:
            method = get_orchestrator(self.session.method)
        except KeyError:
            return None
        spec = next((s for s in method.settings if s.key == "max-depth"), None)
        if spec is None:
            return None
        raw = self.clankit.runtime.config.orchestration.option(method.name, spec.key)
        try:
            return int(spec.coerce(raw))
        except (ConfigError, TypeError, ValueError):
            return int(spec.default)

    def _empty_state(self) -> Content:
        runtime = self.clankit.runtime
        session = self.session
        try:
            needs_model = get_orchestrator(session.method).needs_model
        except KeyError:
            needs_model = True
        no_model = session.model_override is None and runtime.providers.default_model() is None
        if needs_model and no_model:
            return Content.assemble(
                ("Welcome to Clankit.", "bold $text"),
                "\n\n",
                ("No model is set up yet. Press ", "$text"),
                ("f2", "bold $text"),
                (
                    " to open the settings at Providers and add an API key. Several providers "
                    "have free tiers, and local servers such as Ollama need no key at all.",
                    "$text",
                ),
                "\n\n",
                "To look around without a model, type ",
                ("/method demo", "$text"),
                " or start Clankit with ",
                ("clankit --demo", "$text"),
                ".",
            )
        try:
            method = get_orchestrator(session.method)
            method_text = Content.assemble(
                (method.title, "$text"), "   ", (method.summary, "$fg-muted")
            )
        except KeyError:
            method_text = Content(session.method)
        return Content.assemble(
            ("Clankit", "bold $text"),
            "\n",
            "Describe a task and press enter. Type / for commands, f1 for keys.",
            "\n\n",
            ("workspace  ", "$fg-muted"),
            (str(session.workspace), "$text"),
            ("\nmodel      ", "$fg-muted"),
            (self.clankit.model_label(), "$text"),
            ("\nmethod     ", "$fg-muted"),
            method_text,
        )

    # -- painting -------------------------------------------------------------------------------

    def _paint_crumb(self) -> None:
        view = self.session.view
        width = self.crumb.content_region.width or self.size.width
        if self.viewed is None:
            left = Content.assemble(
                ("conversation", "$text"),
                "   ",
                (ellipsize_left(str(self.session.workspace), max(10, width // 2)), "$fg-muted"),
            )
        else:
            path = ellipsize_left(node_path(view, self.viewed), max(10, width - 30))
            head, _, last = path.rpartition(" / ")
            left = Content.assemble(
                (f"{head} / " if head else "", "$fg-muted"), (last, "bold $text")
            )
        try:
            method = get_orchestrator(self.session.method).title
        except KeyError:
            method = self.session.method
        self.crumb.show(left, Content.styled(method, "$fg-muted"))

    def _paint_banner(self, waiting: set[str], now: float) -> None:
        node = self.session.view.nodes.get(self.viewed or "")
        self.banner.display = node is not None
        if node is None:
            return
        state = node_state(node, node.info.id in waiting)
        facts = "   ".join(
            part
            for part in (
                STATE_WORDS[state],
                node.info.model,
                f"{human_tokens(node.tokens)} tokens",
                elapsed_of(node, now),
            )
            if part and part != "-"
        )
        lines = [
            Content.assemble(
                (STATE_GLYPHS[state], STATE_STYLES[state]), " ", (node.info.title, "bold")
            ),
            Content.styled(facts, "$fg-muted"),
        ]
        if node.info.summary:
            lines.append(Content(node.info.summary))
        self.banner.update(Content("\n").join(lines))

    def _paint_rollup(self, waiting: set[str]) -> None:
        view = self.session.view
        show = not self.panel.display and self.size.width >= NARROW and bool(view.nodes)
        self.rollup.display = show
        if show:
            self.rollup.show(
                counts_content(view, waiting),
                Content.assemble(("ctrl+t", "bold $text"), (" tree", "$fg-muted")),
            )

    def _paint_hint(self) -> None:
        view = self.session.view
        parts: list[tuple[str, str]]
        if view.questions:
            placeholder = "Type an answer to the question above"
            parts = [("enter", "answer"), ("ctrl+y", "choose an option")]
        elif self.session.busy:
            placeholder = "Message the agent; it reads this at its next step"
            parts = [("enter", "send"), ("esc", "stop the run"), ("ctrl+j", "new line")]
        else:
            placeholder = "Describe a task, or type / for commands"
            parts = [("enter", "send"), ("ctrl+j", "new line"), ("/", "commands")]
        if self.viewed is not None and not self.session.busy:
            parts.append(("esc", "back"))
        pieces: list[Content | str | tuple[str, str]] = []
        for key, word in parts:
            pieces.extend([(key, "bold $text"), f" {word}   "])
        self.hint.update(Content.assemble(*pieces))
        if self.composer.placeholder != placeholder:
            self.composer.placeholder = placeholder

    def _paint_status(self) -> None:
        session = self.session
        view = session.view
        items: list[Content] = []
        if session.busy:
            run = self.history.current_run()
            took = human_elapsed(time.time() - run.ts) if run else ""
            items.append(
                Content.assemble(
                    (STATE_GLYPHS["running"], STATE_STYLES["running"]), f" working {took}"
                )
            )
        items.append(Content.styled(self.clankit.model_label(), "$text"))
        total = sum(view.total_usage.values()) if view.total_usage else 0
        tokens = sum(
            view.total_usage.get(k, 0)
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        )
        if total:
            items.append(Content(f"{human_tokens(tokens)} tok"))
        cost = session.cost() if tokens else None
        if cost is not None:
            items.append(Content(f"${cost:.2f}"))
        context = session.context_usage()
        if context is not None and context[1]:
            items.append(Content(f"ctx {context[0] * 100 // context[1]}%"))
        requests = pending_requests(view)
        if requests:
            permissions = sum(isinstance(r, PermissionRequested) for r in requests)
            questions = sum(isinstance(r, QuestionAsked) for r in requests)
            words = []
            if permissions:
                words.append(f"{permissions} permission" + ("s" if permissions > 1 else ""))
            if questions:
                words.append(f"{questions} question" + ("s" if questions > 1 else ""))
            pending = Content.assemble(("! ", "bold $warning"), (", ".join(words), "$text"))
            if not self.requests.answering:
                pending = Content.assemble(pending, "  ", ("ctrl+y", "bold $text"), " review")
            items.append(pending)
        if self.size.width < NARROW and view.nodes:
            counts = tree_counts(view, {r.node_id for r in requests})
            items.append(Content(f"tasks {len(view.nodes)} done {counts['done']}"))
        items.append(Content(session.method))
        left = Content("   ").join(items)
        if self._notice is not None and time.time() <= self._notice[1]:
            right = self._notice[0]
        else:
            right = Content.assemble(
                (self.app.theme, "$fg-muted"), "   ", ("f1", "bold $text"), " keys"
            )
        self.status.show(left.stylize_before("$fg-muted"), right)

    # -- layout ---------------------------------------------------------------------------------

    def on_resize(self) -> None:
        self._layout()

    def _layout(self) -> None:
        width = self.size.width
        self.set_class(self.size.height < SHORT, "-short")
        wide = width >= WIDE and self.tree_open
        self.panel.display = wide
        if wide:
            self.panel.styles.width = max(PANEL_MIN, min(PANEL_MAX, width - PROSE_COLUMNS))
        self.post_message(Refresh())

    # -- requests -------------------------------------------------------------------------------

    def _route_request_focus(self) -> None:
        """Hand the panel focus when a new request arrives, unless the user is typing."""
        current = self.requests.request_id
        is_new = current is not None and current != self._last_request
        self._last_request = current
        if current is None:
            if self.requests.has_focus:
                self.composer.focus()
            return
        if is_new and not self.composer.text.strip() and self.focused in (self.composer, None):
            self.requests.focus()
        self.requests.refresh_keys()

    @on(RequestPanel.Permitted)
    def _permit(self, message: RequestPanel.Permitted) -> None:
        self.session.resolve_permission(message.request_id, message.choice)
        self._last_request = None

    @on(RequestPanel.Answered)
    def _answer(self, message: RequestPanel.Answered) -> None:
        self.session.answer_question(message.request_id, message.answer)
        self._last_request = None

    @on(RequestPanel.Typed)
    def _typed(self, message: RequestPanel.Typed) -> None:
        self.composer.focus()
        self.composer.insert(message.character)

    @on(RequestPanel.Released)
    def _released(self) -> None:
        self.composer.focus()

    @on(RequestPanel.ViewNode)
    async def _view_request_node(self, message: RequestPanel.ViewNode) -> None:
        await self.view_node(message.node_id)

    def action_review(self) -> None:
        if self.requests.request is not None:
            self.requests.focus()
        else:
            self.flash("Nothing is waiting for an answer.")

    def on_descendant_focus(self) -> None:
        self._paint_status()

    def on_descendant_blur(self) -> None:
        self._paint_status()

    # -- composer -------------------------------------------------------------------------------

    @on(TextArea.Changed, "#composer")
    def _composer_changed(self) -> None:
        self.completion.show_for(self.composer.text)
        self._paint_status()

    @on(Composer.Submitted)
    async def _submitted(self, message: Composer.Submitted) -> None:
        self.completion.hide()
        await self.submit(message.text)

    async def submit(self, text: str) -> None:
        """Route a line from the composer: a command, an answer, steering or a new prompt."""
        session = self.session
        parsed = commands.parse(text)
        if parsed is not None:
            name, argument = parsed
            self.clankit.run_command(name, argument)
            return
        question = next(iter(session.view.questions.values()), None)
        if question is not None:
            session.answer_question(question.request_id, text.strip())
            return
        if session.busy:
            if session.steer(text):
                self.history.add("steer", text)
            else:
                self.say("The agent cannot take a message right now; try again in a moment.")
            await self.refresh_view()
            return
        await self.show_conversation()
        self.clankit.start_run(text)

    # -- navigation -----------------------------------------------------------------------------

    async def view_node(self, node_id: str | None) -> None:
        """Show `node_id`'s transcript (None, or a root node, shows the conversation)."""
        node = self.session.view.nodes.get(node_id or "")
        target = node_id if node is not None and node.info.parent_id is not None else None
        if target == self.viewed:
            return
        self.viewed = target
        await self.transcript.clear()
        await self.refresh_view()

    async def show_conversation(self) -> None:
        await self.view_node(None)

    @on(TaskTree.Focused)
    async def _tree_focused(self, message: TaskTree.Focused) -> None:
        await self.view_node(message.node_id)

    @on(TaskTree.Peeked)
    def _tree_peeked(self, message: TaskTree.Peeked) -> None:
        view = self.session.view
        waiting = {r.node_id for r in pending_requests(view)}
        self.panel.show_peek(message.node_id, view, waiting, time.time())

    @on(Transcript.UnseenChanged)
    def _unseen(self, message: Transcript.UnseenChanged) -> None:
        self.newbar.display = message.count > 0
        self.newbar.update(
            Content.assemble(
                (f"{message.count} new", "$text"),
                "   ",
                ("ctrl+g", "bold $text"),
                (" jump to latest", "$fg-muted"),
            )
        )

    async def action_escape(self) -> None:
        if self.completion.is_open:
            self.completion.hide()
        elif self.panel.peeking is not None:
            self.panel.show_peek(None, self.session.view, set(), time.time())
        elif self.session.busy:
            if self.session.cancel():
                self.flash("Stopping the run.")
        elif self.viewed is not None:
            await self.show_conversation()
            self.composer.focus()

    def action_tree(self) -> None:
        if self.size.width >= WIDE:
            self.tree_open = not self.tree_open
            self._layout()
        else:
            self.clankit.open_tree()

    def action_expand_all(self) -> None:
        self.transcript.set_expand_all(not bool(self.transcript.expand_all))
        self.post_message(Refresh())

    def action_jump(self) -> None:
        self.transcript.jump_to_end()

"""The live task tree: an indented outline with fixed right-hand columns.

Row grammar (every column has a fixed width, so live updates never reflow the panel)::

    <bar><indent><state> <caret> <label (+N)>   <model> <tokens> <elapsed>

* The state glyph carries the status by shape; its colour only repeats it.
* The model column is muted when a node runs on its parent's model and bright when it does
  not, so cross-model delegation stands out without another colour.
* Finished subtrees collapse into ``(+N)`` on their own, while every path that leads to a
  running, failed or waiting-for-input node stays open; whatever the user opens or closes
  by hand stays that way.
* Indentation is two cells per level for the first levels and one cell after that, and
  never so deep that a label gets fewer than ten cells: at the default depth limit of ten a
  label would otherwise vanish in a 50-column panel.

Keys follow the WAI-ARIA tree pattern: up and down move, right opens or steps into the
first child, left closes or steps out to the parent, home and end jump, ``*`` opens every
sibling. Enter focuses a node (the transcript shows its work) and space peeks at it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from clankit.agent.events import TERMINAL_STATUSES
from clankit.agent.state import Item, NodeView, SessionView
from clankit.tui.format import (
    ACTIVE_STATES,
    CARET_CLOSED,
    CARET_OPEN,
    STATE_GLYPHS,
    STATE_STYLES,
    STATE_WORDS,
    ellipsize,
    first_line,
    human_elapsed,
    human_tokens,
    model_alias,
    node_state,
    pad,
)

MIN_LABEL = 10
_MODEL_COLUMN = 15
_NARROW_MODEL_COLUMN = 10
#: Rows at least this wide get the full model column.
_WIDE_ROW = 56
_TOKENS_COLUMN = 5
_ELAPSED_COLUMN = 5


@dataclass(frozen=True, slots=True)
class Row:
    """Everything one tree row shows; equal rows need no repaint."""

    node_id: str
    depth: int
    title: str
    state: str
    model: str
    model_inherited: bool
    tokens: int
    elapsed: str
    caret: str
    hidden: int


def indent_for(depth: int, label_room: int) -> int:
    """Cells of indentation for a row at `depth`, leaving at least `MIN_LABEL` for the label."""
    levels = min(depth, 4)
    cells = 2 * levels + max(0, depth - levels)
    return max(0, min(cells, label_room - MIN_LABEL))


def render_row(row: Row, width: int, *, cursor: bool = False) -> Content:
    """A row fitted to `width` cells. On the cursor row muted text is brightened, because
    the muted tone does not reach a readable contrast on the selection surface."""
    muted = "$text" if cursor else "$fg-muted"
    show_model = width >= 40
    model_width = _MODEL_COLUMN if width >= _WIDE_ROW else _NARROW_MODEL_COLUMN
    meta_width = (1 + model_width if show_model else 0) + 1 + _TOKENS_COLUMN + 1 + _ELAPSED_COLUMN
    label_room = width - 4 - meta_width
    indent = indent_for(row.depth, label_room)
    suffix = f" (+{row.hidden})" if row.hidden else ""
    label_width = max(1, label_room - indent)
    title = ellipsize(row.title, max(1, label_width - len(suffix)))
    label_style = muted if row.state in ("done", "cancelled") else "$text"
    gap = max(0, label_width - len(title) - len(suffix))
    parts: list[Content | str | tuple[str, str]] = [
        " " * indent,
        (STATE_GLYPHS[row.state], STATE_STYLES[row.state]),
        " ",
        (row.caret or " ", "$fg-subtle" if not cursor else "$text"),
        " ",
        (title, label_style),
        (suffix, muted),
        " " * gap,
    ]
    if show_model:
        model_style = muted if row.model_inherited else "$text"
        parts.extend([" ", (pad(row.model or "-", model_width, right=True), model_style)])
    parts.extend(
        [
            " ",
            (pad(human_tokens(row.tokens), _TOKENS_COLUMN, right=True), muted),
            " ",
            (pad(row.elapsed, _ELAPSED_COLUMN, right=True), muted),
        ]
    )
    return Content.assemble(*parts)


def elapsed_of(node: NodeView, now: float, first_seen: float = 0.0) -> str:
    """How long a node has been (or was) at work; ``-`` before it starts."""
    if node.info.status == "pending":
        return "-"
    start = node.started or first_seen
    if not start:
        return "-"
    end = node.finished if node.info.status in TERMINAL_STATUSES and node.finished else now
    return human_elapsed(max(0.0, end - start))


def descendants(view: SessionView, node: NodeView) -> Iterator[NodeView]:
    for child_id in node.children:
        child = view.nodes.get(child_id)
        if child is not None:
            yield child
            yield from descendants(view, child)


class TreeRow(Widget):
    """One row of the tree."""

    DEFAULT_CSS = """
    TreeRow {
        height: 1;
        border-left: blank;
        padding: 0 1 0 0;
    }
    TreeRow:hover {
        background: $line;
    }
    TreeRow.-viewed {
        border-left: outer $fg-subtle;
    }
    TaskTree:focus > TreeRow.-cursor {
        background: $selection;
        border-left: outer $accent;
    }
    """

    def __init__(self, row: Row) -> None:
        super().__init__()
        self.row = row
        self.node_id = row.node_id

    def render(self) -> Content:
        focused = self.has_class("-cursor") and self.parent is not None and self.parent.has_focus
        return render_row(self.row, self.content_region.width, cursor=focused)

    def update(self, row: Row) -> None:
        if row != self.row:
            self.row = row
            self.refresh()

    def on_click(self) -> None:
        tree = self.parent
        if isinstance(tree, TaskTree):
            tree.move_cursor(self.node_id)


class TaskTree(VerticalScroll, can_focus=True):
    """The tree of nodes, kept in step with a `SessionView` by `sync`."""

    DEFAULT_CSS = """
    TaskTree {
        height: 1fr;
        background: $bg-alt;
        scrollbar-size-vertical: 0;
        scrollbar-background: $bg-alt;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "move(-10)", show=False),
        Binding("pagedown", "move(10)", show=False),
        Binding("home", "edge(0)", show=False),
        Binding("end", "edge(1)", show=False),
        Binding("right", "open", "Open", show=False),
        Binding("left", "close", "Close", show=False),
        Binding("asterisk", "open_siblings", "Open siblings", show=False),
        Binding("enter", "focus_node", "Focus node", show=False),
        Binding("space", "peek", "Peek", show=False),
    ]

    class Focused(Message):
        """The user asked to see a node's work in the transcript."""

        def __init__(self, node_id: str) -> None:
            super().__init__()
            self.node_id = node_id

    class Peeked(Message):
        """The user asked for a quick look at a node (None closes the peek)."""

        def __init__(self, node_id: str | None) -> None:
            super().__init__()
            self.node_id = node_id

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.overrides: dict[str, bool] = {}
        self.cursor: str | None = None
        self.viewed: str | None = None
        self._view: SessionView | None = None
        self._waiting: set[str] = set()
        self._now = 0.0
        self._rows: list[Row] = []
        self._first_seen: dict[str, float] = {}

    @property
    def rows(self) -> list[Row]:
        return list(self._rows)

    # -- building -------------------------------------------------------------------------------

    def expanded(self, view: SessionView, node: NodeView, waiting: set[str]) -> bool:
        if node.info.id in self.overrides:
            return self.overrides[node.info.id]
        if node.info.depth == 0:
            return True
        return any(
            d.info.status not in TERMINAL_STATUSES
            or d.info.status == "failed"
            or d.info.id in waiting
            or d.info.id == self.viewed
            for d in descendants(view, node)
        )

    def build(self, view: SessionView, waiting: set[str], now: float) -> list[Row]:
        rows: list[Row] = []

        def visit(node: NodeView, parent_model: str | None) -> None:
            info = node.info
            if info.status not in TERMINAL_STATUSES and info.status != "pending":
                self._first_seen.setdefault(info.id, now)
            has_children = bool(node.children)
            opened = has_children and self.expanded(view, node, waiting)
            hidden = 0 if opened or not has_children else sum(1 for _ in descendants(view, node))
            tokens = node.tokens
            if has_children and not opened:
                tokens += sum(d.tokens for d in descendants(view, node))
            rows.append(
                Row(
                    node_id=info.id,
                    depth=info.depth,
                    title=info.title,
                    state=_state(node, waiting),
                    model=model_alias(info.model),
                    model_inherited=parent_model is not None and info.model == parent_model,
                    tokens=tokens,
                    elapsed=elapsed_of(node, now, self._first_seen.get(info.id, 0.0)),
                    caret=(CARET_OPEN if opened else CARET_CLOSED) if has_children else "",
                    hidden=hidden,
                )
            )
            if opened:
                for child_id in node.children:
                    child = view.nodes.get(child_id)
                    if child is not None:
                        visit(child, info.model)

        for root_id in view.roots:
            root = view.nodes.get(root_id)
            if root is not None:
                visit(root, None)
        return rows

    async def sync(
        self, view: SessionView, waiting: set[str], now: float, viewed: str | None = None
    ) -> None:
        """Rebuild the rows from `view`; only rows that changed repaint."""
        self._view = view
        self._waiting = waiting
        self._now = now
        self.viewed = viewed
        rows = self.build(view, waiting, now)
        self._rows = rows
        widgets = {w.node_id: w for w in self.query_children(TreeRow)}
        wanted = [row.node_id for row in rows]
        stale = [w for node_id, w in widgets.items() if node_id not in set(wanted)]
        if stale:
            await self.remove_children(stale)
        current = [w.node_id for w in self.query_children(TreeRow)]
        if wanted[: len(current)] == current:
            fresh = [TreeRow(row) for row in rows[len(current) :]]
            if fresh:
                await self.mount_all(fresh)
        else:
            previous: TreeRow | None = None
            for row in rows:
                widget = widgets.get(row.node_id)
                if widget is None:
                    widget = TreeRow(row)
                    if previous is None:
                        await self.mount(widget, before=0)
                    else:
                        await self.mount(widget, after=previous)
                elif previous is None and self.children.index(widget) != 0:
                    self.move_child(widget, before=0)
                elif previous is not None and (
                    self.children.index(widget) != self.children.index(previous) + 1
                ):
                    self.move_child(widget, after=previous)
                previous = widget
        for widget, row in zip(self.query_children(TreeRow), rows, strict=False):
            widget.update(row)
            widget.set_class(row.node_id == viewed and viewed is not None, "-viewed")
        if self.cursor not in wanted:
            self.cursor = wanted[0] if wanted else None
        self._paint_cursor()

    def _resync(self) -> None:
        if self._view is not None:
            self.call_later(self.sync, self._view, self._waiting, self._now, self.viewed)

    # -- cursor ---------------------------------------------------------------------------------

    def _paint_cursor(self) -> None:
        for widget in self.query_children(TreeRow):
            is_cursor = widget.node_id == self.cursor
            if widget.has_class("-cursor") != is_cursor:
                widget.set_class(is_cursor, "-cursor")
                widget.refresh()

    def move_cursor(self, node_id: str) -> None:
        self.cursor = node_id
        self._paint_cursor()
        widget = next((w for w in self.query_children(TreeRow) if w.node_id == node_id), None)
        if widget is not None:
            self.scroll_to_widget(widget, animate=False)

    def on_focus(self) -> None:
        self._paint_cursor()
        for widget in self.query_children(TreeRow):
            widget.refresh()

    def on_blur(self) -> None:
        for widget in self.query_children(TreeRow):
            widget.refresh()

    def _index(self) -> int:
        ids = [row.node_id for row in self._rows]
        return ids.index(self.cursor) if self.cursor in ids else 0

    def action_move(self, step: int) -> None:
        if self._rows:
            index = max(0, min(len(self._rows) - 1, self._index() + step))
            self.move_cursor(self._rows[index].node_id)

    def action_edge(self, end: int) -> None:
        if self._rows:
            self.move_cursor(self._rows[-1 if end else 0].node_id)

    def _cursor_node(self) -> NodeView | None:
        if self._view is None or self.cursor is None:
            return None
        return self._view.nodes.get(self.cursor)

    def action_open(self) -> None:
        node = self._cursor_node()
        if node is None or not node.children:
            return
        row = self._rows[self._index()]
        if row.caret == CARET_CLOSED:
            self.overrides[node.info.id] = True
            self._resync()
        else:
            self.action_move(1)

    def action_close(self) -> None:
        node = self._cursor_node()
        if node is None:
            return
        row = self._rows[self._index()]
        if row.caret == CARET_OPEN:
            self.overrides[node.info.id] = False
            self._resync()
        elif node.info.parent_id:
            self.move_cursor(node.info.parent_id)

    def action_open_siblings(self) -> None:
        node = self._cursor_node()
        if node is None or self._view is None:
            return
        parent = self._view.nodes.get(node.info.parent_id or "")
        siblings = parent.children if parent else self._view.roots
        for sibling_id in siblings:
            sibling = self._view.nodes.get(sibling_id)
            if sibling is not None and sibling.children:
                self.overrides[sibling_id] = True
        self._resync()

    def action_focus_node(self) -> None:
        if self.cursor is not None:
            self.post_message(self.Focused(self.cursor))

    def action_peek(self) -> None:
        if self.cursor is not None:
            self.post_message(self.Peeked(self.cursor))


def _state(node: NodeView, waiting: set[str]) -> str:
    return node_state(node, node.info.id in waiting)


# -- header, peek and panel -------------------------------------------------------------------


def tree_counts(view: SessionView, waiting: set[str]) -> dict[str, int]:
    """Nodes per display group: done, active, failed, waiting for input, pending."""
    counts = {"done": 0, "active": 0, "failed": 0, "needs-input": 0, "pending": 0}
    for node in view.nodes.values():
        state = _state(node, waiting)
        if state == "needs-input":
            counts["needs-input"] += 1
        elif state in ACTIVE_STATES:
            counts["active"] += 1
        elif state in counts:
            counts[state] += 1
    return counts


def counts_content(view: SessionView, waiting: set[str]) -> Content:
    """``tasks 8   ✓ 6  ● 1  ✗ 0`` with the optional groups only when they are not empty."""
    if not view.nodes:
        return Content.styled("tasks", "$fg-muted")
    counts = tree_counts(view, waiting)
    parts: list[Content | str | tuple[str, str]] = [
        ("tasks ", "$fg-muted"),
        (str(len(view.nodes)), "$text"),
        " ",
    ]
    groups = [("done", "done"), ("active", "running"), ("failed", "failed")]
    if counts["needs-input"]:
        groups.append(("needs-input", "needs-input"))
    if counts["pending"]:
        groups.append(("pending", "pending"))
    for key, state in groups:
        parts.extend(
            ["  ", (STATE_GLYPHS[state], STATE_STYLES[state]), " ", (str(counts[key]), "$text")]
        )
    return Content.assemble(*parts)


class TreeHeader(Widget):
    """The panel's first row: node counts on the left, depth against its limit on the right."""

    DEFAULT_CSS = """
    TreeHeader {
        height: 1;
        padding: 0 1 0 1;
        background: $bg-alt;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts = Content("")
        self.depth = ""

    def show(self, counts: Content, depth: str) -> None:
        if counts.plain != self.counts.plain or depth != self.depth:
            self.counts, self.depth = counts, depth
            self.refresh()

    def render(self) -> Content:
        width = self.content_region.width
        right = Content.styled(self.depth, "$fg-muted")
        gap = max(1, width - self.counts.cell_length - right.cell_length)
        return Content.assemble(self.counts, " " * gap, right)


def peek_content(view: SessionView, node_id: str, waiting: set[str], now: float) -> Content:
    """A quick look at a node: what it is, how it is doing, and its latest activity."""
    node = view.nodes.get(node_id)
    if node is None:
        return Content("")
    info = node.info
    state = _state(node, waiting)
    lines: list[Content] = [
        Content.assemble((STATE_GLYPHS[state], STATE_STYLES[state]), " ", (info.title, "bold")),
        Content.styled(
            "   ".join(
                part
                for part in (
                    STATE_WORDS[state],
                    info.model,
                    f"{human_tokens(node.tokens)} tokens",
                    elapsed_of(node, now),
                )
                if part and part != "-"
            ),
            "$fg-muted",
        ),
    ]
    if info.summary:
        lines.append(Content(info.summary))
    elif info.brief:
        lines.append(Content(ellipsize(info.brief, 400)))
    recent = [item for item in node.items if item.kind in ("tool", "assistant", "notice")][-4:]
    if recent:
        lines.append(Content(""))
        lines.extend(_activity(item) for item in recent)
    lines.append(
        Content.assemble(
            ("enter", "bold"),
            (" focus   ", "$fg-muted"),
            ("space", "bold"),
            (" close", "$fg-muted"),
        )
    )
    return Content("\n").join(lines)


def _activity(item: Item) -> Content:
    if item.kind == "tool":
        state = "running" if not item.done else "failed" if item.is_error else "done"
        return Content.assemble(
            (STATE_GLYPHS[state], STATE_STYLES[state]),
            " ",
            Content.styled(item.summary, "$fg-muted"),
        )
    return Content.styled(ellipsize(first_line(item.text), 200), "$fg-muted")


class TreePanel(Vertical):
    """Header, tree and peek, on the chrome surface."""

    DEFAULT_CSS = """
    TreePanel {
        background: $bg-alt;
        padding: 1 0 0 0;
    }
    TreePanel > TreeHeader {
        margin: 0 0 1 0;
    }
    TreePanel > #peek {
        height: auto;
        max-height: 14;
        background: $bg-raised;
        padding: 1 2;
        display: none;
    }
    TreePanel > #empty {
        color: $fg-muted;
        padding: 0 2;
        height: auto;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.header = TreeHeader()
        self.task_tree = TaskTree()
        self.peek = Static("", id="peek")
        self.empty = Static(
            "Subtasks appear here when an agent splits its work.", id="empty", markup=False
        )
        self.peeking: str | None = None

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.empty
        yield self.task_tree
        yield self.peek

    async def sync(
        self,
        view: SessionView,
        waiting: set[str],
        now: float,
        *,
        viewed: str | None,
        depth_limit: int | None,
    ) -> None:
        depth = max((n.info.depth for n in view.nodes.values()), default=0)
        depth_text = f"depth {depth}/{depth_limit}" if depth_limit is not None else ""
        self.header.show(counts_content(view, waiting), depth_text)
        self.empty.display = not view.nodes
        await self.task_tree.sync(view, waiting, now, viewed)
        if self.peeking is not None:
            self.peek.update(peek_content(view, self.peeking, waiting, now))

    def show_peek(
        self, node_id: str | None, view: SessionView, waiting: set[str], now: float
    ) -> None:
        """Open the peek for `node_id`, or close it when it is already open for that node."""
        if node_id is None or node_id == self.peeking:
            self.peeking = None
            self.peek.display = False
            return
        self.peeking = node_id
        self.peek.update(peek_content(view, node_id, waiting, now))
        self.peek.display = True

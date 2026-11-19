"""The transcript: the conversation, or one node's work, as a column of blocks.

`Transcript.sync` reconciles its blocks against a list of `Entry` objects by key, so a
refresh touches only what changed: a streaming answer appends its new text, a finished tool
call repaints one row, a new entry mounts one block. Everything that came from a model or a
tool is untrusted text and is rendered as `Content` (never parsed as markup).

Layout rules (from the design notes): a two-column gutter, one blank row between blocks,
none between consecutive tool rows (they read as a group), user turns on the raised surface
with an accent bar, notices with a bar in their level's colour, tool calls as one aligned
row each that opens in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.content import Content
from textual.geometry import Size
from textual.highlight import highlight
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownFence, MarkdownStream

from clankit.agent.state import Item
from clankit.themes.textual import CodeHighlight
from clankit.tui.entries import Entry
from clankit.tui.format import (
    CARET_CLOSED,
    CARET_OPEN,
    STATE_GLYPHS,
    STATE_STYLES,
    ellipsize,
    first_line,
    human_duration,
    human_elapsed,
    pad,
    plural,
    wrap_cells,
)

#: Lines of tool output shown when a tool row is opened; the rest is summarised.
OUTPUT_PREVIEW_LINES = 40

_TOOL_COLUMN = 8
_RESULT_COLUMN = 32


@dataclass(frozen=True, slots=True)
class Context:
    """Settings that shape every block during one refresh."""

    now: float
    #: True or False after the user opened or closed every tool row at once, else None.
    expand_all: bool | None = None
    #: How a message handed to an agent (not the user's prompt) is captioned.
    message_caption: str = "sent to the agent"


# -- Markdown ---------------------------------------------------------------------------------


class CodeFence(MarkdownFence):
    """Code blocks in the theme's own syntax colours."""

    @classmethod
    def highlight(cls, code: str, language: str, ansi: bool = False, dark: bool = False) -> Content:
        return highlight(code, language=language or None, theme=CodeHighlight)


class Prose(Markdown):
    """Model prose: plain bullets, themed code, no centred headings or rules."""

    BULLETS: ClassVar[list[str]] = ["• ", "◦ ", "‣ ", "• ", "◦ "]
    BLOCKS: ClassVar[dict] = {**Markdown.BLOCKS, "fence": CodeFence, "code_block": CodeFence}


def clean_markdown(text: str) -> str:
    """Model Markdown made safe to render here.

    Textual draws an image as a picture emoji; images cannot be shown in a terminal anyway, so
    they become plain links to their source.
    """
    return text.replace("![", "[")


# -- blocks -----------------------------------------------------------------------------------


class Block(Widget):
    """A transcript block bound to one entry key.

    Subclasses compute a `signature` of what they show and repaint in `apply` only when it
    changes, which keeps a refresh of a long transcript cheap.
    """

    DEFAULT_CSS = """
    Block {
        height: auto;
        margin: 1 0 0 0;
    }
    Block.-grouped {
        margin: 0;
    }
    """

    expandable: ClassVar[bool] = False

    def __init__(self, entry: Entry) -> None:
        super().__init__(classes=f"block -{entry.kind}")
        self.key = entry.key
        self.kind = entry.kind
        self._signature: object = None

    async def sync(self, entry: Entry, ctx: Context) -> None:
        signature = self.signature(entry, ctx)
        if signature != self._signature:
            self._signature = signature
            await self.apply(entry, ctx)

    def signature(self, entry: Entry, ctx: Context) -> object:
        raise NotImplementedError

    async def apply(self, entry: Entry, ctx: Context) -> None:
        raise NotImplementedError

    def toggle(self) -> None:
        """Open or close the block (expandable blocks only)."""

    def context(self) -> Context:
        """A refresh context for a change made from inside the block (a click, a key)."""
        parent = self.parent
        expand_all = parent.expand_all if isinstance(parent, Transcript) else None
        return Context(now=time.time(), expand_all=expand_all)


class TextBlock(Block):
    """A block that shows one piece of `Content`."""

    def __init__(self, entry: Entry) -> None:
        super().__init__(entry)
        self._content = Content("")

    def render(self) -> Content:
        return self._content

    def show(self, content: Content) -> None:
        self._content = content
        self.refresh(layout=True)


class PromptBlock(TextBlock):
    """The user's prompt, or a message the user sent while a run was working."""

    DEFAULT_CSS = """
    PromptBlock {
        background: $bg-raised;
        border-left: outer $accent;
        padding: 0 1;
    }
    """

    def signature(self, entry: Entry, ctx: Context) -> object:
        return entry.run.prompt if entry.run else entry.local.text if entry.local else ""

    async def apply(self, entry: Entry, ctx: Context) -> None:
        text = entry.run.prompt if entry.run else entry.local.text if entry.local else ""
        content = Content(text.strip("\n"))
        if entry.kind == "steer":
            content = Content.assemble(
                content, "\n", ("sent while the agent was working", "$fg-muted")
            )
        self.show(content)


class MessageBlock(TextBlock):
    """Text handed to an agent that is not the user's prompt: a brief, review findings."""

    DEFAULT_CSS = """
    MessageBlock {
        background: $bg-raised;
        border-left: outer $fg-subtle;
        padding: 0 1;
    }
    """

    def signature(self, entry: Entry, ctx: Context) -> object:
        return (entry.item.text if entry.item else "", ctx.message_caption)

    async def apply(self, entry: Entry, ctx: Context) -> None:
        text = entry.item.text.strip("\n") if entry.item else ""
        self.show(Content.assemble((ctx.message_caption, "$fg-muted"), "\n", Content(text)))


class AssistantBlock(Block):
    """Model prose, streamed into Markdown as it arrives."""

    DEFAULT_CSS = """
    AssistantBlock {
        padding: 0 0 0 2;
    }
    """

    def __init__(self, entry: Entry) -> None:
        super().__init__(entry)
        #: The Markdown handed to the widget so far; a stream only ever appends to it.
        self._written = clean_markdown(entry.item.text if entry.item else "")
        self._stream: MarkdownStream | None = None
        self.prose = Prose(self._written)

    def compose(self) -> ComposeResult:
        yield self.prose

    def signature(self, entry: Entry, ctx: Context) -> object:
        item = entry.item
        return (len(item.text), item.done) if item else None

    async def apply(self, entry: Entry, ctx: Context) -> None:
        item = entry.item
        if item is None:
            return
        text = clean_markdown(item.text)
        if not item.done and text.startswith(self._written):
            suffix = text[len(self._written) :]
            if suffix:
                if self._stream is None:
                    self._stream = Markdown.get_stream(self.prose)
                await self._stream.write(suffix)
                self._written = text
            return
        await self._stop_stream()
        if text != self.prose.source:
            await self.prose.update(text)
        self._written = text

    async def _stop_stream(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await stream.stop()

    async def on_unmount(self) -> None:
        await self._stop_stream()


class ThinkingBlock(Block):
    """Model reasoning: muted, open while it streams, one line once it is done."""

    DEFAULT_CSS = """
    ThinkingBlock {
        color: $fg-muted;
        padding: 0 0 0 2;
    }
    ThinkingBlock:hover {
        background: $line;
    }
    """

    expandable = True

    def __init__(self, entry: Entry) -> None:
        super().__init__(entry)
        self.opened: bool | None = None
        self._entry = entry

    def signature(self, entry: Entry, ctx: Context) -> object:
        item = entry.item
        return (len(item.text), item.done, self.opened) if item else None

    async def apply(self, entry: Entry, ctx: Context) -> None:
        self._entry = entry
        self.refresh(layout=True)

    def render(self) -> Content:
        item = self._entry.item
        if item is None:
            return Content("")
        opened = self.opened if self.opened is not None else not item.done
        caret = CARET_OPEN if opened else CARET_CLOSED
        if opened:
            return Content.assemble(f"{caret} reasoning\n", Content(item.text.strip()))
        lines = item.text.strip().splitlines() or [""]
        more = f"  +{len(lines) - 1} lines" if len(lines) > 1 else ""
        room = self.content_region.width - len("▸ reasoning  ") - len(more)
        preview = ellipsize(first_line(item.text), max(10, room))
        return Content.assemble(f"{caret} reasoning  ", Content(preview), more)

    def toggle(self) -> None:
        item = self._entry.item
        currently = self.opened if self.opened is not None else not (item and item.done)
        self.opened = not currently
        self._signature = None
        self.call_later(self.sync, self._entry, self.context())

    def on_click(self) -> None:
        self.toggle()


class ToolHeader(Widget):
    """One aligned row for a tool call: glyph, tool, summary, result, duration."""

    DEFAULT_CSS = """
    ToolHeader {
        height: 1;
        padding: 0 0 0 0;
    }
    ToolHeader:hover {
        background: $line;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.item: Item | None = None
        self.now = 0.0

    def render(self) -> Content:
        item = self.item
        if item is None:
            return Content("")
        return tool_row(item, self.size.width, self.now)

    def on_click(self) -> None:
        block = self.parent
        if isinstance(block, Block):
            block.toggle()


def tool_row(item: Item, width: int, now: float) -> Content:
    """The one-line summary of a tool call, fitted to `width` cells."""
    if not item.done:
        glyph, glyph_style = STATE_GLYPHS["running"], STATE_STYLES["running"]
        result = Content.styled("running", "$fg-muted")
        duration = human_elapsed(now - item.ts) if now > item.ts else ""
    else:
        state = "failed" if item.is_error else "done"
        glyph, glyph_style = STATE_GLYPHS[state], STATE_STYLES[state]
        result = _result(item)
        duration = human_duration(item.duration) if item.duration else ""
    summary = item.summary or item.tool
    if summary.startswith(item.tool + " "):
        summary = summary[len(item.tool) + 1 :]
    tail_width = 2 + 6
    result_width = min(_RESULT_COLUMN, result.cell_length)
    if width < 60:
        result_width = 0
    room = width - 2 - (_TOOL_COLUMN + 1) - tail_width - (result_width + 2 if result_width else 0)
    parts: list[Content | str | tuple[str, str]] = [
        (glyph, glyph_style),
        " ",
        (pad(item.tool, _TOOL_COLUMN), "$text"),
        " ",
        (pad(summary, max(room, 4)), "$fg-muted"),
    ]
    if result_width:
        parts.extend(["  ", result.truncate(result_width, ellipsis=True)])
    parts.extend(["  ", (pad(duration, 6, right=True), "$fg-muted")])
    return Content.assemble(*parts)


def _result(item: Item) -> Content:
    display = item.display or {}
    if display.get("type") == "diff":
        added, removed = diff_stat(str(display.get("diff", "")))
        return Content.assemble((f"+{added}", "$success"), " ", (f"-{removed}", "$error"))
    return Content.styled(first_line(item.output) or "no output", "$fg-muted")


def diff_stat(diff: str) -> tuple[int, int]:
    """Lines added and removed by a unified diff."""
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


class DiffView(Widget):
    """A unified diff with full-width add and delete tints and a +/- gutter.

    Long lines wrap under their gutter instead of running off the edge, so nothing a change
    did is hidden.
    """

    DEFAULT_CSS = """
    DiffView {
        height: auto;
        background: $bg-alt;
        padding: 1 0;
    }
    """

    def __init__(self, diff: str, path: str = "") -> None:
        super().__init__()
        self.lines = [line.expandtabs(4) for line in diff.splitlines()]
        self.path = path
        self._cache: tuple[int, list[Content]] | None = None

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return len(self._layout(width))

    def render(self) -> Content:
        return Content("\n").join(self._layout(self.content_region.width or self.size.width))

    def _layout(self, width: int) -> list[Content]:
        if self._cache is not None and self._cache[0] == width:
            return self._cache[1]
        rendered: list[Content] = []
        code_width = max(8, width - 4)
        for line in self.lines:
            if line.startswith(("---", "+++")):
                continue
            if line.startswith("@@"):
                for piece in wrap_cells(line, code_width + 2):
                    rendered.append(Content.styled(f"  {piece}".ljust(width), "$fg-muted"))
                continue
            sign, code = (line[:1], line[1:]) if line[:1] in "+- " else (" ", line)
            if line.startswith("\\"):
                rendered.append(Content.styled(f"  {line}".ljust(width), "$fg-muted"))
                continue
            tint = {"+": "on $success 14%", "-": "on $error 14%"}.get(sign, "")
            sign_style = {"+": "$success", "-": "$error"}.get(sign, "$fg-subtle")
            highlighted = highlight(code, path=self.path or None, theme=CodeHighlight)
            pieces = highlighted.divide(range(code_width, len(code), code_width)) or [highlighted]
            for index, piece in enumerate(pieces):
                gutter = Content.styled(f" {sign} " if index == 0 else "   ", sign_style)
                row = Content.assemble(gutter, piece)
                row = row.pad_right(max(0, width - row.cell_length))
                rendered.append(row.stylize_before(tint) if tint else row)
        self._cache = (width, rendered)
        return rendered


class OutputView(Static):
    """A tool's output preview in a well."""

    DEFAULT_CSS = """
    OutputView {
        background: $bg-alt;
        color: $text;
        padding: 1 2;
    }
    """


class ToolBlock(Block):
    """A tool call: one row that opens in place to show its diff or output."""

    DEFAULT_CSS = """
    ToolBlock {
        padding: 0 0 0 2;
    }
    ToolBlock.-open > ToolHeader {
        background: $bg-raised;
    }
    ToolBlock > .tool-body {
        margin: 0 0 0 2;
    }
    """

    expandable = True

    def __init__(self, entry: Entry) -> None:
        super().__init__(entry)
        self.header = ToolHeader()
        self.opened: bool | None = None
        self._entry = entry
        self._body_key: object = None

    def compose(self) -> ComposeResult:
        yield self.header

    def default_open(self, item: Item) -> bool:
        display = item.display or {}
        return item.done and (display.get("type") == "diff" or item.is_error)

    def is_open(self, ctx: Context) -> bool:
        item = self._entry.item
        if self.opened is not None:
            return self.opened
        if ctx.expand_all is not None:
            return ctx.expand_all
        return bool(item and self.default_open(item))

    def signature(self, entry: Entry, ctx: Context) -> object:
        item = entry.item
        if item is None:
            return None
        running = None if item.done else int(ctx.now - item.ts)
        return (item.done, item.is_error, len(item.output), running, self.is_open(ctx))

    async def apply(self, entry: Entry, ctx: Context) -> None:
        self._entry = entry
        item = entry.item
        if item is None:
            return
        self.header.item = item
        self.header.now = ctx.now
        self.header.refresh()
        opened = self.is_open(ctx) and item.done
        self.set_class(opened, "-open")
        body_key = (opened, len(item.output), item.done)
        if body_key == self._body_key:
            return
        self._body_key = body_key
        await self.query(".tool-body").remove()
        if opened:
            await self.mount(self._body(item))

    def _body(self, item: Item) -> Widget:
        display = item.display or {}
        if display.get("type") == "diff":
            view = DiffView(str(display.get("diff", "")), str(display.get("path", "")))
            view.add_class("tool-body")
            return view
        lines = item.output.rstrip("\n").splitlines()
        shown = lines[:OUTPUT_PREVIEW_LINES]
        content = Content("\n".join(shown)) if shown else Content.styled("no output", "$fg-muted")
        if len(lines) > len(shown):
            more = plural(len(lines) - len(shown), "more line")
            content = Content.assemble(content, "\n", (more, "$fg-muted"))
        return OutputView(content, classes="tool-body")

    def toggle(self) -> None:
        ctx = self.context()
        self.opened = not self.is_open(ctx)
        self._signature = None
        self.call_later(self.sync, self._entry, ctx)

    def forget_choice(self) -> None:
        """Drop the user's own open/closed choice so the transcript-wide one applies."""
        self.opened = None
        self._signature = None


class NoticeBlock(TextBlock):
    """Something the user should see that is not agent output, barred by its level."""

    DEFAULT_CSS = """
    NoticeBlock {
        border-left: outer $info;
        padding: 0 1;
    }
    NoticeBlock.-level-warning {
        border-left: outer $warning;
    }
    NoticeBlock.-level-error {
        border-left: outer $error;
    }
    NoticeBlock.-level-quiet {
        border-left: outer $fg-subtle;
    }
    """

    _GLYPHS: ClassVar[dict[str, tuple[str, str]]] = {
        "warning": ("! ", "$warning"),
        "error": ("✗ ", "$error"),
    }

    def signature(self, entry: Entry, ctx: Context) -> object:
        return self._parts(entry)

    async def apply(self, entry: Entry, ctx: Context) -> None:
        level, text = self._parts(entry)
        for name in ("info", "warning", "error", "quiet"):
            self.set_class(level == name, f"-level-{name}")
        glyph = self._GLYPHS.get(level)
        self.show(Content.assemble(glyph or "", Content(text)) if glyph else Content(text))

    @staticmethod
    def _parts(entry: Entry) -> tuple[str, str]:
        if entry.item is not None:
            return entry.item.level or "info", entry.item.text
        if entry.local is not None:
            level = entry.local.level if entry.local.level in ("warning", "error") else "quiet"
            return level, entry.local.text
        return "info", ""


class SkillBlock(TextBlock):
    """A skill was proposed for review, or saved to the library."""

    DEFAULT_CSS = """
    SkillBlock {
        border-left: outer $info;
        padding: 0 1;
    }
    """

    def signature(self, entry: Entry, ctx: Context) -> object:
        return entry.local.key if entry.local else None

    async def apply(self, entry: Entry, ctx: Context) -> None:
        local = entry.local
        if local is None:
            return
        description = Content(local.detail) if local.detail else Content("")
        if local.level == "proposed":
            self.show(
                Content.assemble(
                    "Proposed a skill: ",
                    (local.text, "bold"),
                    "\n",
                    description,
                    "\n" if local.detail else "",
                    ("f3", "bold $text"),
                    (" review it in Settings, Skills", "$fg-muted"),
                )
            )
        else:
            self.show(
                Content.assemble(
                    f"Skill {local.level}: ",
                    (local.text, "bold"),
                    "\n" if local.detail else "",
                    description,
                )
            )


class ResolutionBlock(Block):
    """How a permission request or a question was answered: one quiet line."""

    DEFAULT_CSS = """
    ResolutionBlock {
        padding: 0 0 0 2;
        color: $fg-muted;
    }
    """

    def __init__(self, entry: Entry) -> None:
        super().__init__(entry)
        self._entry = entry

    def signature(self, entry: Entry, ctx: Context) -> object:
        return entry.local.key if entry.local else None

    async def apply(self, entry: Entry, ctx: Context) -> None:
        self._entry = entry
        self.refresh(layout=True)

    def render(self) -> Content:
        local = self._entry.local
        if local is None:
            return Content("")
        state = "failed" if local.level == "error" else "done"
        head = Content.assemble(
            (STATE_GLYPHS[state], STATE_STYLES[state]),
            " ",
            (local.text, "$fg-muted"),
            "  ",
            Content(local.detail),
        )
        if not local.context:
            return head.truncate(self.content_region.width, ellipsis=True)
        room = max(10, self.content_region.width - head.cell_length - 3)
        return Content.assemble(head, "   ", (ellipsize(local.context, room), "$fg-muted"))


class RunEndBlock(TextBlock):
    """A quiet marker after each run: how it ended and how long it took."""

    DEFAULT_CSS = """
    RunEndBlock {
        padding: 0 0 0 2;
        color: $fg-muted;
    }
    """

    _WORDS: ClassVar[dict[str, tuple[str, str]]] = {
        "completed": ("done", "done"),
        "failed": ("failed", "failed"),
        "cancelled": ("cancelled", "cancelled"),
    }

    def signature(self, entry: Entry, ctx: Context) -> object:
        run = entry.run
        return (run.status, run.finished_ts) if run else None

    async def apply(self, entry: Entry, ctx: Context) -> None:
        run = entry.run
        if run is None:
            return
        word, state = self._WORDS.get(run.status, (run.status, "done"))
        took = human_elapsed(run.finished_ts - run.ts)
        self.show(
            Content.assemble(
                (STATE_GLYPHS[state], STATE_STYLES[state]), " ", f"{word}  {took}  {run.method}"
            )
        )


_BLOCKS: dict[str, type[Block]] = {
    "prompt": PromptBlock,
    "steer": PromptBlock,
    "message": MessageBlock,
    "assistant": AssistantBlock,
    "thinking": ThinkingBlock,
    "tool": ToolBlock,
    "notice": NoticeBlock,
    "local": NoticeBlock,
    "skill": SkillBlock,
    "resolution": ResolutionBlock,
    "run-end": RunEndBlock,
}


def make_block(entry: Entry) -> Block:
    return _BLOCKS.get(entry.kind, NoticeBlock)(entry)


# -- the transcript ---------------------------------------------------------------------------


class EmptyState(Static):
    """What the transcript shows before there is anything to show."""

    DEFAULT_CSS = """
    EmptyState {
        padding: 1 2 0 2;
        color: $fg-muted;
        height: auto;
    }
    """


class Transcript(VerticalScroll, can_focus=True):
    """A scrolling column of blocks that follows the bottom unless the user scrolls away.

    Keys: up/down move a cursor over the blocks that open (tool calls, reasoning), enter or
    space opens one, page and home/end keys scroll.
    """

    DEFAULT_CSS = """
    Transcript {
        height: 1fr;
        padding: 0 2 1 2;
        background: $background;
    }
    Transcript > Block.-cursor {
        background: $selection;
    }
    Transcript:focus > Block.-cursor {
        border-left: outer $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "cursor(-1)", "Previous", show=False),
        Binding("down", "cursor(1)", "Next", show=False),
        Binding("enter,space", "toggle", "Open or close", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
        Binding("home", "scroll_home", show=False),
        Binding("end", "scroll_end", show=False),
    ]

    class UnseenChanged(Message):
        """The number of blocks added below the visible part changed."""

        def __init__(self, count: int) -> None:
            super().__init__()
            self.count = count

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.unseen = 0
        self.cursor_key: str | None = None
        self._empty = EmptyState("")
        self._entries: dict[str, Entry] = {}
        self.expand_all: bool | None = None

    def compose(self) -> ComposeResult:
        yield self._empty

    @property
    def blocks(self) -> list[Block]:
        return [child for child in self.children if isinstance(child, Block)]

    def set_empty(self, content: Content | None) -> None:
        """Show `content` while there are no entries (None hides it)."""
        if content is None:
            self._empty.display = False
        else:
            self._empty.update(content)
            self._empty.display = not self.blocks

    async def clear(self) -> None:
        """Drop every block, e.g. before showing another node."""
        self.anchor(False)
        self.cursor_key = None
        self._entries.clear()
        await self.remove_children(self.blocks)
        self._set_unseen(0)

    async def sync(self, entries: list[Entry], ctx: Context) -> None:
        """Bring the blocks in line with `entries` (in order)."""
        scrolled_away = self.max_scroll_y > 0 and self.scroll_y < self.max_scroll_y - 1
        self._entries = {entry.key: entry for entry in entries}
        existing = {block.key: block for block in self.blocks}
        stale = [block for key, block in existing.items() if key not in self._entries]
        if stale:
            await self.remove_children(stale)
        added = await self._place(entries, existing)
        previous_kind = ""
        for block, entry in zip(self.blocks, entries, strict=False):
            block.set_class(entry.kind == "tool" and previous_kind == "tool", "-grouped")
            previous_kind = entry.kind
            await block.sync(entry, ctx)
        self._empty.display = not entries and bool(self._empty.content)
        if added and scrolled_away and self.is_anchored:
            self._set_unseen(self.unseen + added)
        self.call_after_refresh(self._follow)

    async def _place(self, entries: list[Entry], existing: dict[str, Block]) -> int:
        """Mount new blocks and move existing ones so the order matches; returns new count."""
        current = self.blocks
        keys = [block.key for block in current]
        wanted = [entry.key for entry in entries]
        if wanted[: len(keys)] == keys:
            new = [make_block(entry) for entry in entries[len(keys) :]]
            if new:
                await self.mount_all(new)
            return len(new)
        added = 0
        previous: Widget = self._empty
        for entry in entries:
            block = existing.get(entry.key)
            if block is None:
                block = make_block(entry)
                await self.mount(block, after=previous)
                added += 1
            elif self.children.index(block) != self.children.index(previous) + 1:
                self.move_child(block, after=previous)
            previous = block
        return added

    def _follow(self) -> None:
        """Start following the bottom once the content is taller than the view."""
        if not self.is_anchored and self.virtual_size.height > self.size.height:
            self.anchor()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if self.unseen and new_value >= self.max_scroll_y - 1:
            self._set_unseen(0)

    def jump_to_end(self) -> None:
        self.anchor()
        self._set_unseen(0)

    def _set_unseen(self, count: int) -> None:
        if count != self.unseen:
            self.unseen = count
            self.post_message(self.UnseenChanged(count))

    # -- cursor ---------------------------------------------------------------------------------

    def _expandable(self) -> list[Block]:
        return [block for block in self.blocks if block.expandable]

    def on_focus(self) -> None:
        candidates = self._expandable()
        if candidates:
            current = next((b for b in candidates if b.key == self.cursor_key), candidates[-1])
            self._move_cursor(current)

    def on_blur(self) -> None:
        for block in self.blocks:
            block.remove_class("-cursor")

    def _move_cursor(self, block: Block) -> None:
        for other in self.blocks:
            other.set_class(other is block, "-cursor")
        self.cursor_key = block.key
        self.scroll_to_widget(block, animate=False)

    def action_cursor(self, step: int) -> None:
        candidates = self._expandable()
        if not candidates:
            if step < 0:
                self.scroll_up(animate=False)
            else:
                self.scroll_down(animate=False)
            return
        keys = [block.key for block in candidates]
        index = keys.index(self.cursor_key) if self.cursor_key in keys else len(keys)
        index = max(0, min(len(keys) - 1, index + step))
        self._move_cursor(candidates[index])

    def action_toggle(self) -> None:
        block = next((b for b in self.blocks if b.key == self.cursor_key), None)
        if block is not None:
            block.toggle()

    def set_expand_all(self, opened: bool) -> None:
        """Open or close every tool row, forgetting rows the user opened one by one."""
        self.expand_all = opened
        for block in self.blocks:
            if isinstance(block, ToolBlock):
                block.forget_choice()

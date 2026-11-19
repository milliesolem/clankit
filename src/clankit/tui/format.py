"""Glyphs, status vocabulary and number formats shared by every part of the terminal UI.

Status is always carried by the shape of a glyph, with colour as a second signal only, so the
interface stays readable in monochrome terminals and for colour-blind users. Every glyph here
is a text-presentation character (none has the Unicode Emoji property), which keeps it one
cell wide and stops terminals from substituting a colour emoji.

Colours are named by theme variable (``$success``) and resolved by Textual at render time, so
a theme switch repaints everything without rebuilding any text.
"""

from __future__ import annotations

from rich.cells import cell_len, chop_cells

from clankit.agent.events import TERMINAL_STATUSES
from clankit.agent.state import NodeView

#: Node states shown in the task tree and the transcript. `verifying` is a running reviewer
#: and `needs-input` a node waiting on a permission or a question.
STATE_GLYPHS: dict[str, str] = {
    "pending": "○",
    "running": "●",
    "waiting": "◐",
    "integrating": "◆",
    "verifying": "◇",
    "done": "✓",
    "failed": "✗",
    "cancelled": "-",
    "needs-input": "!",
}

STATE_STYLES: dict[str, str] = {
    "pending": "$fg-subtle",
    "running": "$accent",
    "waiting": "$accent",
    "integrating": "$accent",
    "verifying": "$info",
    "done": "$success",
    "failed": "$error",
    "cancelled": "$fg-subtle",
    "needs-input": "$warning",
}

STATE_WORDS: dict[str, str] = {
    "pending": "pending",
    "running": "working",
    "waiting": "waiting on subtasks",
    "integrating": "integrating",
    "verifying": "reviewing",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
    "needs-input": "needs input",
}

#: States in which a node's agent is doing something right now.
ACTIVE_STATES = frozenset({"running", "waiting", "integrating", "verifying", "needs-input"})

CARET_OPEN = "▾"
CARET_CLOSED = "▸"
PROMPT_MARK = "\u203a"  # a single right-pointing angle quotation mark
ELLIPSIS = "…"

_REVIEW_ROLES = frozenset({"reviewer", "judge"})
_VENDOR_PREFIXES = ("claude-",)


def node_state(node: NodeView, needs_input: bool = False) -> str:
    """The display state of a node: its status, refined by what it is waiting for."""
    status = node.info.status
    if needs_input and status not in TERMINAL_STATUSES:
        return "needs-input"
    if status == "running" and node.info.role in _REVIEW_ROLES:
        return "verifying"
    return status


def human_tokens(count: int) -> str:
    """Token counts in at most five cells: ``830``, ``41.2k``, ``1.25M``; ``-`` for none."""
    if count <= 0:
        return "-"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M" if count < 10_000_000 else f"{count // 1_000_000}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k" if count < 100_000 else f"{count // 1_000}k"
    return str(count)


def human_elapsed(seconds: float) -> str:
    """Wall time in at most five cells: ``0:04``, ``12:31``, ``1h02``."""
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}"
    return f"{total // 60}:{total % 60:02d}"


def human_duration(seconds: float) -> str:
    """A tool call's duration: tenths below ten seconds, then whole seconds, then m:ss."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return human_elapsed(seconds)


def model_alias(ref: str) -> str:
    """A short name for a ``provider/model`` reference, as the tree and status row show it.

    The provider and well-known vendor prefixes are dropped (``anthropic/claude-sonnet-5-5``
    becomes ``sonnet-5-5``) because the column is narrow and the full reference is one
    keystroke away in the peek view.
    """
    if not ref:
        return ""
    name = ref.rsplit("/", 1)[-1]
    for prefix in _VENDOR_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix) :]
    return name


def ellipsize(text: str, width: int) -> str:
    """`text` cut to `width` terminal cells, ending in an ellipsis when it was cut."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width == 1:
        return ELLIPSIS
    return chop_cells(text, width - 1)[0].rstrip() + ELLIPSIS


def ellipsize_left(text: str, width: int) -> str:
    """`text` cut to `width` cells from the left, keeping its end (for paths)."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    kept = ""
    for char in reversed(text):
        if cell_len(kept) + cell_len(char) > width - 1:
            break
        kept = char + kept
    return ELLIPSIS + kept


def pad(text: str, width: int, *, right: bool = False) -> str:
    """`text` fitted to exactly `width` cells: ellipsized if too long, padded if short."""
    fitted = ellipsize(text, width)
    gap = " " * max(0, width - cell_len(fitted))
    return gap + fitted if right else fitted + gap


def first_line(text: str) -> str:
    """The first line of `text` that has something on it, stripped."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def wrap_cells(text: str, width: int) -> list[str]:
    """Hard-wrap one line of text into pieces of at most `width` cells (for code and diffs)."""
    if width <= 0:
        return [text]
    if not text:
        return [""]
    return chop_cells(text, width)


def plural(count: int, noun: str, plural_form: str = "") -> str:
    return f"{count} {noun if count == 1 else (plural_form or noun + 's')}"

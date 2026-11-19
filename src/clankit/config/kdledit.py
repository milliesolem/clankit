"""Changing a KDL document without rewriting it.

An `Editor` takes a parsed document, collects edits addressed to its nodes (change this value,
remove that node, add a node here) and applies them all at once as splices into the original
text. Whatever no edit touches survives byte for byte: comments, blank lines, disabled ("/-")
nodes, quoting, indentation, line endings and the byte order mark.

Every edit refers to the document as it was parsed, so edits never invalidate each other's
positions, and `result()` refuses to combine edits that would overlap. The editor knows nothing
about what the nodes mean; `clankit.config.kdl` decides which edits express a configuration
change and checks the outcome.

New text follows its surroundings: the indentation of neighbouring nodes, the document's line
endings, and a blank line around a new block where the neighbours are spaced that way. A block
written on one line, ``web { port 8420; host "::1" }``, stays on one line.

Removing a node removes its lines and the comment directly above it. One case is ambiguous: a
comment above a one-line node that is directly followed by another node may be a heading for
the whole group, so there the comment stays.
"""

from __future__ import annotations

import itertools
import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from clankit.config.kdlsyntax import (
    BOM,
    NEWLINE,
    Document,
    NewNode,
    Node,
    Value,
    Written,
    format_value,
)

_DEFAULT_INDENT = "    "

#: What may follow a node on its last line for the node to own that line.
_LINE_TAIL = re.compile(r"[ \t]*;?[ \t]*(?://.*)?", re.DOTALL)
#: A line holding only a `//` comment. A line that contains "*/" might instead be the end of a
#: block comment, and deleting it would open that comment over the rest of the file.
_COMMENT_LINE = re.compile(r"[ \t]*//(?:(?!\*/).)*", re.DOTALL)
_SEPARATOR = re.compile(r"[ \t]*;[ \t]*")
_BLANK_LINE_BETWEEN = re.compile(rf"(?:{NEWLINE.pattern})[ \t]*(?:{NEWLINE.pattern})")


class KdlEditError(Exception):
    """The requested edits contradict each other; no text was produced."""


@dataclass(frozen=True, slots=True)
class _Edit:
    """Replace ``text[start:end]``.

    `lines` marks replacement text made of whole lines, which must begin on a fresh line;
    `spaced` additionally sets it off from its neighbours with blank lines.
    """

    start: int
    end: int
    text: str = ""
    lines: bool = False
    spaced: bool = False

    @property
    def deletes(self) -> bool:
        return not self.text


class Editor:
    """Collects edits to one document; `result()` returns the edited text.

    `parent` is the node whose children block an edit works in, or None for the top level.
    Where a method takes a `NewNode | Node`, a `NewNode` is written out and an existing `Node`
    is moved: its source text is carried to the new place, comments included.
    """

    def __init__(self, document: Document) -> None:
        self._document = document
        self._text = document.text
        breaks = list(NEWLINE.finditer(self._text))
        self._starts = [0, *(m.end() for m in breaks)]
        self._ends = [*(m.start() for m in breaks), len(self._text)]
        self._newline = "\r\n" if "\r\n" in self._text else "\n"
        self._unit = self._indent_unit()
        self._edits: list[_Edit] = []
        self._appended: dict[Node | None, list[NewNode]] = {}
        self._removed: set[Node] = set()

    # -- edits ----------------------------------------------------------------------------------

    def set_value(self, value: Value, new: Written) -> None:
        """Rewrite one value in place, keeping its type annotation."""
        old = self._text[value.start : value.end]
        text = value.annotation + format_value(new)
        end = value.end
        gap = self._aligned_gap(value)
        if gap and not NEWLINE.search(old):
            # The comment after the value lines up with its neighbours: keep it in its column.
            text += " " * max(1, gap - (len(text) - len(old)))
            end += gap
        self._edits.append(_Edit(value.start, end, text))

    def remove(self, parent: Node | None, node: Node) -> None:
        """Delete a node, with its own comment and without leaving a hole in the layout."""
        self._removed.add(node)
        if not self._owns_lines(node):
            self._edits.append(self._removal_within_line(node))
            return
        start, end = self._footprint(node)
        self._edits.append(_Edit(*self._without_stray_blank_lines(parent, start, end)))

    def append(self, parent: Node | None, new: NewNode) -> None:
        """Add a node at the end of a block, creating the braces if the parent has none."""
        self._appended.setdefault(parent, []).append(new)

    def insert_after(self, parent: Node | None, anchor: Node, piece: NewNode | Node) -> None:
        """Put a node directly after its sibling-to-be `anchor`."""
        if self._owns_lines(anchor):
            at = self._next_line(anchor.end)
            self._edits.append(self._line_insertion(at, parent, piece, self._indent_at(anchor)))
        else:
            self._edits.append(_Edit(anchor.end, anchor.end, "; " + self._on_one_line(piece)))
        self._leave(parent, piece)

    def insert_before(self, parent: Node | None, anchor: Node, piece: NewNode | Node) -> None:
        """Put a node before its sibling-to-be `anchor`, above the comment on top of it."""
        if self._owns_lines(anchor):
            at = self._comment_start(self._line_of(anchor.start))
            self._edits.append(self._line_insertion(at, parent, piece, self._indent_at(anchor)))
        else:
            self._edits.append(_Edit(anchor.start, anchor.start, self._on_one_line(piece) + "; "))
        self._leave(parent, piece)

    def result(self) -> str:
        """The document with every edit applied."""
        edits = [*self._edits]
        for parent, pieces in self._appended.items():
            edits.extend(self._appends(parent, pieces))
        edits.sort(key=lambda edit: (edit.start, edit.end))  # stable: ties keep request order
        out = ""
        cursor = 0
        deleting = False
        for edit in edits:
            if edit.start < cursor:
                if not (deleting and edit.deletes):
                    raise KdlEditError(f"edits overlap at offset {edit.start}")
                cursor = max(cursor, edit.end)  # two removals that share blank lines
                continue
            out += self._text[cursor : edit.start]
            out += self._fitted(out, edit) if edit.lines else edit.text
            cursor = edit.end
            deleting = edit.deletes
        out += self._text[cursor:]
        return (BOM if self._document.bom else "") + out

    # -- placing text ---------------------------------------------------------------------------

    def _leave(self, parent: Node | None, piece: NewNode | Node) -> None:
        """A node that was placed somewhere else leaves its old place."""
        if isinstance(piece, Node):
            self.remove(parent, piece)

    def _line_insertion(
        self, at: int, parent: Node | None, piece: NewNode | Node, indent: str
    ) -> _Edit:
        if isinstance(piece, NewNode):
            text = "".join(line + self._newline for line in piece.lines(indent, self._unit))
            block = bool(piece.children)
        elif self._owns_lines(piece):
            start, end = self._footprint(piece)
            text = self._text[start:end]
            if not NEWLINE.search(text[-1:]):  # the last line of a file with no final newline
                text += self._newline
            block = self._line_of(piece.start) != self._line_of(piece.end)
        else:
            text = indent + self._text[piece.start : piece.end] + self._newline
            block = False
        return _Edit(at, at, text, lines=True, spaced=self._spaced(parent, block))

    def _on_one_line(self, piece: NewNode | Node) -> str:
        if isinstance(piece, NewNode):
            return piece.inline()
        return self._text[piece.start : piece.end]

    def _appends(self, parent: Node | None, pieces: Sequence[NewNode]) -> Iterator[_Edit]:
        """Edits that add `pieces` at the end of a block, whatever shape the block is in."""
        if parent is None:
            for piece in pieces:
                yield self._line_insertion(len(self._text), None, piece, "")
            return
        indent = self._indent_at(parent)
        block = parent.block
        if block is None or not self._text[block[0] + 1 : block[1] - 1].strip(" \t"):
            # No braces yet, or "{}" on one line: write a proper block.
            deeper = indent + self._unit
            body = "".join(
                line + self._newline for piece in pieces for line in piece.lines(deeper, self._unit)
            )
            start, end, lead = (parent.end, parent.end, " ") if block is None else (*block, "")
            yield _Edit(start, end, f"{lead}{{{self._newline}{body}{indent}}}")
            return
        close = block[1] - 1
        closing_line = self._starts[self._line_of(close)]
        if not self._text[closing_line:close].strip(" \t"):  # "}" on a line of its own
            child_indent = self._child_indent(parent, block)
            for piece in pieces:
                yield self._line_insertion(closing_line, parent, piece, child_indent)
            return
        text = "; ".join(piece.inline() for piece in pieces)
        staying = [child for child in parent.children if child not in self._removed]
        if staying:
            yield _Edit(staying[-1].end, staying[-1].end, "; " + text)
        else:
            lead = "" if self._text[close - 1] in " \t" else " "
            yield _Edit(close, close, f"{lead}{text} ")

    def _fitted(self, before: str, edit: _Edit) -> str:
        """`edit.text` with the line breaks it needs to sit on lines of its own after `before`."""
        text = edit.text
        content = before.rstrip()
        if content:
            wanted = 2 if edit.spaced and not content.endswith("{") else 1
            have = len(NEWLINE.findall(before[len(content) :]))
            text = self._newline * max(0, wanted - have) + text
        if edit.spaced:
            following = self._text[edit.end :].lstrip(" \t")
            if following and not NEWLINE.match(following) and not following.startswith("}"):
                text += self._newline
        return text

    def _spaced(self, parent: Node | None, block: bool) -> bool:
        """Should a new node be set off with blank lines?

        Follow the neighbours: yes if they are separated by blank lines. Top-level nodes are
        separated by default; inside a block only multi-line nodes ever are.
        """
        siblings = self._siblings(parent)
        if parent is None and len(siblings) < 2:
            return True
        if parent is not None and not block:
            return False
        return any(
            _BLANK_LINE_BETWEEN.search(self._text, first.end, second.start)
            for first, second in itertools.pairwise(siblings)
        )

    # -- removing -------------------------------------------------------------------------------

    def _footprint(self, node: Node) -> tuple[int, int]:
        """The whole lines a node occupies, including the comment that belongs to it."""
        first = self._line_of(node.start)
        start = self._starts[first] if self._comment_is_shared(node) else self._comment_start(first)
        return start, self._next_line(node.end)

    def _comment_is_shared(self, node: Node) -> bool:
        """Might the comment above `node` describe the node after it as well?"""
        first, last = self._line_of(node.start), self._line_of(node.end)
        if first != last or last + 1 >= len(self._starts):
            return False
        following = self._text[self._starts[last + 1] : self._ends[last + 1]].strip()
        return bool(following) and not following.startswith(("//", "}"))

    def _comment_start(self, line: int) -> int:
        """Offset where the run of comment lines directly above `line` begins."""
        while line > 0 and _COMMENT_LINE.fullmatch(
            self._text, self._starts[line - 1], self._ends[line - 1]
        ):
            line -= 1
        return self._starts[line]

    def _without_stray_blank_lines(
        self, parent: Node | None, start: int, end: int
    ) -> tuple[int, int]:
        """Widen a removal so that it leaves neither two blank lines in a row nor a blank line
        at the very start or end of a block."""
        low, high = 0, len(self._text)
        if parent is not None and parent.block is not None:
            low, high = parent.block[0] + 1, parent.block[1] - 1

        def blank(line: int) -> bool:
            if not 0 <= line < len(self._starts):
                return False
            inside = self._starts[line] < high and self._ends[line] >= low
            return inside and not self._text[self._starts[line] : self._ends[line]].strip()

        above = self._line_of(start) - 1
        below = self._line_of(end) if end < len(self._text) else len(self._starts)
        first = not self._text[low:start].strip()
        last = not self._text[end:high].strip()
        if blank(below) and (blank(above) or first):
            while blank(below + 1):
                below += 1
            end = self._next_line(self._starts[below])
        elif blank(above) and last:
            while blank(above - 1):
                above -= 1
            start = self._starts[above]
        return start, end

    def _removal_within_line(self, node: Node) -> _Edit:
        """Remove a node that shares its line: take its separator, or else the space before it."""
        separator = _SEPARATOR.match(self._text, node.end)
        if separator:
            return _Edit(node.start, separator.end())
        start = node.start
        while start > 0 and self._text[start - 1] in " \t":
            start -= 1
        return _Edit(start, node.end)

    # -- layout of the source -------------------------------------------------------------------

    def _siblings(self, parent: Node | None) -> tuple[Node, ...]:
        return self._document.nodes if parent is None else parent.children

    def _line_of(self, offset: int) -> int:
        return bisect_right(self._starts, offset) - 1

    def _line_text(self, line: int) -> str:
        return self._text[self._starts[line] : self._ends[line]]

    def _next_line(self, offset: int) -> int:
        """Offset just past the line that contains `offset`."""
        line = self._line_of(offset) + 1
        return self._starts[line] if line < len(self._starts) else len(self._text)

    def _starts_line(self, node: Node) -> bool:
        return not self._text[self._starts[self._line_of(node.start)] : node.start].strip(" \t")

    def _owns_lines(self, node: Node) -> bool:
        """Is the node alone on its lines, apart from a terminator and a trailing comment?"""
        tail = _LINE_TAIL.fullmatch(self._text, node.end, self._ends[self._line_of(node.end)])
        return self._starts_line(node) and tail is not None

    def _indent_at(self, node: Node) -> str:
        return _indentation(self._line_text(self._line_of(node.start)))

    def _child_indent(self, parent: Node, block: tuple[int, int]) -> str:
        """Indentation for a new child of a block whose closing brace is on its own line."""
        for child in reversed(parent.children):
            if self._starts_line(child):
                return self._indent_at(child)
        for line in range(self._line_of(block[0]) + 1, self._line_of(block[1] - 1)):
            if self._line_text(line).strip():  # a comment or disabled node shows the depth
                return _indentation(self._line_text(line))
        return self._indent_at(parent) + self._unit

    def _indent_unit(self) -> str:
        """One level of indentation as this document writes it."""
        for parent, child in _nested(self._document.nodes):
            outer, inner = self._indent_at(parent), self._indent_at(child)
            deeper = inner != outer and inner.startswith(outer)
            if deeper and self._starts_line(parent) and self._starts_line(child):
                return inner[len(outer) :]
        return _DEFAULT_INDENT

    def _aligned_gap(self, value: Value) -> int:
        """Width of the run of spaces between a value and the comment after it, if that comment
        starts in the same column as a comment on the line above or below; otherwise 0."""
        line = self._line_of(value.end)
        tail = self._text[value.end : self._ends[line]]
        gap = len(tail) - len(tail.lstrip(" "))
        if not gap or not tail.startswith("//", gap):
            return 0
        column = value.end + gap - self._starts[line]
        for other in (line - 1, line + 1):
            if 0 <= other < len(self._starts) and self._line_text(other).startswith("//", column):
                return gap
        return 0


def _indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _nested(nodes: Sequence[Node]) -> Iterator[tuple[Node, Node]]:
    """Every (parent, child) pair in the tree, parents first."""
    for node in nodes:
        for child in node.children:
            yield node, child
        yield from _nested(node.children)

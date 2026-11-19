"""KDL v2 syntax for the configuration layer: a parser adapter and a writer for single nodes.

Reading: `parse()` turns text into `Node` and `Value` objects that remember exactly where they
sit in the source, which is what lets `clankit.config.kdledit` change one value and leave every
other byte alone. The parsing itself is done by kdlquery, and this is the only module that
imports it. Its serializer is never used: it drops comments and mangles multi-line strings.

Writing: `format_value()`, `format_name()` and `NewNode` produce text for values and nodes that
are new. Strings are always quoted, so no value can turn into a keyword or a number on the way
back in, and names are left bare only when they are plain words.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from kdlquery import KDL2CSTParser, KDLParseError
from kdlquery.types import CSTArgEntry, CSTIdentifier, CSTNode, CSTValue

#: What a parsed argument can be.
Scalar = str | int | float | bool | None
#: What can be written as an argument.
Written = str | int | float | bool

BOM = chr(0xFEFF)

# Character tables from the KDL 2.0 specification, as inclusive code point ranges. They are
# spelled as numbers because several of these characters are invisible, and one of them pasted
# into this file would be both unreadable and wrong.
_LINE_BREAKS = ((0x0A, 0x0D), (0x85, 0x85), (0x2028, 0x2029))
_SURROGATES = ((0xD800, 0xDFFF),)
_FORBIDDEN = (
    (0x00, 0x08),
    (0x0E, 0x1F),
    (0x7F, 0x7F),
    (0x200E, 0x200F),
    (0x202A, 0x202E),
    (0x2066, 0x2069),
    (0xFEFF, 0xFEFF),
    *_SURROGATES,
)


def _char_class(*ranges: tuple[int, int]) -> str:
    """The body of a regular expression character class for the given ranges."""
    return "".join(f"\\U{low:08x}-\\U{high:08x}" for low, high in ranges)


#: Every line break KDL recognises. A `//` comment ends at any of them, so line arithmetic on a
#: document must use this and never a bare "\n".
NEWLINE = re.compile(rf"\r\n|[{_char_class(*_LINE_BREAKS)}]")

_KEYWORDS = frozenset({"true", "false", "null", "inf", "-inf", "nan"})

#: Names written without quotes. KDL allows far more, but a plain word is the subset every
#: parser agrees on, and quoting the rest is always valid.
_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")

#: Characters that cannot appear literally between double quotes: the quote and the backslash,
#: tab and line breaks, and the code points KDL forbids anywhere in a document (control
#: characters, direction overrides, the byte order mark, surrogates).
_ESCAPED = re.compile(rf'[\\"\t{_char_class(*_LINE_BREAKS, *_FORBIDDEN)}]')
_SHORT_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}
_SURROGATE = re.compile(f"[{_char_class(*_SURROGATES)}]")


class KdlSyntaxError(ValueError):
    """The text is not valid KDL v2. `message` says what to write instead."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"line {line}, column {column}: {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass(frozen=True, slots=True, eq=False)
class Value:
    """One argument or property value, and where it is written.

    `start` and `end` are offsets into `Document.text` and cover the type annotation, if any.
    `bare` marks a string written as a bare word rather than in quotes.
    """

    value: Scalar
    start: int
    end: int
    line: int
    column: int
    bare: bool
    annotation: str


@dataclass(frozen=True, slots=True, eq=False)
class Node:
    """One node. Nodes compare by identity: two nodes are the same only if they are the same
    place in the document.

    `start` and `end` cover the node from its name (or type annotation) to its last entry or
    closing brace, without the terminating newline, semicolon or trailing comment. `line` and
    `column` point at the name. `block` is the span of the children block, braces included.
    """

    name: str
    args: tuple[Value, ...]
    props: tuple[tuple[str, Value], ...]
    children: tuple[Node, ...]
    start: int
    end: int
    line: int
    column: int
    block: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class Document:
    """A parsed document. `text` excludes the byte order mark; `bom` records that it had one."""

    text: str
    bom: bool
    nodes: tuple[Node, ...]


def parse(text: str) -> Document:
    """Parse KDL v2. Raises `KdlSyntaxError` with a position and advice."""
    bom = text.startswith(BOM)
    body = text[1:] if bom else text  # kdlquery would read the mark as part of the first name
    try:
        tree = KDL2CSTParser().parse(body)
        nodes = tuple(_node(node) for node in tree.nodes)
    except KDLParseError as exc:
        # kdlquery's own message can quote the source, which may be an API key: it is dropped,
        # together with the exception that carries it.
        message = _advice(exc.code, _text_at(body, exc.line, exc.col)) or exc.hint
        raise KdlSyntaxError(message or "this is not valid KDL", exc.line, exc.col) from None
    except RecursionError:
        raise KdlSyntaxError("blocks are nested too deeply", 1, 1) from None
    return Document(body, bom, nodes)


def _node(node: CSTNode) -> Node:
    args = tuple(_value(e.value) for e in node.entries if isinstance(e, CSTArgEntry))
    props = tuple(
        (e.key.value, _value(e.value)) for e in node.entries if not isinstance(e, CSTArgEntry)
    )
    block = node.children_block_span
    return Node(
        name=node.name.value,
        args=args,
        props=props,
        children=tuple(_node(child) for child in node.children),
        start=node.span.start.offset,
        end=node.span.end.offset,
        line=node.name.span.start.line,
        column=node.name.span.start.column,
        block=None if block is None else (block.start.offset, block.end.offset),
    )


def _value(value: CSTValue | CSTIdentifier) -> Value:
    annotation = value.type_annotation if isinstance(value, CSTValue) else None
    return Value(
        value=value.value,
        start=value.span.start.offset,
        end=value.span.end.offset,
        line=value.span.start.line,
        column=value.span.start.column,
        bare=isinstance(value, CSTIdentifier),
        annotation=annotation.raw if annotation else "",
    )


def _text_at(text: str, line: int, column: int) -> str:
    """The rest of the source line from a 1-based position (empty if the position is off)."""
    starts = [0, *(m.end() for m in NEWLINE.finditer(text))]
    if not 1 <= line <= len(starts):
        return ""
    rest = text[starts[line - 1] + column - 1 :]
    stop = NEWLINE.search(rest)
    return rest[: stop.start()] if stop else rest


def _advice(code: str, found: str) -> str:
    """What to write instead, for the mistakes people actually make in a config file.

    `found` is the source text at the error. Only single characters and keywords from fixed
    lists are ever quoted back, never the user's own text.
    """
    char = found[:1]
    if code == "reserved-bare-identifier":
        word = next(
            (k for k in sorted(_KEYWORDS, key=len, reverse=True) if found.startswith(k)), ""
        )
        if word:
            return f'write #{word} instead of {word} (KDL v2 keywords start with "#")'
    if code == "unexpected-character":
        if char == "/":
            return 'text that contains "/" must be in double quotes'
        if char == "#":
            return (
                '"#" starts a keyword such as #true; other text that contains "#", a colour for '
                "example, must be in double quotes"
            )
        if char == "\\":
            return 'text that contains "\\" must be in double quotes, with the backslash doubled'
        return "this character is not allowed here; text with special characters must be in double quotes"
    if code == "number/invalid-trailing":
        return (
            "a value that starts with a digit must be a number; put text such as an address in "
            "double quotes"
        )
    if code == "block-comment/unterminated":
        return (
            '"/*" starts a comment that is never closed; if it is part of a value, put the value '
            "in double quotes"
        )
    if code in ("quoted-string/unterminated", "quoted-string/newline"):
        return "a string is missing its closing double quote"
    if code == "quoted-string/decode":
        return 'a string contains an invalid "\\" escape; write a literal backslash as "\\\\"'
    if code == "children-block/unterminated":
        return 'a "{" is never closed; add the missing "}"'
    if code == "expected-eol":
        if char == "}":
            return 'this "}" has no matching "{"'
        return 'a new setting must start on a new line (or after ";")'
    if code in ("expected-whitespace", "expected-value") and char == "=":
        return 'a setting is written "name value", without "="'
    return ""


def is_writable(text: str) -> bool:
    """Can `text` be stored at all? Lone surrogates have no representation in KDL or UTF-8."""
    return _SURROGATE.search(text) is None


def quote(text: str) -> str:
    """`text` as a double-quoted string on one line, valid whatever it contains.

    Raises `ValueError` for a lone surrogate; check `is_writable` first.
    """
    return '"' + _ESCAPED.sub(_escape, text) + '"'


def _escape(match: re.Match[str]) -> str:
    char = match.group()
    if char in _SHORT_ESCAPES:
        return _SHORT_ESCAPES[char]
    if not is_writable(char):
        raise ValueError(f"U+{ord(char):04X} is a lone surrogate and cannot be written")
    return f"\\u{{{ord(char):x}}}"


def format_name(name: str) -> str:
    """A node name: bare when it is a plain word, quoted otherwise."""
    return name if _PLAIN_NAME.match(name) and name not in _KEYWORDS else quote(name)


def format_value(value: Written) -> str:
    """A value: #true/#false, an integer without a decimal point, a float that stays a float,
    or a quoted string."""
    if isinstance(value, bool):
        return "#true" if value else "#false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "#nan"
        if math.isinf(value):
            return "#inf" if value > 0 else "#-inf"
        return repr(value)  # the shortest text that reads back as the same float
    return quote(value)


@dataclass(frozen=True, slots=True)
class NewNode:
    """A node to be written: ``name args... { children }``."""

    name: str
    args: tuple[Written, ...] = ()
    children: tuple[NewNode, ...] = ()

    def _head(self) -> str:
        return " ".join([format_name(self.name), *map(format_value, self.args)])

    def lines(self, indent: str, unit: str) -> list[str]:
        """The node on lines of its own, children one `unit` deeper."""
        if not self.children:
            return [indent + self._head()]
        inner = [line for child in self.children for line in child.lines(indent + unit, unit)]
        return [f"{indent}{self._head()} {{", *inner, f"{indent}}}"]

    def inline(self) -> str:
        """The node squeezed onto one line, for blocks written ``{ a 1; b 2 }``."""
        if not self.children:
            return self._head()
        return f"{self._head()} {{ {'; '.join(child.inline() for child in self.children)} }}"

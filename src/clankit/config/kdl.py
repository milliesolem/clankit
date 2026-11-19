"""config.kdl: reading it into a `Config`, and writing changes back into the user's own text.

The file is the user's. They annotate it, switch settings off with "/-" and lay it out as they
like, so saving a change must not rewrite it. `dumps(config, previous=text)` therefore edits:
it reads `text`, compares what it says with `config`, and touches only the nodes that differ.
Everything else survives byte for byte, including settings and sections this version of
Clankit does not know.

Reading and writing walk the same description of the file (`clankit.config.kdllayout.ROOT`),
so they cannot disagree about where a setting lives or which of two duplicates counts (the
last one). `_Reader` turns a document into plain data, `_Writer` turns the difference between
the plain data of a document and that of a `Config` into edits, and `dumps` reads its own
result back before returning it: text that does not mean exactly what was asked for is never
handed out.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from clankit.config.kdledit import Editor, KdlEditError
from clankit.config.kdllayout import (
    ROOT,
    Block,
    Keyed,
    Kind,
    Plain,
    Repeated,
    Single,
    config_from_plain,
    named,
    plain_from_config,
    same,
)
from clankit.config.kdlstarter import MARKER, starter_document
from clankit.config.kdlsyntax import (
    Document,
    KdlSyntaxError,
    NewNode,
    Node,
    Scalar,
    Value,
    parse,
)
from clankit.config.schema import Config, ConfigError

__all__ = ["dumps", "loads", "starter_document"]


def loads(text: str, *, warnings: list[str] | None = None) -> Config:
    """Read the text of a configuration file.

    Raises `ConfigError` for anything that cannot be read as intended; each line of the message
    gives a line and column and says what to write instead. Settings and sections Clankit does
    not know are skipped, and reported through `warnings` together with duplicates.
    """
    _, plain, found = _read(text)
    if warnings is not None:
        warnings.extend(found)
    return config_from_plain(plain)


def dumps(config: Config, *, previous: str | None = None) -> str:
    """The text of a configuration file that says `config`.

    With `previous`, the text of the existing file, the result is that text with only the
    differing settings edited. Without, it is a new file that lists just the settings that
    differ from their defaults. Raises `ConfigError` if `config` holds a value the file cannot
    express, if `previous` cannot be read, or if the result would not read back as `config`.
    """
    wanted = plain_from_config(config)
    try:
        document, current, _ = _read(MARKER + "\n" if previous is None else previous)
    except ConfigError as exc:
        raise ConfigError(f"the file must be fixed before settings can be saved:\n{exc}") from None
    writer = _Writer(document)
    writer.block(ROOT, None, document.nodes, current, wanted)
    try:
        text = writer.result()
        reread = _read(text)[1]
    except (KdlEditError, ConfigError) as exc:
        raise ConfigError(f"the change could not be written safely: {exc}") from exc
    if not same(reread, wanted):
        raise ConfigError("the change could not be written safely: it would not read back intact")
    return text


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------


def _read(text: str) -> tuple[Document, Plain, list[str]]:
    """Parse and read `text`: the document, its plain data and its warnings."""
    try:
        document = parse(text)
    except KdlSyntaxError as exc:
        raise ConfigError(str(exc)) from None
    reader = _Reader(document)
    plain = reader.block(ROOT, document.nodes)
    if reader.errors:
        errors = sorted(reader.errors)
        raise ConfigError("\n".join(f"line {line}, column {col}: {m}" for line, col, m in errors))
    return document, plain, [f"line {line}: {message}" for line, message in sorted(reader.warnings)]


def _by_name(nodes: Sequence[Node]) -> dict[str, list[Node]]:
    groups: dict[str, list[Node]] = {}
    for node in nodes:
        groups.setdefault(node.name, []).append(node)
    return groups


def _by_key(nodes: Sequence[Node]) -> dict[Scalar, list[Node]]:
    """Keyed nodes grouped by their argument. The last node of a group is the one that counts,
    and the groups are in the order of those."""
    groups: dict[Scalar, list[Node]] = {}
    for node in nodes:
        groups.setdefault(node.args[0].value, []).append(node)
    return dict(sorted(groups.items(), key=lambda group: group[1][-1].start))


class _Reader:
    """Reads a document into plain data.

    Problems are collected rather than raised one at a time, so that a file with three mistakes
    is fixed in one pass and not three. A value that could not be read is None in the result,
    which the caller discards once `errors` is non-empty.
    """

    def __init__(self, document: Document) -> None:
        self._text = document.text
        self.errors: list[tuple[int, int, str]] = []
        self.warnings: list[tuple[int, str]] = []

    def block(self, schema: Block, nodes: Sequence[Node]) -> Plain:
        groups = _by_name(nodes)
        result: Plain = {}
        for name, part in schema.parts.items():
            found = groups.pop(name, [])
            if isinstance(part, Single):
                result[name] = self._value(self._last(found), part.kind) if found else part.default
            elif isinstance(part, Repeated):
                result[name] = [self._value(node, part.kind) for node in found]
            elif isinstance(part, Keyed):
                result[name] = self._items(part, found)
            else:
                result[name] = self.block(part, self._children(self._last(found)) if found else ())
        for name, found in groups.items():
            if schema.entries is None:
                self._unknown(schema, found)
            elif problem := schema.key_problem(name):
                self._error(found[-1], problem)
            elif isinstance(schema.entries, Kind):
                result[name] = self._value(self._last(found), schema.entries)
            else:
                result[name] = self.block(schema.entries, self._children(self._last(found)))
        return result

    def _last(self, found: Sequence[Node]) -> Node:
        """The node that counts among duplicates, with a warning for each of the others."""
        for node in found[:-1]:
            later = found[-1].line
            self._warn(
                node, f"{named(node.name)} is set again on line {later}; this one is ignored"
            )
        return found[-1]

    def _children(self, node: Node) -> tuple[Node, ...]:
        """The children of a section, with warnings for whatever else is written on its node."""
        if node.args:
            ignored = f"the value after {named(node.name)} is ignored"
            self._warn(node, f"{ignored}; its settings go inside {{ }}")
        self._note_properties(node)
        return node.children

    def _value(self, node: Node, kind: Kind) -> Scalar:
        """The value of a one-value node."""
        name = named(node.name)
        self._note_properties(node)
        if node.block is not None:
            self._warn(node, f"the {{ }} block after {name} is ignored")
        if not node.args:
            self._error(node, f"{name} needs a value: {kind.expects}")
            return None
        if len(node.args) > 1:
            many = f"{name} takes one value, but {len(node.args)} are written"
            self._error(node.args[1], f"{many}; text with spaces in it must be in double quotes")
            return None
        value = node.args[0]
        problem = self._misquoted(value) or kind.problem(value.value)
        if problem:
            unset = " (delete the line to use the default)" if value.value is None else ""
            self._error(value, f"{name} {problem}{unset}")
            return None
        return kind.normalised(value.value)

    def _misquoted(self, value: Value) -> str | None:
        """Bare words that parse, but not as what was meant."""
        if not value.bare:
            return None
        if self._text.startswith("//", value.end):
            return (
                'is cut off at "//", which starts a comment; put the whole value in double quotes'
            )
        if str(value.value).startswith("'"):
            return "must be in double quotes, not single quotes"
        return None

    def _items(self, part: Keyed, found: Sequence[Node]) -> list[Plain]:
        valid = []
        for node in found:
            self._note_properties(node)
            name = named(node.name)
            if len(node.args) != 1:
                self._error(node, f'{name} needs one value, its id: {name} "..." {{ }}')
            elif problem := part.kind.problem(node.args[0].value):
                self._error(node.args[0], f"the id of a {name} {problem}")
            else:
                valid.append(node)
        items = []
        for key, group in _by_key(valid).items():
            for node in group[:-1]:
                later = group[-1].line
                self._warn(node, f"this {node.name} is listed again on line {later} and is ignored")
            items.append({part.key: key, **self.block(part.block, group[-1].children)})
        return items

    def _unknown(self, schema: Block, found: Sequence[Node]) -> None:
        for node in found:
            close = difflib.get_close_matches(node.name, schema.parts, n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            unknown = f"{named(node.name)} is not a known setting {schema.where}{hint}"
            self._warn(node, f"{unknown}; it is ignored")

    def _note_properties(self, node: Node) -> None:
        if node.props:
            key = named(node.props[0][0])
            ignored = f'"{key}=..." after {named(node.name)} is ignored'
            self._warn(node, f"{ignored}; a setting goes on a line of its own, as: {key} ...")

    def _warn(self, node: Node, message: str) -> None:
        self.warnings.append((node.line, message))

    def _error(self, at: Node | Value, message: str) -> None:
        self.errors.append((at.line, at.column, message))


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------


def _rendered(schema: Block, plain: Plain) -> tuple[NewNode, ...]:
    """New nodes for the contents of a block: only what differs from the defaults."""
    nodes: list[NewNode] = []
    for name, part in schema.parts.items():
        value = plain[name]
        if isinstance(part, Single):
            if value is not None and not same(value, part.default):
                nodes.append(NewNode(name, (value,)))
        elif isinstance(part, Repeated):
            nodes.extend(NewNode(name, (element,)) for element in value)
        elif isinstance(part, Keyed):
            nodes.extend(_item(name, part, item) for item in value)
        elif children := _rendered(part, value):
            nodes.append(NewNode(name, (), children))
    if schema.entries is not None:
        entries = {key: value for key, value in plain.items() if key not in schema.parts}
        nodes.extend(_entry(schema, key, value) for key, value in entries.items())
    return tuple(nodes)


def _item(name: str, part: Keyed, item: Plain) -> NewNode:
    return NewNode(name, (item[part.key],), _rendered(part.block, item))


def _entry(schema: Block, key: str, value: Any) -> NewNode:
    if isinstance(schema.entries, Block):
        return NewNode(key, (), _rendered(schema.entries, value))
    return NewNode(key, (value,))


def _alignment(
    old: Sequence[Scalar], new: Sequence[Scalar]
) -> tuple[list[tuple[str, int]], list[int]]:
    """How to turn the list `old` into `new` while disturbing as little as possible.

    Returns where each element of `new` comes from, as (how, index into `old`), and the
    indexes of `old` that are dropped. "keep": the element stays where it is. "edit": the
    element at that index is rewritten in place. "move": that element is wanted in a different
    position. "add": the element is new (index -1). Kept and edited elements are in the same
    relative order in both lists, so everything else can be placed around them; and a move
    never happens without a kept element, since both come from the lists sharing a value.
    """
    sources: list[tuple[str, int]] = [("add", -1)] * len(new)
    changed: list[tuple[range, range]] = []
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                sources[j1 + offset] = ("keep", i1 + offset)
        else:
            changed.append((range(i1, i2), range(j1, j2)))
    spare: dict[Scalar, list[int]] = {}
    for olds, _ in changed:
        for i in olds:
            spare.setdefault(old[i], []).append(i)
    for _, news in changed:
        for j in news:
            if spare.get(new[j]):
                sources[j] = ("move", spare[new[j]].pop(0))
    moved = {i for how, i in sources if how == "move"}
    dropped: list[int] = []
    for olds, news in changed:
        staying = [i for i in olds if i not in moved]
        arriving = [j for j in news if sources[j][0] == "add"]
        for i, j in zip(staying, arriving, strict=False):
            sources[j] = ("edit", i)
        dropped.extend(staying[len(arriving) :])
    return sources, dropped


class _Writer:
    """Expresses the difference between a document's plain data (`old`) and the wanted plain
    data (`new`) as edits to the document.

    Mirrors `_Reader.block`: the same grouping by name, the same last-one-counts rule. `owner`
    is the node whose children `nodes` are, None at the top level.
    """

    def __init__(self, document: Document) -> None:
        self._editor = Editor(document)

    def result(self) -> str:
        return self._editor.result()

    def block(
        self, schema: Block, owner: Node | None, nodes: Sequence[Node], old: Plain, new: Plain
    ) -> None:
        groups = _by_name(nodes)
        for name, part in schema.parts.items():
            found = groups.pop(name, [])
            if isinstance(part, Single):
                self._single(schema, owner, nodes, name, found, old[name], new[name])
            elif isinstance(part, Repeated):
                self._values(owner, name, found, old[name], new[name])
            elif isinstance(part, Keyed):
                self._items(owner, name, part, found, old[name], new[name])
            elif found:
                self.block(part, found[-1], found[-1].children, old[name], new[name])
            elif children := _rendered(part, new[name]):
                self._add(schema, owner, nodes, NewNode(name, (), children))
        if schema.entries is not None:
            self._entries(schema, owner, groups, old, new)

    def _single(
        self,
        schema: Block,
        owner: Node | None,
        nodes: Sequence[Node],
        name: str,
        found: Sequence[Node],
        old: Scalar,
        new: Scalar,
    ) -> None:
        if same(old, new):
            return
        if new is None:
            for node in found:
                self._editor.remove(owner, node)
        elif found:
            self._editor.set_value(found[-1].args[0], new)
        else:
            self._add(schema, owner, nodes, NewNode(name, (new,)))

    def _add(self, schema: Block, owner: Node | None, nodes: Sequence[Node], new: NewNode) -> None:
        """Add a child: at the end, or in an ordered block next to its neighbours in `parts`."""
        present = _by_name(nodes)
        names = list(schema.parts) if schema.ordered else [new.name]
        position = names.index(new.name)
        before = next((name for name in reversed(names[:position]) if name in present), None)
        after = next((name for name in names[position + 1 :] if name in present), None)
        if before is not None:
            self._editor.insert_after(owner, present[before][-1], new)
        elif after is not None:
            self._editor.insert_before(owner, present[after][0], new)
        else:
            self._editor.append(owner, new)

    def _entries(
        self,
        schema: Block,
        owner: Node | None,
        groups: Mapping[str, Sequence[Node]],
        old: Plain,
        new: Plain,
    ) -> None:
        for key, found in groups.items():
            if key not in new:
                for node in found:
                    self._editor.remove(owner, node)
            elif isinstance(schema.entries, Block):
                self.block(schema.entries, found[-1], found[-1].children, old[key], new[key])
            elif not same(old[key], new[key]):
                self._editor.set_value(found[-1].args[0], new[key])
        for key, value in new.items():
            if key not in schema.parts and key not in groups:
                self._editor.append(owner, _entry(schema, key, value))

    def _values(
        self, owner: Node | None, name: str, found: Sequence[Node], old: list[str], new: list[str]
    ) -> None:
        def rewrite(i: int, j: int) -> None:
            if old[i] != new[j]:
                self._editor.set_value(found[i].args[0], new[j])

        self._sequence(
            owner, found, old, new, rewrite, lambda i, j: True, lambda j: NewNode(name, (new[j],))
        )

    def _items(
        self,
        owner: Node | None,
        name: str,
        part: Keyed,
        found: Sequence[Node],
        old: list[Plain],
        new: list[Plain],
    ) -> None:
        groups = _by_key(found)
        nodes = [group[-1] for group in groups.values()]
        old_keys = [item[part.key] for item in old]
        new_keys = [item[part.key] for item in new]

        def rewrite(i: int, j: int) -> None:
            if old_keys[i] != new_keys[j]:
                self._editor.set_value(nodes[i].args[0], new_keys[j])
            self.block(part.block, nodes[i], nodes[i].children, old[i], new[j])

        sources = self._sequence(
            owner,
            nodes,
            old_keys,
            new_keys,
            rewrite,
            lambda i, j: same(old[i], new[j]),
            lambda j: _item(name, part, new[j]),
        )
        # An entry that stays put goes on hiding its earlier duplicates. Any other change would
        # bring them back to life, so they go.
        kept = {old_keys[i] for how, i in sources if how == "keep"}
        for key, group in groups.items():
            if key not in kept:
                for node in group[:-1]:
                    self._editor.remove(owner, node)

    def _sequence(
        self,
        owner: Node | None,
        nodes: Sequence[Node],
        old: Sequence[Scalar],
        new: Sequence[Scalar],
        rewrite: Callable[[int, int], None],
        unchanged: Callable[[int, int], bool],
        fresh: Callable[[int], NewNode],
    ) -> list[tuple[str, int]]:
        """Edit the list written as `nodes`, whose elements are `old`, into `new`.

        `rewrite(i, j)` edits node i in place into element j, `unchanged(i, j)` says whether
        the two are identical, and `fresh(j)` renders element j as a new node. Returns the
        alignment that was applied.
        """
        sources, dropped = _alignment(old, new)
        in_place = [nodes[i] for how, i in sources if how in ("keep", "edit")]
        anchor: Node | None = None
        for j, (how, i) in enumerate(sources):
            if how in ("keep", "edit"):
                anchor = nodes[i]
                rewrite(i, j)
            elif how == "move" and unchanged(i, j):
                self._beside(owner, anchor, in_place[0], nodes[i])
            else:
                if how == "move":
                    dropped.append(i)
                if in_place:
                    self._beside(owner, anchor, in_place[0], fresh(j))
                else:
                    self._editor.append(owner, fresh(j))
        for i in dropped:
            self._editor.remove(owner, nodes[i])
        return sources

    def _beside(
        self, owner: Node | None, anchor: Node | None, first: Node, piece: NewNode | Node
    ) -> None:
        """Place `piece` after `anchor`, the last element before it that stays in place, or
        failing that before `first`, the first element that stays in place."""
        if anchor is None:
            self._editor.insert_before(owner, first, piece)
        else:
            self._editor.insert_after(owner, anchor, piece)

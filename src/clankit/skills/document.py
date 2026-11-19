"""The SKILL.md format: read leniently, write strictly.

A skill is a folder holding a ``SKILL.md`` file: YAML frontmatter between two ``---`` lines
(``name`` and ``description`` plus optional fields) followed by Markdown instructions. The
format is shared by most coding agents, and files in the wild break the written specification
in small ways, so the two directions are deliberately asymmetric:

* `parse` accepts anything whose frontmatter can be recovered. Every scalar stays a string
  (``name: on`` must not become ``True``), and the mistakes people actually make in
  hand-written frontmatter - an unquoted colon, a ``#``, a leading ``[`` - are repaired
  instead of rejected.
* `render` writes only what every client accepts, and raises `SkillError` for anything else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SKILL_FILE = "SKILL.md"
NAME_MAX = 64
DESCRIPTION_MAX = 1024
BODY_MAX_LINES = 500
#: How much of a file is read to find its frontmatter. Bodies are read only on activation.
HEAD_BYTES = 64 * 1024

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_METADATA_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_FRONTMATTER_RE = re.compile(r"\A(?:[ \t]*\n)*---[ \t]*\n(?:(.*?)\n)?---[ \t]*(?:\n|\Z)", re.DOTALL)
_KEY_LINE_RE = re.compile(r"^([A-Za-z_][\w.-]*):[ \t]+(\S.*?)[ \t]*$")
_BLOCK_SCALAR_RE = re.compile(r"[|>][+-]?\d?[+-]?(?:[ \t]+#.*)?")
#: Fields that hold prose. Their one-line values are taken literally, so "C# projects" and
#: "[issue-number]" survive YAML's comment and flow-sequence syntax.
_TEXT_KEYS = frozenset({"description", "when_to_use", "argument-hint"})
_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


class SkillError(Exception):
    """A skill file is unusable, or an operation on the skill library was refused.

    The message is written for whoever has to act on it (the user, or a model that proposed
    the skill): it says what is wrong and what would be accepted instead.
    """


@dataclass(frozen=True, slots=True)
class Document:
    """A parsed SKILL.md: the frontmatter fields and the Markdown that follows them."""

    meta: dict[str, Any]
    body: str


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------


def decode(raw: bytes) -> str:
    """The text of a skill file whatever editor wrote it: no BOM, LF line endings."""
    text = raw.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read(path: Path, *, head_only: bool = False) -> Document:
    """Parse the SKILL.md at `path`. With `head_only`, the body may be cut short.

    Raises `OSError` if the file cannot be read and `SkillError` if it has no usable
    frontmatter.
    """
    with path.open("rb") as handle:
        raw = handle.read(HEAD_BYTES) if head_only else handle.read()
    return parse(decode(raw))


def parse(text: str) -> Document:
    """Split decoded SKILL.md text into frontmatter fields and body.

    Raises `SkillError` when there is no frontmatter block or it cannot be read as a mapping
    even after repair. Unknown fields are kept; callers pick out what they understand.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillError(
            "no frontmatter: the file must start with a '---' line, then 'name:' and "
            "'description:', then a closing '---' line"
        )
    return Document(_load_mapping(match.group(1) or ""), text[match.end() :].lstrip("\n"))


def _load_mapping(source: str) -> dict[str, Any]:
    literal = _quote_values(source, _is_unquoted_prose)
    try:
        data = _load_yaml(literal)
    except yaml.YAMLError:
        try:
            data = _load_yaml(_quote_values(literal, _is_broken))
        except yaml.YAMLError as exc:
            reason = " ".join(str(exc).split())
            raise SkillError(f"the frontmatter is not valid YAML ({reason})") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SkillError("the frontmatter must be a list of 'field: value' lines")
    return {str(key): value for key, value in data.items()}


def _load_yaml(source: str) -> Any:
    # BaseLoader builds only str, list and dict and leaves every scalar a string.
    return yaml.load(source, Loader=yaml.BaseLoader)


def _is_unquoted_prose(key: str, value: str) -> bool:
    return key in _TEXT_KEYS and value[0] not in "\"'"


def _is_broken(key: str, value: str) -> bool:
    try:
        _load_yaml(f"{key}: {value}")
    except yaml.YAMLError:
        return True
    return False


def _quote_values(source: str, should_quote: Callable[[str, str], bool]) -> str:
    """Turn selected top-level ``key: value`` lines into ``key: "value"``.

    Only one-line plain values are touched: block scalars (``|``, ``>``) and values that
    continue on indented lines are valid YAML that quoting would break.
    """
    lines = source.split("\n")
    for index, line in enumerate(lines):
        match = _KEY_LINE_RE.match(line)
        if match is None:
            continue
        key, value = match.groups()
        following = lines[index + 1] if index + 1 < len(lines) else ""
        continues = following[:1] in (" ", "\t") and following.strip() != ""
        if continues or _BLOCK_SCALAR_RE.fullmatch(value) or not should_quote(key, value):
            continue
        lines[index] = f"{key}: {_quoted(value)}"
    return "\n".join(lines)


def text_field(value: Any) -> str:
    """A prose field as one line of text ("" when it is missing or not text)."""
    return " ".join(value.split()) if isinstance(value, str) else ""


def flag_field(value: Any) -> bool | None:
    """A boolean field as agents write them (true/false, yes/no, on/off, 1/0).

    Returns None when the value is not one of those, so the caller can keep its default and
    say so.
    """
    word = value.strip().lower() if isinstance(value, str) else ""
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    return None


def metadata_field(value: Any) -> dict[str, str]:
    """The ``metadata`` mapping as the format defines it: text keys to text values.

    Anything else (a missing or malformed field, nested values) is left out. Nested values
    are never rendered: YAML aliases let a few lines describe a structure that is enormous
    when written out.
    """
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, str)}


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def name_problem(name: str) -> str:
    """Why `name` is not a valid skill name, or "" when it is."""
    if not _NAME_RE.fullmatch(name):
        return (
            f"the name {name!r} is not valid: use lowercase letters, digits and single "
            "hyphens, e.g. 'release-notes'"
        )
    if len(name) > NAME_MAX:
        return f"the name {name!r} is {len(name)} characters long; the limit is {NAME_MAX}"
    return ""


def description_problem(description: str) -> str:
    """Why `description` cannot be written into a skill, or "" when it can."""
    text = " ".join(description.split())
    if not text:
        return "the description is empty: say what the skill does and when to use it"
    if len(text) > DESCRIPTION_MAX:
        return f"the description is {len(text)} characters long; the limit is {DESCRIPTION_MAX}"
    if "<" in text or ">" in text:
        return "the description contains '<' or '>', which several agents reject; reword it"
    if "---" in text:
        return (
            "the description contains '---', which some agents read as the end of the frontmatter"
        )
    return ""


def body_problem(body: str) -> str:
    """Why `body` cannot be a skill's instructions, or "" when it can."""
    if not body.strip():
        return "the body is empty: a skill needs instructions"
    lines = body.count("\n") + 1
    if lines > BODY_MAX_LINES:
        return f"the body is {lines} lines long; the limit is {BODY_MAX_LINES}"
    return ""


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------


def render(
    name: str, description: str, body: str, metadata: Mapping[str, str] | None = None
) -> str:
    """SKILL.md text that every Agent Skills client accepts.

    Only the specification's own fields are written, the description and metadata values are
    always quoted (so no parser can mistake them for numbers, dates or comments), and nothing
    is emitted unless name, description and body are all valid. Raises `SkillError` otherwise.
    """
    problems = [
        p for p in (name_problem(name), description_problem(description), body_problem(body)) if p
    ]
    if problems:
        raise SkillError("; ".join(problems))
    lines = ["---", f"name: {name}", f"description: {_quoted(' '.join(description.split()))}"]
    if metadata:
        lines.append("metadata:")
        for key, value in metadata.items():
            if not _METADATA_KEY_RE.fullmatch(key):
                raise SkillError(
                    f"the metadata key {key!r} is not valid: use letters, digits, '.', '-', '_' and '/'"
                )
            lines.append(f"  {key}: {_quoted(' '.join(value.split()))}")
    lines += ["---", "", body.strip("\n"), ""]
    return "\n".join(lines)


def _quoted(value: str) -> str:
    # A JSON string is a valid YAML double-quoted scalar and stays on one line.
    return json.dumps(value, ensure_ascii=False)

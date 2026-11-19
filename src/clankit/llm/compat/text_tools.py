"""A prompt-based tool protocol for models and backends without dependable native tool calls.

Tools are described in the system prompt; the model asks for one by ending its message with
a tagged block; tool results come back as text in the next user message::

    <use_tool name="write_file">
    <arg name="path">src/app.py</arg>
    <arg name="content">
    def main():
        print("hi")
    </arg>
    </use_tool>

Design choices, each for robustness with weak models:

* One call per message. The block's closing tag is sent as a stop sequence where possible,
  so a model cannot go on to invent the tool's result.
* Argument values are raw text, not JSON: escaping long code inside a JSON string is exactly
  what these models get wrong. Only non-text arguments are written as JSON.
* The tags are not ones any chat template uses for its own tool calls, so servers pass them
  through as ordinary text.

Everything here is hidden behind the `Provider` contract: the rest of Clankit sees ordinary
`ToolUseBlock`s and `ToolResultBlock`s.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from clankit.llm.compat.arguments import INVALID_KEY
from clankit.llm.types import ToolResultBlock, ToolSpec, ToolUseBlock

OPEN = "<use_tool"
CLOSE = "</use_tool>"
ARG_CLOSE = "</arg>"
#: Shown as the tool name when the model wrote a block without one, so that the agent
#: loop's "no such tool" answer tells the model what was wrong.
UNNAMED = "(missing tool name)"

_NAME = r"""\s+(?:name\s*=\s*)?["']?([^"'>\s]+)["']?\s*>"""
_CALL_OPEN = re.compile(r"<use_tool" + _NAME)
_ARG_OPEN = re.compile(r"<arg" + _NAME)
_BARE_OPEN = re.compile(r"<use_tool\s*>")
_BLOCK_END = re.compile(re.escape(ARG_CLOSE) + r"\s*" + re.escape(CLOSE))
_TRAILING_FENCE = re.compile(r"\n?```[\w-]*[ \t]*\n?\s*$")

_INSTRUCTIONS = """\
# Tools

You can act by using the tools listed below. To use a tool, end your message with one block \
in exactly this format:

<use_tool name="TOOL_NAME">
<arg name="ARGUMENT_NAME">value</arg>
<arg name="ANOTHER_ARGUMENT">
a value
that spans several lines
</arg>
</use_tool>

Rules:
- Use at most one tool per message, and write nothing after the closing </use_tool> tag. \
The tool's result arrives in the next message; never write a result yourself.
- Write each value exactly as the tool should receive it. Do not escape quotes or newlines, \
and do not wrap values in quotes or code fences. File contents and other multi-line values \
go between the tags as they are.
- Arguments that are not text (numbers, true/false, lists, objects) are written as JSON.
- When no tool is needed, answer normally without a block."""


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def render_instructions(tools: Sequence[ToolSpec]) -> str:
    """The system-prompt section that teaches the protocol and lists the tools."""
    return "\n\n".join([_INSTRUCTIONS, *(_describe(tool) for tool in tools)])


def render_call(use: ToolUseBlock) -> str:
    """A tool call as the model would have written it (for replaying earlier turns)."""
    lines = [f'<use_tool name="{use.name}">']
    lines += [
        f'<arg name="{key}">{_pad(_as_text(value))}{ARG_CLOSE}' for key, value in use.input.items()
    ]
    lines.append(CLOSE)
    return "\n".join(lines)


def render_result(name: str, result: ToolResultBlock) -> str:
    """A tool result as text for the next user message."""
    status = "error" if result.is_error else "ok"
    return f'<tool_result name="{name}" status="{status}">\n{result.content}\n</tool_result>'


def _describe(tool: ToolSpec) -> str:
    properties = tool.input_schema.get("properties") or {}
    required = set(tool.input_schema.get("required") or ())
    lines = [f"## {tool.name}", tool.description.strip()]
    if not properties:
        lines.append("Arguments: none")
        return "\n".join(lines)
    lines.append("Arguments:")
    for name, schema in properties.items():
        lines.append(_describe_argument(name, schema, name in required))
    return "\n".join(lines)


def _describe_argument(name: str, schema: object, required: bool) -> str:
    schema = schema if isinstance(schema, Mapping) else {}
    kind = schema.get("type")
    kind_text = " or ".join(kind) if isinstance(kind, list) else str(kind or "any")
    need = "required" if required else "optional"
    line = f"- {name} ({kind_text}, {need})"
    description = str(schema.get("description") or "").strip()
    if description:
        line += f": {description}"
    if "enum" in schema:
        line += f" One of: {json.dumps(schema['enum'])}."
    if kind in ("array", "object"):
        line += f" Written as JSON matching: {json.dumps(schema)}"
    return line


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _pad(text: str) -> str:
    """Multi-line values start on their own line; `_unpad` removes exactly that newline."""
    return f"\n{text}" if "\n" in text else text


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def split_reply(text: str) -> tuple[str, str | None]:
    """Split a model reply into the prose before the first tool block and the block itself.

    The block starts at its opening tag and runs to its closing tag, or to the end of the
    text when the closing tag was consumed as the stop sequence. Anything after the first
    block is discarded: one call per message.
    """
    start = text.find(OPEN)
    if start < 0:
        return text, None
    block = text[start:]
    end = _BLOCK_END.search(block)
    if end:
        block = block[: end.end()]
    elif ARG_CLOSE not in block and CLOSE in block:
        block = block[: block.index(CLOSE) + len(CLOSE)]
    return _TRAILING_FENCE.sub("", text[:start]).rstrip(), block


def closed(block: str) -> str:
    """The block with its closing tag, which a stop sequence or an output limit may have
    removed. Earlier turns are replayed in this form so the model always sees whole blocks."""
    block = block.rstrip()
    return block if block.endswith(CLOSE) else f"{block}\n{CLOSE}"


def block_name(block: str) -> str | None:
    """The tool named by a (possibly still incomplete) block, once its opening tag is whole."""
    match = _CALL_OPEN.match(block)
    if match:
        return match.group(1)
    return UNNAMED if _BARE_OPEN.match(block) else None


def parse_block(
    block: str, tools: Mapping[str, ToolSpec], *, complete: bool
) -> tuple[str, dict[str, Any]]:
    """Read a tool block into ``(tool name, input)``.

    A block that cannot be read - no arguments recognisable, a value left open, or a
    response that was cut off (`complete` is False) before the block closed - yields
    ``{"_invalid_json": block}``, which the agent loop answers with an error. Nothing is
    guessed: a half-written value must never reach a tool.
    """
    name = block_name(block) or UNNAMED
    opening = _CALL_OPEN.match(block) or _BARE_OPEN.match(block)
    body = block[opening.end() :] if opening else block[len(OPEN) :]
    closed = body.rstrip().endswith(CLOSE)
    if closed:
        body = body.rstrip()[: -len(CLOSE)]
    if not (closed or complete):
        return name, {INVALID_KEY: block}
    raw_arguments = _read_arguments(body)
    if raw_arguments is None:
        return name, _json_body(body) or {INVALID_KEY: block}
    schema = tools[name].input_schema.get("properties") if name in tools else None
    properties = schema if isinstance(schema, Mapping) else {}
    return name, {key: _coerce(value, properties.get(key)) for key, value in raw_arguments.items()}


def _read_arguments(body: str) -> dict[str, str] | None:
    """The ``<arg>`` elements of a block body in order, or None if the body is not made of
    them."""
    arguments: dict[str, str] = {}
    rest = body.lstrip()
    while rest:
        opening = _ARG_OPEN.match(rest)
        if not opening:
            return None
        end = _value_end(rest, opening.end())
        if end < 0:
            return None
        arguments[opening.group(1)] = _unpad(rest[opening.end() : end])
        rest = rest[end + len(ARG_CLOSE) :].lstrip()
    return arguments


def _value_end(text: str, start: int) -> int:
    """Where the value starting at `start` ends: the first closing tag that is followed by
    another argument or by nothing. A closing tag inside the value (say, in a file that
    documents this format) is thereby read as part of the value."""
    position = start
    while (end := text.find(ARG_CLOSE, position)) >= 0:
        rest = text[end + len(ARG_CLOSE) :].lstrip()
        if not rest or _ARG_OPEN.match(rest):
            return end
        position = end + 1
    return -1


def _unpad(value: str) -> str:
    """Remove the layout newlines around a value, and nothing else.

    The newline after the opening tag is dropped. A value that still spans several lines is
    then exact, trailing newline included (a file should end with one); a one-line value
    loses the newline before its closing tag, so a path written on its own line stays a
    valid path.
    """
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    stripped = value.removesuffix("\n").removesuffix("\r")
    return stripped if "\n" not in stripped else value


def _coerce(raw: str, schema: object) -> Any:
    """Give a raw value the type its schema asks for. Text stays text; anything else is
    read as JSON, and left as text for schema validation to report if that fails."""
    kind = schema.get("type") if isinstance(schema, Mapping) else None
    if kind == "string" or (isinstance(kind, list) and "string" in kind):
        return raw
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    return raw if kind is None and isinstance(value, str) else value


def _json_body(body: str) -> dict[str, Any] | None:
    """Accept a block whose body is one JSON object: some models fall back to that."""
    try:
        value = json.loads(body, strict=False)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None

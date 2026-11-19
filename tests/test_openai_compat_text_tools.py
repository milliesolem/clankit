"""The prompt-based tool protocol: what the model is told, and how its blocks are read."""

from __future__ import annotations

from typing import Any

import pytest

from clankit.llm.compat.arguments import INVALID_KEY
from clankit.llm.compat.text_tools import (
    UNNAMED,
    block_name,
    closed,
    parse_block,
    render_call,
    render_instructions,
    render_result,
    split_reply,
)
from clankit.llm.types import ToolResultBlock, ToolSpec, ToolUseBlock

EDIT = ToolSpec(
    "edit_file",
    "Replace text in a file.\n",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit."},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "count": {"type": "integer", "description": "How many matches to replace."},
            "dry_run": {"type": "boolean"},
            "ranges": {"type": "array", "items": {"type": "integer"}},
            "mode": {"type": "string", "enum": ["exact", "regex"]},
            "label": {"type": ["string", "null"]},
            "anything": {},
        },
        "required": ["path", "old", "new"],
    },
)
PING = ToolSpec("ping", "Check the connection.", {"type": "object", "properties": {}})
TOOLS = {tool.name: tool for tool in (EDIT, PING)}


def parse(block: str, *, complete: bool = True) -> tuple[str, dict[str, Any]]:
    return parse_block(block, TOOLS, complete=complete)


# --------------------------------------------------------------------------------------------
# What the model is told
# --------------------------------------------------------------------------------------------


def test_instructions_teach_the_format_and_describe_every_tool() -> None:
    text = render_instructions([EDIT, PING])
    assert '<use_tool name="TOOL_NAME">' in text and "</use_tool>" in text
    assert "at most one tool per message" in text
    assert (
        "## edit_file\n"
        "Replace text in a file.\n"
        "Arguments:\n"
        "- path (string, required): File to edit.\n"
        "- old (string, required)\n"
        "- new (string, required)\n"
        "- count (integer, optional): How many matches to replace.\n"
        "- dry_run (boolean, optional)\n"
    ) in text
    assert '- ranges (array, optional) Written as JSON matching: {"type": "array"' in text
    assert '- mode (string, optional) One of: ["exact", "regex"].' in text
    assert "- label (string or null, optional)" in text
    assert "- anything (any, optional)" in text
    assert text.endswith("## ping\nCheck the connection.\nArguments: none")


def test_instructions_do_not_change_between_requests() -> None:
    """The system prompt must stay identical for the life of a conversation."""
    assert render_instructions([EDIT, PING]) == render_instructions([EDIT, PING])


def test_results_are_rendered_with_their_tool_and_status() -> None:
    ok = render_result("read_file", ToolResultBlock("t1", "line 1\nline 2"))
    failed = render_result("read_file", ToolResultBlock("t1", "no such file", is_error=True))
    assert ok == '<tool_result name="read_file" status="ok">\nline 1\nline 2\n</tool_result>'
    assert failed == '<tool_result name="read_file" status="error">\nno such file\n</tool_result>'


# --------------------------------------------------------------------------------------------
# Reading well-formed blocks
# --------------------------------------------------------------------------------------------


def test_values_are_raw_text_and_typed_by_the_schema() -> None:
    block = (
        '<use_tool name="edit_file">\n'
        '<arg name="path">src/app.py</arg>\n'
        '<arg name="old">\n'
        'def main():\n    print("a < b && c > d")\n'
        "</arg>\n"
        '<arg name="new">    return "42"</arg>\n'
        '<arg name="count">2</arg>\n'
        '<arg name="dry_run">true</arg>\n'
        '<arg name="ranges">[1, 5]</arg>\n'
        '<arg name="label">null</arg>\n'
        "</use_tool>"
    )
    assert parse(block) == (
        "edit_file",
        {
            "path": "src/app.py",
            "old": 'def main():\n    print("a < b && c > d")\n',
            "new": '    return "42"',  # text stays text, indentation and quotes included
            "count": 2,
            "dry_run": True,
            "ranges": [1, 5],
            "label": "null",  # a string-typed argument is never reinterpreted
        },
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("src/a.py", "src/a.py", id="inline"),
        pytest.param("\nsrc/a.py\n", "src/a.py", id="one-line-on-its-own-line"),
        pytest.param("\r\nsrc/a.py\r\n", "src/a.py", id="crlf"),
        pytest.param("  indented", "  indented", id="leading-spaces-kept"),
        pytest.param("\nline 1\nline 2\n", "line 1\nline 2\n", id="multi-line-keeps-final-newline"),
        pytest.param("\nline 1\nline 2", "line 1\nline 2", id="multi-line-without-final-newline"),
        pytest.param("\n\nline 1\n\n", "\nline 1\n\n", id="blank-lines-inside-are-content"),
        pytest.param("", "", id="empty"),
    ],
)
def test_only_the_layout_newlines_around_a_value_are_removed(body: str, expected: str) -> None:
    _, arguments = parse(f'<use_tool name="edit_file"><arg name="old">{body}</arg></use_tool>')
    assert arguments == {"old": expected}


def test_closing_tag_inside_a_value_does_not_end_it() -> None:
    """A file that documents this very format can still be written."""
    content = 'Write <arg name="x">value</arg> then </use_tool> to finish.\n</arg> is the end tag.'
    block = (
        '<use_tool name="edit_file">\n'
        f'<arg name="old">{content}</arg>\n'
        '<arg name="path">README.md</arg>\n'
        "</use_tool>"
    )
    assert parse(block)[1] == {"old": content, "path": "README.md"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("7", 7), ("2.5", 2.5), ("true", True), ('{"a": [1]}', {"a": [1]}), ("hello", "hello")],
)
def test_untyped_arguments_are_read_as_json_when_they_are_json(raw: str, expected: Any) -> None:
    block = f'<use_tool name="edit_file"><arg name="anything">{raw}</arg></use_tool>'
    assert parse(block)[1] == {"anything": expected}
    unknown = f'<use_tool name="not_a_tool"><arg name="x">{raw}</arg></use_tool>'
    assert parse(unknown) == ("not_a_tool", {"x": expected})


def test_value_of_the_wrong_type_is_left_for_schema_validation_to_report() -> None:
    block = '<use_tool name="edit_file"><arg name="count">several</arg></use_tool>'
    assert parse(block)[1] == {"count": "several"}


@pytest.mark.parametrize(
    "use",
    [
        ToolUseBlock("t1", "edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"}),
        ToolUseBlock("t1", "edit_file", {"old": "a\nb\n", "new": "a\nb", "count": 3}),
        ToolUseBlock("t1", "edit_file", {"dry_run": False, "ranges": [1, 2], "path": " a b "}),
        ToolUseBlock("t1", "ping", {}),
    ],
)
def test_rendered_calls_read_back_exactly(use: ToolUseBlock) -> None:
    rendered = render_call(use)
    assert rendered.startswith(f'<use_tool name="{use.name}">') and rendered.endswith("</use_tool>")
    assert parse(rendered) == (use.name, use.input)


# --------------------------------------------------------------------------------------------
# Finding the block in a reply
# --------------------------------------------------------------------------------------------

CALL = '<use_tool name="ping">\n</use_tool>'


@pytest.mark.parametrize(
    ("reply", "prose", "block"),
    [
        pytest.param("Just an answer.", "Just an answer.", None, id="no-block"),
        pytest.param(f"Checking.\n\n{CALL}", "Checking.", CALL, id="prose-then-block"),
        pytest.param(f"{CALL}\nResult: pong!", "", CALL, id="text-after-is-dropped"),
        pytest.param(f"{CALL}\n{CALL}", "", CALL, id="second-block-is-dropped"),
        pytest.param(f"Here:\n```xml\n{CALL}\n```", "Here:", CALL, id="code-fence-is-removed"),
        pytest.param(
            'Go.\n<use_tool name="edit_file">\n<arg name="path">a</arg>\n',
            "Go.",
            '<use_tool name="edit_file">\n<arg name="path">a</arg>\n',
            id="closing-tag-eaten-by-the-stop-sequence",
        ),
    ],
)
def test_reply_is_split_into_prose_and_first_block(
    reply: str, prose: str, block: str | None
) -> None:
    assert split_reply(reply) == (prose, block)


def test_block_ends_at_its_own_closing_tag_not_one_inside_an_argument() -> None:
    block = '<use_tool name="edit_file">\n<arg name="old">see </use_tool> here</arg>\n</use_tool>'
    assert split_reply(f"{block}\nand more") == ("", block)


@pytest.mark.parametrize(
    ("head", "name"),
    [
        ('<use_tool name="ping">', "ping"),
        ("<use_tool name='ping'>\n<arg", "ping"),
        ("<use_tool name=ping>", "ping"),
        ("<use_tool ping>", "ping"),
        ('<use_tool  name = "ping" >', "ping"),
        ("<use_tool>", UNNAMED),
        ('<use_tool name="pi', None),
        ("<use_tool", None),
    ],
)
def test_tool_name_is_known_once_the_opening_tag_is_whole(head: str, name: str | None) -> None:
    assert block_name(head) == name


def test_closed_adds_the_closing_tag_only_when_it_is_missing() -> None:
    assert closed('<use_tool name="ping">\n') == CALL
    assert closed(CALL + "\n") == CALL


# --------------------------------------------------------------------------------------------
# Imperfect and unreadable blocks
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        pytest.param(
            "<use_tool name='edit_file'>\n<arg name=path>\na.py\n</arg>\n</use_tool>",
            ("edit_file", {"path": "a.py"}),
            id="loose-quoting",
        ),
        pytest.param(
            '<use_tool name="edit_file">\n{"path": "a.py", "count": 1}\n</use_tool>',
            ("edit_file", {"path": "a.py", "count": 1}),
            id="json-body",
        ),
        pytest.param(
            '<use_tool name="edit_file"><arg name="path">a.py</arg>',
            ("edit_file", {"path": "a.py"}),
            id="no-closing-tag",
        ),
        pytest.param('<use_tool name="ping">', ("ping", {}), id="no-arguments"),
        pytest.param(
            '<use_tool>\n<arg name="path">a.py</arg>\n</use_tool>',
            (UNNAMED, {"path": "a.py"}),
            id="no-name",
        ),
        pytest.param(
            '<use_tool name="edit_file"><arg name="path">a</arg><arg name="path">b</arg>',
            ("edit_file", {"path": "b"}),
            id="repeated-argument-last-wins",
        ),
    ],
)
def test_imperfect_blocks_are_read_when_nothing_has_to_be_guessed(
    block: str, expected: tuple[str, dict[str, Any]]
) -> None:
    assert parse(block) == expected


@pytest.mark.parametrize(
    ("block", "name"),
    [
        pytest.param(
            '<use_tool name="edit_file">\n<arg name="path">a.py</arg>\n<arg name="old">\nx = 1\n',
            "edit_file",
            id="value-left-open",
        ),
        pytest.param(
            '<use_tool name="edit_file">\nplease edit a.py\n</use_tool>',
            "edit_file",
            id="prose-instead-of-arguments",
        ),
        pytest.param(
            '<use_tool name="edit_file">\n<arg name="path">a.py</arg>\nand then some\n</use_tool>',
            "edit_file",
            id="text-between-arguments",
        ),
        pytest.param('<use_tool name="edit_file"\n<arg name="path">a.py</arg>', UNNAMED, id="tag"),
        pytest.param('<use_tool name="edit_file">\n[1, 2]\n</use_tool>', "edit_file", id="array"),
    ],
)
def test_unreadable_blocks_are_returned_whole_as_invalid(block: str, name: str) -> None:
    assert parse(block) == (name, {INVALID_KEY: block})


def test_block_cut_off_by_the_output_limit_is_never_run() -> None:
    cut_off = '<use_tool name="edit_file">\n<arg name="path">a.py</arg>\n'
    # The same text is a complete call when the model stopped at the stop sequence...
    assert parse(cut_off, complete=True) == ("edit_file", {"path": "a.py"})
    # ...but not when the response ran out of tokens: more arguments may have been coming.
    assert parse(cut_off, complete=False) == ("edit_file", {INVALID_KEY: cut_off})
    # A block that did close is whole either way.
    whole = closed(cut_off)
    assert parse(whole, complete=False) == ("edit_file", {"path": "a.py"})

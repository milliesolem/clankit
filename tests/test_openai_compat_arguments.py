"""Staged parsing of tool-call arguments: strict first, careful repair, never a guess."""

from __future__ import annotations

import json
from typing import Any

import pytest

from clankit.llm.compat.arguments import INVALID_KEY, parse_arguments, wire_arguments

TRUNCATED_WRITE = '{"path": "src/app.py", "content": "def main():\\n    print('


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param({"city": "Oslo"}, {"city": "Oslo"}, id="object-from-backend"),
        pytest.param(None, {}, id="missing"),
        pytest.param("", {}, id="empty"),
        pytest.param("  \n", {}, id="blank"),
        pytest.param("{}", {}, id="empty-object"),
        pytest.param('{"path": "a.py", "limit": 3}', {"path": "a.py", "limit": 3}, id="strict"),
        pytest.param(' {"path": "a.py"}\n', {"path": "a.py"}, id="surrounding-whitespace"),
        pytest.param('"{\\"cmd\\": \\"ls\\"}"', {"cmd": "ls"}, id="double-encoded"),
        pytest.param(json.dumps(json.dumps('{"cmd": "ls"}')), {"cmd": "ls"}, id="triple-encoded"),
    ],
)
@pytest.mark.parametrize("complete", [True, False])
def test_well_formed_arguments_parse_whether_or_not_the_response_finished(
    raw: Any, expected: dict[str, Any], complete: bool
) -> None:
    assert parse_arguments(raw, complete=complete) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("{'location': 'New York'}", {"location": "New York"}, id="python-quotes"),
        pytest.param(
            '{"recursive": True, "limit": None,}',
            {"recursive": True, "limit": None},
            id="python-literals-and-trailing-comma",
        ),
        pytest.param('```json\n{"cmd": "ls -la"}\n```', {"cmd": "ls -la"}, id="code-fence"),
        pytest.param('```\n{"cmd": "ls"}\n```', {"cmd": "ls"}, id="bare-fence"),
        pytest.param('Sure, the arguments: {"cmd": "ls"}', {"cmd": "ls"}, id="prose-before"),
        pytest.param('{"cmd": "ls"}<|tool_call_end|>', {"cmd": "ls"}, id="stray-token-after"),
        pytest.param(
            '{"content": "line1\nline2\tend"}',
            {"content": "line1\nline2\tend"},
            id="raw-newline-in-string",
        ),
    ],
)
def test_repairs_that_cannot_change_the_meaning_are_made_on_a_clean_finish(
    raw: str, expected: dict[str, Any]
) -> None:
    assert parse_arguments(raw, complete=True) == expected
    # ...and none of them is attempted when the response was cut off.
    assert parse_arguments(raw, complete=False) == {INVALID_KEY: raw}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(TRUNCATED_WRITE, id="cut-off-mid-string"),
        pytest.param('{"path": "a.py"', id="missing-brace"),
        pytest.param('{"path": "a"}{"path": "b"}', id="two-objects"),
        pytest.param("[1, 2]", id="array"),
        pytest.param('"just a string"', id="string"),
        pytest.param("42", id="number"),
        pytest.param("{path: a.py}", id="unquoted"),
        pytest.param("{'a': {1, 2}}", id="python-set"),
        pytest.param("__import__('os').system('true')", id="code"),
    ],
)
@pytest.mark.parametrize("complete", [True, False])
def test_anything_that_is_not_an_object_is_returned_as_invalid(raw: str, complete: bool) -> None:
    assert parse_arguments(raw, complete=complete) == {INVALID_KEY: raw}


def test_truncated_file_write_is_not_turned_into_a_valid_looking_call() -> None:
    """The hazard this module exists for: half a file must never be written."""
    parsed = parse_arguments(TRUNCATED_WRITE, complete=False)
    assert "content" not in parsed and parsed[INVALID_KEY] == TRUNCATED_WRITE


def test_non_string_non_object_arguments_are_invalid() -> None:
    assert parse_arguments([1, 2], complete=True) == {INVALID_KEY: "[1, 2]"}


@pytest.mark.parametrize(
    ("raw", "parsed", "expected"),
    [
        pytest.param('{"a":  1}', {"a": 1}, '{"a":  1}', id="valid-json-kept-as-written"),
        pytest.param("{'a': 1}", {"a": 1}, '{"a": 1}', id="repaired-is-reserialised"),
        pytest.param("", {}, "{}", id="empty-becomes-object"),
        pytest.param({"a": 1}, {"a": 1}, '{"a": 1}', id="object-becomes-string"),
        pytest.param("oops", {INVALID_KEY: "oops"}, '{"_invalid_json": "oops"}', id="invalid"),
        pytest.param('"{\\"a\\": 1}"', {"a": 1}, '{"a": 1}', id="double-encoded-is-unwrapped"),
        pytest.param('{"s": "é"}', {"s": "é"}, '{"s": "é"}', id="non-ascii"),
    ],
)
def test_stored_arguments_are_always_a_json_object(
    raw: Any, parsed: dict[str, Any], expected: str
) -> None:
    wire = wire_arguments(raw, parsed)
    assert wire == expected
    assert json.loads(wire) == parsed

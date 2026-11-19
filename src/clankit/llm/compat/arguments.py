"""Turning what a model wrote as tool-call arguments into a dict, or into an honest failure.

The stages run from strict to lenient and stop at the first that yields a JSON object:

1. an object the backend already parsed is used as it is;
2. nothing at all means "no arguments";
3. strict JSON;
4. repair, limited to rewrites that cannot change what the model meant - and skipped
   entirely when the response was cut off, because "repairing" half of a file write produces
   a valid-looking call that silently writes half a file;
5. a JSON string that itself contains JSON (double encoding) is unwrapped.

Anything else is handed to the agent loop as ``{"_invalid_json": raw}``, which answers the
call with an error the model can act on.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

INVALID_KEY = "_invalid_json"

_FENCE = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```$", re.DOTALL)
_UNPARSED = object()


def parse_arguments(raw: object, *, complete: bool) -> dict[str, Any]:
    """Parse the arguments of one tool call.

    `complete` is False when the response ended before the model finished (output limit,
    interrupted stream). Only strictly valid JSON is accepted then.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {INVALID_KEY: json.dumps(raw)}
    text = raw.strip()
    if not text:
        return {}
    value = _loads(text, strict=True)
    if value is _UNPARSED and complete:
        value = _repair(text)
    value = _unwrap(value)
    return value if isinstance(value, dict) else {INVALID_KEY: raw}


def wire_arguments(raw: object, parsed: dict[str, Any]) -> str:
    """The arguments string to keep in the stored assistant message.

    What the model wrote is kept when it is already a JSON object. Otherwise the parsed form
    is serialised, because several backends parse the arguments of earlier turns and fail
    the whole request - and with it the rest of the conversation - on a malformed one.
    """
    if isinstance(raw, str) and isinstance(_loads(raw.strip(), strict=True), dict):
        return raw
    return json.dumps(parsed, ensure_ascii=False)


def _loads(text: str, *, strict: bool) -> Any:
    try:
        return json.loads(text, strict=strict)
    except ValueError:
        return _UNPARSED


def _repair(text: str) -> Any:
    """Try the meaning-preserving readings of almost-JSON, most literal first."""
    candidates = [text]
    fenced = _FENCE.match(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])  # prose or stray tokens around the object
    for candidate in candidates:
        # Raw newlines and tabs inside strings: common when a model writes code as JSON.
        value = _loads(candidate, strict=False)
        if value is _UNPARSED:
            value = _python_literal(candidate)
        if value is not _UNPARSED:
            return value
    return _UNPARSED


def _python_literal(text: str) -> Any:
    """Read a Python-style dict (single quotes, True/False/None, trailing commas)."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return _UNPARSED
    if not isinstance(value, dict):
        return _UNPARSED
    try:
        return json.loads(json.dumps(value))  # only JSON-representable values qualify
    except (TypeError, ValueError):
        return _UNPARSED


def _unwrap(value: Any) -> Any:
    """Undo up to two levels of a JSON object encoded as a JSON string."""
    for _ in range(2):
        if not isinstance(value, str):
            break
        inner = _loads(value.strip(), strict=True)
        if inner is _UNPARSED:
            break
        value = inner
    return value

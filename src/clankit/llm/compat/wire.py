"""Neutral requests to chat-completions request bodies.

The interesting part is what happens to earlier assistant turns. Each turn this adapter
produced is stored as the backend's own message (`Message.native`), tagged with the provider,
model and tool mode that produced it. It is sent back unchanged only to that same
combination, and only when the backend's profile says it wants its reasoning back. In every
other case the turn is rebuilt from its neutral blocks - text and tool calls - because
reasoning, reasoning details and thought signatures mean nothing to another model and are a
400 on backends that validate message fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from clankit.llm.compat import text_tools
from clankit.llm.compat.profiles import Profile, ToolMode
from clankit.llm.types import (
    ChatRequest,
    Message,
    NativePayload,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)

KIND = "openai-compatible"

#: Gemini refuses a function call without a thought signature; this documented placeholder
#: stands in for calls that another model made.
PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"


def build_body(
    request: ChatRequest, *, provider: str, profile: Profile, mode: ToolMode
) -> dict[str, Any]:
    """The request body for one model call.

    Deliberately minimal: several backends reject fields they do not know, and others
    silently fix sampling parameters, so nothing optional is sent unless the profile says
    the backend understands it.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": build_messages(request, provider=provider, profile=profile, mode=mode),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.tools and mode == "text":
        body["stop"] = [text_tools.CLOSE]
    elif request.tools:
        body["tools"] = [_tool(tool) for tool in request.tools]
        if mode == "sequential" and profile.parallel_flag:
            body["parallel_tool_calls"] = False
    if request.max_tokens:
        body[profile.max_tokens_key] = request.max_tokens
    body.update(profile.effort_params(request.effort))
    return body


def build_messages(
    request: ChatRequest, *, provider: str, profile: Profile, mode: ToolMode
) -> list[dict[str, Any]]:
    """The conversation in wire form, system prompt first."""
    as_text = mode == "text"
    system = request.system
    if as_text and request.tools:
        system = "\n\n".join(filter(None, [system, text_tools.render_instructions(request.tools)]))
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    needs_reasoning = profile.replay == "required" and bool(request.tools) and not as_text
    tool_names: dict[str, str] = {}
    for message in request.messages:
        if message.role == "user":
            messages.extend(_user(message, tool_names, as_text=as_text))
            continue
        tool_names.update({use.id: use.name for use in message.tool_uses})
        wire = _assistant(message, request.model, provider=provider, profile=profile, mode=mode)
        if needs_reasoning:
            wire.setdefault(profile.reasoning_field, "")
        messages.append(wire)
    return _merge_user_runs(messages)


def stored_message(
    provider: str, model: str, mode: ToolMode, message: dict[str, Any]
) -> NativePayload:
    """Wrap a backend's assistant message for `Message.native`, with the identity that
    decides later whether it may be replayed."""
    data = {"provider": provider, "mode": _wire_mode(mode), "message": message}
    return NativePayload(kind=KIND, model=model, data=data)


# --------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------


def _assistant(
    message: Message, model: str, *, provider: str, profile: Profile, mode: ToolMode
) -> dict[str, Any]:
    stored = _replayable(message.native, model, provider=provider, mode=mode)
    if stored is not None and profile.replay != "drop":
        return dict(stored)
    if mode == "text":
        parts = [message.text, *(text_tools.render_call(use) for use in message.tool_uses)]
        return {"role": "assistant", "content": "\n\n".join(filter(None, parts))}
    wire: dict[str, Any] = {"role": "assistant", "content": message.text}
    if message.tool_uses:
        wire["tool_calls"] = [_tool_call(use) for use in message.tool_uses]
        if profile.placeholder_signature:
            wire["tool_calls"][0]["extra_content"] = {
                "google": {"thought_signature": PLACEHOLDER_SIGNATURE}
            }
    return wire


def _replayable(
    native: NativePayload | None, model: str, *, provider: str, mode: ToolMode
) -> Mapping[str, Any] | None:
    """The stored message of a turn, if this provider, model and tool mode produced it."""
    if native is None or native.kind != KIND or native.model != model:
        return None
    data = native.data
    if not isinstance(data, Mapping):
        return None
    if data.get("provider") != provider or data.get("mode") != _wire_mode(mode):
        return None
    message = data.get("message")
    return message if isinstance(message, Mapping) else None


def _user(
    message: Message, tool_names: Mapping[str, str], *, as_text: bool
) -> list[dict[str, Any]]:
    """Tool results first (one "tool" message each), then whatever the user or harness said."""
    results = message.tool_result_blocks
    text = "\n\n".join(block.text for block in message.content if isinstance(block, TextBlock))
    if as_text:
        parts = [
            text_tools.render_result(tool_names.get(result.tool_use_id, "tool"), result)
            for result in results
        ]
        return [{"role": "user", "content": "\n\n".join(filter(None, [*parts, text]))}]
    wire: list[dict[str, Any]] = [
        {"role": "tool", "tool_call_id": result.tool_use_id, "content": result.content}
        for result in results
    ]
    if text or not results:
        wire.append({"role": "user", "content": text})
    return wire


def _merge_user_runs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive user messages: several chat templates insist that roles alternate."""
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and message["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1] = {
                "role": "user",
                "content": f"{merged[-1]['content']}\n\n{message['content']}",
            }
        else:
            merged.append(message)
    return merged


# --------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------


def _tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _tool_call(use: ToolUseBlock) -> dict[str, Any]:
    return {
        "id": use.id,
        "type": "function",
        "function": {"name": use.name, "arguments": json.dumps(use.input, ensure_ascii=False)},
    }


def _wire_mode(mode: ToolMode) -> str:
    """ "native" and "sequential" share one wire format; "text" is the other."""
    return "text" if mode == "text" else "native"

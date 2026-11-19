"""Chat-completion chunks to stream events and a final response.

One accumulator serves every backend. It assumes nothing a chunk does not prove:

* tool-call fragments are joined by their ``index``, never by id; ids and names are set once
  and may arrive late, be repeated on every fragment, or be missing altogether;
* reasoning is read from every field backends use for it and from an inline ``<think>``
  block, and is kept out of the answer text;
* usage may ride on the last content chunk, on a chunk of its own with ``"choices": []``,
  or never arrive;
* fields it does not know are kept, merged across fragments, so the stored assistant
  message can be sent back to the same backend exactly as it was produced.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from clankit.llm.compat import text_tools
from clankit.llm.compat.arguments import parse_arguments, wire_arguments
from clankit.llm.compat.profiles import ToolMode
from clankit.llm.compat.wire import stored_message
from clankit.llm.provider import ProviderError
from clankit.llm.types import (
    Block,
    ChatResponse,
    Message,
    ResponseDone,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStarted,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

#: Message fields backends put reasoning text in, in order of preference for display.
REASONING_FIELDS: tuple[str, ...] = ("reasoning_content", "reasoning", "thinking")

#: The model finished of its own accord; only then may malformed arguments be repaired.
_CLEAN_FINISHES = frozenset({"stop", "tool_calls", "function_call"})
#: The backend abandoned the response; asking again is the remedy.
_FAILED_FINISHES = frozenset({"error", "insufficient_system_resource"})
#: Members of unknown structured fields that arrive as text fragments. Everything else in
#: such fields (ids, types, signatures) arrives whole and is replaced, not appended.
_STREAMED_TEXT = frozenset({"text", "summary"})

#: Characters of a text-protocol block examined for its opening tag before giving up.
_HEAD_LIMIT = 300

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _split_at(buffer: str, marker: str) -> tuple[str, str, bool]:
    """Split `buffer` at `marker`: (text before it, text after it, True).

    If the marker is absent the text is returned as ready, except for a tail that could be
    the beginning of the marker, which is handed back to wait for the next fragment.
    """
    index = buffer.find(marker)
    if index >= 0:
        return buffer[:index], buffer[index + len(marker) :], True
    for size in range(min(len(buffer), len(marker) - 1), 0, -1):
        if buffer.endswith(marker[:size]):
            return buffer[:-size], buffer[-size:], False
    return buffer, "", False


class ThinkSplitter:
    """Separates a leading ``<think>...</think>`` block from the answer that follows it.

    Only a block that opens the response counts. Models reason before they answer, so the
    tag anywhere later is ordinary text - for example an answer that explains these tags.
    """

    def __init__(self) -> None:
        self._state: Literal["start", "thinking", "between", "answer"] = "start"
        self._buffer = ""

    def feed(self, piece: str) -> list[tuple[bool, str]]:
        """Consume a content fragment; return ``(is_reasoning, text)`` parts ready to emit."""
        self._buffer += piece
        parts: list[tuple[bool, str]] = []
        while self._buffer and self._advance(parts):
            pass
        return parts

    def flush(self) -> list[tuple[bool, str]]:
        """Whatever was held back when the stream ended."""
        held, self._buffer = self._buffer, ""
        if not held or self._state == "between":
            return []
        return [(self._state == "thinking", held)]

    def _advance(self, parts: list[tuple[bool, str]]) -> bool:
        """Process the buffer in the current state. False means: wait for more input."""
        if self._state == "start":
            head = self._buffer.lstrip()
            if head.startswith(_THINK_OPEN):
                self._buffer, self._state = head[len(_THINK_OPEN) :], "thinking"
                return True
            if _THINK_OPEN.startswith(head):
                return False
            self._state = "answer"
            return True
        if self._state == "thinking":
            ready, self._buffer, closed = _split_at(self._buffer, _THINK_CLOSE)
            if ready:
                parts.append((True, ready))
            if closed:
                self._state = "between"
            return closed
        if self._state == "between":
            # Drop the blank lines that separate the reasoning from the answer.
            self._buffer = self._buffer.lstrip()
            if self._buffer:
                self._state = "answer"
            return bool(self._buffer)
        parts.append((False, self._buffer))
        self._buffer = ""
        return False


@dataclass(slots=True)
class _Call:
    """One tool call being assembled from fragments."""

    position: int
    id: str = ""
    name: str = ""
    fragments: list[str] = field(default_factory=list)
    #: Set when the backend sent the arguments as an object instead of a JSON string.
    parsed: dict[str, Any] | None = None
    #: Fields of the tool call this adapter does not interpret (thought signatures, ...).
    extra: dict[str, Any] = field(default_factory=dict)
    announced: bool = False

    @property
    def raw_arguments(self) -> object:
        return self.parsed if self.parsed is not None else "".join(self.fragments)

    @property
    def empty(self) -> bool:
        return not (self.name or self.fragments or self.parsed)


class _Turn(NamedTuple):
    """A finished assistant turn: its answer text, its tool calls, and the backend's own
    message for replay."""

    text: str
    uses: list[ToolUseBlock]
    stored: dict[str, Any]


class StreamAccumulator:
    """Folds the chunks of one response into events and, at the end, a `ChatResponse`.

    Feed every chunk to `feed` and emit what it returns; then call `finish`, which returns
    the remaining events ending with `ResponseDone`, or raises `ProviderError` when the
    response is unusable (cut off before it finished, or abandoned by the backend).
    """

    def __init__(
        self, *, provider: str, model: str, mode: ToolMode, tools: Sequence[ToolSpec]
    ) -> None:
        self._provider = provider
        self._model = model
        self._mode: ToolMode = mode
        self._tools = {tool.name: tool for tool in tools}
        #: The reply may end in a text-protocol tool block. Without tools on offer there is
        #: nothing to call, and a block quoted in, say, a summary must stay text.
        self._reads_blocks = mode == "text" and bool(tools)
        self._think = ThinkSplitter()
        self._content: list[str] = []  # as received, for the stored message
        self._answer: list[str] = []  # content without inline reasoning
        self._thinking: list[str] = []
        self._reasoning: dict[str, list[str]] = {}
        self._extras: dict[str, Any] = {}
        self._calls: list[_Call] = []
        self._by_index: dict[int, _Call] = {}
        self._pending_prose = ""  # text protocol: might be the start of a tool block
        self._block_head: str | None = None  # text protocol: how the tool block begins
        self._block_id = ""
        self._usage: Mapping[str, Any] | None = None
        self._finish: str | None = None
        self._reported_model = ""

    # -- input ----------------------------------------------------------------------------------

    def feed(self, chunk: object) -> list[StreamEvent]:
        """Consume one chunk (also accepts a whole non-streamed completion)."""
        if not isinstance(chunk, Mapping):
            return []
        if isinstance(chunk.get("usage"), Mapping):
            self._usage = chunk["usage"]  # cumulative where repeated: the last one wins
        if isinstance(chunk.get("model"), str):
            self._reported_model = chunk["model"]
        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, Mapping):
            return []
        delta = choice.get("delta") or choice.get("message")
        events = self._delta(delta) if isinstance(delta, Mapping) else []
        if isinstance(choice.get("finish_reason"), str):
            self._finish = choice["finish_reason"]
        return events

    def _delta(self, delta: Mapping[str, Any]) -> list[StreamEvent]:
        events = self._reasoning_delta(delta)
        for key, value in delta.items():
            if value is None or key == "role" or key in REASONING_FIELDS:
                continue
            if key == "content":
                events += self._content_delta(_content_text(value))
            elif key == "tool_calls":
                events += self._tool_call_delta(value)
            elif key == "function_call":  # the pre-tools form of a single call
                events += self._tool_call_delta([{"index": 0, "function": value}])
            else:
                self._extras[key] = _merge(key, self._extras.get(key), value)
        return events

    def _reasoning_delta(self, delta: Mapping[str, Any]) -> list[StreamEvent]:
        shown = ""
        for name in REASONING_FIELDS:
            value = delta.get(name)
            if isinstance(value, str) and value:
                self._reasoning.setdefault(name, []).append(value)
                shown = shown or value  # backends that send two fields send the same text
        shown = shown or _details_text(delta.get("reasoning_details"))
        if not shown:
            return []
        self._thinking.append(shown)
        return [ThinkingDelta(shown)]

    def _content_delta(self, piece: str) -> list[StreamEvent]:
        if not piece:
            return []
        self._content.append(piece)
        return self._emit_content(self._think.feed(piece))

    def _emit_content(self, parts: list[tuple[bool, str]]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for is_reasoning, text in parts:
            if is_reasoning:
                self._thinking.append(text)
                events.append(ThinkingDelta(text))
            else:
                self._answer.append(text)
                events += self._answer_delta(text)
        return events

    def _answer_delta(self, text: str) -> list[StreamEvent]:
        if not self._reads_blocks:
            return [TextDelta(text)]
        if self._block_head is not None:
            return self._announce_block(text)
        prose, rest, found = _split_at(self._pending_prose + text, text_tools.OPEN)
        events: list[StreamEvent] = [TextDelta(prose)] if prose else []
        if not found:
            self._pending_prose = rest
            return events
        self._pending_prose, self._block_head = "", text_tools.OPEN
        return events + self._announce_block(rest)

    def _announce_block(self, text: str) -> list[StreamEvent]:
        """Text protocol: report the tool call as soon as the block's opening tag names it.

        The block itself is never streamed; only its first characters are looked at.
        """
        if self._block_id or self._block_head is None or len(self._block_head) > _HEAD_LIMIT:
            return []
        self._block_head += text
        name = text_tools.block_name(self._block_head)
        if name is None:
            return []
        self._block_id = _new_id()
        return [ToolCallStarted(index=0, id=self._block_id, name=name)]

    def _tool_call_delta(self, fragments: object) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for fragment in fragments if isinstance(fragments, list) else ():
            if not isinstance(fragment, Mapping):
                continue
            call = self._call_for(fragment)
            added = self._absorb(call, fragment)
            if self._mode == "sequential" and call.position > 0:
                continue  # one call per turn: later ones are dropped, so never announced
            if not call.announced and call.name:
                events += self._announce(call)
            elif call.announced and added:
                events.append(ToolCallDelta(index=call.position, partial_json=added))
        return events

    def _call_for(self, fragment: Mapping[str, Any]) -> _Call:
        """The call a fragment belongs to. ``index`` decides; ids only break ties."""
        index, call_id = fragment.get("index"), fragment.get("id") or ""
        if isinstance(index, int):
            call = self._by_index.get(index)
            if call is None or (call_id and call.id and call_id != call.id):
                # A new index, or an index reused for a different call.
                call = self._by_index[index] = self._new_call()
            return call
        known = next((c for c in self._calls if call_id and c.id == call_id), None)
        if known is not None:
            return known
        function = fragment.get("function")
        starts_call = call_id or (isinstance(function, Mapping) and function.get("name"))
        return self._new_call() if starts_call or not self._calls else self._calls[-1]

    def _new_call(self) -> _Call:
        call = _Call(position=len(self._calls))
        self._calls.append(call)
        return call

    def _absorb(self, call: _Call, fragment: Mapping[str, Any]) -> str:
        """Merge a fragment into its call; return the argument text it added."""
        added = ""
        for key, value in fragment.items():
            if value is None or key in ("index", "type"):
                continue
            if key == "id":
                call.id = call.id or str(value)
            elif key == "function" and isinstance(value, Mapping):
                call.name = call.name or str(value.get("name") or "")
                arguments = value.get("arguments")
                if isinstance(arguments, str) and arguments:
                    call.fragments.append(arguments)
                    added = arguments
                elif isinstance(arguments, dict):
                    call.parsed = {**(call.parsed or {}), **arguments}
            elif key != "function":
                call.extra[key] = _merge(key, call.extra.get(key), value)
        return added

    def _announce(self, call: _Call) -> list[StreamEvent]:
        call.announced = True
        call.id = call.id or _new_id()
        events: list[StreamEvent] = [
            ToolCallStarted(index=call.position, id=call.id, name=call.name)
        ]
        so_far = call.raw_arguments
        text = so_far if isinstance(so_far, str) else json.dumps(so_far, ensure_ascii=False)
        if text:
            events.append(ToolCallDelta(index=call.position, partial_json=text))
        return events

    # -- output ---------------------------------------------------------------------------------

    def finish(self) -> list[StreamEvent]:
        """The remaining events, ending with `ResponseDone`."""
        events = self._emit_content(self._think.flush())
        if self._finish is None:
            raise ProviderError(
                "the response stream ended before the model finished (no finish reason arrived)",
                kind="network",
                provider=self._provider,
            )
        if self._finish in _FAILED_FINISHES:
            raise ProviderError(
                f"the backend abandoned the response (finish reason {self._finish!r})",
                kind="overloaded",
                provider=self._provider,
            )
        turn = self._text_turn(events) if self._reads_blocks else self._native_turn(events)
        content: list[Block] = []
        thinking = "".join(self._thinking).strip()
        if thinking:
            content.append(ThinkingBlock(thinking))
        if turn.text.strip():
            content.append(TextBlock(turn.text))
        content.extend(turn.uses)
        native = stored_message(self._provider, self._model, self._mode, turn.stored)
        response = ChatResponse(
            message=Message(role="assistant", content=content, native=native),
            stop_reason=self._stop_reason(bool(turn.uses)),
            usage=_usage(self._usage),
            model=self._reported_model or self._model,
            detail=self._detail(),
        )
        return [*events, ResponseDone(response)]

    @property
    def _complete(self) -> bool:
        """The model finished of its own accord, so what it wrote is all it meant to write."""
        return self._finish in _CLEAN_FINISHES

    def _kept_calls(self) -> list[_Call]:
        """The tool calls that count: none after a refusal, one in sequential mode."""
        if self._finish == "content_filter":
            return []
        calls = [call for call in self._calls if not call.empty]
        return calls[:1] if self._mode == "sequential" else calls

    def _native_turn(self, events: list[StreamEvent]) -> _Turn:
        """Assemble a turn whose tool calls arrived as ``tool_calls``. Adds to `events` the
        announcement of any call that could not be announced while streaming."""
        uses: list[ToolUseBlock] = []
        wire_calls: list[dict[str, Any]] = []
        seen: set[str] = set()
        for call in self._kept_calls():
            if not call.announced:
                events += self._announce(call)
            if call.id in seen:
                call.id = _new_id()  # backends that give every call the same id
            seen.add(call.id)
            arguments = parse_arguments(call.raw_arguments, complete=self._complete)
            uses.append(ToolUseBlock(id=call.id, name=call.name, input=arguments))
            function = {
                "name": call.name,
                "arguments": wire_arguments(call.raw_arguments, arguments),
            }
            wire_calls.append(
                {"id": call.id, "type": "function", "function": function, **call.extra}
            )
        stored = self._stored("".join(self._content))
        stored.update(self._extras)
        if wire_calls:
            stored["tool_calls"] = wire_calls
        return _Turn("".join(self._answer), uses, stored)

    def _text_turn(self, events: list[StreamEvent]) -> _Turn:
        """Assemble a turn whose tool call, if any, is a text-protocol block in the reply."""
        if self._pending_prose:  # looked like the start of a block, but was not
            events.append(TextDelta(self._pending_prose))
        prose, block = text_tools.split_reply("".join(self._answer))
        if block is None or self._finish == "content_filter":
            return _Turn(prose, [], self._stored(prose))
        name, arguments = text_tools.parse_block(block, self._tools, complete=self._complete)
        if not self._block_id:
            self._block_id = _new_id()
            events.append(ToolCallStarted(index=0, id=self._block_id, name=name))
        # The model's own block is never shown: progress consumers get the parsed call.
        events.append(ToolCallDelta(index=0, partial_json=json.dumps(arguments)))
        use = ToolUseBlock(id=self._block_id, name=name, input=arguments)
        content = "\n\n".join(filter(None, [prose, text_tools.closed(block)]))
        return _Turn(prose, [use], self._stored(content))

    def _stored(self, content: str) -> dict[str, Any]:
        """The start of the message kept for replay: content plus reasoning as received."""
        stored: dict[str, Any] = {"role": "assistant", "content": content}
        stored.update({name: "".join(parts) for name, parts in self._reasoning.items()})
        return stored

    def _stop_reason(self, has_tool_calls: bool) -> StopReason:
        if self._finish == "length":
            return "max_tokens"
        if self._finish == "content_filter":
            return "refusal"
        if has_tool_calls:
            return "tool_use"  # whatever the finish reason says: "stop" is common here
        return "end_turn" if self._finish in _CLEAN_FINISHES else "other"

    def _detail(self) -> str:
        if self._finish == "content_filter":
            return "the provider's content filter stopped the response"
        if self._finish in _CLEAN_FINISHES or self._finish == "length":
            return ""
        return f"finish reason: {self._finish}"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _new_id() -> str:
    #: Generated ids stay out of the "call_..." namespace providers use for their own ids.
    return f"tool_{uuid.uuid4().hex[:16]}"


def _content_text(value: object) -> str:
    """Content as text: a string, or the text parts of a list of content parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part["text"]
            for part in value
            if isinstance(part, Mapping)
            and part.get("type", "text") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def _details_text(details: object) -> str:
    """Readable text of OpenRouter-style ``reasoning_details``, for display."""
    if not isinstance(details, list):
        return ""
    texts = (d.get("text") or d.get("summary") for d in details if isinstance(d, Mapping))
    return "".join(text for text in texts if isinstance(text, str))


def _merge(key: str, old: Any, new: Any) -> Any:
    """Combine a fragment of a field this adapter does not interpret with what came before."""
    if old is None:
        return new
    if isinstance(old, str) and isinstance(new, str):
        return old + new if key in _STREAMED_TEXT else new
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for name, value in new.items():
            if value is not None:
                merged[name] = _merge(name, merged.get(name), value)
        return merged
    if isinstance(old, list) and isinstance(new, list):
        return _merge_lists(old, new)
    return new


def _merge_lists(old: list[Any], new: list[Any]) -> list[Any]:
    """Append, except that entries carrying an ``index`` continue the entry with that index."""
    merged = list(old)
    for entry in new:
        index = entry.get("index") if isinstance(entry, dict) else None
        position = next(
            (
                i
                for i, existing in enumerate(merged)
                if index is not None
                and isinstance(existing, dict)
                and existing.get("index") == index
            ),
            None,
        )
        if position is None:
            merged.append(entry)
        else:
            merged[position] = _merge("", merged[position], entry)
    return merged


def _usage(raw: Mapping[str, Any] | None) -> Usage:
    """Token counts in Clankit's terms: cached prompt tokens are reported apart from the rest."""
    if not raw:
        return Usage()
    prompt = _count(raw.get("prompt_tokens"))
    details = raw.get("prompt_tokens_details")
    cached = _count(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    cached = min(prompt, cached or _count(raw.get("prompt_cache_hit_tokens")))
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=_count(raw.get("completion_tokens")),
        cache_read_tokens=cached,
    )


def _count(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0

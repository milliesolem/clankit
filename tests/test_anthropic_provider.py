"""The Anthropic adapter, driven through the real SDK against a mock HTTP transport.

Nothing here touches the network: `FakeAPI` stands in for the Messages API, answering with
server-sent-event bodies shaped like the real ones and recording every request it receives.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import anthropic
import httpx2
import pytest

from clankit.agent.session import Session
from clankit.config.schema import ModelEntry, ProviderConfig
from clankit.llm.anthropic_provider import AnthropicProvider, capabilities_for
from clankit.llm.provider import ProviderError
from clankit.llm.registry import build_provider
from clankit.llm.types import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelRef,
    NativePayload,
    ResponseDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from clankit.runtime import Runtime
from clankit.tools.base import Tool, ToolContext, ToolOutput

OPUS = "claude-opus-5-5"
HAIKU_4_5 = "claude-haiku-4-5"
SYSTEM = "You are a test agent."
FALLBACK_BETA = "server-side-fallback-2026-07-01"
ADAPTIVE = {"type": "adaptive", "display": "summarized"}
EPHEMERAL = {"type": "ephemeral"}
SIGNATURE = "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds+/Zm9=="
THOUGHT = "The user wants the README — read it first."
REQUEST_ID = "req_011CSHoEeqs5C35K2UUqR7Fy"

READ = ToolSpec(
    "read_file",
    "Read a file from the workspace.",
    {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
)
WRITE = ToolSpec(
    "write_file",
    "Write a file in the workspace.",
    {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)

#: The streaming example from the API documentation: a thinking block, then a tool call.
DOCUMENTED_STREAM = b"""\
event: message_start
data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-opus-5","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user is asking about"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" the weather in San Francisco."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \\"San Francisco, CA\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}

"""


# --------------------------------------------------------------------------------------------
# Server-sent events, as the Messages API writes them
# --------------------------------------------------------------------------------------------


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_start(model: str = OPUS, **usage: int) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_014p7gG3wDgGV9EUtLvnow3U",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 1, **usage},
        },
    }


def block(index: int, start: dict[str, Any], *deltas: dict[str, Any]) -> list[dict[str, Any]]:
    """One content block: its start event, its deltas, its stop event."""
    return [
        {"type": "content_block_start", "index": index, "content_block": start},
        *({"type": "content_block_delta", "index": index, "delta": d} for d in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def json_delta(fragment: str) -> dict[str, Any]:
    return {"type": "input_json_delta", "partial_json": fragment}


def text_block(index: int, *chunks: str) -> list[dict[str, Any]]:
    deltas = [{"type": "text_delta", "text": chunk} for chunk in chunks]
    return block(index, {"type": "text", "text": ""}, *deltas)


def thinking_block(index: int, *chunks: str, signature: str = SIGNATURE) -> list[dict[str, Any]]:
    """With no chunks this is what `display: "omitted"` produces: one empty delta."""
    deltas = [{"type": "thinking_delta", "thinking": chunk} for chunk in chunks or ("",)]
    deltas.append({"type": "signature_delta", "signature": signature})
    return block(index, {"type": "thinking", "thinking": "", "signature": ""}, *deltas)


def tool_block(index: int, call_id: str, name: str, *fragments: str) -> list[dict[str, Any]]:
    start = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
    return block(index, start, *map(json_delta, fragments))


def message_end(
    stop_reason: str = "end_turn", *, stop_details: dict[str, Any] | None = None, **usage: Any
) -> list[dict[str, Any]]:
    delta = {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details}
    return [
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 12, **usage}},
        {"type": "message_stop"},
    ]


def reply(
    *blocks: list[dict[str, Any]],
    model: str = OPUS,
    stop: str = "end_turn",
    start_usage: dict[str, int] | None = None,
    **end: Any,
) -> httpx2.Response:
    """A complete streamed response: message_start, a ping, the blocks, the ending."""
    events = [message_start(model, **(start_usage or {})), {"type": "ping"}]
    for content_block in blocks:
        events.extend(content_block)
    events.extend(message_end(stop, **end))
    return stream_response(sse(*events))


def stream_response(body: bytes) -> httpx2.Response:
    headers = {"content-type": "text/event-stream", "request-id": REQUEST_ID}
    return httpx2.Response(200, headers=headers, content=body)


def error_response(
    status: int,
    error_type: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
) -> httpx2.Response:
    error: dict[str, Any] = {"type": error_type, "message": message}
    if details:
        error["details"] = details
    body = {"type": "error", "error": error, "request_id": REQUEST_ID}
    return httpx2.Response(status, headers={"request-id": REQUEST_ID, **(headers or {})}, json=body)


class OpenStream(httpx2.AsyncByteStream):
    """A response body that sends `head` and then either stalls forever or fails."""

    def __init__(self, head: bytes, *, failure: Exception | None = None) -> None:
        self._head = head
        self._failure = failure
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        if self._failure is not None:
            raise self._failure
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def open_response(stream: OpenStream) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)


#: A response that got as far as its first text delta.
HALF_A_REPLY = sse(message_start(), *text_block(0, "Hel")[:2])


class FakeAPI:
    """Answers requests from a queue (the last answer repeats) and records what was asked."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self._replies: list[httpx2.Response | Exception] = []

    def queue(self, *replies: httpx2.Response | Exception) -> None:
        self._replies.extend(replies)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        # A response built from bytes is fully buffered, so serving it again is safe.
        answer = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


Connect = Callable[..., AnthropicProvider]


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
async def connect(api: FakeAPI) -> AsyncIterator[Connect]:
    """Build providers wired to the fake API; keyword arguments override the configuration."""
    made: list[AnthropicProvider] = []

    def _connect(**overrides: Any) -> AnthropicProvider:
        settings: dict[str, Any] = {"id": "anthropic", "kind": "anthropic", "api_key": "sk-test"}
        config = ProviderConfig(**{**settings, **overrides})
        client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(api.handle))
        made.append(AnthropicProvider(config, http_client=client))
        return made[-1]

    yield _connect
    for provider in made:
        await provider.aclose()


def ask(model: str = OPUS, **overrides: Any) -> ChatRequest:
    fields: dict[str, Any] = {"model": model, "system": SYSTEM, "messages": [Message.user("hi")]}
    return ChatRequest(**{**fields, **overrides})


async def run(provider: AnthropicProvider, request: ChatRequest) -> tuple[list, ChatResponse]:
    """Everything streamed before the end, and the final response.

    Asserts the shape every stream must have: exactly one `ResponseDone`, in last place.
    """
    *deltas, done = [event async for event in provider.stream(request)]
    assert isinstance(done, ResponseDone)
    assert not any(isinstance(event, ResponseDone) for event in deltas)
    return deltas, done.response


async def failure(provider: AnthropicProvider, request: ChatRequest) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        await provider.complete(request)
    return caught.value


# --------------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------------


async def test_text_streams_in_order_and_ends_with_one_response_done(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "Hel", "lo ", "there")))
    deltas, response = await run(connect(), ask())

    assert deltas == [TextDelta("Hel"), TextDelta("lo "), TextDelta("there")]
    assert response.message.role == "assistant"
    assert response.message.content == [TextBlock("Hello there")]
    assert response.message.native == NativePayload(
        kind="anthropic", model=OPUS, data=[{"type": "text", "text": "Hello there"}]
    )
    assert (response.stop_reason, response.model, response.detail) == ("end_turn", OPUS, "")


async def test_thinking_and_tool_use_in_one_response(api: FakeAPI, connect: Connect) -> None:
    api.queue(stream_response(DOCUMENTED_STREAM))
    weather = ToolSpec("get_weather", "Look up the weather.", {"type": "object"})
    deltas, response = await run(connect(), ask("claude-opus-5", tools=[weather]))

    call_id = "toolu_01T1x1fJ34qAmk2tNTrN7Up6"
    signature = "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds"
    thought = "The user is asking about the weather in San Francisco."
    assert deltas == [
        ThinkingDelta("The user is asking about"),
        ThinkingDelta(" the weather in San Francisco."),
        ToolCallStarted(index=0, id=call_id, name="get_weather"),
        ToolCallDelta(index=0, partial_json='{"location":'),
        ToolCallDelta(index=0, partial_json=' "San Francisco, CA"}'),
    ]
    assert response.stop_reason == "tool_use"
    assert response.message.content == [
        ThinkingBlock(thought),
        ToolUseBlock(call_id, "get_weather", {"location": "San Francisco, CA"}),
    ]
    assert response.message.native == NativePayload(
        kind="anthropic",
        model="claude-opus-5",
        data=[
            {"type": "thinking", "thinking": thought, "signature": signature},
            {
                "type": "tool_use",
                "id": call_id,
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ],
    )
    assert response.usage == Usage(input_tokens=472, output_tokens=89)


async def test_parallel_tool_calls_are_numbered_in_call_order(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(
        reply(
            thinking_block(0, "Two files."),
            text_block(1, "Reading both."),
            tool_block(2, "toolu_a", "read_file", '{"path": "a"}'),
            tool_block(3, "toolu_b", "read_file", '{"pa', 'th": "b"}'),
            stop="tool_use",
        )
    )
    deltas, response = await run(connect(), ask(tools=[READ]))

    assert deltas == [
        ThinkingDelta("Two files."),
        TextDelta("Reading both."),
        ToolCallStarted(0, "toolu_a", "read_file"),
        ToolCallDelta(0, '{"path": "a"}'),
        ToolCallStarted(1, "toolu_b", "read_file"),
        ToolCallDelta(1, '{"pa'),
        ToolCallDelta(1, 'th": "b"}'),
    ]
    assert response.message.tool_uses == [
        ToolUseBlock("toolu_a", "read_file", {"path": "a"}),
        ToolUseBlock("toolu_b", "read_file", {"path": "b"}),
    ]


async def test_withheld_thinking_is_kept_but_not_streamed(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(thinking_block(0), text_block(1, "Done.")))
    deltas, response = await run(connect(), ask())

    assert deltas == [TextDelta("Done.")]
    assert response.message.content == [ThinkingBlock(""), TextBlock("Done.")]
    assert response.message.native is not None
    assert response.message.native.data[0] == {
        "type": "thinking",
        "thinking": "",
        "signature": SIGNATURE,
    }


async def test_unknown_blocks_survive_in_the_native_payload_only(
    api: FakeAPI, connect: Connect
) -> None:
    redacted = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4="}
    search = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}
    found = {
        "type": "web_search_result",
        "url": "https://example.com",
        "title": "Example",
        "encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1",
        "page_age": None,
    }
    result = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [found]}
    novel = {"type": "block_from_the_future", "payload": {"nested": [1, 2]}}
    api.queue(
        reply(
            block(0, redacted),
            block(1, search, json_delta('{"query": "kdl"}')),
            block(2, result),
            block(3, novel),
            text_block(4, "Found it."),
        )
    )
    deltas, response = await run(connect(), ask())

    assert deltas == [TextDelta("Found it.")]  # a server tool is not ours to run or report
    assert response.message.content == [TextBlock("Found it.")]
    assert response.message.native is not None
    assert response.message.native.data == [
        redacted,
        {**search, "input": {"query": "kdl"}},
        result,
        novel,
        {"type": "text", "text": "Found it."},
    ]


async def test_extra_fields_on_known_blocks_round_trip(api: FakeAPI, connect: Connect) -> None:
    start = {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "read_file",
        "input": {},
        "caller": {"type": "direct"},
    }
    api.queue(reply(block(0, start, json_delta('{"path": "a"}')), stop="tool_use"))
    response = await connect().complete(ask(tools=[READ]))

    assert response.message.native is not None
    assert response.message.native.data == [{**start, "input": {"path": "a"}}]


# --------------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------------


async def test_native_turn_is_replayed_verbatim(api: FakeAPI, connect: Connect) -> None:
    api.queue(
        reply(
            thinking_block(0, THOUGHT[:10], THOUGHT[10:]),
            tool_block(1, "toolu_01", "read_file", '{"path": "README.md"}'),
            stop="tool_use",
        ),
        reply(text_block(0, "It is empty.")),
    )
    provider = connect()
    first = await provider.complete(ask(tools=[READ]))
    assert first.message.native is not None
    history = [
        Message.user("hi"),
        first.message,
        Message.tool_results([ToolResultBlock("toolu_01", "# Readme")]),
    ]
    await provider.complete(ask(tools=[READ], messages=history))

    sent = api.bodies[1]["messages"]
    assert sent[1] == {"role": "assistant", "content": first.message.native.data}
    assert sent[1]["content"][0] == {
        "type": "thinking",
        "thinking": THOUGHT,
        "signature": SIGNATURE,
    }
    # Byte for byte: the signature and the reasoning text appear in the body as they arrived.
    raw = api.requests[1].content
    assert SIGNATURE.encode() in raw
    assert json.dumps(THOUGHT, ensure_ascii=False).encode() in raw
    assert sent[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "# Readme"}],
    }


async def test_native_turn_wins_over_neutral_blocks_even_across_claude_models(
    api: FakeAPI, connect: Connect
) -> None:
    native = [
        {"type": "thinking", "thinking": "", "signature": SIGNATURE},
        {"type": "text", "text": "From Fable."},
    ]
    turn = Message(
        role="assistant",
        content=[TextBlock("a neutral view that must not be used")],
        native=NativePayload("anthropic", "claude-fable-5-2", native),
    )
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask("claude-haiku-5-5", messages=[Message.user("hi"), turn]))

    assert api.bodies[0]["messages"][1] == {"role": "assistant", "content": native}


async def test_turn_from_another_provider_is_rebuilt_without_its_reasoning(
    api: FakeAPI, connect: Connect
) -> None:
    foreign = Message(
        role="assistant",
        content=[
            ThinkingBlock("reasoning that is not portable"),
            TextBlock("Checking."),
            TextBlock("  "),
            ToolUseBlock("call:0.read", "read_file", {"path": "a.txt"}),
        ],
        native=NativePayload("openai-compatible", "deepseek-chat", {"role": "assistant"}),
    )
    history = [
        Message.user("hi"),
        foreign,
        Message.tool_results([ToolResultBlock("call:0.read", "contents")]),
        Message.assistant("Plain text, no native payload."),
        Message.user("thanks"),
    ]
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask(messages=history))

    sent = api.bodies[0]["messages"]
    # The id is rewritten to the characters this API accepts, on the call and on its result.
    call = {
        "type": "tool_use",
        "id": "call_0_read",
        "name": "read_file",
        "input": {"path": "a.txt"},
    }
    assert sent[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Checking."}, call],
    }
    assert sent[2]["content"][0]["tool_use_id"] == "call_0_read"
    assert sent[3] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Plain text, no native payload."}],
    }


async def test_tool_results_stay_in_one_user_message_and_keep_the_error_flag(
    api: FakeAPI, connect: Connect
) -> None:
    ids = ("toolu_a", "toolu_b", "toolu_c")
    calls = Message("assistant", [ToolUseBlock(i, "read_file", {"path": i}) for i in ids])
    results = Message.tool_results(
        [
            ToolResultBlock("toolu_a", "file contents"),
            ToolResultBlock("toolu_b", "No such file: b.txt", is_error=True),
            ToolResultBlock("toolu_c", "", is_error=True),
        ],
        note="Also check c.txt",
    )
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask(messages=[Message.user("go"), calls, results]))

    sent = api.bodies[0]["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "user"]
    failed = {"type": "tool_result", "tool_use_id": "toolu_b", "content": "No such file: b.txt"}
    # The API rejects an empty error result, so an empty one is given a marker.
    silent = {"type": "tool_result", "tool_use_id": "toolu_c", "content": "(no output)"}
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_a", "content": "file contents"},
        {**failed, "is_error": True},
        {**silent, "is_error": True},
        {"type": "text", "text": "Also check c.txt"},
    ]


async def test_messages_without_content_are_not_sent(api: FakeAPI, connect: Connect) -> None:
    refused = Message(role="assistant", content=[], native=NativePayload("anthropic", OPUS, []))
    history = [Message.user("first"), refused, Message.user("   "), Message.user("second")]
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask(messages=history))

    assert api.bodies[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]


# --------------------------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------------------------


async def test_request_for_a_current_model(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask(tools=[WRITE, READ], effort="xhigh"))

    request, body = api.requests[0], api.bodies[0]
    assert (request.method, str(request.url.copy_with(query=None))) == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert request.headers["authorization"] == "Bearer sk-test"
    assert (body["model"], body["stream"], body["max_tokens"]) == (OPUS, True, 64_000)
    assert body["thinking"] == ADAPTIVE
    assert body["output_config"] == {"effort": "xhigh"}
    # Caching: one breakpoint after tools + system, and automatic caching for the conversation.
    assert body["system"] == [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]
    assert body["cache_control"] == EPHEMERAL
    assert body["tools"] == [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
            "eager_input_streaming": True,
        }
        for spec in (READ, WRITE)
    ]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert not {"temperature", "top_p", "top_k", "tool_choice", "stop_sequences"} & body.keys()


async def test_system_and_tools_are_byte_identical_across_calls(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "ok")))
    provider = connect()
    await provider.complete(ask(tools=[WRITE, READ]))
    longer = [Message.user("hi"), Message.assistant("ok"), Message.user("and now?")]
    await provider.complete(ask(tools=[READ, WRITE], messages=longer, effort="low"))

    def rendered(index: int, key: str) -> str:
        return json.dumps(api.bodies[index][key])  # key order survives the round trip

    assert rendered(0, "system") == rendered(1, "system")
    assert rendered(0, "tools") == rendered(1, "tools")
    assert json.dumps(api.bodies[0]["messages"][0]) == json.dumps(api.bodies[1]["messages"][0])


async def test_empty_system_and_no_tools_are_left_out(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask(system="", tools=[]))
    assert not {"system", "tools"} & api.bodies[0].keys()


def budget(tokens: int, **extra: Any) -> dict[str, Any]:
    return {"type": "enabled", "budget_tokens": tokens, **extra}


@pytest.mark.parametrize(
    ("model", "effort", "thinking", "sent_effort", "max_tokens"),
    [
        # Current models: adaptive thinking with visible summaries, every effort level.
        ("claude-opus-5-5", "xhigh", ADAPTIVE, "xhigh", 64_000),
        ("claude-fable-5-2", None, ADAPTIVE, None, 64_000),
        ("claude-haiku-5-5", "max", ADAPTIVE, "max", 64_000),
        # The 4.6 generation has no "xhigh": the nearest level below it is sent. It returns
        # reasoning summaries unasked, so no display setting is needed either.
        ("claude-sonnet-4-6", "xhigh", {"type": "adaptive"}, "high", 64_000),
        ("claude-opus-4-6", "max", {"type": "adaptive"}, "max", 64_000),
        # Older models: a thinking budget instead of adaptive, effort only where it exists.
        ("claude-opus-4-5", "max", budget(32_768), "high", 64_000),
        ("claude-haiku-4-5-20251001", "low", budget(2_048), None, 64_000),
        ("claude-haiku-4-5", None, budget(16_384), None, 64_000),
        # Unknown models: nothing optional, and an output ceiling they are sure to accept.
        ("claude-opus-6", "high", None, None, 64_000),
        ("some-local-model", "high", None, None, 8_192),
    ],
)
async def test_parameters_follow_the_capability_table(
    api: FakeAPI,
    connect: Connect,
    model: str,
    effort: str | None,
    thinking: dict[str, Any] | None,
    sent_effort: str | None,
    max_tokens: int,
) -> None:
    api.queue(reply(text_block(0, "ok"), model=model))
    await connect(options={"refusal-fallback": False}).complete(ask(model, effort=effort))

    body = api.bodies[0]
    assert body.get("thinking") == thinking
    assert body.get("output_config") == ({"effort": sent_effort} if sent_effort else None)
    assert body["max_tokens"] == max_tokens
    assert not {"temperature", "top_p", "top_k", "tool_choice", "fallbacks"} & body.keys()
    assert "anthropic-beta" not in api.requests[0].headers


def test_capability_patterns_put_specific_models_before_their_family() -> None:
    assert capabilities_for("claude-opus-5-5").off == ""
    assert capabilities_for("claude-opus-5").off == "disabled"
    assert capabilities_for("claude-sonnet-5-5").off == "between_tools"
    assert capabilities_for("claude-sonnet-5").off_caps_effort is False
    assert capabilities_for("claude-opus-4-5-20251101").thinking == "budget"
    assert capabilities_for("gpt-4o") == capabilities_for("anything-else")


@pytest.mark.parametrize(
    ("model", "thinking", "sent_effort"),
    [
        ("claude-opus-4-8", {"type": "disabled"}, "max"),
        # Disabling is only accepted up to "high" on these, so the effort comes down with it.
        ("claude-opus-5", {"type": "disabled"}, "high"),
        ("claude-sonnet-5-5", {"type": "between_tools"}, "high"),
        # No off switch to send: these either cannot stop thinking or think only when asked.
        ("claude-opus-5-5", None, "max"),
        (HAIKU_4_5, None, None),
        ("some-local-model", None, None),
    ],
)
async def test_thinking_off_uses_the_switch_each_model_accepts(
    api: FakeAPI,
    connect: Connect,
    model: str,
    thinking: dict[str, Any] | None,
    sent_effort: str | None,
) -> None:
    api.queue(reply(text_block(0, "ok"), model=model))
    await connect(options={"thinking": "off"}).complete(ask(model, effort="max"))

    assert api.bodies[0].get("thinking") == thinking
    assert api.bodies[0].get("output_config", {}).get("effort") == sent_effort


async def test_thinking_options_reach_models_the_table_does_not_know(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "ok")))
    provider = connect(options={"thinking": "adaptive", "thinking-display": "omitted"})
    for model in ("claude-opus-6", OPUS, "claude-opus-4-6", HAIKU_4_5):
        await provider.complete(ask(model))

    assert [body["thinking"] for body in api.bodies] == [
        {"type": "adaptive", "display": "omitted"},
        {"type": "adaptive", "display": "omitted"},
        {"type": "adaptive", "display": "omitted"},
        # A model known to reject adaptive keeps the form it accepts.
        budget(16_384, display="omitted"),
    ]


async def test_max_tokens_precedence(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok")))
    declared = [ModelEntry("local", max_output=32_000), ModelEntry("tiny", max_output=2_000)]
    plain = connect(models=declared)
    await plain.complete(ask("local"))
    await plain.complete(ask("tiny"))
    configured = connect(models=declared, options={"max-tokens": 20_000})
    await configured.complete(ask("local"))
    await configured.complete(ask("local", max_tokens=512))

    assert [body["max_tokens"] for body in api.bodies] == [32_000, 2_000, 20_000, 512]


async def test_thinking_budget_always_fits_inside_max_tokens(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "ok"), model=HAIKU_4_5))
    provider = connect()
    await provider.complete(ask(HAIKU_4_5, max_tokens=4_000))
    await provider.complete(ask(HAIKU_4_5, max_tokens=1_500))

    assert api.bodies[0]["thinking"] == budget(2_976)
    assert "thinking" not in api.bodies[1]  # no room for the smallest budget and an answer


@pytest.mark.parametrize(
    ("options", "named"),
    [
        ({"thinking": "on"}, "'thinking'"),
        ({"thinking-display": "full"}, "'thinking-display'"),
        ({"max-tokens": 0}, "'max-tokens'"),
        ({"max-tokens": "lots"}, "'max-tokens'"),
        ({"refusal-fallback": "maybe"}, "'refusal-fallback'"),
    ],
)
def test_invalid_options_are_rejected_with_the_option_named(
    options: dict[str, Any], named: str
) -> None:
    config = ProviderConfig(id="anthropic", kind="anthropic", api_key="sk-test", options=options)
    with pytest.raises(ProviderError) as caught:
        AnthropicProvider(config)

    assert caught.value.kind == "bad_request"
    assert named in caught.value.message


async def test_options_written_as_text_are_understood(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok")))
    await connect(options={"max-tokens": "9000", "refusal-fallback": "off"}).complete(ask())

    assert api.bodies[0]["max_tokens"] == 9_000
    assert "fallbacks" not in api.bodies[0]


async def test_unknown_effort_level_is_refused_before_any_request(
    api: FakeAPI, connect: Connect
) -> None:
    error = await failure(connect(), ask(effort="extreme"))

    assert error.kind == "bad_request"
    assert "extreme" in error.message and "xhigh" in error.message
    assert api.requests == []


async def test_other_endpoints_get_only_widely_supported_fields(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "ok")))
    provider = connect(id="gateway", base_url="https://llm.example/anthropic")
    replayed = [{"type": "text", "text": "Sure."}]
    turn = Message("assistant", [TextBlock("Sure.")], NativePayload("anthropic", OPUS, replayed))
    history = [Message.user("hi"), turn, Message.user("go on")]
    await provider.complete(ask(tools=[READ], messages=history))

    request, body = api.requests[0], api.bodies[0]
    assert str(request.url) == "https://llm.example/anthropic/v1/messages"
    assert "anthropic-beta" not in request.headers
    assert not {"cache_control", "fallbacks"} & body.keys()
    assert "eager_input_streaming" not in body["tools"][0]
    # Explicit breakpoints instead of automatic caching: the system prompt and the tail.
    assert body["system"][0]["cache_control"] == EPHEMERAL
    assert body["messages"][-1]["content"] == [
        {"type": "text", "text": "go on", "cache_control": EPHEMERAL}
    ]
    assert body["messages"][1]["content"] == replayed
    assert replayed == [{"type": "text", "text": "Sure."}]  # replayed content is never mutated


# --------------------------------------------------------------------------------------------
# Refusal fallback
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    ],
)
async def test_fallback_is_requested_for_models_that_support_it(
    api: FakeAPI, connect: Connect, model: str
) -> None:
    api.queue(reply(text_block(0, "ok"), model=model))
    await connect().complete(ask(model, tools=[READ]))

    assert api.bodies[0]["fallbacks"] == "default"
    assert api.requests[0].headers["anthropic-beta"] == FALLBACK_BETA
    assert api.bodies[0]["tools"][0]["name"] == "read_file"


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-5-5",  # has refusals but no fallback: a fallback list is rejected
        "claude-opus-4-8",
        "claude-mythos-5-2",
        "claude-opus-5-5-preview",  # only the exact ids qualify
        "some-local-model",
    ],
)
async def test_fallback_is_never_sent_to_other_models(
    api: FakeAPI, connect: Connect, model: str
) -> None:
    api.queue(reply(text_block(0, "ok"), model=model))
    await connect().complete(ask(model))

    assert "fallbacks" not in api.bodies[0]
    assert "anthropic-beta" not in api.requests[0].headers


async def test_fallback_can_be_switched_off(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok")))
    await connect(options={"refusal-fallback": False}).complete(ask())

    assert "fallbacks" not in api.bodies[0]
    assert "anthropic-beta" not in api.requests[0].headers


async def test_fallback_is_dropped_for_good_when_the_api_turns_it_down(
    api: FakeAPI, connect: Connect
) -> None:
    unknown_beta = f"Unexpected value(s) `{FALLBACK_BETA}` for the `anthropic-beta` header."
    api.queue(
        error_response(400, "invalid_request_error", unknown_beta),
        reply(text_block(0, "ok")),
    )
    provider = connect()
    deltas, response = await run(provider, ask())
    await provider.complete(ask())

    assert (deltas, response.message.text) == ([TextDelta("ok")], "ok")
    assert ["fallbacks" in body for body in api.bodies] == [True, False, False]
    assert ["anthropic-beta" in r.headers for r in api.requests] == [True, False, False]


async def test_an_unrelated_rejection_is_not_retried(api: FakeAPI, connect: Connect) -> None:
    api.queue(error_response(400, "invalid_request_error", "messages: roles must alternate"))
    error = await failure(connect(), ask())

    assert error.kind == "bad_request"
    assert len(api.requests) == 1


def fallback_reply() -> httpx2.Response:
    """Opus 5.5 is declined part-way through; Opus 4.8 takes over in the same response."""
    marker = {
        "type": "fallback",
        "from": {"model": OPUS},
        "to": {"model": "claude-opus-4-8"},
        "trigger": {"type": "refusal", "category": "cyber"},
    }
    cache = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    iterations = [
        {"type": "message", "input_tokens": 25, "output_tokens": 9, **cache},
        {
            "type": "fallback_message",
            "model": "claude-opus-4-8",
            "input_tokens": 31,
            "output_tokens": 40,
            **cache,
        },
    ]
    return reply(
        thinking_block(0, "Scan the ports.", signature="c2lnLWRlY2xpbmVk"),
        text_block(1, "I'll start by"),
        tool_block(2, "toolu_declined", "read_file", '{"path": "/etc/'),
        block(3, marker),
        thinking_block(4, "A safer approach.", signature="c2lnLWZhbGxiYWNr"),
        text_block(5, "Here is a safe way."),
        tool_block(6, "toolu_ok", "read_file", '{"path": "notes.md"}'),
        stop="tool_use",
        iterations=iterations,
    )


async def test_fallback_answer_keeps_only_what_may_be_replayed(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(fallback_reply())
    response = await connect().complete(ask(tools=[READ]))

    assert (response.model, response.stop_reason) == ("claude-opus-4-8", "tool_use")
    assert "claude-opus-4-8" in response.detail and OPUS in response.detail
    # The declined model's reasoning and unfinished tool call are gone; its text, the marker
    # and everything the fallback model wrote remain.
    native = response.message.native
    assert native is not None and native.model == "claude-opus-4-8"
    assert [b["type"] for b in native.data] == ["text", "fallback", "thinking", "text", "tool_use"]
    assert native.data[1]["to"] == {"model": "claude-opus-4-8"}
    assert response.message.content == [
        TextBlock("I'll start by"),
        ThinkingBlock("A safer approach."),
        TextBlock("Here is a safe way."),
        ToolUseBlock("toolu_ok", "read_file", {"path": "notes.md"}),
    ]


async def test_fallback_marker_in_history_keeps_the_beta_named(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(fallback_reply(), reply(text_block(0, "ok"), model="claude-haiku-5-5"))
    provider = connect()
    first = await provider.complete(ask(tools=[READ]))
    assert first.message.native is not None
    history = [
        Message.user("hi"),
        first.message,
        Message.tool_results([ToolResultBlock("toolu_ok", "notes")]),
    ]
    # The conversation moves to a model without fallback support.
    await provider.complete(ask("claude-haiku-5-5", tools=[READ], messages=history))

    assert api.bodies[1]["messages"][1]["content"] == first.message.native.data
    assert api.requests[1].headers["anthropic-beta"] == FALLBACK_BETA
    assert "fallbacks" not in api.bodies[1]


# --------------------------------------------------------------------------------------------
# Stop reasons and usage
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api_reason", "stop_reason"),
    [
        ("end_turn", "end_turn"),
        ("stop_sequence", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("pause_turn", "pause"),
        ("a_reason_from_the_future", "other"),
    ],
)
async def test_stop_reasons(
    api: FakeAPI, connect: Connect, api_reason: str, stop_reason: str
) -> None:
    api.queue(reply(text_block(0, "Partial"), stop=api_reason))
    response = await connect().complete(ask())

    assert response.stop_reason == stop_reason
    assert response.message.text == "Partial"
    assert ("a_reason_from_the_future" in response.detail) == (stop_reason == "other")


async def test_refusal_carries_an_explanation_and_no_content(
    api: FakeAPI, connect: Connect
) -> None:
    details = {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request could enable cyber harm.",
    }
    api.queue(
        reply(
            thinking_block(0, "Hmm."),
            text_block(1, "Sure, the exploit"),
            tool_block(2, "toolu_cut", "write_file", '{"path": "x.py", "cont'),
            stop="refusal",
            stop_details=details,
        )
    )
    _, response = await run(connect(), ask(tools=[WRITE]))

    assert response.stop_reason == "refusal"
    assert "cyber" in response.detail and "could enable cyber harm" in response.detail
    # Partial output of a declined turn is neither acted on nor replayed.
    assert response.message.content == []
    assert response.message.native == NativePayload("anthropic", OPUS, [])


async def test_refusal_without_details_still_explains_itself(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(stop="refusal"))
    response = await connect().complete(ask())

    assert response.stop_reason == "refusal"
    assert response.detail.startswith("The model declined this request.")


async def test_refusal_by_the_whole_fallback_chain_says_so(api: FakeAPI, connect: Connect) -> None:
    tried = {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 9}
    details = {"type": "refusal", "category": "bio", "explanation": None}
    api.queue(reply(stop="refusal", stop_details=details, iterations=[tried]))
    response = await connect().complete(ask())

    assert response.stop_reason == "refusal"
    assert "bio" in response.detail and "fallback model was tried" in response.detail


async def test_filled_context_window_is_reported_as_an_overflow(
    api: FakeAPI, connect: Connect
) -> None:
    api.queue(reply(text_block(0, "Cut o"), stop="model_context_window_exceeded"))
    error = await failure(connect(), ask())

    assert error.kind == "context_overflow"
    assert not error.retryable


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(sse(message_start(), *text_block(0, "Hel")), id="no-stop-reason"),
        pytest.param(HALF_A_REPLY, id="cut-mid-block"),
        pytest.param(b"", id="empty"),
        pytest.param(b": keep-alive\n\n", id="comments-only"),
    ],
)
async def test_stream_that_ends_early_is_a_retryable_failure(
    api: FakeAPI, connect: Connect, body: bytes
) -> None:
    api.queue(stream_response(body))
    events: list[StreamEvent] = []
    with pytest.raises(ProviderError) as caught:
        async for event in connect().stream(ask()):
            events.append(event)

    assert (caught.value.kind, caught.value.retryable) == ("network", True)
    assert not any(isinstance(event, ResponseDone) for event in events)


async def test_usage_keeps_cached_tokens_out_of_input_tokens(
    api: FakeAPI, connect: Connect
) -> None:
    start = {"cache_read_input_tokens": 4_000, "cache_creation_input_tokens": 300}
    api.queue(reply(text_block(0, "ok"), start_usage=start, output_tokens=77))
    response = await connect().complete(ask())

    assert response.usage == Usage(
        input_tokens=25, output_tokens=77, cache_read_tokens=4_000, cache_write_tokens=300
    )
    assert response.usage.prompt_tokens == 4_325


async def test_final_usage_overrides_the_opening_estimate(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok"), input_tokens=40))
    response = await connect().complete(ask())
    assert response.usage == Usage(input_tokens=40, output_tokens=12)


# --------------------------------------------------------------------------------------------
# Tool input
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fragments",
    [
        pytest.param(("[1, ", "2]"), id="array"),
        pytest.param(('"README.md"',), id="string"),
        pytest.param(('{"path": "a.txt"}', " trailing"), id="trailing-text"),
        pytest.param(('{"path": "a.txt", "cont', 'ent": "unterminated'), id="cut-off"),
    ],
)
async def test_tool_input_that_is_not_a_json_object_is_returned_raw(
    api: FakeAPI, connect: Connect, fragments: tuple[str, ...]
) -> None:
    api.queue(
        reply(
            tool_block(0, "toolu_ok", "read_file", '{"path": "b.txt"}'),
            tool_block(1, "toolu_bad", "write_file", *fragments),
            stop="tool_use",
        )
    )
    response = await connect().complete(ask(tools=[READ, WRITE]))

    # The call is kept, so the agent can answer it with an error result.
    raw = "".join(fragments)
    assert response.message.tool_uses == [
        ToolUseBlock("toolu_ok", "read_file", {"path": "b.txt"}),
        ToolUseBlock("toolu_bad", "write_file", {"_invalid_json": raw}),
    ]
    # What goes back to the API is still a valid object.
    assert response.message.native is not None
    assert response.message.native.data[1]["input"] == {"_invalid_json": raw}


async def test_tool_input_cut_off_by_the_output_limit_is_not_mistaken_for_complete(
    api: FakeAPI, connect: Connect
) -> None:
    raw = '{"path": "big.txt", "content": "line 1\\nline 2'
    api.queue(reply(tool_block(0, "toolu_01", "write_file", raw), stop="max_tokens"))
    response = await connect().complete(ask(tools=[WRITE]))

    assert response.stop_reason == "max_tokens"
    assert response.message.tool_uses == [
        ToolUseBlock("toolu_01", "write_file", {"_invalid_json": raw})
    ]


async def test_tool_call_without_arguments_has_empty_input(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(tool_block(0, "toolu_01", "list_files"), stop="tool_use"))
    response = await connect().complete(ask())
    assert response.message.tool_uses == [ToolUseBlock("toolu_01", "list_files", {})]


async def test_tool_input_delivered_whole_with_the_block_is_kept(
    api: FakeAPI, connect: Connect
) -> None:
    start = {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "a"}}
    api.queue(reply(block(0, start), stop="tool_use"))
    deltas, response = await run(connect(), ask(tools=[READ]))

    assert deltas == [ToolCallStarted(0, "toolu_01", "read_file")]
    assert response.message.tool_uses == [ToolUseBlock("toolu_01", "read_file", {"path": "a"})]


@pytest.mark.parametrize(
    "fragments",
    [
        pytest.param(('{"content": "raw\nnewline"}',), id="control-character"),
        pytest.param(("not json at all",), id="garbage"),
        pytest.param(('"READ', 'ME.md"'), id="fragmented-scalar"),
    ],
)
async def test_tool_input_the_sdk_cannot_follow_fails_the_call_retryably(
    api: FakeAPI, connect: Connect, fragments: tuple[str, ...]
) -> None:
    # The SDK abandons a response as soon as its tool input stops being a prefix of valid
    # JSON, so there is no complete turn to return: the request has to be made again.
    api.queue(reply(tool_block(0, "toolu_01", "write_file", *fragments), stop="tool_use"))
    error = await failure(connect(), ask(tools=[WRITE]))

    assert (error.kind, error.retryable) == ("network", True)
    assert "could not be decoded" in error.message


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------

TOO_LONG = "prompt is too long: 1200000 tokens > 1000000 maximum"
OVER_LIMIT = "input length and `max_tokens` exceed context limit: 990000 + 64000 > 1000000"
BUSY = "Too many requests"


@pytest.mark.parametrize(
    ("status", "error_type", "message", "headers", "kind", "retry_after"),
    [
        (401, "authentication_error", "invalid x-api-key", {}, "auth", None),
        (402, "billing_error", "Your credit balance is too low", {}, "auth", None),
        (403, "permission_error", "This key may not use that model", {}, "auth", None),
        (404, "not_found_error", "model: claude-nope-1", {}, "not_found", None),
        (400, "invalid_request_error", "messages: roles must alternate", {}, "bad_request", None),
        (422, "invalid_request_error", "tools.0.name: too long", {}, "bad_request", None),
        (400, "invalid_request_error", TOO_LONG, {}, "context_overflow", None),
        (400, "invalid_request_error", OVER_LIMIT, {}, "context_overflow", None),
        (
            413,
            "request_too_large",
            "Request exceeds the maximum size",
            {},
            "context_overflow",
            None,
        ),
        (429, "rate_limit_error", BUSY, {"retry-after": "7"}, "rate_limit", 7.0),
        (429, "rate_limit_error", BUSY, {"retry-after-ms": "1500"}, "rate_limit", 1.5),
        (429, "rate_limit_error", BUSY, {}, "rate_limit", None),
        (500, "api_error", "Internal server error", {}, "overloaded", None),
        (504, "timeout_error", "Gateway timeout", {}, "overloaded", None),
        (529, "overloaded_error", "Overloaded", {"retry-after": "3"}, "overloaded", 3.0),
    ],
)
async def test_api_errors_are_translated(
    api: FakeAPI,
    connect: Connect,
    status: int,
    error_type: str,
    message: str,
    headers: dict[str, str],
    kind: str,
    retry_after: float | None,
) -> None:
    api.queue(error_response(status, error_type, message, headers=headers))
    error = await failure(connect(), ask())

    assert (error.kind, error.status, error.retry_after) == (kind, status, retry_after)
    assert error.provider == "anthropic"
    assert error.retryable == (kind in ("rate_limit", "overloaded"))
    assert message in error.message and REQUEST_ID in error.message
    assert len(api.requests) == 1  # the SDK does not retry behind the agent loop's back


async def test_spend_limit_is_not_treated_as_a_passing_rate_limit(
    api: FakeAPI, connect: Connect
) -> None:
    details = {"error_code": "enforced_spend_limit_reached"}
    api.queue(error_response(429, "rate_limit_error", "Monthly limit reached", details=details))
    error = await failure(connect(), ask())

    assert (error.kind, error.retryable) == ("auth", False)


async def test_error_page_from_a_gateway_is_classified_by_status(
    api: FakeAPI, connect: Connect
) -> None:
    page = "<html><body>" + "bad gateway " * 200 + "</body></html>"
    api.queue(httpx2.Response(502, headers={"content-type": "text/html"}, content=page.encode()))
    error = await failure(connect(), ask())

    assert (error.kind, error.status) == ("overloaded", 502)
    assert len(error.message) < 400


async def test_error_event_in_the_middle_of_a_stream(api: FakeAPI, connect: Connect) -> None:
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    api.queue(stream_response(HALF_A_REPLY + sse(overloaded)))
    events: list[StreamEvent] = []
    with pytest.raises(ProviderError) as caught:
        async for event in connect().stream(ask()):
            events.append(event)

    assert events == [TextDelta("Hel")]
    assert (caught.value.kind, caught.value.status) == ("overloaded", None)
    assert caught.value.retryable
    assert "Overloaded" in caught.value.message


@pytest.mark.parametrize(
    ("raised", "mentions"),
    [
        (httpx2.ConnectError("All connection attempts failed"), "All connection attempts failed"),
        (httpx2.ReadTimeout("The read operation timed out"), "timed out"),
    ],
)
async def test_connection_failures_are_network_errors(
    api: FakeAPI, connect: Connect, raised: Exception, mentions: str
) -> None:
    api.queue(raised)
    error = await failure(connect(), ask())

    assert (error.kind, error.retryable) == ("network", True)
    assert mentions in error.message


async def test_connection_lost_in_the_middle_of_a_stream(api: FakeAPI, connect: Connect) -> None:
    body = OpenStream(HALF_A_REPLY, failure=httpx2.ReadError("Connection reset by peer"))
    api.queue(open_response(body))
    error = await failure(connect(), ask())

    assert error.kind == "network"
    assert "Connection reset by peer" in error.message
    assert body.closed


async def test_malformed_event_data_is_a_retryable_failure(api: FakeAPI, connect: Connect) -> None:
    api.queue(stream_response(b"event: message_start\ndata: {not json\n\n"))
    error = await failure(connect(), ask())
    assert (error.kind, error.retryable) == ("network", True)


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def model_entry(model_id: str, name: str, created: str, **extra: Any) -> dict[str, Any]:
    entry = {"type": "model", "id": model_id, "display_name": name, "created_at": created}
    return {**entry, "lifecycle": "active", **extra}


async def test_list_models_newest_first(api: FakeAPI, connect: Connect) -> None:
    listing = [
        model_entry(
            "claude-sonnet-4-5-20250929",
            "Claude Sonnet 4.5",
            "2025-09-29T00:00:00Z",
            lifecycle="deprecated",
            retires_at="2026-11-30T00:00:00Z",
            max_input_tokens=200_000,
            max_tokens=64_000,
        ),
        model_entry(
            "claude-opus-5-5",
            "Claude Opus 5.5",
            "2026-09-10T00:00:00Z",
            max_input_tokens=1_000_000,
            max_tokens=128_000,
        ),
        model_entry("claude-haiku-5-5", "Claude Haiku 5.5", "2026-10-02T00:00:00Z"),
    ]
    page = {"data": listing, "has_more": False, "first_id": "x", "last_id": "y"}
    api.queue(httpx2.Response(200, json=page))
    models = await connect().list_models()

    assert (api.requests[0].method, api.requests[0].url.path) == ("GET", "/v1/models")
    assert [m.id for m in models] == [
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-sonnet-4-5-20250929",
    ]
    haiku, opus, sonnet = models
    assert (opus.provider, opus.name) == ("anthropic", "Claude Opus 5.5")
    assert (opus.context_window, opus.max_output) == (1_000_000, 128_000)
    assert (opus.supports_tools, opus.free, opus.note) == (True, False, "")
    assert (haiku.context_window, haiku.max_output) == (None, None)  # unknown, not guessed
    assert sonnet.note == "Deprecated; retires 2026-11-30."


async def test_list_models_failure_is_a_provider_error(api: FakeAPI, connect: Connect) -> None:
    api.queue(error_response(401, "authentication_error", "invalid x-api-key"))
    with pytest.raises(ProviderError) as caught:
        await connect().list_models()
    assert caught.value.kind == "auth"


# --------------------------------------------------------------------------------------------
# Cancellation and lifecycle
# --------------------------------------------------------------------------------------------


async def test_closing_the_generator_closes_the_http_stream(api: FakeAPI, connect: Connect) -> None:
    body = OpenStream(HALF_A_REPLY)
    api.queue(open_response(body))
    stream = connect().stream(ask())

    assert await anext(stream) == TextDelta("Hel")
    assert not body.closed
    await stream.aclose()  # type: ignore[attr-defined]
    assert body.closed


async def test_cancelling_the_task_closes_the_http_stream(api: FakeAPI, connect: Connect) -> None:
    body = OpenStream(HALF_A_REPLY)
    api.queue(open_response(body))
    received = asyncio.Event()

    async def consume() -> None:
        async for _ in connect().stream(ask()):
            received.set()

    task = asyncio.create_task(consume())
    await received.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed


async def test_aclose_closes_the_http_client() -> None:
    client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(lambda _: None))
    config = ProviderConfig(id="anthropic", kind="anthropic", api_key="sk-test")
    provider = AnthropicProvider(config, http_client=client)

    await provider.aclose()
    assert client.is_closed


# --------------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------------


async def test_registry_builds_the_adapter_from_configuration() -> None:
    provider = build_provider(ProviderConfig(id="work", kind="anthropic", api_key="sk-test"))

    assert isinstance(provider, AnthropicProvider)
    assert (provider.id, provider.kind) == ("work", "anthropic")
    await provider.aclose()


def test_missing_api_key_is_an_auth_error_that_says_where_to_put_one() -> None:
    config = ProviderConfig(id="anthropic", kind="anthropic", api_key_env="ANTHROPIC_API_KEY")
    with pytest.raises(ProviderError) as caught:
        AnthropicProvider(config)

    assert caught.value.kind == "auth"
    assert "ANTHROPIC_API_KEY" in caught.value.message


async def test_api_key_from_the_environment(
    api: FakeAPI, connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    api.queue(reply(text_block(0, "ok")))
    await connect(api_key="", api_key_env="ANTHROPIC_API_KEY").complete(ask())

    assert api.requests[0].headers["authorization"] == "Bearer sk-from-env"


async def test_configuration_decides_the_endpoint_and_key_not_the_sdk_environment(
    api: FakeAPI, connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    api.queue(reply(text_block(0, "ok")))
    await connect().complete(ask())

    assert api.requests[0].url.host == "api.anthropic.com"
    assert api.requests[0].headers["authorization"] == "Bearer sk-test"


async def test_keyless_server_needs_no_key(api: FakeAPI, connect: Connect) -> None:
    api.queue(reply(text_block(0, "ok"), model="local"))
    provider = connect(id="local", api_key="", keyless=True, base_url="http://localhost:8080")
    response = await provider.complete(ask("local"))

    assert response.message.text == "ok"
    assert api.requests[0].url.host == "localhost"


# --------------------------------------------------------------------------------------------
# Under the agent loop
# --------------------------------------------------------------------------------------------


class Echo(Tool):
    name = "echo"
    description = "Echo text back."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(tool_input["text"])


async def test_agent_tool_round_keeps_the_prefix_stable(
    api: FakeAPI, connect: Connect, runtime: Runtime, session: Session
) -> None:
    runtime.providers.register(connect())
    api.queue(
        reply(
            thinking_block(0, THOUGHT),
            tool_block(1, "toolu_01", "echo", '{"text": "ping"}'),
            stop="tool_use",
        ),
        reply(text_block(0, "pong")),
    )
    node = session.new_node("test", role="root")
    agent = session.agent(node, system=SYSTEM, tools=[Echo()], model=ModelRef("anthropic", OPUS))
    result = await agent.run("say ping")

    assert (result.text, result.stop) == ("pong", "end_turn")
    first, second = api.bodies
    assert json.dumps(first["system"]) == json.dumps(second["system"])
    assert json.dumps(first["tools"]) == json.dumps(second["tools"])
    assert second["messages"][:1] == first["messages"]
    assert second["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": THOUGHT,
        "signature": SIGNATURE,
    }
    assert second["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "ping"}
    ]


async def test_agent_loop_does_the_retrying(
    api: FakeAPI, connect: Connect, runtime: Runtime, session: Session
) -> None:
    runtime.providers.register(connect())
    api.queue(
        error_response(429, "rate_limit_error", BUSY, headers={"retry-after-ms": "10"}),
        reply(text_block(0, "Here now.")),
    )
    node = session.new_node("test", role="root")
    agent = session.agent(node, system=SYSTEM, tools=[], model=ModelRef("anthropic", OPUS))
    result = await agent.run("hello?")

    assert result.text == "Here now."
    assert len(api.requests) == 2  # one failure, one retry: nothing retried twice over

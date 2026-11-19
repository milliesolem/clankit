"""The OpenAI-compatible adapter, driven through its public interface against scripted backends.

Every test talks to the real adapter and the real ``openai`` SDK; only the network is
replaced, by an `httpx2.MockTransport` that answers with the SSE bodies real backends send.
The long tails of argument parsing, the text tool protocol, error phrasings and model lists
are covered next to their modules (``test_openai_compat_*.py``); here each of them is shown
working through the adapter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest

from clankit.agent.session import Session
from clankit.config.schema import ProviderConfig
from clankit.llm.openai_compat import OpenAICompatibleProvider
from clankit.llm.provider import ProviderError
from clankit.llm.registry import build_provider
from clankit.llm.types import (
    ChatRequest,
    ChatResponse,
    Message,
    NativePayload,
    ResponseDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStarted,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from clankit.tools.base import Tool, ToolContext, ToolOutput

BASE = "https://api.example.test/v1"
Reply = Callable[[], httpx2.Response]
Handler = Callable[[httpx2.Request], httpx2.Response]

READ = ToolSpec(
    "read_file",
    "Read a file.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to read."}},
        "required": ["path"],
        "additionalProperties": False,
    },
)
WRITE = ToolSpec(
    "write_file",
    "Write a file.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "integer"},
            "backup": {"type": "boolean"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)
TOOLS = [READ, WRITE]


# --------------------------------------------------------------------------------------------
# A scripted backend
# --------------------------------------------------------------------------------------------


def chunk(delta: dict[str, Any] | None = None, finish: str | None = None, **extra: Any) -> dict:
    """One chat-completion chunk with a single choice."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "served-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }


def call(index: int | None, id: str | None = None, name: str | None = None, args: Any = None) -> dict:
    """A chunk carrying one tool-call fragment; whatever is None is left out of it."""
    function = {key: value for key, value in (("name", name), ("arguments", args)) if value is not None}
    fields = (("index", index), ("id", id), ("function", function or None))
    return chunk({"tool_calls": [{key: value for key, value in fields if value is not None}]})


def wire_call(id: str, name: str, arguments: str, **extra: Any) -> dict[str, Any]:
    """A tool call as it appears in a request's assistant message."""
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}, **extra}


def sse(*frames: dict | str, done: bool = True) -> bytes:
    """Encode chunks as an SSE body. Strings are written as they are (comments, raw frames)."""
    parts = [f if isinstance(f, str) else f"data: {json.dumps(f)}\n\n" for f in frames]
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def streaming(*frames: dict | str, done: bool = True) -> Reply:
    headers = {"content-type": "text/event-stream"}
    return lambda: httpx2.Response(200, headers=headers, content=sse(*frames, done=done))


def answering(status: int, payload: Any, headers: dict[str, str] | None = None) -> Reply:
    return lambda: httpx2.Response(status, json=payload, headers=headers)


def says(text: str) -> Reply:
    """A backend turn that simply answers with `text`."""
    return streaming(chunk({"role": "assistant", "content": text}), chunk(finish="stop"))


def typed(text: str, finish: str = "stop", size: int = 7) -> Reply:
    """A backend turn that streams `text` a few characters at a time."""
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    return streaming(*(chunk({"content": piece}) for piece in pieces), chunk(finish=finish))


def oai_error(message: str, **fields: Any) -> dict[str, Any]:
    kind = fields.pop("type", "invalid_request_error")
    return {"error": {"message": message, "type": kind, **fields}}


class Backend:
    """Answers each request with the next scripted reply (the last one repeats)."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return reply()

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.body["messages"]


Connect = Callable[..., OpenAICompatibleProvider]


@pytest.fixture
async def connect() -> AsyncIterator[Connect]:
    """Build adapters wired to a `Backend`; they are closed when the test ends."""
    made: list[OpenAICompatibleProvider] = []

    def _connect(
        backend: Handler,
        provider_id: str = "custom",
        *,
        base_url: str = BASE,
        api_key: str = "sk-test",
        **config: Any,
    ) -> OpenAICompatibleProvider:
        cfg = ProviderConfig(id=provider_id, base_url=base_url, api_key=api_key, **config)
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(backend))
        provider = OpenAICompatibleProvider(cfg, http_client=client)
        made.append(provider)
        return provider

    yield _connect
    for provider in made:
        await provider.aclose()


def ask(*messages: Message, model: str = "test-model", **fields: Any) -> ChatRequest:
    fields.setdefault("system", "You are terse.")
    return ChatRequest(model=model, messages=list(messages) or [Message.user("hi")], **fields)


async def run(
    provider: OpenAICompatibleProvider, request: ChatRequest
) -> tuple[list[StreamEvent], ChatResponse]:
    """Run one call; return the events before the final one, and the final response."""
    events = [event async for event in provider.stream(request)]
    assert [isinstance(event, ResponseDone) for event in events].count(True) == 1
    final = events[-1]
    assert isinstance(final, ResponseDone), "ResponseDone must be the last event"
    return events[:-1], final.response


async def failure(
    provider: OpenAICompatibleProvider, request: ChatRequest | None = None
) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        await run(provider, request or ask())
    assert caught.value.provider == provider.id
    return caught.value


def stored(response: ChatResponse) -> dict[str, Any]:
    """The backend's own assistant message as kept for replay."""
    native = response.message.native
    assert native is not None and native.kind == "openai-compatible"
    return native.data["message"]


def text_of(events: list[StreamEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


# --------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------


async def test_request_is_minimal_with_the_system_prompt_first(connect: Connect) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend), ask(Message.user("hello"), tools=[READ], max_tokens=512))

    request = backend.requests[0]
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    function = {"name": "read_file", "description": "Read a file.", "parameters": READ.input_schema}
    assert backend.body == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [{"type": "function", "function": function}],
        "max_tokens": 512,
    }


async def test_optional_fields_are_left_out_when_not_asked_for(connect: Connect) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend), ask(system=""))
    assert set(backend.body) == {"model", "messages", "stream", "stream_options"}
    assert backend.messages == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("provider_id", "key"),
    [
        ("custom", "max_tokens"),
        ("deepseek", "max_tokens"),
        ("groq", "max_completion_tokens"),
        ("cerebras", "max_completion_tokens"),
    ],
)
async def test_output_cap_uses_the_key_the_backend_expects(
    connect: Connect, provider_id: str, key: str
) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend, provider_id), ask(max_tokens=100))
    assert backend.body[key] == 100
    assert {"max_tokens", "max_completion_tokens"} & set(backend.body) == {key}


@pytest.mark.parametrize(
    ("provider_id", "effort", "expected"),
    [
        ("deepseek", "low", {"reasoning_effort": "low"}),
        ("deepseek", "medium", {"reasoning_effort": "high"}),
        ("deepseek", "xhigh", {"reasoning_effort": "high"}),
        ("deepseek", "max", {"reasoning_effort": "max"}),
        ("openrouter", "xhigh", {"reasoning": {"effort": "xhigh"}}),
        ("gemini", "medium", {"reasoning_effort": "medium"}),
        ("gemini", "max", {"reasoning_effort": "high"}),
        ("cerebras", "xhigh", {"reasoning_effort": "high"}),
        ("groq", "high", {}),
        ("ollama", "high", {}),
        ("custom", "high", {}),
        ("deepseek", None, {}),
    ],
)
async def test_effort_is_mapped_where_known_and_dropped_elsewhere(
    connect: Connect, provider_id: str, effort: Any, expected: dict[str, Any]
) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend, provider_id), ask(effort=effort))
    sent = {k: v for k, v in backend.body.items() if k in ("reasoning_effort", "reasoning")}
    assert sent == expected


async def test_ambient_openai_identity_never_reaches_another_provider(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-private")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-private")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "admin-private")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example.test/v1")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Corp: private\nAuthorization: Bearer ambient")
    backend = Backend(says("ok"), answering(200, {"data": []}))
    provider = connect(backend)
    await run(provider, ask())
    await provider.list_models()

    for request in backend.requests:
        assert request.url.host == "api.example.test"
        assert request.headers["authorization"] == "Bearer sk-test"
        assert not {"openai-organization", "openai-project", "x-corp"} & set(request.headers)
        assert "private" not in str(request.headers) and "ambient" not in str(request.headers)


async def test_keyless_server_gets_no_authorization_header(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient")
    backend = Backend(says("local"), answering(200, {"data": [{"id": "qwen3.8:27b"}]}))
    local = "http://localhost:11434/v1"
    provider = connect(backend, "ollama", base_url=local, api_key="", keyless=True)

    _, response = await run(provider, ask())
    models = await provider.list_models()
    assert response.message.text == "local" and [m.id for m in models] == ["qwen3.8:27b"]
    assert [str(r.url) for r in backend.requests] == [f"{local}/chat/completions", f"{local}/models"]
    assert all("authorization" not in request.headers for request in backend.requests)


async def test_keyless_server_started_with_a_key_gets_it(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMA_API_KEY", "from-env")
    backend = Backend(says("ok"))
    provider = connect(backend, "llamacpp", api_key="", keyless=True, api_key_env="LLAMA_API_KEY")
    await run(provider, ask())
    assert backend.requests[0].headers["authorization"] == "Bearer from-env"


async def test_missing_key_is_reported_before_anything_is_sent(connect: Connect) -> None:
    backend = Backend(says("never"))
    error = await failure(connect(backend, "deepseek", api_key="", api_key_env="DEEPSEEK_API_KEY"))
    assert error.kind == "auth" and "DEEPSEEK_API_KEY" in error.message
    assert backend.requests == []


async def test_missing_base_url_is_explained(connect: Connect) -> None:
    error = await failure(connect(Backend(says("never")), base_url=""))
    assert error.kind == "bad_request" and "base-url" in error.message


async def test_base_url_placeholder_is_filled_from_option_or_environment(
    connect: Connect, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = "https://api.cloudflare.test/accounts/{account_id}/ai/v1"
    backend = Backend(says("ok"))

    options = {"account-id": "abc"}
    await run(connect(backend, "cloudflare", base_url=template, options=options), ask())
    assert str(backend.requests[-1].url).startswith("https://api.cloudflare.test/accounts/abc/ai/")

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "from-env")
    await run(connect(backend, "cloudflare", base_url=template), ask())
    assert "/accounts/from-env/" in str(backend.requests[-1].url)

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    error = await failure(connect(backend, "cloudflare", base_url=template))
    assert error.kind == "bad_request"
    assert "account-id" in error.message and "CLOUDFLARE_ACCOUNT_ID" in error.message


# --------------------------------------------------------------------------------------------
# Streaming: text, reasoning, usage
# --------------------------------------------------------------------------------------------


async def test_text_is_streamed_and_assembled(connect: Connect) -> None:
    usage = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    backend = Backend(
        streaming(
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "Hel"}),
            chunk({"content": "lo."}),
            chunk({"content": ""}, finish="stop", usage=usage),
        )
    )
    events, response = await run(connect(backend), ask())
    assert events == [TextDelta("Hel"), TextDelta("lo.")]
    assert response.message.content == [TextBlock("Hello.")]
    assert response.stop_reason == "end_turn" and response.detail == ""
    assert response.usage == Usage(input_tokens=7, output_tokens=2)
    assert response.model == "served-model"
    assert stored(response) == {"role": "assistant", "content": "Hello."}


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
async def test_reasoning_fields_become_thinking_not_text(connect: Connect, field: str) -> None:
    backend = Backend(
        streaming(
            chunk({"role": "assistant", "content": None, field: ""}),
            chunk({field: "Compare "}),
            chunk({field: "digits."}),
            chunk({"content": "9.8"}),
            chunk(finish="stop"),
        )
    )
    events, response = await run(connect(backend), ask())
    assert events == [ThinkingDelta("Compare "), ThinkingDelta("digits."), TextDelta("9.8")]
    assert response.message.thinking == "Compare digits." and response.message.text == "9.8"
    assert stored(response) == {"role": "assistant", "content": "9.8", field: "Compare digits."}


async def test_reasoning_sent_twice_in_one_delta_is_shown_once(connect: Connect) -> None:
    """OpenRouter sends the same reasoning as `reasoning` and inside `reasoning_details`."""
    detail = {"type": "reasoning.text", "id": "r1", "format": "unknown", "index": 0}
    first = {**detail, "text": "Let me ", "signature": None}
    second = {**detail, "text": "think.", "signature": "sig"}
    backend = Backend(
        streaming(
            ": OPENROUTER PROCESSING\n\n",
            chunk({"reasoning": "Let me ", "reasoning_details": [first]}),
            chunk({"reasoning": "think.", "reasoning_details": [second]}),
            chunk({"content": "Done."}, finish="stop"),
        )
    )
    events, response = await run(connect(backend, "openrouter"), ask())
    assert events == [ThinkingDelta("Let me "), ThinkingDelta("think."), TextDelta("Done.")]
    # The details are kept whole for replay: fragments of one entry are joined by index.
    whole = {**detail, "text": "Let me think.", "signature": "sig"}
    assert stored(response)["reasoning_details"] == [whole]


async def test_reasoning_details_alone_are_still_shown(connect: Connect) -> None:
    summary = {"type": "reasoning.summary", "summary": "Weighed it.", "index": 0}
    backend = Backend(
        streaming(chunk({"reasoning_details": [summary]}), chunk({"content": "Yes."}, finish="stop"))
    )
    events, response = await run(connect(backend, "openrouter"), ask())
    assert events[0] == ThinkingDelta("Weighed it.") and response.message.thinking == "Weighed it."


async def test_inline_think_block_is_split_out_across_chunk_boundaries(connect: Connect) -> None:
    pieces = ["\n<th", "ink>\nplan", "ning</thi", "nk>\n\nThe ", "answer </think> stays."]
    backend = Backend(streaming(*(chunk({"content": p}) for p in pieces), chunk(finish="stop")))
    events, response = await run(connect(backend), ask())
    assert [type(e) for e in events] == [ThinkingDelta, ThinkingDelta, TextDelta, TextDelta]
    assert response.message.thinking == "planning"
    assert response.message.text == text_of(events) == "The answer </think> stays."
    # what the backend sent is kept as it was
    assert stored(response)["content"] == "".join(pieces)


async def test_think_tag_after_the_answer_began_is_ordinary_text(connect: Connect) -> None:
    text = "Models wrap reasoning in <think>...</think> tags."
    _, response = await run(connect(Backend(typed(text))), ask())
    assert response.message.text == text and response.message.thinking == ""


async def test_reasoning_that_never_closes_is_all_thinking(connect: Connect) -> None:
    _, response = await run(connect(Backend(typed("<think>still going", "length"))), ask())
    assert response.message.thinking == "still going" and response.message.text == ""
    assert response.stop_reason == "max_tokens"


DEEPSEEK_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 40,
    "total_tokens": 140,
    "prompt_tokens_details": {"cached_tokens": 64},
    "prompt_cache_hit_tokens": 64,
    "prompt_cache_miss_tokens": 36,
}
OPENAI_USAGE = {
    "prompt_tokens": 5,
    "completion_tokens": 1,
    "prompt_tokens_details": {"cached_tokens": 2},
}


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        pytest.param(  # DeepSeek: on the last chunk, which still has a choice
            [chunk({"content": "x"}), chunk({"content": ""}, finish="stop", usage=DEEPSEEK_USAGE)],
            Usage(input_tokens=36, output_tokens=40, cache_read_tokens=64),
            id="deepseek-on-the-finish-chunk",
        ),
        pytest.param(  # OpenAI: a chunk of its own with no choices
            [chunk({"content": "x"}, finish="stop"), {"choices": [], "usage": OPENAI_USAGE}],
            Usage(input_tokens=3, output_tokens=1, cache_read_tokens=2),
            id="openai-on-a-chunk-without-choices",
        ),
        pytest.param(  # OpenRouter: the finish chunk repeated, with usage
            [
                chunk({"content": "x"}, finish="stop"),
                chunk({"content": "", "role": "assistant"}, finish="stop", usage=OPENAI_USAGE),
            ],
            Usage(input_tokens=3, output_tokens=1, cache_read_tokens=2),
            id="openrouter-on-a-repeated-finish-chunk",
        ),
        pytest.param(
            [chunk({"content": "x"}, usage={"prompt_cache_hit_tokens": 4}), chunk(finish="stop")],
            Usage(),
            id="hit-count-without-a-prompt-count",
        ),
        pytest.param([chunk({"content": "x"}, finish="stop")], Usage(), id="missing"),
        pytest.param(
            [chunk({"content": "x"}, finish="stop", usage=None), {"choices": [], "usage": None}],
            Usage(),
            id="null",
        ),
    ],
)
async def test_usage_is_found_wherever_the_backend_puts_it(
    connect: Connect, frames: list[dict], expected: Usage
) -> None:
    _, response = await run(connect(Backend(streaming(*frames))), ask())
    assert response.usage == expected and response.message.text == "x"


async def test_stream_noise_is_tolerated(connect: Connect) -> None:
    """Keep-alive comments, CRLF framing, empty choices, and no [DONE] at the end."""
    body = (
        ": keep-alive\r\n\r\n"
        + f"data: {json.dumps(chunk({'content': 'a'}))}\r\n\r\n"
        + ": keep-alive\n\n"
        + f"data: {json.dumps({'choices': []})}\n\n"
        + f"data: {json.dumps(chunk({'content': 'b'}, finish='stop'))}\n\n"
    )
    _, response = await run(connect(Backend(streaming(body, done=False))), ask())
    assert response.message.text == "ab" and response.stop_reason == "end_turn"


async def test_whole_completion_instead_of_a_stream_is_accepted(connect: Connect) -> None:
    """Some gateways ignore `stream: true` and answer with one JSON document."""
    message = {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [wire_call("c1", "read_file", '{"path": "a"}')],
    }
    completion = {
        "model": "served-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    events, response = await run(connect(Backend(answering(200, completion))), ask(tools=TOOLS))
    assert response.message.text == text_of(events) == "Looking."
    assert response.message.tool_uses == [ToolUseBlock("c1", "read_file", {"path": "a"})]
    assert response.stop_reason == "tool_use" and response.usage.output_tokens == 2


# --------------------------------------------------------------------------------------------
# Streaming: tool calls
# --------------------------------------------------------------------------------------------


async def test_interleaved_tool_call_fragments_are_joined_by_index(connect: Connect) -> None:
    backend = Backend(
        streaming(
            chunk({"role": "assistant", "content": "Reading both."}),
            call(0, "call_A", "read_file", ""),
            call(0, args='{"path": "a'),
            call(1, "call_B", "read_file", ""),
            call(0, args='.py"}'),
            call(1, args='{"path": "b.py"}'),
            chunk({"content": ""}, finish="tool_calls"),
        )
    )
    events, response = await run(connect(backend), ask(tools=TOOLS))
    assert events == [
        TextDelta("Reading both."),
        ToolCallStarted(0, "call_A", "read_file"),
        ToolCallDelta(0, '{"path": "a'),
        ToolCallStarted(1, "call_B", "read_file"),
        ToolCallDelta(0, '.py"}'),
        ToolCallDelta(1, '{"path": "b.py"}'),
    ]
    assert response.message.tool_uses == [
        ToolUseBlock("call_A", "read_file", {"path": "a.py"}),
        ToolUseBlock("call_B", "read_file", {"path": "b.py"}),
    ]
    assert response.stop_reason == "tool_use"
    assert stored(response)["tool_calls"] == [
        wire_call("call_A", "read_file", '{"path": "a.py"}'),
        wire_call("call_B", "read_file", '{"path": "b.py"}'),
    ]


async def test_missing_ids_are_generated_and_distinct(connect: Connect) -> None:
    backend = Backend(
        streaming(
            call(0, name="read_file", args='{"path": "a"}'),
            call(1, name="read_file", args='{"path": "b"}'),
            chunk(finish="tool_calls"),
        )
    )
    events, response = await run(connect(backend), ask(tools=TOOLS))
    ids = [use.id for use in response.message.tool_uses]
    assert all(ids) and len(set(ids)) == 2
    assert [e.id for e in events if isinstance(e, ToolCallStarted)] == ids
    assert [c["id"] for c in stored(response)["tool_calls"]] == ids


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        pytest.param(
            [
                call(None, "c1", "read_file", '{"pa'),
                call(None, args='th": "a"}'),
                call(None, "c2", "read_file", '{"path": "b"}'),
            ],
            [("c1", {"path": "a"}), ("c2", {"path": "b"})],
            id="no-index-at-all",
        ),
        pytest.param(
            [call(0, "c1", "read_file", '{"pa'), call(0, "c1", "read_file", 'th": "a"}')],
            [("c1", {"path": "a"})],
            id="id-and-name-repeated-on-every-fragment",
        ),
        pytest.param(
            [call(0, "c1", "read_file", '{"path": "a"}'), call(0, "c2", "read_file", '{"path": "b"}')],
            [("c1", {"path": "a"}), ("c2", {"path": "b"})],
            id="one-index-reused-for-separate-calls",
        ),
        pytest.param(
            [call(0, "c1", args='{"path"'), call(0, name="read_file", args=': "a"}')],
            [("c1", {"path": "a"})],
            id="name-arriving-after-the-arguments-began",
        ),
        pytest.param(
            [call(None, name="read_file", args={"path": "a"})],
            [(None, {"path": "a"})],
            id="arguments-as-an-object-and-no-id",
        ),
    ],
)
async def test_tool_call_fragments_in_unusual_shapes(
    connect: Connect, frames: list[dict], expected: list[tuple[str | None, dict]]
) -> None:
    backend = Backend(streaming(*frames, chunk(finish="tool_calls")))
    events, response = await run(connect(backend), ask(tools=TOOLS))
    uses = response.message.tool_uses
    assert [(u.name, u.input) for u in uses] == [("read_file", i) for _, i in expected]
    assert [u.id for u in uses if u.id.startswith("c")] == [i for i, _ in expected if i]
    assert all(use.id for use in uses)
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    assert [(e.index, e.id, e.name) for e in started] == [
        (n, use.id, "read_file") for n, use in enumerate(uses)
    ]
    # The stored message is well formed whatever arrived: string arguments, one id per call.
    assert stored(response)["tool_calls"] == [
        wire_call(use.id, "read_file", json.dumps(use.input)) for use in uses
    ]


async def test_calls_sharing_one_id_are_given_distinct_ids(connect: Connect) -> None:
    backend = Backend(
        streaming(
            call(0, "call_0", "read_file", '{"path": "a"}'),
            call(1, "call_0", "read_file", '{"path": "b"}'),
            chunk(finish="tool_calls"),
        )
    )
    _, response = await run(connect(backend), ask(tools=TOOLS))
    first, second = response.message.tool_uses
    assert first.id == "call_0" and second.id and second.id != first.id


async def test_call_without_a_name_is_kept_for_the_loop_to_answer(connect: Connect) -> None:
    backend = Backend(streaming(call(0, "c1", args='{"path": "a"}'), chunk(finish="tool_calls")))
    events, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.message.tool_uses == [ToolUseBlock("c1", "", {"path": "a"})]
    assert events == [ToolCallStarted(0, "c1", ""), ToolCallDelta(0, '{"path": "a"}')]


async def test_empty_tool_call_placeholders_are_ignored(connect: Connect) -> None:
    backend = Backend(
        streaming(
            chunk({"content": "No tools needed.", "tool_calls": []}),
            chunk({"tool_calls": [{"index": 0}]}),
            chunk({"tool_calls": None}, finish="stop"),
        )
    )
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.message.tool_uses == [] and response.stop_reason == "end_turn"
    assert "tool_calls" not in stored(response)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", {}),
        ("{'path': 'a.py', 'backup': True}", {"path": "a.py", "backup": True}),
        ('"{\\"path\\": \\"a.py\\"}"', {"path": "a.py"}),
        ('{"path": "a"}{"path": "b"}', {"_invalid_json": '{"path": "a"}{"path": "b"}'}),
        ("not json at all", {"_invalid_json": "not json at all"}),
    ],
)
async def test_malformed_arguments_are_repaired_or_passed_on_as_invalid(
    connect: Connect, raw: str, expected: dict
) -> None:
    backend = Backend(streaming(call(0, "c1", "write_file", raw), chunk(finish="tool_calls")))
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.message.tool_uses == [ToolUseBlock("c1", "write_file", expected)]
    # Whatever the model wrote, what is replayed to the backend is a valid JSON object.
    assert json.loads(stored(response)["tool_calls"][0]["function"]["arguments"]) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"path": "src/app.py", "content": "def main():\\n    print(',
        "{'path': 'a.py'}",  # repairable on a clean finish, but this response was cut off
    ],
)
async def test_truncated_call_is_never_repaired(connect: Connect, raw: str) -> None:
    backend = Backend(
        streaming(
            call(0, "c1", "read_file", '{"path": "a"}'),
            call(1, "c2", "write_file", raw),
            chunk(finish="length"),
        )
    )
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.stop_reason == "max_tokens"
    assert response.message.tool_uses == [
        ToolUseBlock("c1", "read_file", {"path": "a"}),  # a call that did complete is intact
        ToolUseBlock("c2", "write_file", {"_invalid_json": raw}),
    ]


# --------------------------------------------------------------------------------------------
# Finish reasons
# --------------------------------------------------------------------------------------------

ONE_CALL = call(0, "c1", "read_file", '{"path": "a"}')


async def test_finish_reason_stop_with_tool_calls_is_a_tool_turn(connect: Connect) -> None:
    _, response = await run(connect(Backend(streaming(ONE_CALL, chunk(finish="stop")))), ask())
    assert response.stop_reason == "tool_use"


async def test_finish_reason_tool_calls_without_calls_ends_the_turn(connect: Connect) -> None:
    backend = Backend(streaming(chunk({"content": "Nothing to do."}), chunk(finish="tool_calls")))
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.stop_reason == "end_turn"


async def test_unknown_finish_reason_is_reported_not_fatal(connect: Connect) -> None:
    _, response = await run(connect(Backend(typed("partial", "eos_token"))), ask())
    assert response.stop_reason == "other" and "eos_token" in response.detail
    assert response.message.text == "partial"


async def test_unknown_finish_reason_does_not_license_repairing_arguments(connect: Connect) -> None:
    raw = "{'path': 'a'}"
    backend = Backend(streaming(call(0, "c1", "read_file", raw), chunk(finish="aborted")))
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.message.tool_uses[0].input == {"_invalid_json": raw}


async def test_content_filter_is_a_refusal_and_runs_no_tools(connect: Connect) -> None:
    backend = Backend(streaming(chunk({"content": "I can"}), ONE_CALL, chunk(finish="content_filter")))
    _, response = await run(connect(backend), ask(tools=TOOLS))
    assert response.stop_reason == "refusal" and "content filter" in response.detail
    assert response.message.tool_uses == [] and "tool_calls" not in stored(response)


async def test_stream_that_ends_without_a_finish_reason_is_retryable(connect: Connect) -> None:
    frames = [chunk({"content": "half an ans"}), call(0, "c1", "read_file", '{"pa')]
    error = await failure(connect(Backend(streaming(*frames, done=False))), ask(tools=TOOLS))
    assert error.kind == "network" and error.retryable
    assert "before the model finished" in error.message


async def test_empty_stream_is_an_error_not_an_empty_answer(connect: Connect) -> None:
    assert (await failure(connect(Backend(streaming())))).kind == "network"


@pytest.mark.parametrize("reason", ["insufficient_system_resource", "error"])
async def test_backend_giving_up_mid_response_is_retryable(connect: Connect, reason: str) -> None:
    backend = Backend(streaming(chunk({"content": "par"}), chunk({"content": ""}, finish=reason)))
    error = await failure(connect(backend, "deepseek"))
    assert error.kind == "overloaded" and error.retryable and reason in error.message


# --------------------------------------------------------------------------------------------
# Conversion and replay
# --------------------------------------------------------------------------------------------

NOTE = "[Message from the user while you were working]\nAlso update the docs."


def tool_round(native: NativePayload | None = None) -> list[Message]:
    """A history with one finished tool round whose results are followed by a note."""
    uses = [
        ToolUseBlock("t1", "read_file", {"path": "a.py"}),
        ToolUseBlock("t2", "read_file", {"path": "b.py"}),
    ]
    results = [ToolResultBlock("t1", "alpha"), ToolResultBlock("t2", "no such file", is_error=True)]
    return [
        Message.user("Fix the bug."),
        Message(role="assistant", content=[TextBlock("Reading."), *uses], native=native),
        Message.tool_results(results, note=NOTE),
    ]


async def test_tool_results_become_tool_messages_followed_by_the_text(connect: Connect) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend), ask(*tool_round(), tools=TOOLS))
    assert backend.messages[1:] == [
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                wire_call("t1", "read_file", '{"path": "a.py"}'),
                wire_call("t2", "read_file", '{"path": "b.py"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "alpha"},
        {"role": "tool", "tool_call_id": "t2", "content": "no such file"},
        {"role": "user", "content": NOTE},
    ]


async def test_results_without_text_add_no_user_message_and_user_runs_are_joined(
    connect: Connect,
) -> None:
    history = [
        *tool_round()[:2],
        Message.tool_results([ToolResultBlock("t1", "alpha"), ToolResultBlock("t2", "beta")]),
        Message.user("Thanks."),
        Message.user("One more thing."),
    ]
    backend = Backend(says("ok"))
    await run(connect(backend), ask(*history, tools=TOOLS))
    roles = [m["role"] for m in backend.messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "user"]
    assert backend.messages[-1]["content"] == "Thanks.\n\nOne more thing."


REASONED_TOOL_TURN = streaming(
    chunk({"role": "assistant", "content": "", "reasoning_content": ""}),
    chunk({"reasoning_content": "I should read it."}),
    call(0, "call_0", "read_file", '{"path": "a.py"}'),
    chunk({"content": ""}, finish="tool_calls", usage=DEEPSEEK_USAGE),
)
REBUILT_CALLS = [wire_call("call_0", "read_file", '{"path": "a.py"}')]


async def continued(
    connect: Connect,
    first_id: str,
    second_id: str,
    *,
    second_model: str = "m1",
    tools: list[ToolSpec] = TOOLS,
    **second_config: Any,
) -> dict[str, Any]:
    """Produce a reasoned tool turn on one provider (model "m1"), then send the conversation
    on to another. Returns the assistant message as the second backend received it."""
    first = connect(Backend(REASONED_TOOL_TURN), first_id)
    _, response = await run(first, ask(Message.user("go"), model="m1", tools=TOOLS))
    results = Message.tool_results([ToolResultBlock("call_0", "contents")])
    backend = Backend(says("ok"))
    second = connect(backend, second_id, **second_config)
    history = [Message.user("go"), response.message, results]
    await run(second, ask(*history, model=second_model, tools=tools))
    return backend.messages[2]


async def test_same_provider_and_model_get_their_turn_back_verbatim(connect: Connect) -> None:
    assert await continued(connect, "deepseek", "deepseek") == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should read it.",
        "tool_calls": REBUILT_CALLS,
    }


async def test_another_model_gets_the_turn_rebuilt_without_reasoning(connect: Connect) -> None:
    replayed = await continued(connect, "openrouter", "openrouter", second_model="m2")
    assert replayed == {"role": "assistant", "content": "", "tool_calls": REBUILT_CALLS}


async def test_another_provider_gets_the_turn_rebuilt_without_reasoning(connect: Connect) -> None:
    replayed = await continued(connect, "deepseek", "openrouter")
    assert replayed == {"role": "assistant", "content": "", "tool_calls": REBUILT_CALLS}


@pytest.mark.parametrize("provider_id", ["groq", "custom"])
async def test_strict_and_unknown_backends_never_get_reasoning_back(
    connect: Connect, provider_id: str
) -> None:
    replayed = await continued(connect, provider_id, provider_id)
    assert replayed == {"role": "assistant", "content": "", "tool_calls": REBUILT_CALLS}


async def test_deepseek_gets_reasoning_content_on_every_assistant_turn_when_tools_are_sent(
    connect: Connect,
) -> None:
    """Its own turns carry the original; turns from elsewhere get an empty field, because a
    missing one is an HTTP 400."""
    thinking = [{"type": "thinking", "thinking": "", "signature": "c2ln"}]
    foreign = NativePayload("anthropic", "claude-opus-5-5", thinking)
    first = connect(Backend(REASONED_TOOL_TURN), "deepseek")
    _, own = await run(first, ask(model="deepseek-flash", tools=TOOLS))
    history = [
        *tool_round(native=foreign),
        own.message,
        Message.tool_results([ToolResultBlock("call_0", "contents")]),
        Message.assistant("A turn with no native payload at all."),
        Message.user("continue"),
    ]
    backend = Backend(says("ok"))
    await run(connect(backend, "deepseek"), ask(*history, model="deepseek-flash", tools=TOOLS))
    assistants = [m for m in backend.messages if m["role"] == "assistant"]
    assert [m["reasoning_content"] for m in assistants] == ["", "I should read it.", ""]
    assert "signature" not in json.dumps(backend.messages)


async def test_deepseek_without_tools_sends_no_placeholder_reasoning(connect: Connect) -> None:
    assert "reasoning_content" not in await continued(connect, "openrouter", "deepseek", tools=[])


async def test_profile_is_found_by_host_and_by_option(connect: Connect) -> None:
    host = "https://api.deepseek.com"
    by_host = await continued(connect, "openrouter", "my-deepseek", base_url=host)
    by_option = await continued(connect, "openrouter", "proxy", options={"profile": "deepseek"})
    neither = await continued(connect, "openrouter", "proxy")
    assert by_host["reasoning_content"] == "" and by_option["reasoning_content"] == ""
    assert "reasoning_content" not in neither


async def test_unknown_profile_option_is_rejected_with_the_known_names(connect: Connect) -> None:
    error = await failure(connect(Backend(says("never")), options={"profile": "nope"}))
    assert error.kind == "bad_request" and "deepseek" in error.message


async def test_unknown_fields_survive_a_verbatim_replay(connect: Connect) -> None:
    """Gemini's thought signature rides on the tool call and must come back with it."""
    signature = {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    signed = wire_call("fc1", "read_file", '{"path": "a"}', index=0, extra_content=signature)
    turn = streaming(chunk({"tool_calls": [signed]}), chunk(finish="tool_calls"))
    _, response = await run(connect(Backend(turn), "gemini"), ask(model="g", tools=TOOLS))
    results = Message.tool_results([ToolResultBlock("fc1", "x")])
    backend = Backend(says("ok"))
    history = [Message.user("go"), response.message, results]
    await run(connect(backend, "gemini"), ask(*history, model="g", tools=TOOLS))
    expected = wire_call("fc1", "read_file", '{"path": "a"}', extra_content=signature)
    assert backend.messages[2]["tool_calls"] == [expected]


async def test_gemini_gets_a_placeholder_signature_on_calls_another_model_made(
    connect: Connect,
) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend, "gemini"), ask(*tool_round(), tools=TOOLS))
    first, second = backend.messages[2]["tool_calls"]
    placeholder = {"google": {"thought_signature": "skip_thought_signature_validator"}}
    assert first["extra_content"] == placeholder
    assert "extra_content" not in second  # only the first call of a turn carries one


async def test_stored_turn_survives_persistence_and_replay_leaves_it_untouched(
    connect: Connect,
) -> None:
    first = connect(Backend(REASONED_TOOL_TURN), "deepseek")
    _, response = await run(first, ask(model="m", tools=TOOLS))
    saved = json.dumps(response.message.to_dict())
    restored = Message.from_dict(json.loads(saved))
    results = Message.tool_results([ToolResultBlock("call_0", "x")])
    backend = Backend(says("ok"))
    history = [Message.user("go"), restored, results]
    await run(connect(backend, "deepseek"), ask(*history, model="m", tools=TOOLS))
    assert backend.messages[2]["reasoning_content"] == "I should read it."
    assert json.dumps(restored.to_dict()) == saved


# --------------------------------------------------------------------------------------------
# Tool modes
# --------------------------------------------------------------------------------------------

TWO_CALLS = streaming(
    call(0, "c1", "read_file", '{"path": "a"}'),
    call(1, "c2", "read_file", '{"path": "b"}'),
    call(0, args=""),
    chunk(finish="tool_calls"),
)


@pytest.mark.parametrize(
    ("provider_id", "flag_sent"), [("openrouter", True), ("deepseek", False), ("custom", False)]
)
async def test_sequential_mode_keeps_only_the_first_call(
    connect: Connect, provider_id: str, flag_sent: bool
) -> None:
    backend = Backend(TWO_CALLS)
    provider = connect(backend, provider_id, options={"tool-mode": "sequential"})
    events, response = await run(provider, ask(tools=TOOLS))
    assert ("parallel_tool_calls" in backend.body) is flag_sent
    assert backend.body.get("parallel_tool_calls", False) is False
    assert response.message.tool_uses == [ToolUseBlock("c1", "read_file", {"path": "a"})]
    assert events == [ToolCallStarted(0, "c1", "read_file"), ToolCallDelta(0, '{"path": "a"}')]
    assert stored(response)["tool_calls"] == [wire_call("c1", "read_file", '{"path": "a"}')]


async def test_native_mode_keeps_every_call_and_sends_no_flag(connect: Connect) -> None:
    backend = Backend(TWO_CALLS)
    _, response = await run(connect(backend, "openrouter"), ask(tools=TOOLS))
    assert "parallel_tool_calls" not in backend.body
    assert len(response.message.tool_uses) == 2


async def test_tool_mode_can_be_set_per_model(connect: Connect) -> None:
    options = {"tool-mode": "native", "model-tool-modes": {"small-model": "sequential"}}
    provider = connect(Backend(TWO_CALLS), options=options)
    _, big = await run(provider, ask(model="big-model", tools=TOOLS))
    _, small = await run(provider, ask(model="small-model", tools=TOOLS))
    assert (len(big.message.tool_uses), len(small.message.tool_uses)) == (2, 1)


async def test_invalid_tool_mode_names_the_valid_ones(connect: Connect) -> None:
    backend = Backend(says("never"))
    error = await failure(connect(backend, options={"tool-mode": "xml"}))
    assert error.kind == "bad_request" and "native, sequential, text" in error.message
    assert backend.requests == []


# --------------------------------------------------------------------------------------------
# The text tool protocol, end to end
# --------------------------------------------------------------------------------------------

TEXT_MODE = {"tool-mode": "text"}
WRITE_CALL = (
    "I'll create the file.\n\n"
    '<use_tool name="write_file">\n'
    '<arg name="path">src/app.py</arg>\n'
    '<arg name="content">\n'
    "def main():\n"
    '    print("hi </done>")\n'
    "</arg>\n"
    '<arg name="mode">420</arg>\n'
    '<arg name="backup">true</arg>\n'
)
READ_CALL = '<use_tool name="read_file">\n<arg name="path">a.py</arg>\n'


async def test_text_mode_offers_tools_through_the_prompt(connect: Connect) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend, options=TEXT_MODE), ask(tools=TOOLS))
    assert "tools" not in backend.body and backend.body["stop"] == ["</use_tool>"]
    system = backend.messages[0]["content"]
    assert system.startswith("You are terse.\n\n# Tools\n")
    assert '<use_tool name="TOOL_NAME">' in system
    described = "## read_file\nRead a file.\nArguments:\n- path (string, required): Path to read."
    assert described in system and "- mode (integer, optional)" in system


async def test_text_mode_without_tools_changes_nothing_in_the_request(connect: Connect) -> None:
    backend = Backend(says(f"A summary that quotes a block: {READ_CALL}"))
    _, response = await run(connect(backend, options=TEXT_MODE), ask())
    assert set(backend.body) == {"model", "messages", "stream", "stream_options"}
    assert backend.messages[0] == {"role": "system", "content": "You are terse."}
    # ...and with nothing on offer, nothing in the reply is taken for a call.
    assert response.message.tool_uses == [] and READ_CALL in response.message.text


async def test_text_mode_parses_the_block_and_hides_it_from_the_answer(connect: Connect) -> None:
    """The closing tag is the stop sequence, so the backend never sends it."""
    provider = connect(Backend(typed(WRITE_CALL)), options=TEXT_MODE)
    events, response = await run(provider, ask(tools=TOOLS))

    (use,) = response.message.tool_uses
    assert use.name == "write_file" and use.id
    assert use.input == {
        "path": "src/app.py",
        "content": 'def main():\n    print("hi </done>")\n',
        "mode": 420,
        "backup": True,
    }
    assert response.stop_reason == "tool_use"
    assert response.message.text == "I'll create the file."
    assert text_of(events).strip() == "I'll create the file." and "<" not in text_of(events)
    progress = [e for e in events if isinstance(e, (ToolCallStarted, ToolCallDelta))]
    assert progress[0] == ToolCallStarted(0, use.id, "write_file")
    assert [json.loads(e.partial_json) for e in progress[1:]] == [use.input]


async def test_text_mode_round_trip_renders_calls_and_results_as_text(connect: Connect) -> None:
    backend = Backend(typed(WRITE_CALL), says("All done."))
    provider = connect(backend, options=TEXT_MODE)
    _, first = await run(provider, ask(Message.user("make it"), tools=TOOLS))
    result = ToolResultBlock(first.message.tool_uses[0].id, "wrote 2 lines")
    history = [
        Message.user("make it"),
        first.message,
        Message.tool_results([result], note="Also add tests."),
    ]
    _, second = await run(provider, ask(*history, tools=TOOLS))

    assert second.message.text == "All done." and second.stop_reason == "end_turn"
    assert [m["role"] for m in backend.messages] == ["system", "user", "assistant", "user"]
    assert all(set(m) == {"role", "content"} for m in backend.messages)
    # the model sees its own block again, closed
    assert backend.messages[2]["content"] == WRITE_CALL.rstrip() + "\n</use_tool>"
    assert backend.messages[3]["content"] == (
        '<tool_result name="write_file" status="ok">\nwrote 2 lines\n</tool_result>\n\n'
        "Also add tests."
    )


async def test_text_mode_renders_turns_from_another_model(connect: Connect) -> None:
    backend = Backend(says("ok"))
    await run(connect(backend, options=TEXT_MODE), ask(*tool_round(), tools=TOOLS))
    assert backend.messages[2]["content"] == (
        "Reading.\n\n"
        '<use_tool name="read_file">\n<arg name="path">a.py</arg>\n</use_tool>\n\n'
        '<use_tool name="read_file">\n<arg name="path">b.py</arg>\n</use_tool>'
    )
    assert backend.messages[3]["content"] == (
        '<tool_result name="read_file" status="ok">\nalpha\n</tool_result>\n\n'
        '<tool_result name="read_file" status="error">\nno such file\n</tool_result>\n\n'
        f"{NOTE}"
    )


async def test_switching_tool_mode_rebuilds_earlier_turns(connect: Connect) -> None:
    """A native turn cannot be replayed verbatim into a text-mode conversation."""
    replayed = await continued(connect, "ollama", "ollama", options=TEXT_MODE)
    assert set(replayed) == {"role", "content"}
    assert replayed["content"] == READ_CALL + "</use_tool>"


@pytest.mark.parametrize(
    ("reply", "finish", "expected", "stop_reason"),
    [
        pytest.param(
            f"{READ_CALL}</use_tool>\nResult: it worked!\n{READ_CALL}</use_tool>",
            "stop",
            {"path": "a.py"},
            "tool_use",
            id="text-and-a-second-block-after-the-first-are-dropped",
        ),
        pytest.param(
            '<use_tool name="read_file">\n<arg name="path">a.py\n',
            "stop",
            {"_invalid_json": '<use_tool name="read_file">\n<arg name="path">a.py\n'},
            "tool_use",
            id="value-left-open-is-handed-to-the-loop",
        ),
        pytest.param(
            READ_CALL,
            "length",
            {"_invalid_json": READ_CALL},
            "max_tokens",
            id="cut-off-block-is-never-run",
        ),
    ],
)
async def test_text_mode_with_imperfect_output(
    connect: Connect, reply: str, finish: str, expected: dict, stop_reason: str
) -> None:
    provider = connect(Backend(typed(reply, finish)), options=TEXT_MODE)
    events, response = await run(provider, ask(tools=TOOLS))
    (use,) = response.message.tool_uses
    assert (use.name, use.input) == ("read_file", expected)
    assert response.stop_reason == stop_reason
    assert response.message.text == text_of(events) == ""
    assert stored(response)["content"].count("<use_tool") == 1


async def test_text_mode_plain_answer_and_false_starts(connect: Connect) -> None:
    text = "Use a < b here, or <used> tags; nothing to call. <use"
    provider = connect(Backend(typed(text, size=3)), options=TEXT_MODE)
    events, response = await run(provider, ask(tools=TOOLS))
    assert response.message.text == text_of(events) == text
    assert response.message.tool_uses == [] and response.stop_reason == "end_turn"


async def test_text_mode_with_inline_reasoning(connect: Connect) -> None:
    reply = f"<think>Need the file.</think>\n\nReading.\n{READ_CALL}"
    provider = connect(Backend(typed(reply)), "ollama", options=TEXT_MODE)
    _, response = await run(provider, ask(tools=TOOLS))
    assert response.message.thinking == "Need the file." and response.message.text == "Reading."
    assert response.message.tool_uses[0].input == {"path": "a.py"}
    assert stored(response)["content"] == f"Reading.\n\n{READ_CALL}</use_tool>"


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------

TOO_LONG = "This model's maximum context length is 131072 tokens. However, you requested 140211."
TPM = "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 12000"


@pytest.mark.parametrize(
    ("status", "payload", "headers", "kind", "retry_after"),
    [
        (401, oai_error("Authentication Fails"), {}, "auth", None),
        (403, oai_error("Key lacks permission"), {}, "auth", None),
        (404, oai_error("The model `nope` does not exist"), {}, "not_found", None),
        (400, oai_error("Model Not Exist"), {}, "not_found", None),
        (400, oai_error("tools[0].function.name: invalid characters"), {}, "bad_request", None),
        (400, oai_error(TOO_LONG), {}, "context_overflow", None),
        (413, oai_error(TPM), {}, "context_overflow", None),
        (422, oai_error("Invalid parameter"), {}, "bad_request", None),
        (429, oai_error("Rate limit reached"), {"retry-after": "7"}, "rate_limit", 7.0),
        (429, oai_error("Rate Limit Reached"), {}, "rate_limit", None),
        (500, oai_error("Internal error"), {}, "overloaded", None),
        (503, oai_error("Server overloaded"), {"retry-after": "2"}, "overloaded", 2.0),
        (502, "<html><body>Bad gateway</body></html>", {}, "overloaded", None),
    ],
)
async def test_http_errors_are_classified_and_never_retried_by_the_sdk(
    connect: Connect,
    status: int,
    payload: Any,
    headers: dict[str, str],
    kind: str,
    retry_after: float | None,
) -> None:
    backend = Backend(answering(status, payload, headers))
    error = await failure(connect(backend))
    assert (error.kind, error.status, error.retry_after) == (kind, status, retry_after)
    assert error.retryable is (kind in ("rate_limit", "overloaded"))
    assert len(backend.requests) == 1, "the agent loop retries; a second layer would multiply waits"


async def test_insufficient_balance_says_so_and_is_not_retried(connect: Connect) -> None:
    backend = Backend(answering(402, oai_error("Insufficient Balance")))
    error = await failure(connect(backend, "deepseek"))
    assert error.kind == "auth" and not error.retryable and error.status == 402
    assert "insufficient balance" in error.message and "retrying will not help" in error.message
    assert len(backend.requests) == 1


async def test_exhausted_quota_is_not_a_rate_limit(connect: Connect) -> None:
    """OpenRouter's daily cap on free models arrives as a 429 that waiting will not clear."""
    message = "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 per day"
    backend = Backend(answering(429, {"error": {"code": 429, "message": message}}))
    error = await failure(connect(backend, "openrouter"))
    assert error.kind == "auth" and not error.retryable and error.status == 429
    assert "quota exhausted" in error.message and "Not retried" in error.message


async def test_quota_wording_with_a_short_wait_is_still_a_rate_limit(connect: Connect) -> None:
    """Gemini words every 429 as an exceeded quota, and wraps its errors in a list."""
    message = "You exceeded your current quota, please check your plan. Please retry in 34.5s."
    payload = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message}}]
    error = await failure(connect(Backend(answering(429, payload)), "gemini"))
    assert error.kind == "rate_limit" and error.retry_after == 34.5


async def test_invalid_tool_call_rejected_by_the_backend_is_retried(connect: Connect) -> None:
    attempt = {"reason": "not valid JSON", "attempted_arguments": "{'location': 'New York'}"}
    payload = oai_error("Invalid tool call generated", failed_generation=attempt)
    error = await failure(connect(Backend(answering(400, payload)), "groq"))
    assert error.kind == "overloaded" and error.retryable and "invalid tool call" in error.message


@pytest.mark.parametrize(
    ("frame", "kind"),
    [
        pytest.param(
            {"error": {"message": "timeout waiting", "type": "server_error", "code": "x"}},
            "overloaded",
            id="deepseek-error-frame",
        ),
        pytest.param(
            chunk({"content": ""}, "error", error={"code": 429, "message": "Rate limit exceeded"}),
            "rate_limit",
            id="openrouter-error-chunk",
        ),
        pytest.param({"error": {"message": TOO_LONG}}, "context_overflow", id="overflow"),
    ],
)
async def test_error_inside_a_successful_stream_is_raised(
    connect: Connect, frame: dict, kind: str
) -> None:
    backend = Backend(streaming(chunk({"content": "partial "}), frame, done=False))
    provider = connect(backend, "openrouter")
    seen: list[StreamEvent] = []
    with pytest.raises(ProviderError) as caught:
        async for event in provider.stream(ask()):
            seen.append(event)
    assert seen == [TextDelta("partial ")]
    assert caught.value.kind == kind and caught.value.provider == "openrouter"


async def test_error_document_with_status_200_is_raised(connect: Connect) -> None:
    payload = oai_error("The server is busy, try later", type="server_error")
    error = await failure(connect(Backend(answering(200, payload))))
    assert error.kind == "overloaded" and "busy" in error.message


async def test_malformed_frame_is_a_retryable_decoding_failure(connect: Connect) -> None:
    backend = Backend(streaming(chunk({"content": "a"}), "data: {not json}\n\n"))
    error = await failure(connect(backend))
    assert error.kind == "network" and "decoded" in error.message


async def test_connection_failures_are_network_errors(connect: Connect) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("All connection attempts failed", request=request)

    def stall(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    local = connect(refuse, "ollama", base_url="http://localhost:11434/v1", api_key="", keyless=True)
    refused = await failure(local)
    assert refused.kind == "network" and refused.retryable
    assert "localhost:11434" in refused.message and "is the server running?" in refused.message
    timed_out = await failure(connect(stall))
    assert timed_out.kind == "network" and "timed out" in timed_out.message


async def test_connection_lost_mid_stream_is_a_network_error(connect: Connect) -> None:
    class Dropping(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse(chunk({"content": "par"}), done=False)
            raise httpx2.ReadError("connection reset by peer")

    def handler(request: httpx2.Request) -> httpx2.Response:
        headers = {"content-type": "text/event-stream"}
        return httpx2.Response(200, headers=headers, stream=Dropping())

    error = await failure(connect(handler))
    assert error.kind == "network" and error.retryable


# --------------------------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "data": [
        {
            "id": "qwen/qwen3-coder:free",
            "name": "Qwen: Qwen3 Coder (free)",
            "context_length": 262144,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "top_provider": {"context_length": 262144, "max_completion_tokens": 65536},
            "supported_parameters": ["max_tokens", "tools", "tool_choice", "reasoning"],
            "expiration_date": "2026-12-31",
        },
        {
            "id": "deepseek/deepseek-v4.1-flash",
            "name": "DeepSeek: V4.1 Flash",
            "context_length": 1048576,
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            "top_provider": {"max_completion_tokens": None},
            "supported_parameters": ["max_tokens", "temperature"],
            "expiration_date": None,
        },
        {
            "id": "openrouter/auto",
            "context_length": 2000000,
            "pricing": {"prompt": "-1", "completion": "-1"},
        },
        {"id": "google/lyria-3", "architecture": {"output_modalities": ["text", "audio"]}},
    ]
}


async def test_list_models_reads_what_openrouter_reports(connect: Connect) -> None:
    backend = Backend(answering(200, OPENROUTER_MODELS))
    provider = connect(backend, "openrouter", base_url="https://openrouter.ai/api/v1")
    free, paid, router = await provider.list_models()

    request = backend.requests[0]
    assert (request.method, str(request.url)) == ("GET", "https://openrouter.ai/api/v1/models")
    assert request.headers["authorization"] == "Bearer sk-test"
    assert free.to_dict() == {
        "ref": "openrouter/qwen/qwen3-coder:free",
        "provider": "openrouter",
        "id": "qwen/qwen3-coder:free",
        "name": "Qwen: Qwen3 Coder (free)",
        "context_window": 262144,
        "max_output": 65536,
        "supports_tools": True,
        "free": True,
        "note": "expires 2026-12-31",
        "input_price": 0.0,
        "output_price": 0.0,
    }
    assert (paid.free, paid.input_price, paid.output_price) == (False, 0.15, 0.6)
    assert (paid.supports_tools, paid.max_output, paid.note) == (False, None, "")
    assert (router.id, router.free, router.input_price) == ("openrouter/auto", None, None)


@pytest.mark.parametrize(
    ("provider_id", "payload", "expected"),
    [
        ("lmstudio", {"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model"}]}, "qwen3.8-27b"),
        ("gemini", {"data": [{"id": "models/gemini-4-hydrogen"}]}, "gemini-4-hydrogen"),
    ],
)
async def test_list_models_with_ids_only(
    connect: Connect, provider_id: str, payload: Any, expected: str
) -> None:
    (model,) = await connect(Backend(answering(200, payload)), provider_id).list_models()
    assert (model.provider, model.id) == (provider_id, expected)
    assert (model.context_window, model.free, model.supports_tools) == (None, None, None)


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        (answering(401, oai_error("Invalid API key")), "auth"),
        (answering(404, {"detail": "Not Found"}), "not_found"),
        (answering(503, oai_error("Loading model")), "overloaded"),
        (answering(200, {"object": "error", "message": "nope"}), "unknown"),
    ],
)
async def test_list_models_failures_raise_provider_errors(
    connect: Connect, reply: Reply, kind: str
) -> None:
    with pytest.raises(ProviderError) as caught:
        await connect(Backend(reply)).list_models()
    assert caught.value.kind == kind and caught.value.provider == "custom"


# --------------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------------


class Gated(httpx2.AsyncByteStream):
    """A response body that sends one chunk and then waits until it is closed."""

    def __init__(self) -> None:
        self.closed = False
        self.sent = asyncio.Event()
        self._released = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield sse(chunk({"content": "first"}), done=False)
        self.sent.set()
        await self._released.wait()

    async def aclose(self) -> None:
        self.closed = True
        self._released.set()

    def backend(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=self)


async def test_cancelling_the_consumer_closes_the_http_stream(connect: Connect) -> None:
    body = Gated()
    provider = connect(body.backend)
    seen: list[StreamEvent] = []

    async def consume() -> None:
        async for event in provider.stream(ask()):
            seen.append(event)

    task = asyncio.create_task(consume())
    await body.sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [TextDelta("first")] and body.closed


async def test_abandoning_the_stream_early_closes_the_http_stream(connect: Connect) -> None:
    body = Gated()
    stream = connect(body.backend).stream(ask())
    assert await anext(stream) == TextDelta("first")
    await stream.aclose()
    assert body.closed


@pytest.mark.parametrize("used", [True, False])
async def test_aclose_closes_the_http_client(used: bool) -> None:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(Backend(says("ok"))))
    config = ProviderConfig(id="custom", base_url=BASE, api_key="k")
    provider = OpenAICompatibleProvider(config, http_client=client)
    if used:
        assert (await provider.complete(ask())).message.text == "ok"
    await provider.aclose()
    assert client.is_closed


async def test_registry_builds_this_adapter_for_openai_compatible_providers() -> None:
    config = ProviderConfig(id="deepseek", base_url="https://api.deepseek.com", api_key="k")
    provider = build_provider(config)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert (provider.id, provider.kind) == ("deepseek", "openai-compatible")
    await provider.aclose()


class ReadFile(Tool):
    name = READ.name
    description = READ.description
    input_schema = READ.input_schema
    read_only = True

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(f"contents of {tool_input['path']}")


@pytest.mark.parametrize("mode", ["native", "text"])
async def test_agent_loop_runs_a_tool_round_through_the_adapter(
    connect: Connect, session: Session, mode: str
) -> None:
    """The whole path: the model asks for a tool, the loop runs it, the result goes back."""
    turns = {"native": REASONED_TOOL_TURN, "text": typed(f"Looking.\n{READ_CALL}")}
    backend = Backend(turns[mode], says("It defines main()."))
    provider = connect(backend, "scripted", options={"tool-mode": mode})
    session.runtime.providers.register(provider)

    node = session.new_node("test", role="root")
    agent = session.agent(node, system="You are a test agent.", tools=[ReadFile()])
    result = await agent.run("What is in a.py?")

    assert result.text == "It defines main()." and len(backend.requests) == 2
    assert agent.messages[2].tool_result_blocks[0].content == "contents of a.py"
    assert "contents of a.py" in json.dumps(backend.messages[-1])
    assert agent.usage.output_tokens == (40 if mode == "native" else 0)

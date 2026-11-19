"""The Anthropic adapter: Claude models over the Messages API, through the official SDK.

Three rules shape this module.

* **The request prefix never changes within a conversation.** `tools` and `system` are rendered
  the same way on every call (tools sorted by name, nothing time-dependent) and assistant turns
  go back exactly as they arrived. Prompt caching is a prefix match, and current models bind
  each reasoning block to the conversation prefix that produced it, so one changed byte costs a
  cache rebuild at best and a rejected request at worst.
* **No parameter is sent on a guess.** Models differ in which `thinking` configurations they
  accept, whether `effort` exists and how much they may write. `_CAPABILITIES` records the
  differences per model family, and a model the table does not know gets the bare minimum, so
  an unfamiliar model id is never rejected because of something this adapter added. The same
  caution applies to endpoints: features only the first-party API is known to accept are left
  out when the provider points at another base URL.
* **Every failure is a `ProviderError`.** The SDK's own retries are switched off because the
  agent loop already retries retryable errors with backoff.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, aclosing
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import anthropic
import httpx2
from anthropic.types import Message as SDKMessage
from anthropic.types import ModelInfo as SDKModelInfo
from anthropic.types import RawContentBlockDeltaEvent
from anthropic.types.beta import BetaMessage, BetaRawContentBlockDeltaEvent

from clankit.config.schema import ProviderConfig
from clankit.llm.provider import ErrorKind, Provider, ProviderError
from clankit.llm.types import (
    EFFORT_LEVELS,
    Block,
    ChatRequest,
    ChatResponse,
    Effort,
    Message,
    ModelInfo,
    NativePayload,
    ResponseDone,
    StopReason,
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

__all__ = ["AnthropicProvider", "ModelCapabilities", "capabilities_for"]

log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com"
_EPHEMERAL: Final[dict[str, str]] = {"type": "ephemeral"}

#: Room for a long answer without lowballing: responses are always streamed, so a generous
#: ceiling carries no timeout risk, and a truncated turn costs a whole extra model call.
_DEFAULT_MAX_TOKENS = 64_000
#: The smallest thinking budget the API accepts, also kept free for the visible answer.
_MIN_THINKING_BUDGET = 1_024
#: Thinking budgets for models that need one. Effort is the only depth control the rest of
#: Clankit knows, so it picks the budget.
_THINKING_BUDGETS: Final[dict[str, int]] = {
    "low": 2_048,
    "medium": 8_192,
    "high": 16_384,
    "xhigh": 24_576,
    "max": 32_768,
}

#: Server-side refusal fallback: when a safety classifier declines a request, the API re-runs
#: it on the model Anthropic recommends for that refusal category, inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Brotli is never offered: httpx2's decoder crashes on the older `brotli` bindings some
#: systems carry (no `output_buffer_limit` keyword), with no graceful fallback. gzip and
#: deflate decode through the standard library everywhere.
_SAFE_ENCODINGS: Final[dict[str, str]] = {"Accept-Encoding": "gzip, deflate"}
#: The only models that accept the `fallbacks` parameter. Matched exactly, never by pattern:
#: sending it to any other model is a rejected request.
_FALLBACK_MODELS = frozenset(
    {
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    }
)

_STOP_REASONS: Final[dict[str, StopReason]] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "pause_turn": "pause",
}

#: `error.type` of an API error body. Preferred over the HTTP status because an error that
#: arrives in the middle of a stream has status 200.
_KIND_BY_ERROR_TYPE: Final[dict[str, ErrorKind]] = {
    "authentication_error": "auth",
    "permission_error": "auth",
    "billing_error": "auth",
    "not_found_error": "not_found",
    "rate_limit_error": "rate_limit",
    "request_too_large": "context_overflow",
    "invalid_request_error": "bad_request",
    "overloaded_error": "overloaded",
    "api_error": "overloaded",
    "timeout_error": "overloaded",
}
_ERROR_LABELS: Final[dict[ErrorKind, str]] = {
    "auth": "the API rejected this account or key (check the key, its permissions and billing)",
    "rate_limit": "rate limited",
    "overloaded": "the API is temporarily unavailable",
    "context_overflow": "the conversation no longer fits in the model's context window",
    "bad_request": "the API rejected the request",
    "not_found": "not found (check the model id and, for a custom endpoint, its base URL)",
    "network": "the connection to the API failed",
    "unknown": "the API reported an error",
}
#: How the API words "the prompt is larger than the context window" in a 400.
_CONTEXT_OVERFLOW = re.compile(
    r"prompt is too long|context (window|limit|length)|too many tokens", re.IGNORECASE
)
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
#: Block types that may carry a cache breakpoint.
_CACHEABLE_BLOCKS = frozenset({"text", "tool_use", "tool_result"})
_EMPTY_TOOL_RESULT = "(no output)"
#: Sent as the key to servers that need none; the SDK refuses to send a request without one.
_KEYLESS_API_KEY = "none"

_SDKStream = anthropic.AsyncMessageStream[Any] | anthropic.BetaAsyncMessageStream[Any]
_SDKStreamManager = (
    anthropic.AsyncMessageStreamManager[Any] | anthropic.BetaAsyncMessageStreamManager[Any]
)


# --------------------------------------------------------------------------------------------
# Model capabilities
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Which optional request parameters one family of models accepts.

    The defaults describe a model about which nothing is known: nothing optional is sent.
    """

    #: How thinking is switched on: "adaptive" (``thinking.type = adaptive``), "budget"
    #: (``type = enabled`` with ``budget_tokens``), or "" when no form is known to work.
    thinking: Literal["adaptive", "budget", ""] = ""
    #: The ``thinking.type`` that switches thinking off. Empty when sending nothing is the
    #: right way to ask: the model thinks only on request, or cannot stop thinking at all.
    off: Literal["disabled", "between_tools", ""] = ""
    #: Whether the off switch is rejected above effort "high".
    off_caps_effort: bool = False
    #: Accepted ``output_config.effort`` levels; empty when the parameter does not exist.
    effort: tuple[str, ...] = ()
    #: The largest ``max_tokens`` the model accepts.
    max_output: int = 8_192
    #: Whether reasoning summaries come back without asking. Newer models withhold them
    #: unless ``thinking.display`` says otherwise; where they are the default, the setting is
    #: only sent to hide them.
    summarizes: bool = False


_ALWAYS_THINKING = ModelCapabilities("adaptive", effort=EFFORT_LEVELS, max_output=128_000)
_OFF_ANY_EFFORT = replace(_ALWAYS_THINKING, off="disabled")
_OFF_UP_TO_HIGH = replace(_OFF_ANY_EFFORT, off_caps_effort=True)
_BEFORE_XHIGH = replace(_OFF_ANY_EFFORT, effort=("low", "medium", "high", "max"), summarizes=True)
_BUDGET_ONLY = ModelCapabilities("budget", max_output=64_000, summarizes=True)

#: First match wins, so a specific id must come before the family pattern that would also
#: match it ("claude-opus-5-5*" before "claude-opus-5*"). Patterns end in "*" to cover dated
#: snapshot ids such as "claude-haiku-4-5-20251001".
_CAPABILITIES: tuple[tuple[str, ModelCapabilities], ...] = (
    # Thinking is always on: adaptive is the only accepted setting and there is no off switch.
    ("claude-fable-5*", _ALWAYS_THINKING),
    ("claude-mythos-5*", _ALWAYS_THINKING),
    ("claude-opus-5-5*", _ALWAYS_THINKING),
    # "disabled" is rejected here; thinking is switched off with "between_tools" instead.
    ("claude-sonnet-5-5*", replace(_OFF_UP_TO_HIGH, off="between_tools")),
    # Thinking may be disabled, but only at effort "high" or below.
    ("claude-opus-5*", _OFF_UP_TO_HIGH),
    ("claude-haiku-5-5*", _OFF_UP_TO_HIGH),
    # Thinking may be disabled at any effort. Adaptive is the only way to switch it on, and
    # Opus 4.8 and 4.7 do not think at all unless it is sent.
    ("claude-sonnet-5*", _OFF_ANY_EFFORT),
    ("claude-opus-4-8*", _OFF_ANY_EFFORT),
    ("claude-opus-4-7*", _OFF_ANY_EFFORT),
    # The 4.6 generation predates the "xhigh" effort level and still shows its reasoning
    # summaries without being asked.
    ("claude-opus-4-6*", _BEFORE_XHIGH),
    ("claude-sonnet-4-6*", _BEFORE_XHIGH),
    # Older models think only when given an explicit token budget, and reject adaptive.
    # Of these, only Opus 4.5 has an effort parameter, and only up to "high".
    ("claude-opus-4-5*", replace(_BUDGET_ONLY, effort=EFFORT_LEVELS[:3])),
    ("claude-haiku-4-5*", _BUDGET_ONLY),
    ("claude-sonnet-4-5*", _BUDGET_ONLY),
    # A Claude model this table has not heard of, most likely a newer one. Every Claude model
    # still served writes at least 64K tokens; nothing else is assumed.
    ("claude-*", ModelCapabilities(max_output=64_000)),
)
_UNKNOWN_MODEL = ModelCapabilities()


def capabilities_for(model: str) -> ModelCapabilities:
    """What is known about the request parameters `model` accepts."""
    for pattern, capabilities in _CAPABILITIES:
        if fnmatch.fnmatchcase(model, pattern):
            return capabilities
    return _UNKNOWN_MODEL


# --------------------------------------------------------------------------------------------
# Provider options
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Options:
    """`ProviderConfig.options`, validated. `AnthropicProvider` documents what each means."""

    thinking: str = "auto"
    display: str = "summarized"
    max_tokens: int | None = None
    refusal_fallback: bool = True


def _read_options(config: ProviderConfig) -> _Options:
    return _Options(
        thinking=_choice(config, "thinking", ("auto", "adaptive", "off"), "auto"),
        display=_choice(config, "thinking-display", ("summarized", "omitted"), "summarized"),
        max_tokens=_positive_int(config, "max-tokens"),
        refusal_fallback=_flag(config, "refusal-fallback", default=True),
    )


def _choice(config: ProviderConfig, key: str, choices: tuple[str, ...], default: str) -> str:
    value = config.options.get(key, default)
    if value not in choices:
        raise _bad_option(config, key, f"one of {', '.join(choices)}", value)
    return str(value)


def _positive_int(config: ProviderConfig, key: str) -> int | None:
    value = config.options.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _bad_option(config, key, "a whole number of tokens above zero", value)
    return value


def _flag(config: ProviderConfig, key: str, *, default: bool) -> bool:
    value = config.options.get(key, default)
    if isinstance(value, str) and value.lower() in ("true", "false", "yes", "no", "on", "off"):
        return value.lower() in ("true", "yes", "on")
    if not isinstance(value, bool):
        raise _bad_option(config, key, "true or false", value)
    return value


def _bad_option(config: ProviderConfig, key: str, expected: str, value: object) -> ProviderError:
    return ProviderError(
        f"option {key!r} must be {expected}, but is {value!r}; correct it in the configuration",
        kind="bad_request",
        provider=config.id,
    )


def _api_key(config: ProviderConfig) -> str:
    """The key to authenticate with.

    Always explicit: given no key, the SDK would go looking through environment variables and
    credential files of its own, and Clankit's configuration is meant to be the only source.
    """
    key = config.resolved_api_key
    if key:
        return key
    if config.keyless:
        return _KEYLESS_API_KEY
    hint = f" or set ${config.api_key_env}" if config.api_key_env else ""
    raise ProviderError(
        f"no API key for {config.name or config.id}: add one in settings{hint}",
        kind="auth",
        provider=config.id,
    )


def _is_first_party(base_url: str) -> bool:
    return not base_url or urlsplit(base_url).hostname == urlsplit(_API_URL).hostname


# --------------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------------


class AnthropicProvider(Provider):
    """Claude models, or any server that speaks the Anthropic Messages API.

    Switches read from `ProviderConfig.options`:

    * ``thinking``: "auto" (default) follows the capability table; "adaptive" also sends
      adaptive thinking to models the table does not know; "off" switches thinking off where
      the model allows it (a model that always thinks keeps thinking, with its reasoning
      hidden).
    * ``thinking-display``: "summarized" (default) returns readable summaries of the reasoning,
      "omitted" hides them. Thinking happens and is billed the same either way.
    * ``max-tokens``: the output ceiling for requests that do not set their own.
    * ``refusal-fallback``: true (default) lets the API answer a request that a safety
      classifier declined with a fallback model. Only sent to the models that support it.
    """

    kind = "anthropic"

    def __init__(
        self, config: ProviderConfig, *, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        super().__init__(config.id)
        self._config = config
        self._options = _read_options(config)
        #: Gateways and compatible servers reject fields they do not know, so features only
        #: Anthropic's own API is known to accept are sent to that API alone.
        self._first_party = _is_first_party(config.base_url)
        #: Set once the API has turned the fallback beta down (not enabled for the account).
        self._fallback_refused = False
        self._client = anthropic.AsyncAnthropic(
            api_key=_api_key(config),
            base_url=config.base_url or _API_URL,
            max_retries=0,
            http_client=http_client,
            default_headers=_SAFE_ENCODINGS,
        )

    # -- Provider -------------------------------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        try:
            # `aclosing` matters: closing this generator does not by itself close the one it
            # iterates, and that inner one is what holds the HTTP response open.
            async with aclosing(self._stream(request)) as events:
                async for event in events:
                    yield event
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            models = [model async for model in self._client.models.list(limit=1000)]
        except Exception as exc:
            raise self._translate(exc) from exc
        models.sort(key=_released, reverse=True)
        return [self._model_info(model) for model in models]

    async def aclose(self) -> None:
        await self._client.close()

    # -- streaming ------------------------------------------------------------------------------

    async def _stream(self, request: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        """Translate the SDK's event stream; SDK and transport exceptions pass through.

        Leaving the `async with` closes the HTTP response, which is what makes closing this
        generator or cancelling its task stop the request.
        """
        # Content-block index of each client tool call -> (its position among the tool calls,
        # the fragments of its JSON input as streamed).
        calls: dict[int, tuple[int, list[str]]] = {}
        started = False
        async with AsyncExitStack() as stack:
            stream = await self._open(stack, request)
            # The stream yields raw API events and, derived from them, convenience events
            # ("text", "thinking", "input_json"). Only raw ones are handled, or every delta
            # would be reported twice. Unknown event types are ignored.
            async for event in stream:
                if event.type == "message_start":
                    started = True
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        calls[event.index] = (len(calls), [])
                        yield ToolCallStarted(index=len(calls) - 1, id=block.id, name=block.name)
                elif event.type == "content_block_delta":
                    translated = _translate_delta(event, calls)
                    if translated is not None:
                        yield translated
            if not started:
                raise self._incomplete()
            message = stream.current_message_snapshot
        streamed = {index: "".join(fragments) for index, (_, fragments) in calls.items()}
        yield ResponseDone(self._response(request, message, streamed))

    async def _open(self, stack: AsyncExitStack, request: ChatRequest) -> _SDKStream:
        """Send the request and return its event stream, registered with `stack` for closing.

        Refusal fallback is a beta that is on by default here. Should the API turn it down,
        the request is sent again without it and the feature stays off for this provider:
        an optional safety net must not be what makes every call fail.
        """
        params = self._params(request)
        try:
            return await stack.enter_async_context(self._start(params))
        except anthropic.BadRequestError as exc:
            if "fallbacks" not in params or "fallback" not in str(exc).lower():
                raise
            log.warning(
                "%s: the API does not accept refusal fallbacks (%s); continuing without them",
                self.id,
                exc.api_message or exc.message,
            )
            self._fallback_refused = True
            return await stack.enter_async_context(self._start(self._params(request)))

    def _start(self, params: dict[str, Any]) -> _SDKStreamManager:
        if "betas" in params:
            return self._client.beta.messages.stream(**params)
        return self._client.messages.stream(**params)

    # -- request --------------------------------------------------------------------------------

    def _params(self, request: ChatRequest) -> dict[str, Any]:
        """Keyword arguments for the SDK's `messages.stream`.

        Deliberately absent: sampling parameters and a forced `tool_choice`, which current
        models reject.
        """
        if request.effort is not None and request.effort not in EFFORT_LEVELS:
            raise self._error(
                f"unknown effort level {request.effort!r}; use one of {', '.join(EFFORT_LEVELS)}",
                "bad_request",
            )
        capabilities = capabilities_for(request.model)
        max_tokens = self._max_tokens(request, capabilities)
        messages = _wire_messages(request.messages)
        params: dict[str, Any] = {"model": request.model, "max_tokens": max_tokens}
        if request.system.strip():
            # Tools render before the system prompt, so this one breakpoint caches both.
            params["system"] = [
                {"type": "text", "text": request.system, "cache_control": _EPHEMERAL}
            ]
        if request.tools:
            tools = sorted(request.tools, key=lambda spec: spec.name)
            params["tools"] = [self._tool(spec) for spec in tools]
        thinking = self._thinking(capabilities, request.effort, max_tokens)
        if thinking is not None:
            params["thinking"] = thinking
        effort = self._effort(request.effort, capabilities)
        if effort is not None:
            params["output_config"] = {"effort": effort}
        if self._first_party:
            # Automatic caching: the API keeps a breakpoint on the last block of the growing
            # conversation, so each call reads everything the previous one wrote.
            params["cache_control"] = _EPHEMERAL
        else:
            messages = _with_tail_breakpoint(messages)
        params["messages"] = messages
        params.update(self._fallback_params(request.model, messages))
        return params

    def _tool(self, spec: ToolSpec) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        if self._first_party:
            # Stream tool input as it is written instead of in one burst once the server has
            # buffered it: a large file body would otherwise look like a stalled connection.
            # The price is that the server no longer validates the JSON; `_tool_input` does.
            tool["eager_input_streaming"] = True
        return tool

    def _max_tokens(self, request: ChatRequest, capabilities: ModelCapabilities) -> int:
        if request.max_tokens:
            return request.max_tokens
        if self._options.max_tokens:
            return self._options.max_tokens
        declared = self._config.model(request.model)
        ceiling = (declared.max_output if declared else None) or capabilities.max_output
        return min(_DEFAULT_MAX_TOKENS, ceiling)

    def _thinking(
        self, capabilities: ModelCapabilities, effort: Effort | None, max_tokens: int
    ) -> dict[str, Any] | None:
        """The `thinking` parameter, or None to leave the model on its default."""
        mode = self._options.thinking
        if mode == "off":
            return {"type": capabilities.off} if capabilities.off else None
        if capabilities.thinking == "budget":
            # No effort means the same as "high", as it does on models that have the setting.
            budget = min(_THINKING_BUDGETS[effort or "high"], max_tokens - _MIN_THINKING_BUDGET)
            if budget < _MIN_THINKING_BUDGET:
                return None  # no room for the smallest budget and an answer
            thinking: dict[str, Any] = {"type": "enabled", "budget_tokens": budget}
        elif capabilities.thinking == "adaptive" or mode == "adaptive":
            thinking = {"type": "adaptive"}
        else:
            return None
        if not (capabilities.summarizes and self._options.display == "summarized"):
            thinking["display"] = self._options.display
        return thinking

    def _effort(self, requested: Effort | None, capabilities: ModelCapabilities) -> str | None:
        """The closest effort level the model accepts that is not above the requested one."""
        if requested is None or not capabilities.effort:
            return None
        ceiling = EFFORT_LEVELS.index(requested)
        if self._options.thinking == "off" and capabilities.off_caps_effort:
            ceiling = min(ceiling, EFFORT_LEVELS.index("high"))
        accepted = [lvl for lvl in EFFORT_LEVELS[: ceiling + 1] if lvl in capabilities.effort]
        return accepted[-1] if accepted else None

    def _fallback_params(self, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        offered = (
            self._options.refusal_fallback
            and self._first_party
            and not self._fallback_refused
            and model in _FALLBACK_MODELS
        )
        if offered:
            return {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
        if _has_fallback_block(messages):
            # An earlier turn was answered by a fallback model. Its marker block is part of
            # the history and is only accepted while the beta is named.
            return {"betas": [_FALLBACK_BETA]}
        return {}

    # -- response -------------------------------------------------------------------------------

    def _response(
        self, request: ChatRequest, message: SDKMessage | BetaMessage, streamed: dict[int, str]
    ) -> ChatResponse:
        stop = message.stop_reason
        if stop is None:
            raise self._incomplete()
        if stop == "model_context_window_exceeded":
            # The context filled up while the model was writing. Reported as an overflow so
            # the agent compacts and asks again; a half-written turn is no use to it.
            raise self._error(
                f"{_ERROR_LABELS['context_overflow']}: the response was cut off when the "
                "window filled up",
                "context_overflow",
            )
        # A refused turn may stop anywhere, even inside a tool call or before a reasoning
        # block is signed. None of it is safe to act on or to send back, so none is kept.
        blocks = [] if stop == "refusal" else _replayable(_content(message, streamed))
        model = message.model or request.model
        return ChatResponse(
            message=Message(
                role="assistant",
                content=_neutral_blocks(blocks),
                native=NativePayload(kind=self.kind, model=model, data=blocks),
            ),
            stop_reason=_STOP_REASONS.get(stop, "other"),
            usage=_usage(message.usage),
            model=model,
            detail=_detail(request.model, message),
        )

    def _model_info(self, model: SDKModelInfo) -> ModelInfo:
        """One entry of the Models API. Only `id` is relied on: compatible servers return
        sparser entries than Anthropic does."""
        retires = getattr(model, "retires_at", None)
        note = ""
        if getattr(model, "lifecycle", None) == "deprecated":
            note = "Deprecated."
            if isinstance(retires, datetime):
                note = f"Deprecated; retires {retires:%Y-%m-%d}."
        return ModelInfo(
            provider=self.id,
            id=model.id,
            name=getattr(model, "display_name", None) or model.id,
            context_window=getattr(model, "max_input_tokens", None),
            max_output=getattr(model, "max_tokens", None),
            # Facts about Claude models; another server's models are left unknown.
            supports_tools=True if self._first_party else None,
            free=False if self._first_party else None,
            note=note,
        )

    # -- errors ---------------------------------------------------------------------------------

    def _error(
        self,
        message: str,
        kind: ErrorKind,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> ProviderError:
        return ProviderError(
            message, kind=kind, provider=self.id, status=status, retry_after=retry_after
        )

    def _incomplete(self) -> ProviderError:
        return self._error(
            "the response stream ended before the model finished (no stop reason arrived)",
            "network",
        )

    def _translate(self, exc: Exception) -> ProviderError:
        """Turn anything the SDK or the transport raised into a `ProviderError`."""
        if isinstance(exc, anthropic.APIStatusError):
            return self._status_error(exc)
        if isinstance(exc, (anthropic.APITimeoutError, httpx2.TimeoutException)):
            return self._error(f"{_ERROR_LABELS['network']}: the request timed out", "network")
        if isinstance(exc, (anthropic.APIConnectionError, httpx2.HTTPError, httpx2.StreamError)):
            # The SDK wraps failures to connect; a connection lost mid-stream arrives raw.
            cause = exc.__cause__ if isinstance(exc, anthropic.APIConnectionError) else exc
            return self._error(
                f"{_ERROR_LABELS['network']} ({self._client.base_url}): {_brief(cause or exc)}",
                "network",
            )
        if isinstance(exc, (ValueError, RuntimeError)):
            # The stream itself was unusable: malformed events, or tool input so broken that
            # the SDK gave up on the response. Asking again usually yields a clean one.
            return self._error(
                f"the response stream could not be decoded: {_brief(exc)}", "network"
            )
        return self._error(f"unexpected {type(exc).__name__}: {_brief(exc)}", "unknown")

    def _status_error(self, exc: anthropic.APIStatusError) -> ProviderError:
        detail = _brief(exc.api_message or exc.message)
        kind = _KIND_BY_ERROR_TYPE.get(exc.type or "") or _kind_for_status(exc.status_code)
        if kind == "bad_request" and _CONTEXT_OVERFLOW.search(detail):
            kind = "context_overflow"
        if (exc.error_code or "").endswith("spend_limit_reached"):
            # Arrives as a 429 but does not clear by waiting: the account has to be fixed.
            kind = "auth"
        message = f"{_ERROR_LABELS[kind]}: {detail}"
        if exc.request_id:
            message += f" (request {exc.request_id})"
        return self._error(
            message,
            kind,
            # A stream that fails part-way has already answered 200; that is not the error.
            status=exc.status_code if exc.status_code >= 400 else None,
            retry_after=_retry_after(exc.response.headers),
        )


# --------------------------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------------------------


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """The conversation in the API's format.

    Messages that end up with no content are left out: the API rejects an empty message, and
    a refused or empty assistant turn is exactly that. The rule is the same on every call, so
    the prefix the following turns were produced under does not shift.
    """
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant":
            content = _assistant_content(message)
        else:
            content = _user_content(message)
        if content:
            wire.append({"role": message.role, "content": content})
    return wire


def _assistant_content(message: Message) -> list[dict[str, Any]]:
    """An assistant turn for replay.

    A turn this adapter produced goes back verbatim, whichever Claude model wrote it:
    reasoning blocks are only valid unmodified, and the API itself drops the ones the current
    model cannot read. A turn from another kind of provider is rebuilt from its text and tool
    calls; its reasoning is not portable.
    """
    native = message.native
    if (
        native is not None
        and native.kind == AnthropicProvider.kind
        and isinstance(native.data, list)
    ):
        return native.data
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text.strip():
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append(
                {
                    "type": "tool_use",
                    "id": _wire_id(block.id),
                    "name": block.name,
                    "input": block.input,
                }
            )
    return content


def _user_content(message: Message) -> list[dict[str, Any]]:
    """A user turn. All tool results of a round arrive in one message and stay in one: split
    across messages they teach the model to stop making parallel calls."""
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": _wire_id(block.tool_use_id),
                # The API rejects an empty error result, and a marker reads better than
                # nothing in either case.
                "content": block.content or _EMPTY_TOOL_RESULT,
            }
            if block.is_error:
                result["is_error"] = True
            content.append(result)
        elif isinstance(block, TextBlock) and block.text.strip():
            content.append({"type": "text", "text": block.text})
    return content


def _wire_id(tool_use_id: str) -> str:
    """Tool-call ids minted by other providers may hold characters this API rejects. Ids of
    its own pass through unchanged."""
    return _INVALID_ID_CHARS.sub("_", tool_use_id)


def _with_tail_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the end of the conversation as a cache breakpoint.

    For servers without automatic caching, this is what lets each call reuse the previous
    one's prefix. The marked block is a copy: replayed content is never modified in place.
    """
    if not messages:
        return messages
    last = messages[-1]
    *head, tail = last["content"]
    if tail.get("type") not in _CACHEABLE_BLOCKS:
        return messages
    marked = {**last, "content": [*head, {**tail, "cache_control": _EPHEMERAL}]}
    return [*messages[:-1], marked]


def _has_fallback_block(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "fallback"
        for message in messages
        if message["role"] == "assistant"
        for block in message["content"]
    )


# --------------------------------------------------------------------------------------------
# Response helpers
# --------------------------------------------------------------------------------------------


def _translate_delta(
    event: RawContentBlockDeltaEvent | BetaRawContentBlockDeltaEvent,
    calls: dict[int, tuple[int, list[str]]],
) -> StreamEvent | None:
    """The neutral event for one content delta, or None when there is nothing to report:
    an empty fragment (withheld reasoning streams as one), a signature, a server tool's input."""
    delta = event.delta
    if delta.type == "text_delta" and delta.text:
        return TextDelta(delta.text)
    if delta.type == "thinking_delta" and delta.thinking:
        return ThinkingDelta(delta.thinking)
    if delta.type == "input_json_delta" and delta.partial_json and event.index in calls:
        position, fragments = calls[event.index]
        fragments.append(delta.partial_json)
        return ToolCallDelta(index=position, partial_json=delta.partial_json)
    return None


def _content(message: SDKMessage | BetaMessage, streamed: dict[int, str]) -> list[dict[str, Any]]:
    """The response's content blocks as plain JSON, with client tool input parsed strictly.

    Block types this adapter does not know come through untouched.
    """
    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(message.content):
        data: dict[str, Any] = block.to_dict(mode="json", warnings=False)
        if data.get("type") == "tool_use":
            data["input"] = _tool_input(streamed.get(index, ""), data.get("input"))
        blocks.append(data)
    return blocks


def _tool_input(streamed: str, initial: object) -> dict[str, Any]:
    """A tool call's arguments, or ``{"_invalid_json": raw}`` when they are not a JSON object.

    Parsed from the text as streamed rather than taken from the SDK, whose tolerant parser
    quietly turns truncated or malformed input into a smaller object that looks valid.
    """
    if not streamed:
        # No fragments arrived: the block carried its whole input from the start.
        return initial if isinstance(initial, dict) else {}
    try:
        parsed = json.loads(streamed)
    except ValueError:
        return {"_invalid_json": streamed}
    return parsed if isinstance(parsed, dict) else {"_invalid_json": streamed}


def _replayable(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The blocks of a response that may be sent back as an assistant turn.

    When a fallback model took over mid-response, a "fallback" block marks the hand-over.
    Before it only the declined model's text may be echoed: its reasoning and tool calls were
    abandoned, and a tool call there must not run. The marker and everything after it stay.
    Text blocks without text are dropped everywhere, because the API rejects them as input.
    """
    boundary = max(
        (index for index, block in enumerate(blocks) if block.get("type") == "fallback"),
        default=None,
    )
    if boundary is not None:
        kept = [block for block in blocks[:boundary] if block.get("type") == "text"]
        blocks = kept + blocks[boundary:]
    return [
        block
        for block in blocks
        if block.get("type") != "text" or str(block.get("text") or "").strip()
    ]


def _neutral_blocks(blocks: list[dict[str, Any]]) -> list[Block]:
    """The provider-neutral view: text, reasoning summaries and tool calls. Every other block
    type (server tools, fallback markers, redacted reasoning) lives on in the native payload
    only."""
    content: list[Block] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            content.append(TextBlock(block["text"]))
        elif kind == "thinking":
            content.append(ThinkingBlock(block.get("thinking") or ""))
        elif kind == "tool_use":
            content.append(ToolUseBlock(id=block["id"], name=block["name"], input=block["input"]))
    return content


def _usage(usage: object) -> Usage:
    """Token counts. The API's `input_tokens` already excludes cached tokens."""
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=getattr(usage, "output_tokens", None) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _detail(requested: str, message: SDKMessage | BetaMessage) -> str:
    """What a person should know about an unusual ending, or an empty string."""
    # A fallback block marks only a hand-over inside this response. Later turns of the same
    # conversation are served by the fallback model with no block at all, so the usage
    # breakdown is the dependable signal.
    iterations = getattr(message.usage, "iterations", None) or []
    fell_back = any(getattr(entry, "type", "") == "fallback_message" for entry in iterations)
    if message.stop_reason == "refusal":
        return _refusal_detail(message, fell_back=fell_back)
    if fell_back:
        return (
            f"Answered by the fallback model {message.model} because {requested} declined "
            "this conversation (server-side refusal fallback)."
        )
    if message.stop_reason not in _STOP_REASONS:
        return f"The API ended the response with an unfamiliar stop reason: {message.stop_reason}."
    return ""


def _refusal_detail(message: SDKMessage | BetaMessage, *, fell_back: bool) -> str:
    details = message.stop_details
    category = getattr(details, "category", None)
    explanation = getattr(details, "explanation", None)
    recommended = getattr(details, "recommended_model", None)
    text = "The model declined this request"
    if category:
        text += f" (safety category: {category})"
    text += f": {explanation.rstrip('.')}." if explanation else "."
    if fell_back:
        text += " A fallback model was tried and declined as well."
    if recommended:
        text += f" Asking {recommended} directly may work."
    return text + " Rephrase the request or switch to another model."


def _released(model: object) -> float:
    """Sort key for newest-first model lists; models without a release date sort last."""
    created = getattr(model, "created_at", None)
    return created.timestamp() if isinstance(created, datetime) else 0.0


# --------------------------------------------------------------------------------------------
# Error helpers
# --------------------------------------------------------------------------------------------


def _kind_for_status(status: int) -> ErrorKind:
    """Classification by HTTP status, for error bodies without a recognisable `error.type`
    (gateways, proxies, compatible servers)."""
    if status in (401, 402, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 413:
        return "context_overflow"
    if status == 429:
        return "rate_limit"
    if status == 408 or status >= 500:
        return "overloaded"
    if status >= 400:
        return "bad_request"
    return "unknown"


def _retry_after(headers: httpx2.Headers) -> float | None:
    """Seconds the server asked us to wait, from `retry-after-ms` or `retry-after`."""
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            seconds = float(headers.get(header, "")) * scale
        except ValueError:
            continue
        if seconds > 0:
            return seconds
    return None


def _brief(value: object, limit: int = 300) -> str:
    """One short line: error bodies from gateways can be whole HTML pages."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}..."

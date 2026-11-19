"""The provider interface every model backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar, Literal

from clankit.llm.types import ChatRequest, ChatResponse, ModelInfo, ResponseDone, StreamEvent

ErrorKind = Literal[
    "auth",  # missing/invalid credentials
    "billing",  # out of credit, spend limit reached, quota exhausted; retrying will not help
    "rate_limit",  # 429
    "overloaded",  # 5xx / 529, transient
    "context_overflow",  # prompt larger than the model's context window
    "bad_request",  # the request is malformed; retrying will not help
    "not_found",  # unknown model or endpoint
    "network",  # connection failure or timeout
    "unknown",
]


class ProviderError(Exception):
    """A model call failed.

    Adapters translate SDK exceptions into this single type so the agent loop can decide
    between retrying, compacting, and giving up without knowing any SDK.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = "unknown",
        provider: str = "",
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind: ErrorKind = kind
        self.provider = provider
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in ("rate_limit", "overloaded", "network")

    def __str__(self) -> str:
        prefix = f"{self.provider}: " if self.provider else ""
        return f"{prefix}{self.message}"


class Provider(ABC):
    """A model backend.

    `id` is the name the provider has in the configuration ("anthropic", "deepseek",
    "openrouter", "my-local-server"). `kind` names the wire protocol and is what
    `Message.native.kind` is compared against.
    """

    kind: ClassVar[str] = ""

    def __init__(self, provider_id: str) -> None:
        self.id = provider_id

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Run one model call.

        Yields `TextDelta` / `ThinkingDelta` / `ToolCallStarted` / `ToolCallDelta` events as they
        arrive and then exactly one `ResponseDone`, which is always the last event.

        Contract for implementers:
        * Implement as an async generator.
        * Raise `ProviderError` for every failure, including failures that surface mid-stream.
        * Tool-call arguments in the final message are parsed JSON objects (dicts). If the
          model produced arguments that are not a JSON object, still return the tool call with
          ``input={"_invalid_json": "<raw text>"}`` so the agent loop can answer it with an
          error result; never drop a tool call. The one exception: when the response stream
          itself cannot be decoded (the SDK aborts mid-stream), raise a retryable
          ``ProviderError(kind="network")`` so the request is re-issued.
        * The response message carries `native` so it can be replayed verbatim.
        * Be safe to cancel: closing the generator must close the HTTP stream.
        """
        raise NotImplementedError

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run one model call and return only the final response."""
        final: ChatResponse | None = None
        async for event in self.stream(request):
            if isinstance(event, ResponseDone):
                final = event.response
        if final is None:
            raise ProviderError("stream ended without a response", provider=self.id)
        return final

    async def list_models(self) -> list[ModelInfo]:
        """Models this backend currently offers. May hit the network. Empty if unsupported."""
        return []

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""

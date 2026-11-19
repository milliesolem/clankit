"""A deterministic provider driven by a script. Used by tests and by offline demos.

A script is a sequence of *steps*. Each model call consumes the next step. A step is one of:

* a ``str``                      -> the assistant replies with that text and ends its turn
* a ``ToolUseBlock`` / list of blocks -> the assistant emits those blocks
* a ``ChatResponse``             -> returned as-is
* an ``Exception``               -> raised (use ``ProviderError`` to exercise error handling)
* a callable ``(ChatRequest) -> step`` -> evaluated lazily, so a step can inspect the request

Helpers `say()` and `call()` build the common steps::

    provider = ScriptedProvider([
        call("read_file", path="README.md"),
        say("The README is empty."),
    ])

When several agents run concurrently against one provider the order of calls is not
deterministic. For those cases pass ``router=`` instead: a callable ``(ChatRequest) -> step``
that decides the reply from the request (for example from the system prompt or last message).
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import (
    Block,
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    NativePayload,
    ResponseDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStarted,
    ToolUseBlock,
    Usage,
)

Step = Any  # str | Block | list[Block] | ChatResponse | Exception | Callable[[ChatRequest], Step]

_ids = itertools.count(1)


def say(text: str) -> list[Block]:
    """The assistant says `text` and ends its turn."""
    return [TextBlock(text)]


def call(tool: str, /, **tool_input: Any) -> ToolUseBlock:
    """The assistant calls `tool` with the given input."""
    return ToolUseBlock(id=f"call_{next(_ids)}", name=tool, input=dict(tool_input))


def think(text: str) -> ThinkingBlock:
    return ThinkingBlock(text)


class ScriptedProvider(Provider):
    kind = "scripted"

    def __init__(
        self,
        steps: Iterable[Step] = (),
        *,
        provider_id: str = "scripted",
        router: Callable[[ChatRequest], Step] | None = None,
        models: Iterable[str] = ("test-model",),
        usage: Usage | None = None,
    ) -> None:
        super().__init__(provider_id)
        self._steps: list[Step] = list(steps)
        self._router = router
        self._models = list(models)
        self._usage = usage or Usage(input_tokens=10, output_tokens=5)
        #: Every request received, in order. Tests assert against this.
        self.requests: list[ChatRequest] = []

    # -- scripting ------------------------------------------------------------------------------

    def push(self, *steps: Step) -> None:
        self._steps.extend(steps)

    @property
    def remaining(self) -> int:
        return len(self._steps)

    # -- Provider -------------------------------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        # Snapshot the request: the agent loop keeps appending to the same message list.
        self.requests.append(
            ChatRequest(
                model=request.model,
                system=request.system,
                messages=list(request.messages),
                tools=list(request.tools),
                max_tokens=request.max_tokens,
                effort=request.effort,
            )
        )
        step = self._next_step(request)
        response = self._to_response(step, request)

        index = 0
        for block in response.message.content:
            if isinstance(block, ThinkingBlock):
                yield ThinkingDelta(block.text)
            elif isinstance(block, TextBlock):
                # Stream in two fragments so consumers exercise delta accumulation.
                half = max(1, len(block.text) // 2)
                for piece in (block.text[:half], block.text[half:]):
                    if piece:
                        yield TextDelta(piece)
            elif isinstance(block, ToolUseBlock):
                yield ToolCallStarted(index=index, id=block.id, name=block.name)
                yield ToolCallDelta(index=index, partial_json="{}")
                index += 1
        yield ResponseDone(response)

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(provider=self.id, id=m, name=m, context_window=200_000, supports_tools=True)
            for m in self._models
        ]

    # -- internals ------------------------------------------------------------------------------

    def _next_step(self, request: ChatRequest) -> Step:
        if self._router is not None:
            step = self._router(request)
            if step is not None:
                return step
        if not self._steps:
            raise ProviderError(
                "ScriptedProvider ran out of steps "
                f"(call #{len(self.requests)}, model {request.model!r})",
                kind="bad_request",
                provider=self.id,
            )
        return self._steps.pop(0)

    def _to_response(self, step: Step, request: ChatRequest) -> ChatResponse:
        while callable(step) and not isinstance(step, (ChatResponse, Exception)):
            step = step(request)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ChatResponse):
            return step
        if isinstance(step, str):
            blocks: list[Block] = [TextBlock(step)]
        elif isinstance(step, (TextBlock, ThinkingBlock, ToolUseBlock)):
            blocks = [step]
        elif isinstance(step, (list, tuple)):
            blocks = list(step)
        else:
            raise TypeError(f"unsupported script step: {step!r}")

        has_tool_use = any(isinstance(b, ToolUseBlock) for b in blocks)
        message = Message(
            role="assistant",
            content=blocks,
            native=NativePayload(kind=self.kind, model=request.model, data=None),
        )
        return ChatResponse(
            message=message,
            stop_reason="tool_use" if has_tool_use else "end_turn",
            usage=Usage(
                self._usage.input_tokens,
                self._usage.output_tokens,
                self._usage.cache_read_tokens,
                self._usage.cache_write_tokens,
            ),
            model=request.model,
        )

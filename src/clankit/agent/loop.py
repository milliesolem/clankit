"""The agent loop: call the model, run the tools it asks for, repeat until it is done.

An `Agent` is one conversation with one model. Orchestration methods are built by composing
agents; nothing here knows about task trees.

Invariants this module maintains:

* The message history is append-only, and `system` and `tools` never change after the first
  call. (Reasoning blocks are bound to their conversation prefix; prompt caches are prefix
  matches.) The single exception is compaction, which replaces the whole history at once.
* Every tool call in an assistant turn is answered, in ONE user message, even when a tool
  crashes, permission is denied, or the run is cancelled half way through a tool round.
* Tool inputs are validated against the tool's schema before the tool runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import jsonschema

from clankit.agent import context as ctxmod
from clankit.agent.events import (
    AgentTextDelta,
    AgentThinkingDelta,
    AssistantMessage,
    Event,
    Notice,
    ToolFinished,
    ToolStarted,
    UserMessage,
)
from clankit.config.schema import Config
from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import (
    ChatRequest,
    ChatResponse,
    Effort,
    Message,
    ModelInfo,
    ModelRef,
    ResponseDone,
    TextDelta,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from clankit.tools.base import Tool, ToolContext, ToolOutput

log = logging.getLogger(__name__)

AgentStop = Literal["end_turn", "finished", "max_turns", "refusal"]

#: Characters of tool output included in the `ToolFinished` event (UIs show a preview).
PREVIEW_CHARS = 4000

_message_ids = itertools.count(1)

INTERRUPTED = (
    "[Interrupted by the user before this tool call finished. Its effects may be partial.]"
)
NOT_RUN = "[Not run: the user interrupted the turn before this tool call started.]"
TRUNCATED_CALL = (
    "This tool call was not run: your response reached the output limit while you were still "
    "writing its input, so the input is incomplete. Do the work in smaller steps (for a large "
    "file, write a first part and append the rest with further edits)."
)
REPEATING = (
    "[Harness note] You have made the same tool call with the same input three times in a row. "
    "Repeating it will not change the result. Take a different approach, or stop and report "
    "exactly what is blocking you."
)
CONTINUE_AFTER_LIMIT = (
    "Your response reached the output limit and was cut off. Continue exactly where you "
    "stopped; do not repeat what you already wrote."
)


class AgentHost(Protocol):
    """What an agent needs from its surroundings. `Session` is the real implementation."""

    config: Config

    def provider_for(self, model: ModelRef) -> tuple[Provider, str]:
        """The provider that serves `model` and the model id to send to it."""
        ...

    def model_info(self, model: ModelRef) -> ModelInfo | None: ...

    def emit(self, event: Event) -> Event: ...

    def record_usage(self, node_id: str, usage: Usage, context_tokens: int) -> None: ...

    def model_slot(self, model: ModelRef) -> AbstractAsyncContextManager[None]:
        """Held for the duration of one model call; limits calls in flight per provider."""
        ...

    async def before_turn(self, node_id: str) -> None:
        """Called before every model call. May wait (budget prompt) or raise to stop the run."""
        ...


@dataclass(slots=True)
class AgentResult:
    """How one `Agent.run` ended.

    `stop` is "end_turn" when the model simply finished talking, "finished" when a control
    tool ended the run (then `payload` holds that tool's `finish` dict), "max_turns" when the
    turn limit stopped it, and "refusal" when the model declined (see `detail`).
    """

    text: str
    stop: AgentStop
    payload: dict[str, Any] | None = None
    detail: str = ""


class Agent:
    def __init__(
        self,
        *,
        host: AgentHost,
        node_id: str,
        model: ModelRef,
        system: str,
        tools: Sequence[Tool],
        ctx: ToolContext,
        effort: Effort | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.host = host
        self.node_id = node_id
        self.model = model
        self.system = system
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.ctx = ctx
        self.effort: Effort | None = effort or (host.config.agent.effort or None)  # type: ignore[assignment]
        self.max_turns = max_turns or host.config.agent.max_turns

        self.messages: list[Message] = []
        self.usage = Usage()
        #: Size of the most recent prompt plus its reply: how full the context window is.
        self.context_tokens = 0
        self._specs = [t.spec() for t in tools]
        self._validators = {t.name: jsonschema.Draft202012Validator(t.input_schema) for t in tools}
        self._inbox: list[str] = []
        self._task_text = ""
        self._last_calls: tuple[tuple[str, str], ...] = ()
        self._repeats = 0

    # -- public API -----------------------------------------------------------------------------

    def steer(self, text: str) -> None:
        """Queue a message for the model. It is delivered at the next turn boundary: alongside
        the next tool results, or as a new user message if the model was about to stop."""
        if text.strip():
            self._inbox.append(text)

    async def run(self, prompt: str) -> AgentResult:
        """Give the agent a prompt and run it until it stops.

        May be called again on the same agent to continue the conversation. Raises
        `ProviderError` when the model cannot be reached after retries, and propagates
        `asyncio.CancelledError`; in both cases the history is left in a state from which
        `run` can be called again.
        """
        if not self.messages:
            self._task_text = prompt
        self.messages.append(Message.user(prompt))
        self.host.emit(UserMessage(node_id=self.node_id, text=prompt))

        turns = 0
        truncations = 0
        while True:
            if turns >= self.max_turns:
                self._notice("warning", f"Stopped after {turns} model calls (the max-turns limit).")
                return AgentResult(text=self._last_text(), stop="max_turns")

            await self.host.before_turn(self.node_id)
            await self._compact_if_needed()
            response = await self._call_model()
            turns += 1

            if response.stop_reason == "refusal":
                # Not appended: the refused turn is discarded, and the history stays valid
                # (it ends with the user's message) for a rephrased follow-up.
                return AgentResult(text="", stop="refusal", detail=response.detail)
            self.messages.append(response.message)

            if response.stop_reason == "pause":
                continue  # the provider paused a long turn; sending the history back resumes it

            tool_uses = response.message.tool_uses

            if response.stop_reason == "max_tokens":
                truncations += 1
                if tool_uses:
                    results = [
                        ToolResultBlock(u.id, TRUNCATED_CALL, is_error=True) for u in tool_uses
                    ]
                    self.messages.append(Message.tool_results(results, note=self._drain_inbox()))
                    continue
                if truncations <= 3:
                    self.messages.append(Message.user(CONTINUE_AFTER_LIMIT))
                    continue
                return AgentResult(text=response.message.text, stop="end_turn")

            if not tool_uses:
                pending = self._drain_inbox()
                if pending:
                    self.messages.append(Message.user(pending))
                    continue
                return AgentResult(text=response.message.text, stop="end_turn")

            results, finish = await self._run_tool_round(tool_uses)
            note = self._drain_inbox()
            if self._is_repeating(tool_uses):
                note = f"{note}\n\n{REPEATING}" if note else REPEATING
            self.messages.append(Message.tool_results(results, note=note))
            if finish is not None:
                return AgentResult(text=response.message.text, stop="finished", payload=finish)

    # -- model call -----------------------------------------------------------------------------

    async def _call_model(self) -> ChatResponse:
        provider, model_id = self.host.provider_for(self.model)
        attempts = 0
        compacted = False
        while True:
            request = ChatRequest(
                model=model_id,
                system=self.system,
                messages=self.messages,
                tools=self._specs,
                effort=self.effort,
            )
            message_id = f"m{next(_message_ids)}"
            streamed = False
            try:
                response: ChatResponse | None = None
                async with (
                    self.host.model_slot(self.model),
                    contextlib.aclosing(provider.stream(request)) as stream,
                ):
                    async for event in stream:
                        if isinstance(event, TextDelta):
                            streamed = True
                            self.host.emit(
                                AgentTextDelta(
                                    node_id=self.node_id, message_id=message_id, text=event.text
                                )
                            )
                        elif isinstance(event, ThinkingDelta):
                            streamed = True
                            self.host.emit(
                                AgentThinkingDelta(
                                    node_id=self.node_id, message_id=message_id, text=event.text
                                )
                            )
                        elif isinstance(event, ResponseDone):
                            response = event.response
                if response is None:
                    raise ProviderError(
                        "the model stream ended without a response",
                        provider=provider.id,
                        kind="network",
                    )
            except ProviderError as exc:
                if streamed:
                    # Withdraw the partial message: an AssistantMessage with no content tells
                    # subscribers to drop what the deltas built up.
                    self.host.emit(
                        AssistantMessage(node_id=self.node_id, message_id=message_id, text="")
                    )
                if exc.kind == "context_overflow" and not compacted and len(self.messages) > 1:
                    compacted = True
                    await self._compact(emergency=True)
                    continue
                attempts += 1
                if exc.retryable and attempts <= ctxmod.MAX_RETRIES:
                    delay = exc.retry_after or ctxmod.backoff_delay(attempts)
                    self._notice(
                        "warning",
                        f"{exc} - retrying in {delay:.0f}s (attempt {attempts}/{ctxmod.MAX_RETRIES})",
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except asyncio.CancelledError:
                if streamed:
                    self.host.emit(
                        AssistantMessage(node_id=self.node_id, message_id=message_id, text="")
                    )
                raise

            message = response.message
            if response.stop_reason == "refusal":
                # A refused turn's partial output must be discarded, not shown or replayed.
                if streamed:
                    self.host.emit(
                        AssistantMessage(node_id=self.node_id, message_id=message_id, text="")
                    )
            elif message.text or message.thinking:
                self.host.emit(
                    AssistantMessage(
                        node_id=self.node_id,
                        message_id=message_id,
                        text=message.text,
                        thinking=message.thinking,
                    )
                )
            self.usage.add(response.usage)
            reported = response.usage.prompt_tokens + response.usage.output_tokens
            self.context_tokens = reported or ctxmod.estimate_tokens(self.system, self.messages)
            self.host.record_usage(self.node_id, response.usage, self.context_tokens)
            return response

    # -- tools ----------------------------------------------------------------------------------

    async def _run_tool_round(
        self, tool_uses: list[ToolUseBlock]
    ) -> tuple[list[ToolResultBlock], dict[str, Any] | None]:
        """Run every tool call of one assistant turn and return their results in call order.

        Consecutive read-only calls run concurrently; any other call runs alone. If the run is
        cancelled, unfinished calls are answered with an "interrupted" result so the history
        stays valid, and the cancellation is re-raised.
        """
        outputs: dict[str, ToolOutput] = {}
        started: set[str] = set()
        try:
            for batch in self._batches(tool_uses):
                started.update(u.id for u in batch)
                if len(batch) == 1:
                    outputs[batch[0].id] = await self._run_tool(batch[0])
                else:
                    done = await asyncio.gather(*(self._run_tool(u) for u in batch))
                    outputs.update({u.id: out for u, out in zip(batch, done, strict=True)})
        except asyncio.CancelledError:
            results = [
                ToolResultBlock(
                    u.id,
                    outputs[u.id].content
                    if u.id in outputs
                    else (INTERRUPTED if u.id in started else NOT_RUN),
                    is_error=outputs[u.id].is_error if u.id in outputs else True,
                )
                for u in tool_uses
            ]
            self.messages.append(Message.tool_results(results))
            raise

        finish = next(
            (outputs[u.id].finish for u in tool_uses if outputs[u.id].finish is not None), None
        )
        results = [
            ToolResultBlock(u.id, outputs[u.id].content, outputs[u.id].is_error) for u in tool_uses
        ]
        return results, finish

    def _batches(self, tool_uses: list[ToolUseBlock]) -> list[list[ToolUseBlock]]:
        batches: list[list[ToolUseBlock]] = []
        for use in tool_uses:
            tool = self.tools.get(use.name)
            parallel_safe = tool is not None and tool.read_only
            if (
                parallel_safe
                and batches
                and all(
                    (t := self.tools.get(u.name)) is not None and t.read_only for u in batches[-1]
                )
            ):
                batches[-1].append(use)
            else:
                batches.append([use])
        return batches

    async def _run_tool(self, use: ToolUseBlock) -> ToolOutput:
        tool = self.tools.get(use.name)
        if tool is None:
            output = ToolOutput.error(
                f"There is no tool named {use.name!r}. Available tools: {', '.join(sorted(self.tools))}."
            )
            self._report_tool(use, use.name, output, 0.0)
            return output

        problem = self._validate(use)
        if problem:
            output = ToolOutput.error(problem)
            self._report_tool(use, use.name, output, 0.0)
            return output

        try:
            summary = tool.summarize(use.input)
        except Exception:
            summary = tool.name
        self.host.emit(
            ToolStarted(
                node_id=self.node_id,
                call_id=use.id,
                tool=tool.name,
                summary=summary,
                input=use.input,
            )
        )
        began = time.monotonic()
        try:
            ask = tool.permission(use.input, self.ctx)
            if ask is not None:
                decision = await self.ctx.permissions.check(
                    ask, node_id=self.node_id, tool=tool.name
                )
                if not decision.allowed:
                    reason = f" Reason: {decision.reason}" if decision.reason else ""
                    output = ToolOutput.error(
                        f"Permission denied: {ask.summary}.{reason} "
                        "Do not retry this action unchanged; take another approach or explain what you need."
                    )
                    self._finish_tool(use, output, began)
                    return output
            output = await tool.run(use.input, self.ctx)
        except asyncio.CancelledError:
            self._finish_tool(use, ToolOutput.error(INTERRUPTED), began)
            raise
        except Exception as exc:
            log.exception("tool %s crashed", tool.name)
            output = ToolOutput.error(
                f"The {tool.name} tool failed unexpectedly: {type(exc).__name__}: {exc}"
            )

        limit = self.host.config.agent.tool_output_limit
        if len(output.content) > limit:
            output.content = ctxmod.elide_middle(output.content, limit)
        self._finish_tool(use, output, began)
        return output

    def _validate(self, use: ToolUseBlock) -> str:
        if "_invalid_json" in use.input and len(use.input) == 1:
            return (
                "The arguments for this call were not valid JSON, so it was not run. "
                f"Received: {str(use.input['_invalid_json'])[:500]}"
            )
        errors = sorted(
            self._validators[use.name].iter_errors(use.input), key=lambda e: list(e.path)
        )
        if not errors:
            return ""
        lines = []
        for err in errors[:5]:
            where = ".".join(str(p) for p in err.path) or "input"
            lines.append(f"- {where}: {err.message}")
        return (
            f"Invalid input for {use.name}; the call was not run.\n"
            + "\n".join(lines)
            + f"\nExpected schema: {json.dumps(self.tools[use.name].input_schema)[:1500]}"
        )

    def _report_tool(
        self, use: ToolUseBlock, summary: str, output: ToolOutput, duration: float
    ) -> None:
        self.host.emit(
            ToolStarted(
                node_id=self.node_id,
                call_id=use.id,
                tool=use.name,
                summary=summary,
                input=use.input,
            )
        )
        self.host.emit(
            ToolFinished(
                node_id=self.node_id,
                call_id=use.id,
                output=output.content[:PREVIEW_CHARS],
                is_error=output.is_error,
                duration=duration,
            )
        )

    def _finish_tool(self, use: ToolUseBlock, output: ToolOutput, began: float) -> None:
        self.host.emit(
            ToolFinished(
                node_id=self.node_id,
                call_id=use.id,
                output=output.content[:PREVIEW_CHARS],
                is_error=output.is_error,
                duration=time.monotonic() - began,
                display=output.display,
            )
        )

    # -- context --------------------------------------------------------------------------------

    def _context_window(self) -> int:
        info = self.host.model_info(self.model)
        if info is not None and info.context_window:
            return info.context_window
        return self.host.config.agent.context_window

    async def compact(self) -> bool:
        """Summarise the conversation into a fresh one now. Returns False if there is too
        little history to be worth it. Only call between runs, never during one."""
        if len(self.messages) <= 2:
            return False
        await self._compact(emergency=False)
        return True

    async def _compact_if_needed(self) -> None:
        threshold = self.host.config.agent.compact_at * self._context_window()
        if self.context_tokens > threshold and len(self.messages) > 2:
            try:
                await self._compact(emergency=False)
            except ProviderError as exc:
                if exc.kind != "context_overflow":
                    raise
                # The history no longer fits even for a summary request: shorten it first.
                await self._compact(emergency=True)

    async def _compact(self, *, emergency: bool) -> None:
        """Replace the history with a summary and carry on in a fresh conversation.

        Normally the summary is requested as a continuation of the conversation itself (same
        system prompt and tools, so the prompt cache is hit). In an emergency - the history
        no longer fits in the model's context - a rendered, shortened transcript is summarised
        in a separate conversation instead.
        """
        provider, model_id = self.host.provider_for(self.model)
        before = self.context_tokens
        if emergency:
            window_chars = int(self._context_window() * 0.5 * ctxmod.CHARS_PER_TOKEN)
            transcript = ctxmod.render_transcript(self.messages, budget=window_chars)
            request = ChatRequest(
                model=model_id,
                system="You write faithful, detailed summaries of agent work transcripts.",
                messages=[
                    Message.user(
                        f"<transcript>\n{transcript}\n</transcript>\n\n{ctxmod.SUMMARY_PROMPT}"
                    )
                ],
                effort=self.effort,
            )
        else:
            request = ChatRequest(
                model=model_id,
                system=self.system,
                messages=[*self.messages, Message.user(ctxmod.SUMMARY_PROMPT)],
                tools=self._specs,
                effort=self.effort,
            )
        response = await provider.complete(request)
        self.usage.add(response.usage)
        summary = ctxmod.extract_summary(response.message.text)
        if not summary:
            if not emergency:
                # The model called a tool or said nothing; summarise a rendering instead.
                await self._compact(emergency=True)
                return
            raise ProviderError(
                "compaction produced no summary", provider=provider.id, kind="unknown"
            )

        self.messages = [Message.user(ctxmod.continuation_message(self._task_text, summary))]
        self.context_tokens = ctxmod.estimate_tokens(self.system, self.messages)
        self.host.record_usage(self.node_id, response.usage, self.context_tokens)
        self._notice(
            "info",
            f"Context compacted: about {before:,} tokens summarised to {self.context_tokens:,}.",
        )

    # -- helpers --------------------------------------------------------------------------------

    def _is_repeating(self, tool_uses: list[ToolUseBlock]) -> bool:
        """True on the third consecutive identical tool round (then the count starts over)."""
        calls = tuple((u.name, json.dumps(u.input, sort_keys=True, default=str)) for u in tool_uses)
        if calls == self._last_calls:
            self._repeats += 1
        else:
            self._last_calls = calls
            self._repeats = 1
        if self._repeats >= 3:
            self._repeats = 0
            return True
        return False

    def _drain_inbox(self) -> str | None:
        if not self._inbox:
            return None
        text = "\n\n".join(self._inbox)
        self._inbox.clear()
        return f"[Message from the user while you were working]\n{text}"

    def _last_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.text:
                return message.text
        return ""

    def _notice(self, level: Literal["info", "warning", "error"], text: str) -> None:
        self.host.emit(Notice(level=level, text=text, node_id=self.node_id))

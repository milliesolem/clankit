"""Provider-neutral message, request and streaming types.

Everything above the provider adapters speaks these types. Adapters translate to and from the
wire format of their API.

Two rules shape the design:

* **Histories are append-only.** Agents never rewrite an earlier message. Current Anthropic
  models bind each reasoning block to the exact conversation prefix that produced it, and every
  provider's prompt cache is a prefix match, so editing history is both an error and a cost.
* **Assistant messages keep their native payload.** Reasoning blocks carry signatures that must
  be replayed byte-for-byte. `Message.native` stores the provider's own representation; an
  adapter replays it verbatim when `native.kind` matches its own kind and otherwise rebuilds the
  message from the neutral blocks (dropping reasoning, which is not portable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause", "other"]

EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


# --------------------------------------------------------------------------------------------
# Content blocks
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Model reasoning, for display only. `text` may be empty when the provider withholds it.

    The replayable form (with signature) lives in `Message.native`.
    """

    text: str
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


def block_to_dict(block: Block) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    raise TypeError(f"not a content block: {block!r}")


def block_from_dict(data: dict[str, Any]) -> Block:
    kind = data.get("type")
    if kind == "text":
        return TextBlock(text=data["text"])
    if kind == "thinking":
        return ThinkingBlock(text=data.get("text", ""))
    if kind == "tool_use":
        return ToolUseBlock(id=data["id"], name=data["name"], input=dict(data.get("input") or {}))
    if kind == "tool_result":
        return ToolResultBlock(
            tool_use_id=data["tool_use_id"],
            content=data.get("content", ""),
            is_error=bool(data.get("is_error", False)),
        )
    raise ValueError(f"unknown content block type: {kind!r}")


# --------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NativePayload:
    """A provider's own representation of an assistant message, kept for verbatim replay.

    `kind` is the adapter kind that produced it ("anthropic", "openai-compatible").
    `data` is JSON-serialisable and opaque to everything except that adapter.
    """

    kind: str
    model: str
    data: Any


@dataclass(slots=True)
class Message:
    role: Role
    content: list[Block]
    native: NativePayload | None = None

    # -- constructors -------------------------------------------------------------------------

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text)])

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock], *, note: str | None = None) -> Message:
        """A user message answering every tool call of the previous assistant turn.

        All results of one assistant turn go into ONE user message. `note` is appended as a
        trailing text block (used for steering text and harness reminders), which keeps the
        history append-only.
        """
        content: list[Block] = list(results)
        if note:
            content.append(TextBlock(note))
        return cls(role="user", content=content)

    # -- accessors ----------------------------------------------------------------------------

    @property
    def text(self) -> str:
        """All text blocks joined; reasoning and tool blocks are excluded."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def thinking(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, ThinkingBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_result_blocks(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    # -- persistence --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": [block_to_dict(b) for b in self.content],
        }
        if self.native is not None:
            data["native"] = {
                "kind": self.native.kind,
                "model": self.native.model,
                "data": self.native.data,
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        native = data.get("native")
        return cls(
            role=data["role"],
            content=[block_from_dict(b) for b in data.get("content", [])],
            native=NativePayload(native["kind"], native.get("model", ""), native.get("data"))
            if native
            else None,
        )


# --------------------------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    """Token accounting. `input_tokens` counts only uncached prompt tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Size of the prompt as the model saw it (cached or not)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            int(data.get("input_tokens", 0)),
            int(data.get("output_tokens", 0)),
            int(data.get("cache_read_tokens", 0)),
            int(data.get("cache_write_tokens", 0)),
        )


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model addressed as ``provider/model``, e.g. ``anthropic/claude-opus-5-5``.

    Only the FIRST slash separates provider from model, so model ids that contain slashes
    (``openrouter/qwen/qwen3-coder:free``) survive.
    """

    provider: str
    model: str

    @classmethod
    def parse(cls, ref: str) -> ModelRef:
        provider, sep, model = ref.strip().partition("/")
        if not sep or not provider or not model:
            raise ValueError(f"model must be written as 'provider/model', got {ref!r}")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(slots=True)
class ModelInfo:
    """What is known about a model. Unknown facts are None, never guessed."""

    provider: str
    id: str
    name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    supports_tools: bool | None = None
    free: bool | None = None
    note: str = ""
    #: US dollars per million tokens, when known.
    input_price: float | None = None
    output_price: float | None = None

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.provider, self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "provider": self.provider,
            "id": self.id,
            "name": self.name or self.id,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "supports_tools": self.supports_tools,
            "free": self.free,
            "note": self.note,
            "input_price": self.input_price,
            "output_price": self.output_price,
        }

    def cost(self, usage: Usage) -> float | None:
        """Estimated US dollars for `usage`, or None when prices are unknown.

        Cache reads are priced at a tenth of the input price and cache writes at 1.25x, which
        is how the major providers bill them; treat the result as an estimate.
        """
        if self.input_price is None or self.output_price is None:
            return None
        return (
            usage.input_tokens * self.input_price
            + usage.cache_read_tokens * self.input_price * 0.1
            + usage.cache_write_tokens * self.input_price * 1.25
            + usage.output_tokens * self.output_price
        ) / 1_000_000


# --------------------------------------------------------------------------------------------
# Requests and responses
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ChatRequest:
    """One model call.

    `system` and `tools` must stay identical across the requests of one conversation.
    Sampling parameters are deliberately absent: current Anthropic models reject them, and
    agents should not depend on them.
    """

    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int | None = None
    effort: Effort | None = None


@dataclass(slots=True)
class ChatResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str = ""
    # Human-readable detail for unusual endings (refusal explanation, fallback model, ...).
    detail: str = ""


# --------------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """The model began emitting a tool call; arguments are still streaming."""

    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A fragment of a tool call's JSON arguments (for progress display only)."""

    index: int
    partial_json: str


@dataclass(frozen=True, slots=True)
class ResponseDone:
    """Always the final event of a stream. Carries the complete, parsed response."""

    response: ChatResponse


StreamEvent = TextDelta | ThinkingDelta | ToolCallStarted | ToolCallDelta | ResponseDone

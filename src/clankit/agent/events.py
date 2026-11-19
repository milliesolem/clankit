"""Everything a user interface can observe, as serialisable events.

The engine publishes events on a session's `EventBus`; the terminal UI, the web UI, the headless
printer and the session log are all just subscribers. No UI ever reaches into engine objects.

Events are small dataclasses. `to_dict()` gives the JSON wire form used by the web UI's event
stream and by the session log; `event_from_dict()` reverses it.

Events marked ``transient`` (streaming deltas) are neither logged nor replayed: every delta
sequence is followed by a complete `AssistantMessage`, which is what replays use.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

log = logging.getLogger(__name__)

NodeStatus = Literal[
    "pending",  # created, waiting for a free slot or for a dependency
    "running",  # its agent is working (sizing up the task, or building as a leaf)
    "waiting",  # delegated; its subtasks are running
    "integrating",  # subtasks returned; reviewing, resolving conflicts, gluing together
    "done",
    "failed",
    "cancelled",
]
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "cancelled"})

RunStatus = Literal["completed", "failed", "cancelled"]
NoticeLevel = Literal["info", "warning", "error"]
PermissionChoice = Literal["allow", "allow_session", "deny"]


@dataclass(slots=True)
class NodeInfo:
    """One node of the task tree: a unit of work owned by one agent.

    `role` is free-form but conventional: "root", "branch" (has delegated), "leaf", and for other
    orchestration methods "planner", "worker", "reviewer", "judge", "candidate", "explorer".
    """

    id: str
    parent_id: str | None
    depth: int
    title: str
    role: str = "leaf"
    model: str = ""
    status: NodeStatus = "pending"
    brief: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeInfo:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


# --------------------------------------------------------------------------------------------
# Event base
# --------------------------------------------------------------------------------------------

_REGISTRY: dict[str, type[Event]] = {}


@dataclass(kw_only=True)
class Event:
    kind: ClassVar[str] = "event"
    transient: ClassVar[bool] = False

    seq: int = 0  # assigned by the bus, strictly increasing from 1 within a session
    ts: float = 0.0  # assigned by the bus, unix seconds

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("kind")
        if isinstance(kind, str):
            _REGISTRY[kind] = cls

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        for f in fields(self):
            value = getattr(self, f.name)
            data[f.name] = value.to_dict() if isinstance(value, NodeInfo) else value
        return data


def event_from_dict(data: dict[str, Any]) -> Event:
    cls = _REGISTRY.get(data.get("kind", ""))
    if cls is None:
        raise ValueError(f"unknown event kind: {data.get('kind')!r}")
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    if cls is NodeAdded and isinstance(kwargs.get("node"), dict):
        kwargs["node"] = NodeInfo.from_dict(kwargs["node"])
    return cls(**kwargs)


# --------------------------------------------------------------------------------------------
# Run lifecycle
# --------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class RunStarted(Event):
    """The user submitted a prompt and an orchestrator started working on it."""

    kind: ClassVar[str] = "run_started"
    run_id: str
    prompt: str
    method: str
    model: str


@dataclass(kw_only=True)
class RunFinished(Event):
    kind: ClassVar[str] = "run_finished"
    run_id: str
    status: RunStatus
    answer: str = ""
    error: str = ""


# --------------------------------------------------------------------------------------------
# Task tree
# --------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class NodeAdded(Event):
    kind: ClassVar[str] = "node_added"
    node: NodeInfo


@dataclass(kw_only=True)
class NodeChanged(Event):
    """Partial update: only the fields that are not None changed."""

    kind: ClassVar[str] = "node_changed"
    node_id: str
    status: NodeStatus | None = None
    title: str | None = None
    role: str | None = None
    model: str | None = None
    summary: str | None = None


# --------------------------------------------------------------------------------------------
# Agent transcript (always scoped to a node)
# --------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class UserMessage(Event):
    """Text handed to a node's agent: the user's prompt for the root, a brief for a subtask."""

    kind: ClassVar[str] = "user_message"
    node_id: str
    text: str


@dataclass(kw_only=True)
class AgentTextDelta(Event):
    kind: ClassVar[str] = "text_delta"
    transient: ClassVar[bool] = True
    node_id: str
    message_id: str
    text: str


@dataclass(kw_only=True)
class AgentThinkingDelta(Event):
    kind: ClassVar[str] = "thinking_delta"
    transient: ClassVar[bool] = True
    node_id: str
    message_id: str
    text: str


@dataclass(kw_only=True)
class AssistantMessage(Event):
    """A complete assistant turn's prose. Follows (and supersedes) its deltas.

    Published for every model response that contains text or reasoning, including responses
    that also call tools.
    """

    kind: ClassVar[str] = "assistant_message"
    node_id: str
    message_id: str
    text: str
    thinking: str = ""


@dataclass(kw_only=True)
class ToolStarted(Event):
    kind: ClassVar[str] = "tool_started"
    node_id: str
    call_id: str
    tool: str
    summary: str  # one line, e.g. "edit src/app.py" or "$ pytest -q"
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ToolFinished(Event):
    """`output` is a preview (truncated for display); the model saw the full result.

    `display` is an optional structured payload for richer rendering:
    ``{"type": "diff", "path": str, "diff": str}`` for edits and writes.
    """

    kind: ClassVar[str] = "tool_finished"
    node_id: str
    call_id: str
    output: str
    is_error: bool = False
    duration: float = 0.0
    display: dict[str, Any] | None = None


@dataclass(kw_only=True)
class UsageChanged(Event):
    """Token usage after a model call. `usage` is the node's running total, `total` the session's.

    `context_tokens` is the size of the node's latest prompt, i.e. how full its context is.
    """

    kind: ClassVar[str] = "usage_changed"
    node_id: str
    usage: dict[str, int]
    total: dict[str, int]
    context_tokens: int = 0


# --------------------------------------------------------------------------------------------
# Interaction
# --------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class PermissionRequested(Event):
    """An agent wants to do something the permission policy does not allow outright.

    Answer with `Session.resolve_permission(request_id, choice)`.
    """

    kind: ClassVar[str] = "permission_requested"
    request_id: str
    node_id: str
    tool: str
    category: str  # "read" | "edit" | "shell" | "network"
    summary: str
    detail: str = ""
    #: What "allow for this session" would permit from now on, e.g. "shell:pytest *".
    session_rule: str = ""


@dataclass(kw_only=True)
class PermissionResolved(Event):
    kind: ClassVar[str] = "permission_resolved"
    request_id: str
    choice: PermissionChoice


@dataclass(kw_only=True)
class QuestionAsked(Event):
    """The root agent needs a decision from the user.

    Answer with `Session.answer_question(request_id, text)`. `options` are suggestions; free
    text is always acceptable.
    """

    kind: ClassVar[str] = "question_asked"
    request_id: str
    node_id: str
    question: str
    options: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class QuestionAnswered(Event):
    kind: ClassVar[str] = "question_answered"
    request_id: str
    answer: str


# --------------------------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Notice(Event):
    """Something the user should see that is not agent output (retries, compaction, limits)."""

    kind: ClassVar[str] = "notice"
    level: NoticeLevel
    text: str
    node_id: str = ""


@dataclass(kw_only=True)
class SkillSaved(Event):
    """A skill was written by the skill generator.

    "proposed" means it is staged and waiting for the user's approval; "created" and
    "updated" mean it is live in the skill library.
    """

    kind: ClassVar[str] = "skill_saved"
    name: str
    path: str
    action: Literal["proposed", "created", "updated"] = "created"
    description: str = ""


# --------------------------------------------------------------------------------------------
# Bus
# --------------------------------------------------------------------------------------------

Listener = Callable[[Event], None]


class EventBus:
    """Synchronous fan-out of a session's events, with a bounded replay buffer.

    `publish` stamps the event and calls every listener in the publishing thread; listeners
    must return quickly and must not raise (exceptions are logged and swallowed, so a broken
    UI cannot take the engine down). Subscribing and publishing are thread-safe, which lets
    web-server threads subscribe while the engine publishes from its event-loop thread; such
    listeners should only hand the event to a thread-safe queue.
    """

    def __init__(self, buffer_size: int = 8192) -> None:
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []
        self._buffer: deque[Event] = deque(maxlen=buffer_size)
        self._seq = 0

    @property
    def last_seq(self) -> int:
        return self._seq

    def publish(self, event: Event) -> Event:
        with self._lock:
            self._seq += 1
            event.seq = self._seq
            event.ts = time.time()
            self._buffer.append(event)
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # a subscriber bug must never break the engine
                log.exception("event listener failed on %s", event.kind)
        return event

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register `listener`; returns a function that unsubscribes it."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def since(self, seq: int) -> list[Event] | None:
        """Buffered events with a sequence number greater than `seq`.

        Returns None when events after `seq` have already been evicted, in which case the
        caller must resynchronise from a full snapshot instead.
        """
        with self._lock:
            if seq >= self._seq:
                return []
            if not self._buffer or self._buffer[0].seq > seq + 1:
                return None
            return [e for e in self._buffer if e.seq > seq]

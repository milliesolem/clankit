"""A session's observable state, built by folding events.

`SessionView` is the single view model behind both user interfaces: feed it every event from
the bus (`view.apply(event)`) and read the task tree and transcripts back out. The web UI
sends `snapshot()` to a browser when it connects and then streams events; the browser folds
those events with the same rules, so the two stay identical.

Folding rules for transcripts (each node has an ordered list of `Item`):

* `UserMessage`            -> a "user" item.
* `AgentThinkingDelta`     -> create or extend the item ``<message_id>:thinking`` (not done).
* `AgentTextDelta`         -> create or extend the item ``<message_id>:text`` (not done).
* `AssistantMessage`       -> finalise both items with the complete text. An item whose final
                              text is empty is removed (that is how a failed, partially
                              streamed message is withdrawn).
* `ToolStarted`            -> a "tool" item with id ``call_id`` (not done).
* `ToolFinished`           -> completes that item.
* `Notice`                 -> a "notice" item on its node, or on the newest root node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clankit.agent.events import (
    TERMINAL_STATUSES,
    AgentTextDelta,
    AgentThinkingDelta,
    AssistantMessage,
    Event,
    NodeAdded,
    NodeChanged,
    NodeInfo,
    Notice,
    PermissionRequested,
    PermissionResolved,
    QuestionAnswered,
    QuestionAsked,
    RunFinished,
    RunStarted,
    SkillSaved,
    ToolFinished,
    ToolStarted,
    UsageChanged,
    UserMessage,
)


@dataclass(slots=True)
class Item:
    """One entry of a node's transcript.

    `kind` is "user", "assistant", "thinking", "tool" or "notice". Tool items use `tool`,
    `summary`, `input`, `output`, `is_error`, `duration` and `display`; notices use `level`.
    """

    id: str
    kind: str
    seq: int
    ts: float
    text: str = ""
    done: bool = True
    tool: str = ""
    summary: str = ""
    input: dict[str, Any] | None = None
    output: str = ""
    is_error: bool = False
    duration: float = 0.0
    display: dict[str, Any] | None = None
    level: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "seq": self.seq,
            "ts": self.ts,
            "text": self.text,
            "done": self.done,
        }
        if self.kind == "tool":
            data.update(
                tool=self.tool,
                summary=self.summary,
                input=self.input,
                output=self.output,
                is_error=self.is_error,
                duration=self.duration,
                display=self.display,
            )
        elif self.kind == "notice":
            data["level"] = self.level
        return data


@dataclass(slots=True)
class NodeView:
    info: NodeInfo
    children: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    context_tokens: int = 0
    started: float = 0.0
    finished: float = 0.0

    def item(self, item_id: str) -> Item | None:
        for item in reversed(self.items):
            if item.id == item_id:
                return item
        return None

    @property
    def tokens(self) -> int:
        u = self.usage
        return sum(
            u.get(k, 0)
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.info.to_dict(),
            "children": list(self.children),
            "items": [i.to_dict() for i in self.items],
            "usage": dict(self.usage),
            "context_tokens": self.context_tokens,
            "started": self.started,
            "finished": self.finished,
        }


class SessionView:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeView] = {}
        self.roots: list[str] = []
        self.running = False
        self.run: RunStarted | None = None
        self.last_run: RunFinished | None = None
        self.total_usage: dict[str, int] = {}
        self.permissions: dict[str, PermissionRequested] = {}
        self.questions: dict[str, QuestionAsked] = {}
        self.skills_saved: list[SkillSaved] = []
        self.last_seq = 0

    # -- reading --------------------------------------------------------------------------------

    def node(self, node_id: str) -> NodeView | None:
        return self.nodes.get(node_id)

    def conversation(self) -> list[Item]:
        """The top-level transcript: the items of every root node, in event order."""
        items = [item for root in self.roots for item in self.nodes[root].items]
        items.sort(key=lambda i: i.seq)
        return items

    def walk(self, node_id: str | None = None, depth: int = 0) -> list[tuple[int, NodeView]]:
        """Depth-first listing of the tree as (indent level, node) pairs."""
        out: list[tuple[int, NodeView]] = []
        ids = [node_id] if node_id else self.roots
        for nid in ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            out.append((depth, node))
            for child in node.children:
                out.extend(self.walk(child, depth + 1))
        return out

    def counts(self) -> dict[str, int]:
        """Number of nodes per status."""
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.info.status] = counts.get(node.info.status, 0) + 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        return {
            "seq": self.last_seq,
            "running": self.running,
            "run": self.run.to_dict() if self.run else None,
            "last_run": self.last_run.to_dict() if self.last_run else None,
            "roots": list(self.roots),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "total_usage": dict(self.total_usage),
            "permissions": [p.to_dict() for p in self.permissions.values()],
            "questions": [q.to_dict() for q in self.questions.values()],
        }

    # -- folding --------------------------------------------------------------------------------

    def apply(self, event: Event) -> None:
        if event.seq:
            self.last_seq = max(self.last_seq, event.seq)

        if isinstance(event, RunStarted):
            self.running = True
            self.run = event
        elif isinstance(event, RunFinished):
            self.running = False
            self.last_run = event
            self.permissions.clear()
            self.questions.clear()
        elif isinstance(event, NodeAdded):
            info = event.node
            self.nodes[info.id] = NodeView(info=info)
            parent = self.nodes.get(info.parent_id) if info.parent_id else None
            if parent is not None:
                parent.children.append(info.id)
            else:
                self.roots.append(info.id)
        elif isinstance(event, NodeChanged):
            node = self.nodes.get(event.node_id)
            if node is None:
                return
            if event.status is not None:
                node.info.status = event.status
                if event.status == "running" and not node.started:
                    node.started = event.ts
                if event.status in TERMINAL_STATUSES:
                    node.finished = event.ts
                else:
                    node.finished = 0.0
            if event.title is not None:
                node.info.title = event.title
            if event.role is not None:
                node.info.role = event.role
            if event.model is not None:
                node.info.model = event.model
            if event.summary is not None:
                node.info.summary = event.summary
        elif isinstance(event, UserMessage):
            self._append(
                event.node_id,
                Item(id=f"u{event.seq}", kind="user", seq=event.seq, ts=event.ts, text=event.text),
            )
        elif isinstance(event, AgentThinkingDelta):
            self._extend(event.node_id, f"{event.message_id}:thinking", "thinking", event)
        elif isinstance(event, AgentTextDelta):
            self._extend(event.node_id, f"{event.message_id}:text", "assistant", event)
        elif isinstance(event, AssistantMessage):
            self._finalise(
                event.node_id, f"{event.message_id}:thinking", "thinking", event.thinking, event
            )
            self._finalise(
                event.node_id, f"{event.message_id}:text", "assistant", event.text, event
            )
        elif isinstance(event, ToolStarted):
            self._append(
                event.node_id,
                Item(
                    id=event.call_id,
                    kind="tool",
                    seq=event.seq,
                    ts=event.ts,
                    done=False,
                    tool=event.tool,
                    summary=event.summary,
                    input=event.input,
                ),
            )
        elif isinstance(event, ToolFinished):
            node = self.nodes.get(event.node_id)
            item = node.item(event.call_id) if node else None
            if item is not None:
                item.done = True
                item.output = event.output
                item.is_error = event.is_error
                item.duration = event.duration
                item.display = event.display
        elif isinstance(event, UsageChanged):
            node = self.nodes.get(event.node_id)
            if node is not None:
                node.usage = dict(event.usage)
                node.context_tokens = event.context_tokens
            self.total_usage = dict(event.total)
        elif isinstance(event, PermissionRequested):
            self.permissions[event.request_id] = event
        elif isinstance(event, PermissionResolved):
            self.permissions.pop(event.request_id, None)
        elif isinstance(event, QuestionAsked):
            self.questions[event.request_id] = event
        elif isinstance(event, QuestionAnswered):
            self.questions.pop(event.request_id, None)
        elif isinstance(event, Notice):
            target = (
                event.node_id
                if event.node_id in self.nodes
                else (self.roots[-1] if self.roots else "")
            )
            if target:
                self._append(
                    target,
                    Item(
                        id=f"n{event.seq}",
                        kind="notice",
                        seq=event.seq,
                        ts=event.ts,
                        text=event.text,
                        level=event.level,
                    ),
                )
        elif isinstance(event, SkillSaved):
            self.skills_saved.append(event)

    # -- helpers --------------------------------------------------------------------------------

    def _append(self, node_id: str, item: Item) -> None:
        node = self.nodes.get(node_id)
        if node is not None:
            node.items.append(item)

    def _extend(
        self, node_id: str, item_id: str, kind: str, event: AgentTextDelta | AgentThinkingDelta
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        item = node.item(item_id)
        if item is None:
            node.items.append(
                Item(id=item_id, kind=kind, seq=event.seq, ts=event.ts, text=event.text, done=False)
            )
        else:
            item.text += event.text

    def _finalise(
        self, node_id: str, item_id: str, kind: str, text: str, event: AssistantMessage
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        item = node.item(item_id)
        if not text:
            if item is not None:
                node.items.remove(item)
            return
        if item is None:
            node.items.append(Item(id=item_id, kind=kind, seq=event.seq, ts=event.ts, text=text))
        else:
            # Adopt the final event's position so a replay (which never sees deltas) builds
            # exactly the same transcript as the live fold.
            item.text = text
            item.done = True
            item.seq = event.seq
            item.ts = event.ts

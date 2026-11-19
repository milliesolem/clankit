"""What the transcript shows, as a flat list of keyed entries.

The transcript is rebuilt from state, not from events: every refresh asks this module for
the entries of the node in view and the transcript widget reconciles its blocks against
them by key. Three sources are merged:

* the folded `SessionView` items (prose, reasoning, tool calls, notices);
* the run log kept here (`History`): each run's prompt and how it ended, which the view does
  not record as items;
* things only this interface knows: command output, steering messages, and how permission
  requests and questions were answered.

The root view orders everything by event time. Sequence numbers alone cannot do that: a
resumed session starts numbering its new events from 1 again, while timestamps keep
growing.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from clankit.agent.events import (
    Event,
    PermissionRequested,
    PermissionResolved,
    QuestionAnswered,
    QuestionAsked,
    RunFinished,
    RunStarted,
    SkillSaved,
)
from clankit.agent.state import Item, SessionView

_local_ids = itertools.count(1)

_CHOICE_WORDS = {
    "allow": "allowed once",
    "allow_session": "allowed for this session",
    "deny": "denied",
}


@dataclass(slots=True)
class RunRecord:
    """One prompt run: what was asked and, once it ended, how it ended."""

    run_id: str
    prompt: str
    method: str
    ts: float
    seq: float
    finished_ts: float = 0.0
    finished_seq: float = 0.0
    status: str = ""
    error: str = ""

    @property
    def finished(self) -> bool:
        return bool(self.status)


@dataclass(slots=True)
class LocalEntry:
    """A transcript entry that exists only in this interface.

    `kind` is "message" (command output; `level` says how serious), "steer" (a message sent
    to the agent mid-run), "resolution" (how a request was answered: `text` is the verdict,
    `detail` what it was about, `context` the question for an answer) or "skill" (a skill
    was saved or proposed; `level` is the SkillSaved action).
    """

    key: str
    kind: str
    ts: float
    seq: float
    text: str
    level: str = "info"
    detail: str = ""
    context: str = ""


@dataclass(slots=True)
class History:
    """The run log and interface-only entries of one session."""

    runs: list[RunRecord] = field(default_factory=list)
    local: list[LocalEntry] = field(default_factory=list)
    _requests: dict[str, Event] = field(default_factory=dict)
    _last_seq: float = 0.0

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> History:
        history = cls()
        for event in events:
            history.observe(event)
        return history

    def observe(self, event: Event) -> None:
        """Fold one event into the log; events that do not concern it are ignored."""
        self._last_seq = max(self._last_seq, float(event.seq))
        if isinstance(event, RunStarted):
            self.runs.append(
                RunRecord(
                    run_id=event.run_id,
                    prompt=event.prompt,
                    method=event.method,
                    ts=event.ts,
                    seq=event.seq,
                )
            )
        elif isinstance(event, RunFinished):
            run = next((r for r in reversed(self.runs) if r.run_id == event.run_id), None)
            if run is not None:
                run.status = event.status
                run.error = event.error
                run.finished_ts = event.ts
                run.finished_seq = event.seq
        elif isinstance(event, (PermissionRequested, QuestionAsked)):
            self._requests[event.request_id] = event
        elif isinstance(event, PermissionResolved):
            request = self._requests.pop(event.request_id, None)
            if isinstance(request, PermissionRequested):
                self._append(
                    "resolution",
                    _CHOICE_WORDS.get(event.choice, event.choice),
                    event,
                    detail=request.summary,
                    level="error" if event.choice == "deny" else "info",
                )
        elif isinstance(event, QuestionAnswered):
            request = self._requests.pop(event.request_id, None)
            if isinstance(request, QuestionAsked):
                answer = event.answer.strip() or "(no answer)"
                self._append(
                    "resolution", "answered", event, detail=answer, context=request.question
                )
        elif isinstance(event, SkillSaved):
            self._append("skill", event.name, event, detail=event.description, level=event.action)

    def add(self, kind: str, text: str, level: str = "info") -> LocalEntry:
        """Record something the interface itself wants to show, stamped now."""
        entry = LocalEntry(
            key=f"local{next(_local_ids)}",
            kind=kind,
            ts=time.time(),
            seq=self._last_seq + 0.5,
            text=text,
            level=level,
        )
        self.local.append(entry)
        return entry

    def current_run(self) -> RunRecord | None:
        return self.runs[-1] if self.runs and not self.runs[-1].finished else None

    def _append(
        self,
        kind: str,
        text: str,
        event: Event,
        *,
        detail: str,
        level: str = "info",
        context: str = "",
    ) -> None:
        self.local.append(
            LocalEntry(
                key=f"local{next(_local_ids)}",
                kind=kind,
                ts=event.ts,
                seq=event.seq,
                text=text,
                level=level,
                detail=detail,
                context=context,
            )
        )


@dataclass(frozen=True, slots=True)
class Entry:
    """One transcript block. `key` is stable across refreshes; `kind` picks the widget.

    Kinds: "prompt", "message" (text handed to an agent that is not the user's prompt: a
    subtask brief, review findings), "assistant", "thinking", "tool", "notice", "run-end",
    "steer", "local" (command output), "resolution", "skill".
    """

    key: str
    kind: str
    ts: float
    seq: float
    item: Item | None = None
    run: RunRecord | None = None
    local: LocalEntry | None = None


def item_kind(item: Item) -> str:
    return {"user": "message"}.get(item.kind, item.kind)


def root_entries(view: SessionView, history: History, *, show_thinking: bool) -> list[Entry]:
    """The conversation as the root view shows it, in event-time order."""
    entries: list[Entry] = []
    runs = history.runs
    for root_id in view.roots:
        node = view.nodes.get(root_id)
        if node is None:
            continue
        for item in node.items:
            if item.kind == "thinking" and not show_thinking:
                continue
            kind = item_kind(item)
            if item.kind == "user":
                run = _run_at(runs, item.ts)
                if run is None:
                    kind = "prompt"
                elif _repeats_prompt(item.text, run.prompt):
                    continue
            entries.append(Entry(item.id, kind, item.ts, item.seq, item=item))
    for run in runs:
        entries.append(Entry(f"run-{run.run_id}", "prompt", run.ts, run.seq, run=run))
        if run.finished:
            entries.append(
                Entry(f"end-{run.run_id}", "run-end", run.finished_ts, run.finished_seq, run=run)
            )
    for local in history.local:
        kind = "local" if local.kind == "message" else local.kind
        entries.append(Entry(local.key, kind, local.ts, local.seq, local=local))
    entries.sort(key=lambda e: (e.ts, e.seq))
    return entries


def node_entries(view: SessionView, node_id: str, *, show_thinking: bool) -> list[Entry]:
    """One node's transcript, in the order its agent produced it."""
    node = view.nodes.get(node_id)
    if node is None:
        return []
    return [
        Entry(item.id, item_kind(item), item.ts, item.seq, item=item)
        for item in node.items
        if show_thinking or item.kind != "thinking"
    ]


def _run_at(runs: list[RunRecord], ts: float) -> RunRecord | None:
    """The latest run that had started by `ts`."""
    found = None
    for run in runs:
        if run.ts <= ts:
            found = run
        else:
            break
    return found


def _repeats_prompt(text: str, prompt: str) -> bool:
    """Does an agent's user message just carry the run's prompt (perhaps after a recap)?"""
    prompt = prompt.strip()
    return bool(prompt) and text.rstrip().endswith(prompt)

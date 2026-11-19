"""What a workflow run hands back: its result and a compact account of how it went.

The account is written for the agent that wrote the script. It must be short enough to keep
that agent's context clean, and honest enough that nothing is lost silently: every failed
agent, every slot that came back empty, every warning and the script's own log lines are
listed (the log tail is cut, and the cut is stated). When a run did not complete, the
results of the agents that did finish are included, so the work they did is not wasted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

OutcomeStatus = Literal["completed", "failed", "stopped"]

#: Log lines shown in the account; earlier ones are counted, not shown.
LOG_TAIL = 20
#: Characters of one finished agent's result shown when a run did not complete.
PARTIAL_ITEM = 600
#: Entries shown per list (failures, empty results, warnings), and characters per entry.
LIST_ITEMS = 30
LIST_ITEM_CHARS = 500


@dataclass(slots=True)
class AgentRecord:
    """One `agent()` call that started (or was reused from an earlier run)."""

    node_id: str
    label: str
    phase: str
    line: int
    status: str = "pending"  # pending | running | done | failed | reused | cancelled
    value: Any = None
    error: str = ""


@dataclass(slots=True)
class Outcome:
    """The end of one workflow run."""

    name: str
    status: OutcomeStatus
    result: Any
    error: str
    node_id: str
    tokens: int
    budget: int
    records: list[AgentRecord] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    empties: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    # -- summaries ------------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def tally(self) -> str:
        """Like "14 agents (12 done, 1 failed, 1 reused)"."""
        counts = self.counts()
        parts = [
            f"{counts[s]} {s}" for s in ("done", "failed", "reused", "cancelled") if counts.get(s)
        ]
        total = len(self.records)
        noun = "agent" if total == 1 else "agents"
        return f"{total} {noun}" + (f" ({', '.join(parts)})" if parts else "")

    def headline(self) -> str:
        spend = f"{self.tokens:,} of {self.budget:,} tokens"
        if self.status == "completed":
            return f'Workflow "{self.name}" completed: {self.tally()}; {spend}.'
        if self.status == "stopped":
            return (
                f'Workflow "{self.name}" stopped before it finished: {self.error}. '
                f"{self.tally()}; {spend}."
            )
        return f'Workflow "{self.name}" failed: {self.error}\n{self.tally()}; {spend}.'

    def phase_line(self) -> str:
        parts = []
        for title in self.phases:
            records = [r for r in self.records if r.phase == title]
            failed = sum(1 for r in records if r.status == "failed")
            text = f"{title} {len(records)} agent{'' if len(records) == 1 else 's'}"
            parts.append(text + (f", {failed} failed" if failed else ""))
        return "Phases: " + "; ".join(parts) + "." if parts else ""

    # -- rendering ------------------------------------------------------------------------

    def render(self, *, limit: int = 30_000) -> str:
        """The account and the result as one text of at most about `limit` characters."""
        sections = [self.headline()]
        if line := self.phase_line():
            sections.append(line)
        sections.extend(
            _listing(title, items)
            for title, items in (
                ("Failures", self.failures),
                ("Results that came back empty", self.empties),
                ("Warnings", self.warnings),
            )
            if items
        )
        if self.logs:
            shown = self.logs[-LOG_TAIL:]
            hidden = len(self.logs) - len(shown)
            title = f"Log (last {len(shown)} of {len(self.logs)} lines)" if hidden else "Log"
            sections.append(_listing(title, shown))
        head = "\n\n".join(sections)
        room = max(limit - len(head), 2_000)
        if self.status == "completed":
            return f"{head}\n\n<result>\n{_clip(render_value(self.result), room)}\n</result>"
        tail = self._partial(room)
        return f"{head}\n\n{tail}" if tail else head

    def _partial(self, room: int) -> str:
        parts = []
        if self.result is not None:
            parts.append(
                "The script had assigned `result` before it ended:\n"
                f"<result>\n{_clip(render_value(self.result), room // 2)}\n</result>"
            )
        finished = [r for r in self.records if r.status in ("done", "reused")]
        if finished:
            lines = [
                f"- {r.label}"
                + (f" [{r.phase}]" if r.phase else "")
                + f": {_clip(render_value(r.value), PARTIAL_ITEM)}"
                for r in finished
            ]
            parts.append(_clip_lines("Results of the agents that finished:", lines, room // 2))
        return "\n\n".join(parts)


def render_value(value: Any) -> str:
    """A result as text: strings as they are, everything else as indented JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=repr)


def _listing(title: str, items: list[str]) -> str:
    lines = [f"- {_clip(item, LIST_ITEM_CHARS)}" for item in items[:LIST_ITEMS]]
    if len(items) > LIST_ITEMS:
        lines.append(f"- ({len(items) - LIST_ITEMS} more not shown)")
    return f"{title}:\n" + "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated: {limit:,} of {len(text):,} characters shown]"


def _clip_lines(title: str, lines: list[str], limit: int) -> str:
    shown: list[str] = []
    size = len(title)
    for line in lines:
        if size + len(line) > limit:
            break
        shown.append(line)
        size += len(line) + 1
    if len(shown) < len(lines):
        shown.append(f"- ({len(lines) - len(shown)} more not shown)")
    return "\n".join([title, *shown])

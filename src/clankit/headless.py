"""Running a prompt without a UI: ``clankit run "..."``.

Progress goes to standard error as a compact activity log (one line per tool call, node
status changes indented by depth); the final answer goes to standard output, so the command
composes in pipelines. ``--json`` instead prints every event as one JSON object per line on
standard output. Nobody can answer questions or approve actions here, so anything the
permission policy would ask about is refused: choose a permission mode (or allow rules)
that covers what the run needs.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, TextIO

from clankit.agent.events import (
    Event,
    NodeAdded,
    NodeChanged,
    Notice,
    RunFinished,
    ToolFinished,
    ToolStarted,
)

if TYPE_CHECKING:
    from clankit.agent.session import Session


class ProgressPrinter:
    """Writes a readable activity log of a session to a stream."""

    def __init__(self, session: Session, stream: TextIO) -> None:
        self._session = session
        self._stream = stream
        self._summaries: dict[str, str] = {}

    def _indent(self, node_id: str) -> str:
        node = self._session.view.node(node_id)
        return "  " * (node.info.depth if node else 0)

    def __call__(self, event: Event) -> None:
        line = self._line(event)
        if line:
            print(line, file=self._stream, flush=True)

    def _line(self, event: Event) -> str:
        if isinstance(event, NodeAdded) and event.node.parent_id:
            model = f"  [{event.node.model}]" if event.node.model else ""
            return f"{'  ' * event.node.depth}+ {event.node.title}{model}"
        if isinstance(event, NodeChanged) and event.status in ("done", "failed", "cancelled"):
            node = self._session.view.node(event.node_id)
            if node is None or node.info.parent_id is None:
                return ""
            mark = {"done": "ok", "failed": "FAILED", "cancelled": "cancelled"}[event.status]
            return f"{self._indent(event.node_id)}= {node.info.title}: {mark}"
        if isinstance(event, ToolStarted):
            self._summaries[event.call_id] = event.summary
            return ""
        if isinstance(event, ToolFinished):
            summary = self._summaries.pop(event.call_id, "tool")
            status = "  (error)" if event.is_error else ""
            return f"{self._indent(event.node_id)}  {summary}{status}"
        if isinstance(event, Notice):
            return f"{event.level}: {event.text}"
        return ""


async def run_headless(
    session: Session, prompt: str, *, as_json: bool = False, quiet: bool = False
) -> int:
    """Run `prompt` in `session`; return a process exit code (0 completed, 1 failed, 130 cancelled)."""
    finished: list[RunFinished] = []
    session.bus.subscribe(lambda e: finished.append(e) if isinstance(e, RunFinished) else None)
    if as_json:
        session.bus.subscribe(
            lambda e: print(json.dumps(e.to_dict(), ensure_ascii=False, default=str), flush=True)
        )
    elif not quiet:
        session.bus.subscribe(ProgressPrinter(session, sys.stderr))

    answer = await session.run(prompt)
    outcome = finished[-1] if finished else None
    if not as_json and answer:
        print(answer)
    if outcome is None or outcome.status == "failed":
        if not as_json and outcome is not None and outcome.error:
            print(f"clankit: {outcome.error}", file=sys.stderr)
        return 1
    return 130 if outcome.status == "cancelled" else 0

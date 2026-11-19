"""The sessions the web interface shows, and the history of runs in each.

Everything here touches engine objects and therefore runs on the runtime's event loop, called
through `Engine.submit`.

The conversation view shows every run as the prompt the user typed. `SessionView` keeps only
the current and the last run, and the root agent's own user messages are not always the
user's words (a resumed conversation starts with a recap, a review loop feeds findings back),
so the index records each `RunStarted` / `RunFinished` of the sessions it shows, and reads them
back from the session log when a stored session is resumed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clankit.agent.events import Event, RunFinished, RunStarted
from clankit.web.context import ApiError

if TYPE_CHECKING:
    from clankit.agent.session import Session
    from clankit.runtime import Runtime

NEW_SESSION_TITLE = "New session"


@dataclass(slots=True)
class RunRecord:
    """One prompt the user gave a session, and how the run it started ended."""

    seq: int
    run_id: str
    prompt: str
    method: str
    status: str = ""
    finished_seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "method": self.method,
            "status": self.status,
            "finished_seq": self.finished_seq,
        }


@dataclass(slots=True)
class _Watched:
    session: Session
    runs: list[RunRecord] = field(default_factory=list)
    opened: float = field(default_factory=time.time)
    unsubscribe: Callable[[], None] | None = None

    def record(self, event: Event) -> None:
        """Bus listener (runs on the loop thread, which publishes every event)."""
        if isinstance(event, RunStarted):
            self.runs.append(_started(event))
        elif isinstance(event, RunFinished):
            _finish(self.runs, event)


def _started(event: RunStarted) -> RunRecord:
    return RunRecord(event.seq, event.run_id, event.prompt, event.method)


def _finish(runs: list[RunRecord], event: RunFinished) -> None:
    for run in reversed(runs):
        if run.run_id == event.run_id:
            run.status = event.status
            run.finished_seq = event.seq
            return


def title_of(text: str, limit: int = 80) -> str:
    """A one-line title from a prompt, cut at a word boundary when it is too long."""
    line = " ".join(text.split())
    if len(line) <= limit:
        return line
    cut = line[: limit - 1].rsplit(" ", 1)[0] or line[: limit - 1]
    return cut.rstrip(",.;:") + "…"


class SessionIndex:
    """The live sessions of one workspace that the web interface has shown, newest last."""

    def __init__(self, runtime: Callable[[], Runtime], workspace: Path) -> None:
        self._runtime = runtime
        self.workspace = workspace.resolve()
        self._watched: dict[str, _Watched] = {}

    @property
    def runtime(self) -> Runtime:
        return self._runtime()

    # -- lookup ---------------------------------------------------------------------------------

    def get(self, session_id: str) -> Session:
        """A live session of this workspace. Raises a 404 `ApiError` for any other id."""
        session = self.runtime.get_session(session_id)
        if session is None or session.workspace != self.workspace:
            raise ApiError(404, f"There is no open session {session_id!r}.")
        self.watch(session)
        return session

    def current(self) -> Session | None:
        """The session a newly opened tab shows: the one opened most recently."""
        live = self.runtime.sessions
        for session_id in reversed(list(self._watched)):
            if session_id in live:
                return live[session_id]
        mine = [s for s in live.values() if s.workspace == self.workspace]
        if not mine:
            return None
        self.watch(mine[-1])
        return mine[-1]

    def runs(self, session: Session) -> list[RunRecord]:
        return self.watch(session).runs

    # -- opening --------------------------------------------------------------------------------

    def watch(self, session: Session) -> _Watched:
        """Start recording the runs of `session` (idempotent)."""
        watched = self._watched.get(session.id)
        if watched is None:
            watched = self._watched[session.id] = _Watched(session)
            watched.unsubscribe = session.bus.subscribe(watched.record)
        return watched

    def create(self, *, method: str | None = None, like: Session | None = None) -> Session:
        """A new session in this workspace.

        It keeps the per-session overrides of `like` (the session the user is leaving), so a
        server started with ``--demo`` or ``--method`` stays that way; `method` overrides it.
        """
        from clankit.orchestration import method_names

        if method is not None and method not in method_names():
            raise ApiError(
                400, f"Unknown method {method!r}. Available: {', '.join(method_names())}."
            )
        session = self.runtime.create_session(self.workspace, interactive=True)
        if like is not None:
            session.method_override = like.method_override
            session.model_override = like.model_override
        if method is not None:
            session.method_override = method
        self.watch(session)
        return session

    def resume(self, session_id: str) -> Session:
        """Reopen a stored session of this workspace (or switch to it if it is open)."""
        live = self.runtime.get_session(session_id)
        if live is not None:
            return self.get(session_id)
        if session_id not in {meta.id for meta in self.runtime.session_store.list(self.workspace)}:
            raise ApiError(404, f"There is no stored session {session_id!r} for this folder.")
        try:
            session = self.runtime.resume_session(session_id, interactive=True)
            _, events = self.runtime.session_store.load(session_id)
        except FileNotFoundError as exc:
            raise ApiError(404, f"There is no stored session {session_id!r}.") from exc
        _continue_numbering(session)
        watched = self.watch(session)
        for event in events:
            if isinstance(event, RunStarted):
                watched.runs.append(_started(event))
            elif isinstance(event, RunFinished):
                _finish(watched.runs, event)
        return session

    # -- listing --------------------------------------------------------------------------------

    def title(self, session: Session) -> str:
        runs = self.runs(session)
        if runs:
            return title_of(runs[0].prompt) or NEW_SESSION_TITLE
        return title_of(session.turns[0].prompt) if session.turns else NEW_SESSION_TITLE

    def listing(self) -> list[dict[str, Any]]:
        """Stored and open sessions of this workspace, most recently used first."""
        rows: dict[str, dict[str, Any]] = {}
        for meta in self.runtime.session_store.list(self.workspace):
            rows[meta.id] = {
                "id": meta.id,
                "title": meta.title or NEW_SESSION_TITLE,
                "updated": meta.updated,
                "turns": meta.turns,
                "live": False,
            }
        for session in self.runtime.sessions.values():
            if session.workspace != self.workspace:
                continue
            watched = self.watch(session)
            row = rows.setdefault(session.id, {"id": session.id, "updated": watched.opened})
            row.update(title=self.title(session), turns=len(session.turns), live=True)
        return sorted(rows.values(), key=lambda row: row["updated"], reverse=True)

    def close(self) -> None:
        for watched in self._watched.values():
            if watched.unsubscribe is not None:
                watched.unsubscribe()
        self._watched.clear()


def _continue_numbering(session: Session) -> None:
    """Make a resumed session's new events number on from its restored ones.

    `Session.restore` replays the log into the view but leaves the event bus counting from 1,
    so new events would sort before the restored transcript and look older than the snapshot
    a browser already holds. Until the engine continues the numbering itself, the counter is
    advanced here, before anything new is published.
    """
    bus = session.bus
    restored = session.view.last_seq
    if bus.last_seq < restored:
        with bus._lock:
            bus._seq = max(bus._seq, restored)

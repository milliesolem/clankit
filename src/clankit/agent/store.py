"""Session logs on disk: one JSON-lines file per session.

The first line is a header (``{"kind": "session", ...}``); every following line is one
non-transient event. Replaying the events through a `SessionView` rebuilds the transcript
and task tree, which is how a session is resumed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from clankit import paths
from clankit.agent.events import Event, event_from_dict

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionMeta:
    id: str
    workspace: str
    created: float
    updated: float
    title: str
    turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace": self.workspace,
            "created": self.created,
            "updated": self.updated,
            "title": self.title,
            "turns": self.turns,
        }


class SessionLog:
    """Append-only writer for one session's events."""

    def __init__(self, path: Path, *, session_id: str, workspace: Path) -> None:
        self.path = path
        self._header = {"kind": "session", "id": session_id, "workspace": str(workspace)}
        self._file: IO[str] | None = None
        self._closed = False

    def append(self, event: Event) -> None:
        if not event.transient:
            self._write(event.to_dict())

    def close(self) -> None:
        self._closed = True
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def _open(self) -> None:
        """Create the file on first use, so sessions in which nothing happened leave no trace."""
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        if new:
            self._file.write(json.dumps({**self._header, "created": time.time()}) + "\n")

    def _write(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            if self._file is None:
                self._open()
            assert self._file is not None
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except OSError:
            log.exception("could not write session log %s", self.path)


class SessionStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or paths.sessions_dir()

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.jsonl"

    def open(self, session_id: str, workspace: Path) -> SessionLog:
        return SessionLog(self.path_for(session_id), session_id=session_id, workspace=workspace)

    def load(self, session_id: str) -> tuple[dict[str, Any], list[Event]]:
        """Header and events of a stored session. Unreadable lines are skipped."""
        path = self.path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"no stored session {session_id!r}")
        header: dict[str, Any] = {}
        events: list[Event] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    if record.get("kind") == "session":
                        header = record
                    else:
                        events.append(event_from_dict(record))
                except (ValueError, TypeError, KeyError):
                    continue
        return header, events

    def list(self, workspace: Path | None = None) -> list[SessionMeta]:
        """Stored sessions, most recently used first; optionally only those of one workspace."""
        metas: list[SessionMeta] = []
        if not self.directory.exists():
            return metas
        for path in self.directory.glob("*.jsonl"):
            meta = self._meta(path)
            if meta is None or meta.turns == 0:
                continue
            if workspace is not None and Path(meta.workspace) != workspace:
                continue
            metas.append(meta)
        metas.sort(key=lambda m: m.updated, reverse=True)
        return metas

    def _meta(self, path: Path) -> SessionMeta | None:
        header: dict[str, Any] = {}
        title = ""
        turns = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if '"kind": "session"' in line or '"kind": "run_started"' in line:
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if record.get("kind") == "session":
                            header = record
                        elif record.get("kind") == "run_started":
                            turns += 1
                            title = title or " ".join(str(record.get("prompt", "")).split())[:80]
            updated = path.stat().st_mtime
        except OSError:
            return None
        if not header:
            return None
        return SessionMeta(
            id=str(header.get("id", path.stem)),
            workspace=str(header.get("workspace", "")),
            created=float(header.get("created", updated)),
            updated=updated,
            title=title,
            turns=turns,
        )

"""The resume journal: reusing finished agents when an edited script runs again.

A workflow that fails half way (a bug in the synthesis step, a limit reached) should not
have to pay again for the twenty agents that already finished. When the agent that wrote the
script fixes it and runs it with ``resume``, every agent call whose *position* in the script
and whose *inputs* are unchanged returns its earlier result instead of running.

Position is the structural call path (``root/parallel:0/slot:3/agent:0``), which the engine
derives from the script's own control flow, not from the order in which concurrent agents
happened to start; inputs are a hash of the prompt, schema, model, tools, write scope and
effort. Editing one prompt therefore re-runs exactly the calls it affects.

One conservative rule covers what hashes cannot see: an agent that can change the workspace
(it has bash or the write tools) and has to run again may change files that later agents
read, so once such an agent misses the journal, nothing that starts after it is reused.

Entries live in memory for the session, keyed by workflow name; only successful results are
kept. Values are copied in and out, so a script that modifies a returned dict cannot change
what a later run gets back.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


class Journal:
    """Finished agent results of the latest run of each workflow, for one session."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def begin(self, name: str, *, resume: bool) -> JournalRun:
        """Start recording a run of workflow `name`; with `resume`, reuse the previous one."""
        previous = self._runs.get(name) if resume else None
        return JournalRun(self, name, previous, requested=resume)

    def _commit(self, name: str, entries: dict[str, Any]) -> None:
        self._runs[name] = entries


class JournalRun:
    """The journal as seen by one run: lookups against the previous run, records for the next."""

    def __init__(
        self, journal: Journal, name: str, previous: dict[str, Any] | None, *, requested: bool
    ) -> None:
        self._journal = journal
        self._name = name
        self._previous = previous
        self._entries: dict[str, Any] = {}
        self._open = previous is not None
        #: True when resuming was asked for; `available` says whether there was anything.
        self.requested = requested
        self.available = bool(previous)
        self.hits = 0

    @staticmethod
    def key(path: str, inputs: Any) -> str:
        digest = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False).encode()
        ).hexdigest()
        return f"{path}#{digest[:20]}"

    def lookup(self, key: str, *, mutating: bool) -> tuple[bool, Any]:
        """(True, value) when `key` finished in the previous run and may be reused."""
        if self._open and self._previous is not None and key in self._previous:
            self.hits += 1
            value = self._previous[key]
            self._entries[key] = value
            return True, copy.deepcopy(value)
        if mutating:
            self._open = False
        return False, None

    def record(self, key: str, value: Any) -> None:
        self._entries[key] = copy.deepcopy(value)

    def commit(self) -> None:
        """Make this run's results the ones a later resume reuses."""
        self._journal._commit(self._name, self._entries)

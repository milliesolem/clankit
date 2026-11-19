"""Loading, saving and watching the configuration file."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from clankit import paths
from clankit.config.schema import Config, ConfigError

log = logging.getLogger(__name__)

Listener = Callable[[Config], None]


class ConfigStore:
    """Owns the current `Config` and its file.

    All changes go through `update()`: it applies a mutation to a copy of the configuration,
    asks `clankit.config.kdl` to express the difference as minimal edits to the existing file
    text (so the user's comments, layout and disabled blocks survive), checks that the result
    parses back to exactly the intended configuration, writes it atomically with owner-only
    permissions (the file holds API keys), swaps the copy in and notifies listeners.

    If the file changed on disk since it was last read (the user edited it, or another
    Clankit process saved), `update()` re-reads it first and applies the mutation to that,
    so neither side's changes are lost.

    A store without a path lives in memory only. Thread-safe: the web UI updates settings
    from server threads.
    """

    def __init__(self, path: Path | None = None, config: Config | None = None) -> None:
        self.path = path
        self.config = config or Config()
        #: Non-fatal problems found when the file was last read (unknown settings and such).
        self.warnings: list[str] = []
        self._text: str | None = None  # the file as last read or written
        self._stamp: tuple[int, int] | None = None  # (mtime_ns, size) of the file at that time
        self._listeners: list[Listener] = []
        self._lock = threading.RLock()

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | None = None) -> ConfigStore:
        """Load the configuration file, creating a commented starter file if none exists."""
        store = cls(path or paths.config_file())
        store.reload()
        return store

    @classmethod
    def in_memory(cls, config: Config | None = None) -> ConfigStore:
        return cls(None, config)

    # -- reading and writing --------------------------------------------------------------------

    def reload(self) -> Config:
        """Re-read the file. Raises `ConfigError` (with line and column) if it is malformed."""
        from clankit.config import kdl

        with self._lock:
            if self.path is None:
                return self.config
            if not self.path.exists():
                self._write(kdl.starter_document())
            text = self._read()
            warnings: list[str] = []
            self.config = kdl.loads(text, warnings=warnings)
            self.warnings = warnings
            return self.config

    def changed_on_disk(self) -> bool:
        """Has the file been modified since this store last read or wrote it?"""
        with self._lock:
            return self.path is not None and self._file_stamp() != self._stamp

    def update(self, mutate: Callable[[Config], None]) -> Config:
        """Apply `mutate` to a copy of the config, persist it, and make it current.

        Raises `ConfigError` and changes nothing if the mutation produces a configuration
        that cannot be written and read back faithfully, or if the file on disk has become
        unreadable in the meantime.
        """
        from clankit.config import kdl

        with self._lock:
            if self.changed_on_disk():
                self.reload()
            candidate = self.config.copy()
            mutate(candidate)
            text = kdl.dumps(candidate, previous=self._text)
            try:
                reread = kdl.loads(text)
            except ConfigError as exc:
                raise ConfigError(f"refusing to save an invalid configuration: {exc}") from exc
            if reread != candidate:
                raise ConfigError(
                    "refusing to save: the configuration would not read back as written"
                )
            if self.path is not None:
                self._write(text)
            else:
                self._text = text
            self.config = candidate
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(candidate)
            except Exception:
                log.exception("config listener failed")
        return candidate

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    # -- internals ------------------------------------------------------------------------------

    def _file_stamp(self) -> tuple[int, int] | None:
        assert self.path is not None
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _read(self) -> str:
        assert self.path is not None
        # newline="" keeps the file's own line endings so a save does not rewrite them.
        with self.path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        self._text = text
        self._stamp = self._file_stamp()
        return text

    def _write(self, text: str) -> None:
        assert self.path is not None
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            directory.chmod(0o700)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".config-", suffix=".kdl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._text = text
        self._stamp = self._file_stamp()

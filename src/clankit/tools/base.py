"""The tool interface, the context tools run in, and the file ledger agents share.

A tool is the only way an agent touches the world. Dedicated tools (rather than "just use
bash") exist where the harness needs a typed hook: to gate an action behind permission, to
reject a stale edit, to render a diff, or to know that a call is read-only and can run
concurrently with others.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from clankit.llm.types import ToolSpec

if TYPE_CHECKING:
    from clankit.agent.events import Event

PermissionCategory = Literal["read", "edit", "shell", "network"]


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ToolOutput:
    """What a tool call produced.

    `content` is what the model reads. `display` is an optional structured payload for UIs
    (``{"type": "diff", "path": ..., "diff": ...}``). `finish` is set only by control tools
    that end the agent's run (for example ``report``); the agent loop stops after the current
    tool round and hands the dict to its caller as `AgentResult.payload`.
    """

    content: str
    is_error: bool = False
    display: dict[str, Any] | None = None
    finish: dict[str, Any] | None = None

    @classmethod
    def error(cls, message: str) -> ToolOutput:
        return cls(content=message, is_error=True)


# --------------------------------------------------------------------------------------------
# Permissions
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PermissionAsk:
    """A tool's description of the sensitive thing it is about to do.

    `target` is what allow/deny rules are matched against: a workspace-relative path for
    "read"/"edit" (absolute when outside the workspace), the command line for "shell", the
    URL for "network". `session_rule` is the pattern to remember if the user answers "allow
    for this session" (for example ``pytest *`` for the command ``pytest -q tests``); empty
    means the exact target.
    """

    category: PermissionCategory
    summary: str
    target: str
    detail: str = ""
    session_rule: str = ""


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""


class PermissionGate(Protocol):
    async def check(self, ask: PermissionAsk, *, node_id: str, tool: str) -> PermissionDecision:
        """Decide whether the action may proceed, asking the user if policy requires it."""
        ...


class AllowAll:
    """A gate that permits everything. For tests and for the "auto" permission mode."""

    async def check(self, ask: PermissionAsk, *, node_id: str, tool: str) -> PermissionDecision:
        return PermissionDecision(True)


# --------------------------------------------------------------------------------------------
# File ledger
# --------------------------------------------------------------------------------------------


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class FileLedger:
    """Session-wide record of which agent has seen and changed which file.

    Many agents edit one working tree at the same time. The ledger gives the file tools what
    they need to do that safely:

    * a per-path lock, so two writes to one file never interleave;
    * read stamps per agent, so an edit can be refused when the file changed after the agent
      last looked at it (another agent, the user, or a shell command touched it);
    * the set of files each agent changed, for its report to its parent.
    """

    #: Files larger than this are not checkpointed (and so cannot be restored by undo).
    CHECKPOINT_LIMIT = 4 * 1024 * 1024

    def __init__(self) -> None:
        self._locks: dict[Path, asyncio.Lock] = {}
        self._seen: dict[tuple[str, Path], tuple[int, int] | None] = {}
        self._changed: dict[str, list[Path]] = {}
        self._checkpoints: list[dict[Path, bytes | None]] = []

    def lock(self, path: Path) -> asyncio.Lock:
        return self._locks.setdefault(path, asyncio.Lock())

    def note_read(self, node_id: str, path: Path) -> None:
        self._seen[(node_id, path)] = _stamp(path)

    def has_read(self, node_id: str, path: Path) -> bool:
        return (node_id, path) in self._seen

    def is_stale(self, node_id: str, path: Path) -> bool:
        """True if `path` changed on disk since `node_id` last read or wrote it."""
        return self._seen.get((node_id, path)) != _stamp(path)

    def note_write(self, node_id: str, path: Path) -> None:
        self._seen[(node_id, path)] = _stamp(path)
        changed = self._changed.setdefault(node_id, [])
        if path not in changed:
            changed.append(path)

    def changed_by(self, node_id: str) -> list[Path]:
        return list(self._changed.get(node_id, ()))

    # -- checkpoints (undo) ---------------------------------------------------------------------

    def begin_run(self) -> None:
        """Start a new checkpoint. Called by the session when a run starts."""
        self._checkpoints.append({})

    def checkpoint(self, path: Path) -> None:
        """Remember `path` as it is now, if this run has not touched it yet.

        File tools call this immediately BEFORE creating, overwriting or editing a file.
        """
        if not self._checkpoints:
            self._checkpoints.append({})
        current = self._checkpoints[-1]
        if path in current:
            return
        try:
            if path.stat().st_size > self.CHECKPOINT_LIMIT:
                return
            current[path] = path.read_bytes()
        except FileNotFoundError:
            current[path] = None  # the file did not exist: undo deletes it
        except OSError:
            return

    def originals(self) -> dict[Path, bytes | None]:
        """Pre-run contents of every file changed through the file tools in the current run
        (None for files that did not exist). The basis for harness-computed diffs."""
        return dict(self._checkpoints[-1]) if self._checkpoints else {}

    def undo(self) -> list[Path]:
        """Restore every file changed through the file tools in the most recent run that
        changed any. Returns the restored paths (empty if there is nothing to undo).

        Changes made by shell commands are not tracked and are not reverted.
        """
        while self._checkpoints:
            snapshot = self._checkpoints.pop()
            if not snapshot:
                continue
            for path, content in snapshot.items():
                try:
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                except OSError:
                    continue
            for key in [k for k in self._seen if k[1] in snapshot]:
                del self._seen[key]  # every agent must re-read a restored file before editing
            return list(snapshot)
        return []


# --------------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class ToolContext:
    """Everything a tool may use. One per agent (per task node).

    `write_scope` is how a parent keeps sibling subtasks from colliding: a tuple of glob
    patterns, relative to the workspace, that this agent may create or modify. None means the
    whole workspace. File tools refuse writes outside the scope and tell the agent to report
    the need to its parent instead.
    """

    workspace: Path
    node_id: str
    files: FileLedger
    permissions: PermissionGate
    emit: Callable[[Event], Any]
    write_scope: tuple[str, ...] | None = None
    #: Scopes of every scoped ancestor. A path must satisfy all of them: a child can never
    #: write more than its parent could.
    ancestor_scopes: tuple[tuple[str, ...], ...] = ()
    config: Any = None  # clankit.config.schema.Config
    skills: Any = None  # clankit.skills.library.SkillLibrary
    env: dict[str, str] = field(default_factory=dict)

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Absolute, normalised path for a model-supplied path (relative to the workspace)."""
        p = Path(os.path.expanduser(os.fspath(path)))
        if not p.is_absolute():
            p = self.workspace / p
        return Path(os.path.normpath(p))

    def in_workspace(self, path: Path) -> bool:
        try:
            real = path.resolve()
            return real == self.workspace.resolve() or real.is_relative_to(self.workspace.resolve())
        except OSError:
            return False

    def relative(self, path: Path) -> str:
        """Workspace-relative display form of `path` (absolute if outside the workspace)."""
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def in_write_scope(self, path: Path) -> bool:
        """May this agent create or modify `path`? Checks its own scope and every ancestor's."""
        rel = self.relative(path)
        scopes = self.ancestor_scopes
        if self.write_scope is not None:
            scopes = (*scopes, self.write_scope)
        return all(any(scope_matches(p, rel) for p in scope) for scope in scopes)

    def for_child(self, node_id: str, write_scope: tuple[str, ...] | None = None) -> ToolContext:
        """A context for a subtask's agent: same session, its own node id and write scope."""
        ancestors = self.ancestor_scopes
        if self.write_scope is not None:
            ancestors = (*ancestors, self.write_scope)
        return ToolContext(
            workspace=self.workspace,
            node_id=node_id,
            files=self.files,
            permissions=self.permissions,
            emit=self.emit,
            write_scope=write_scope,
            ancestor_scopes=ancestors,
            config=self.config,
            skills=self.skills,
            env=self.env,
        )


def scope_matches(pattern: str, rel_path: str) -> bool:
    """Does workspace-relative `rel_path` fall under the ownership glob `pattern`?

    Accepted forms: an exact file (``src/app.py``), a directory (``src/api`` or ``src/api/``,
    meaning everything beneath it), and globs where ``*`` stays within one path segment and
    ``**`` crosses segments (``src/**/*.py``, ``tests/test_api_*.py``).
    """
    pattern = pattern.strip().lstrip("./")
    rel_path = rel_path.lstrip("./")
    if not pattern:
        return False
    if pattern.endswith("/"):
        pattern = pattern.rstrip("/")
    if rel_path == pattern or rel_path.startswith(pattern + "/"):
        return True
    return _glob_match(pattern.split("/"), rel_path.split("/"))


def _glob_match(pattern: list[str], parts: list[str]) -> bool:
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_glob_match(rest, parts[i:]) for i in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], head) and _glob_match(rest, parts[1:])


def scopes_overlap(a: tuple[str, ...] | None, b: tuple[str, ...] | None) -> bool:
    """Conservative test for whether two write scopes can touch a common path.

    Used to decide whether two sibling subtasks may run at the same time. Errs on the side of
    "overlap" (which only costs parallelism): scopes overlap when either is unrestricted, or
    when for some pair of patterns the literal path before the first wildcard of one is a
    prefix of the other's. ``src/a.py`` and ``src/b.py`` do not overlap; ``src/api`` and
    ``src/api/routes.py`` do; ``src/*.py`` and ``src/b.py`` do.
    """
    if a is None or b is None:
        return True
    for pa in a:
        for pb in b:
            la = _literal_prefix(pa)
            lb = _literal_prefix(pb)
            if not la or not lb or la == lb:
                return True
            if la.startswith(lb + "/") or lb.startswith(la + "/"):
                return True
    return False


def _has_wildcard(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _literal_prefix(pattern: str) -> str:
    pattern = pattern.strip().lstrip("./").rstrip("/")
    out: list[str] = []
    for segment in pattern.split("/"):
        if _has_wildcard(segment):
            break
        out.append(segment)
    return "/".join(out)


# --------------------------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------------------------


class Tool(ABC):
    """Base class for tools.

    Subclasses set `name`, `description` and `input_schema` (a JSON Schema object with
    ``"additionalProperties": false``) as class attributes, or as instance attributes when the
    text depends on runtime state. The agent loop validates every call against `input_schema`
    before `run` is invoked, so `run` may rely on required keys being present and well-typed.
    """

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    #: True when the tool never changes anything. Read-only calls of one assistant turn run
    #: concurrently; everything else runs one at a time, in the order the model asked.
    read_only: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def summarize(self, tool_input: dict[str, Any]) -> str:
        """One line describing this call, shown in transcripts."""
        return self.name

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        """What to ask the permission gate before running, or None if nothing is sensitive."""
        return None

    @abstractmethod
    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        """Do the work. Report expected failures as ``ToolOutput.error(...)``, not exceptions:
        the message goes to the model, so say what went wrong and what to do instead."""
        raise NotImplementedError

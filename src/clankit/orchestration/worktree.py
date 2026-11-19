"""Isolated copies of the workspace, for attempts that must not see each other's changes.

Best of N runs several agents on the same task at once. Each needs the user's files as they
are now, and none may see another's edits. A git worktree gives exactly that: a separate
checkout that shares the repository's object store, cheap to create and to throw away.

What this module does to the user's repository, and what it never does:

* The starting state is captured from the working tree itself - uncommitted edits and
  untracked files included, ignored files left out - by running ``git add --all`` against a
  temporary index file and storing the result as a commit that no branch points to. The
  user's index, branches, stash and files are never touched; git's garbage collection
  removes the unreferenced objects in time.
* Worktrees live under Clankit's state directory, never inside the workspace (where the next
  snapshot and every search would pick them up), and are checked out with hooks disabled.
  Removal runs to completion even when the run is cancelled half way through it.
* A worktree's changes come back as a git patch and are applied with ``git apply``, which
  changes every file or none: a change cannot land half way, and if the user edited the same
  lines in the meantime it fails instead of overwriting their work. Applied to the workspace,
  every file is checkpointed in the session's ledger first, so undo restores it.

Agents in a worktree get a ledger of their own. Sharing the session's would put worktree
files into the run's undo checkpoint, and undo would then recreate deleted worktrees.

Without git, the fallback is to run attempts one after another in the workspace itself and
to put back what the file tools changed in between: `capture_files`, `restore_files` and
`write_files` do that with the run's checkpoints.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from clankit.agent import prompts
from clankit.orchestration.common import CheckResult, run_check
from clankit.paths import state_dir
from clankit.tools.base import FileLedger, ToolContext

if TYPE_CHECKING:
    from clankit.agent.events import NodeInfo
    from clankit.agent.session import Session
    from clankit.config.schema import Config
    from clankit.runtime import Runtime

#: Seconds allowed for git commands whose cost does not grow with the size of the project.
GIT_SECONDS = 60.0
#: Seconds allowed for commands that read or write every file: snapshots, checkouts, patches.
COPY_SECONDS = 600.0

#: Checking out a worktree would run the repository's post-checkout hook; nothing should run.
_NO_HOOKS = ("core.hooksPath=/dev/null",)
#: The snapshot commit needs an author; the user may not have configured one.
_IDENTITY = {
    "GIT_AUTHOR_NAME": "Clankit",
    "GIT_AUTHOR_EMAIL": "clankit@localhost",
    "GIT_COMMITTER_NAME": "Clankit",
    "GIT_COMMITTER_EMAIL": "clankit@localhost",
}
#: Inherited variables that would point git at another repository, index or work tree (they
#: are set while git runs hooks, so Clankit started from a hook would otherwise obey them).
_REDIRECTS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


class WorktreeError(Exception):
    """Git could not do what was asked. The message is written for the user."""


# --------------------------------------------------------------------------------------------
# Running git
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Result:
    code: int
    out: bytes
    err: str

    @property
    def text(self) -> str:
        return self.out.decode("utf-8", "replace").strip()


def _environment(extra: Mapping[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _REDIRECTS}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_LITERAL_PATHSPECS="1")
    env.update(extra)
    return env


async def _git(
    args: Sequence[str],
    cwd: Path,
    *,
    config: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
    seconds: float = GIT_SECONDS,
    check: bool = True,
) -> _Result:
    """Run one git command. Raises WorktreeError when git is missing, times out, or (with
    `check`) fails; the process is killed on timeout and on cancellation."""
    options = [item for pair in config for item in ("-c", pair)]
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *options,
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_environment(env or {}),
        )
    except FileNotFoundError:
        if shutil.which("git") is None:
            raise WorktreeError("git is not installed (or not on PATH).") from None
        raise WorktreeError(f"{cwd} does not exist.") from None
    except OSError as exc:
        raise WorktreeError(f"git could not be started: {exc}") from None
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout=seconds)
    except BaseException as exc:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        if isinstance(exc, TimeoutError):
            raise WorktreeError(
                f"git {args[0]} did not finish within {seconds:.0f} seconds."
            ) from None
        raise
    result = _Result(process.returncode or 0, out, err.decode("utf-8", "replace").strip())
    if check and result.code != 0:
        reason = result.err.splitlines()[-1] if result.err else f"exit code {result.code}"
        raise WorktreeError(f"git {args[0]} failed: {reason}")
    return result


async def run_to_completion[T](awaitable: Awaitable[T]) -> T:
    """Await `awaitable` to the end even if the calling task is cancelled meanwhile.

    For cleanup and for changes that must not stop half way. A cancellation that arrives is
    held back until the work is done and then re-raised, so it is delayed, never lost.
    """
    task = asyncio.ensure_future(awaitable)
    interrupted: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            interrupted = exc
    if interrupted is not None:
        if not task.cancelled():
            task.exception()  # retrieved, so a failure here is not reported as unhandled
        raise interrupted
    return task.result()


# --------------------------------------------------------------------------------------------
# Snapshots and worktrees
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The user's working tree at one moment, stored as a commit that no ref points to.

    `prefix` is the workspace's path inside the repository ("" when it is the top level).
    """

    repo: Path
    commit: str
    prefix: str = ""


@dataclass(frozen=True, slots=True)
class Worktree:
    """A checkout of a snapshot. `workspace` is where an agent works: the same place in the
    copy as the session's workspace is in the repository. `files` is the copy's own ledger."""

    root: Path
    snapshot: Snapshot
    files: FileLedger = field(default_factory=FileLedger, compare=False, repr=False)

    @property
    def workspace(self) -> Path:
        return self.root / self.snapshot.prefix if self.snapshot.prefix else self.root


async def repository_root(path: Path) -> Path | None:
    """Top level of the git work tree containing `path`; None outside one, in a bare
    repository, or when git is not installed."""
    located = await _locate(path)
    return located[0] if located is not None else None


async def _locate(path: Path) -> tuple[Path, str] | None:
    """(top level of the work tree, `path` relative to it) - as git sees them, so symbolic
    links in the path are already resolved."""
    try:
        result = await _git(["rev-parse", "--show-toplevel", "--show-prefix"], path, check=False)
    except WorktreeError:
        return None
    lines = result.out.decode("utf-8", "surrogateescape").splitlines()
    if result.code != 0 or not lines or not lines[0]:
        return None
    prefix = lines[1].rstrip("/") if len(lines) > 1 else ""
    return Path(lines[0]), prefix


async def take_snapshot(workspace: Path) -> Snapshot:
    """Capture the working tree of the repository containing `workspace`, as it is now.

    Tracked files with their uncommitted changes and untracked files that are not ignored
    are included, exactly what ``git add --all`` would stage. Raises WorktreeError outside a
    repository or when git cannot read the files.
    """
    located = await _locate(workspace)
    if located is None:
        raise WorktreeError(f"{workspace} is not inside a git repository.")
    repo, prefix = located
    head = await _git(["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], repo, check=False)
    parent = head.text if head.code == 0 else ""
    with tempfile.TemporaryDirectory(prefix="clankit-index-") as scratch:
        env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
        if parent:
            # Start from HEAD so that tracked files matching an ignore pattern stay included.
            await _git(["read-tree", parent], repo, env=env)
        await _git(["add", "--all"], repo, env=env, seconds=COPY_SECONDS)
        tree = (await _git(["write-tree"], repo, env=env)).text
    args = ["commit-tree", "--no-gpg-sign", "-m", "Clankit: snapshot of the working tree"]
    if parent:
        args += ["-p", parent]
    commit = (await _git([*args, tree], repo, env=_IDENTITY)).text
    return Snapshot(repo, commit, prefix)


class WorktreeSet:
    """Worktrees made from one snapshot, all removed together.

    Use it as ``async with WorktreeSet(snapshot) as trees: tree = await trees.add("a")``.
    Leaving the block removes every worktree, whether it ends normally, with an error or by
    cancellation, and the removal finishes even if the task is cancelled again meanwhile.
    """

    def __init__(self, snapshot: Snapshot, *, directory: Path | None = None) -> None:
        self.snapshot = snapshot
        self._directory = directory or state_dir() / "worktrees"
        self._roots: list[Path] = []
        # One checkout at a time: they all register themselves in the same repository.
        self._adding = asyncio.Lock()

    async def __aenter__(self) -> WorktreeSet:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def add(self, label: str) -> Worktree:
        """Check the snapshot out into a new directory. `label` names it, for people looking."""
        self._directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "worktree"
        root = self._directory / f"{slug}-{secrets.token_hex(4)}"
        self._roots.append(root)  # first, so that an interrupted checkout is still removed
        async with self._adding:
            await _git(
                ["worktree", "add", "--detach", "--quiet", str(root), self.snapshot.commit],
                self.snapshot.repo,
                config=_NO_HOOKS,
                seconds=COPY_SECONDS,
            )
        return Worktree(root, self.snapshot)

    async def close(self) -> None:
        roots, self._roots = self._roots, []
        if roots:
            await run_to_completion(self._remove(roots))

    async def _remove(self, roots: list[Path]) -> None:
        repo = self.snapshot.repo
        for root in roots:
            with contextlib.suppress(WorktreeError):
                await _git(
                    ["worktree", "remove", "--force", "--force", str(root)], repo, check=False
                )
            if root.exists():
                await asyncio.to_thread(shutil.rmtree, root, True)
        with contextlib.suppress(WorktreeError):
            await _git(["worktree", "prune"], repo, check=False)


# --------------------------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------------------------


async def diff(tree: Worktree) -> str:
    """Everything changed in the worktree since its snapshot, as a git patch.

    Covers the workspace part of the copy only; untracked files are included, ignored ones
    are not, and binary changes are encoded so that `apply_patch` reproduces them exactly.
    Paths are relative to the repository's top level. Bytes that are not UTF-8 survive the
    round trip through `str` (they are carried as surrogate escapes).
    """
    await _git(["add", "--all"], tree.root, seconds=COPY_SECONDS)
    pathspec = ["--", tree.snapshot.prefix] if tree.snapshot.prefix else []
    result = await _git(
        [
            "diff",
            "--cached",
            "--binary",
            "--no-renames",  # every path appears on its own, so each can be checkpointed
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            tree.snapshot.commit,
            *pathspec,
        ],
        tree.root,
        config=("core.quotePath=false",),
        seconds=COPY_SECONDS,
    )
    return result.out.decode("utf-8", "surrogateescape")


def readable(patch: str) -> str:
    """`patch` as text a model or a person can be shown (undecodable bytes replaced)."""
    return patch.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


async def patch_stat(patch: str, snapshot: Snapshot) -> list[str]:
    """One line per file the patch changes, ``path  +added -removed`` (or ``(binary)``),
    with paths relative to the workspace."""
    lines = []
    for name, added, removed in await _numstat(patch, snapshot.repo):
        if snapshot.prefix:
            name = name.removeprefix(f"{snapshot.prefix}/")
        lines.append(f"{name}  (binary)" if added is None else f"{name}  +{added} -{removed}")
    return lines


async def apply_patch(
    patch: str, root: Path, *, files: FileLedger | None = None, node_id: str = ""
) -> list[Path]:
    """Apply a patch made by `diff` to the checkout whose top level is `root`.

    Git applies every file or none. Raises WorktreeError, with git's reason, when the patch
    does not apply (typically because a file changed since the snapshot was taken); nothing
    has changed then. With `files`, every path is checkpointed before the change and
    recorded as written by `node_id` after it, so undo restores it and harness diffs show
    it. Returns the absolute paths the patch touched.
    """
    data = patch.encode("utf-8", "surrogateescape")
    if not data.strip():
        return []
    targets = await patch_paths(patch, root)
    command = ["apply", "--whitespace=nowarn", "-"]
    if files is None:
        await run_to_completion(_git(command, root, stdin=data, seconds=COPY_SECONDS))
        return targets
    async with contextlib.AsyncExitStack() as stack:
        for path in sorted(set(targets)):
            await stack.enter_async_context(files.lock(path))
        for path in targets:
            files.checkpoint(path)
        await run_to_completion(_git(command, root, stdin=data, seconds=COPY_SECONDS))
        for path in targets:
            files.note_write(node_id, path)
    return targets


async def patch_paths(patch: str, root: Path) -> list[Path]:
    """The absolute paths a patch made by `diff` changes, for the checkout at `root`."""
    return [root / name for name, _, _ in await _numstat(patch, root)]


async def _numstat(patch: str, root: Path) -> list[tuple[str, int | None, int | None]]:
    """(path, lines added, lines removed) per file; the counts are None for binary files."""
    data = patch.encode("utf-8", "surrogateescape")
    if not data.strip():
        return []
    result = await _git(["apply", "--numstat", "-z", "-"], root, stdin=data)
    entries = []
    for record in result.out.split(b"\0"):
        parts = record.split(b"\t", 2)
        if len(parts) != 3:
            continue
        added, removed, name = parts
        entries.append(
            (
                os.fsdecode(name),
                int(added) if added.isdigit() else None,
                int(removed) if removed.isdigit() else None,
            )
        )
    return entries


#: Patch lines that describe what happened to a file; kept whole in a `change_key`.
_KEPT_AS_IS = ("--- ", "+++ ", "old mode", "new mode", "new file", "deleted file", "Binary file")


def change_key(patch: str) -> str:
    """A form of a patch that is equal for changes differing only in whitespace.

    Used to merge candidates that made the same change. Kept: which files changed and how,
    binary data verbatim, context lines (which pin where a change sits) and the changed
    lines with their indentation, which carries meaning in some languages. Dropped: blob
    hashes, hunk positions, trailing whitespace, blank changed lines, and differences in
    the amount of whitespace between words.
    """
    key: list[str] = []
    binary = False
    for line in patch.splitlines():
        if line.startswith("diff "):
            binary = False
            key.append(line)
        elif binary or line.startswith(_KEPT_AS_IS):
            key.append(line.rstrip())
        elif line.startswith("GIT binary patch"):
            binary = True
            key.append(line)
        elif line[:1] in (" ", "+", "-"):
            body = line[1:].rstrip()
            indent = body[: len(body) - len(body.lstrip())]
            words = " ".join(body.split())
            if words:
                key.append(f"{line[0]}{indent}{words}")
    return "\n".join(key)


# --------------------------------------------------------------------------------------------
# Agents in a worktree
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Relocated:
    """The attributes `prompts.base_prompt` and `run_check` read from a session, with the
    workspace moved into a worktree. Neither helper takes a directory of its own."""

    workspace: Path
    config: Config
    runtime: Runtime


def _relocated(session: Session, tree: Worktree) -> Session:
    return cast("Session", _Relocated(tree.workspace, session.config, session.runtime))


def tool_context(session: Session, node: NodeInfo, tree: Worktree) -> ToolContext:
    """A tool context like `Session.tool_context` gives, rooted in the worktree: relative
    paths resolve there and shell commands start there. It uses the worktree's own ledger."""
    return ToolContext(
        workspace=tree.workspace,
        node_id=node.id,
        files=tree.files,
        permissions=session.permissions,
        emit=session.emit,
        config=session.config,
        skills=session.runtime.skills,
    )


def system_prompt(session: Session, tree: Worktree, role: str) -> str:
    """A system prompt like `prompts.system_prompt` builds, naming the worktree as the
    working directory (and reading project instructions from the copy)."""
    base = prompts.base_prompt(_relocated(session, tree))
    role = role.strip()
    return f"{base}\n\n{role}" if role else base


async def run_check_in(
    session: Session, ctx: ToolContext, tree: Worktree, command: str
) -> CheckResult:
    """`run_check` with the worktree as the working directory."""
    return await run_check(_relocated(session, tree), ctx, command)


# --------------------------------------------------------------------------------------------
# Without git: restoring files from the run's checkpoints
# --------------------------------------------------------------------------------------------


def capture_files(paths: Iterable[Path]) -> dict[Path, bytes | None]:
    """The current contents of `paths` (None for a file that does not exist)."""
    captured: dict[Path, bytes | None] = {}
    for path in paths:
        try:
            captured[path] = path.read_bytes()
        except FileNotFoundError:
            captured[path] = None
    return captured


def restore_files(files: FileLedger, paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Put `paths` back as they were when the current run started, from its checkpoints.

    Returns (restored, not restorable). A file cannot be restored when the file tools never
    checkpointed it: it was changed only by shell commands, or it is too large to keep. Only
    call this while no agent is working on these files.
    """
    originals = files.originals()
    restored: list[Path] = []
    missing: list[Path] = []
    for path in paths:
        if path in originals and _put(path, originals[path]):
            restored.append(path)
        else:
            missing.append(path)
    return restored, missing


def write_files(files: FileLedger, contents: Mapping[Path, bytes | None], node_id: str) -> None:
    """Write captured contents back (None deletes), checkpointing each file first and
    recording it as written by `node_id`, as the file tools would."""
    for path, content in contents.items():
        files.checkpoint(path)
        if not _put(path, content):
            raise OSError(f"could not write {path}")
        files.note_write(node_id, path)


def _put(path: Path, content: bytes | None) -> bool:
    """Make `path` hold `content` (None: not exist), replacing it atomically."""
    try:
        if content is None:
            path.unlink(missing_ok=True)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.restore")
        try:
            temporary.write_bytes(content)
            if path.exists():
                shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        return False
    return True

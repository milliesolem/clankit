"""The search tools: glob (find files by name) and grep (find text in files).

Both use ripgrep when it is installed, because it is fast and honours ``.gitignore``, and
otherwise a pure-Python implementation that takes the same options and prints the same
output. Either way a search:

* never descends into ``.git`` or the directories that hold dependencies, caches and build
  output, which bury real results;
* never follows symbolic links, so a search cannot wander out of the workspace;
* (grep) never prints from files that look like secrets unless such a file is named
  directly, which goes through the same permission question as reading it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shutil
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from clankit import paths
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput
from clankit.tools.files import output_budget, read_permission, secret_name

__all__ = ["GlobTool", "GrepTool", "glob_matcher", "ripgrep"]

#: Directory names no search descends into (below the directory being searched; naming one
#: of them as the search path still works).
SKIP_DIRS = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    ".next",
    "dist",
    "build",
    "target",
)
#: Files one glob call lists.
GLOB_LIMIT = 200
#: Entries (files or lines) one grep call returns unless `head_limit` says otherwise.
DEFAULT_HEAD_LIMIT = 200
MAX_HEAD_LIMIT = 2000
MAX_CONTEXT = 20
#: Characters of one matching line that grep shows.
MAX_LINE_CHARS = 400
#: Seconds a search may take before it is stopped.
SEARCH_TIMEOUT = 30.0

#: Longest single line of ripgrep output read in one piece (a JSON record for a match in a
#: minified file can be very long).
_STREAM_LIMIT = 16 * 1024 * 1024
_RG_COMMON = ("--no-config", "--no-require-git", "--hidden", "--no-messages")


class _SearchFailed(Exception):
    """A search that cannot be carried out. The message is the tool result."""


def ripgrep() -> str | None:
    """Path of the ``rg`` executable, or None when the pure-Python search has to do."""
    return shutil.which("rg")


# --------------------------------------------------------------------------------------------
# Globs
# --------------------------------------------------------------------------------------------


def glob_matcher(pattern: str) -> Callable[[str], bool]:
    """A predicate over "/"-separated relative paths with ripgrep's reading of `pattern`.

    That reading is gitignore's: ``*`` and ``?`` stay within one path segment, ``**``
    crosses segments, ``{a,b}`` lists alternatives, a pattern without a slash matches a
    name at any depth, a pattern with one is relative to the directory searched, and a
    leading ``!`` inverts the match.
    """
    negated = pattern.startswith("!")
    alternatives = _expand_braces(pattern.removeprefix("!"))
    regex = re.compile("|".join(f"(?:{_glob_regex(option)})" for option in alternatives))
    return lambda rel: (regex.fullmatch(rel) is None) == negated


def _normalise_glob(pattern: str) -> str:
    """Tidy the forms models write: ``./src/*.py`` and ``src/`` (meaning all of ``src``)."""
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    return f"{pattern}**" if pattern.endswith("/") else pattern


def _expand_braces(pattern: str) -> list[str]:
    start = pattern.find("{")
    while start != -1:
        group = _brace_group(pattern, start)
        if group is None:
            break
        end, commas = group
        if commas:
            cuts = [start, *commas, end]
            return [
                expanded
                for left, right in pairwise(cuts)
                for expanded in _expand_braces(
                    pattern[:start] + pattern[left + 1 : right] + pattern[end + 1 :]
                )
            ]
        start = pattern.find("{", start + 1)
    return [pattern]


def _brace_group(pattern: str, start: int) -> tuple[int, list[int]] | None:
    """For the ``{`` at `start`: the index of its ``}`` and of its top-level commas."""
    depth = 0
    commas: list[int] = []
    for index in range(start, len(pattern)):
        char = pattern[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index, commas
        elif char == "," and depth == 1:
            commas.append(index)
    return None


def _glob_regex(pattern: str) -> str:
    anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
    pattern = pattern.lstrip("/")
    out = [] if anchored else ["(?:.*/)?"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index) and (index == 0 or pattern[index - 1] == "/"):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        elif char == "[" and (character_class := _glob_class(pattern, index)) is not None:
            regex, index = character_class
            out.append(regex)
        else:
            out.append(re.escape(char))
            index += 1
    return "".join(out)


def _glob_class(pattern: str, start: int) -> tuple[str, int] | None:
    """The ``[...]`` at `start` as a regex class and the index after it; None if it is not
    a well-formed class, in which case the bracket is an ordinary character."""
    end = pattern.find("]", start + 2)
    if end == -1:
        return None
    body = pattern[start + 1 : end]
    negated = body[0] in "!^"
    listed = body[1:] if negated else body
    members = "".join(f"\\{char}" if char in "\\[]^" else char for char in listed)
    if not members:
        return None
    return f"[{'^' if negated else ''}{members}]", end + 1


# --------------------------------------------------------------------------------------------
# Walking and running
# --------------------------------------------------------------------------------------------


def _walk(root: Path, stop: threading.Event) -> Iterator[str]:
    """Relative paths of the regular files under `root`, as ripgrep lists them when there
    are no ignore files: hidden files included, symbolic links and `SKIP_DIRS` left out."""
    pending = [""]
    while pending and not stop.is_set():
        folder = pending.pop()
        try:
            with os.scandir(root / folder) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in SKIP_DIRS:
                    pending.append(f"{folder}{entry.name}/")
            elif entry.is_file(follow_symlinks=False):
                yield f"{folder}{entry.name}"


def _path_order(rel: str) -> list[str]:
    """Sort key that orders paths as ``rg --sort path`` does: segment by segment."""
    return rel.split("/")


async def _in_thread[T](work: Callable[[threading.Event], T]) -> T:
    """Run the pure-Python search off the event loop, telling it to stop when the caller
    gives up (a thread cannot be cancelled, but it can be asked)."""
    stop = threading.Event()
    try:
        return await asyncio.wait_for(asyncio.to_thread(work, stop), SEARCH_TIMEOUT)
    except TimeoutError:
        raise _SearchFailed(_TIMED_OUT) from None
    finally:
        stop.set()


async def _rg(rg: str, args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run ripgrep to the end: (exit code, stdout, stderr). For output of bounded size."""
    process = await asyncio.create_subprocess_exec(
        rg,
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), SEARCH_TIMEOUT)
    except TimeoutError:
        raise _SearchFailed(_TIMED_OUT) from None
    finally:
        await _reap(process)
    return process.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def _reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


_TIMED_OUT = (
    f"The search was stopped after {SEARCH_TIMEOUT:.0f} seconds. Narrow it: give a path, a "
    "glob, or a more specific pattern."
)


def _rg_failure(stderr: str, pattern: str) -> _SearchFailed:
    """Turn ripgrep's complaint into a message the model can act on."""
    detail = next(
        (
            line.removeprefix("error:").strip()
            for line in stderr.splitlines()
            if line[:6] == "error:"
        ),
        "",
    )
    if "regex parse error" in stderr:
        return _SearchFailed(_bad_regex(pattern, detail or "it could not be parsed"))
    if "error parsing glob" in stderr:
        return _SearchFailed(f"Invalid glob: {stderr.strip().removeprefix('rg: ')}")
    return _SearchFailed(f"The search failed: {stderr.strip() or 'ripgrep reported an error'}")


def _bad_regex(pattern: str, detail: str) -> str:
    return (
        f"Invalid regular expression {pattern!r}: {detail}. The pattern is a regular "
        "expression, so put a backslash before characters meant literally, such as "
        "( ) [ ] { } . * + ? | ^ $"
    )


def _place(ctx: ToolContext, path: Path) -> str:
    """`path` as named in a sentence: the workspace root reads better by name than as "."."""
    return "the workspace" if path == ctx.resolve(".") else ctx.relative(path)


def _skip_globs() -> list[str]:
    return [f"--glob=!{name}/" for name in SKIP_DIRS]


# --------------------------------------------------------------------------------------------
# glob
# --------------------------------------------------------------------------------------------


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by name. Returns the paths that match a glob pattern, most recently "
        "modified first. `*` matches within one path segment and `**` across segments; a "
        "pattern without a slash matches names at any depth (`*.py`, `conftest.py`), one "
        "with a slash is relative to the directory searched (`src/**/*.ts`, "
        "`tests/test_*.py`); `{a,b}` gives alternatives (`*.{js,ts}`). Use it to learn the "
        "layout of a project or to locate a file whose name you roughly know; to search "
        f"inside files use grep. At most {GLOB_LIMIT} paths are returned. Files ignored by "
        ".gitignore, and dependency, cache and build directories (.git, node_modules, "
        ".venv, dist, ...) are left out unless you name such a directory as path."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The glob pattern to match."},
            "path": {
                "type": "string",
                "description": "Directory to search in. Default: the workspace root.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        where = f" in {tool_input['path']}" if tool_input.get("path") else ""
        return f"glob {tool_input['pattern']}{where}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        root, _ = _glob_request(tool_input, ctx)
        return read_permission(ctx, root, verb="List files in")

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        root, pattern = _glob_request(tool_input, ctx)
        shown_root = _place(ctx, root)
        if not pattern:
            return ToolOutput.error('pattern names no files. Use "*" to list a directory.')
        if not root.is_dir():
            kind = "is a file, not a directory" if root.exists() else "does not exist"
            return ToolOutput.error(
                f"Cannot search {shown_root}: it {kind}. path must be a directory; leave it "
                "out to search the whole workspace."
            )
        try:
            found = await _list_files(root, pattern)
        except _SearchFailed as exc:
            return ToolOutput.error(str(exc))
        if not found:
            return ToolOutput(f"No files match {pattern!r} in {shown_root}.")
        newest = sorted(found, key=lambda rel: (-_modified(root / rel), rel))
        lines = [ctx.relative(root / rel) for rel in newest[:GLOB_LIMIT]]
        if len(newest) > GLOB_LIMIT:
            lines.append(
                f"[Showing the {GLOB_LIMIT} most recently modified of {len(newest):,} "
                "matching files. Narrow the pattern or the path to see the others.]"
            )
        return ToolOutput("\n".join(lines))


def _glob_request(tool_input: dict[str, Any], ctx: ToolContext) -> tuple[Path, str]:
    """The directory to search and the pattern relative to it.

    Models often put the directory into the pattern (``/repo/src/**/*.py``); the leading
    wildcard-free part of an absolute pattern therefore becomes the directory, so that the
    permission check sees where the search really goes.
    """
    pattern = tool_input["pattern"].strip()
    root = ctx.resolve(tool_input.get("path") or ".")
    expanded = os.path.expanduser(pattern)
    if os.path.isabs(expanded):
        parts = Path(expanded).parts
        wild = (i for i, part in enumerate(parts) if any(ch in part for ch in "*?[{"))
        literal = max(1, next(wild, len(parts) - 1))
        root = ctx.resolve(Path(*parts[:literal]))
        rest = "/".join(parts[literal:])
        # The leading slash keeps what is left anchored to that directory.
        pattern = f"/{rest}{'/' if pattern.endswith('/') else ''}" if rest else ""
    return root, _normalise_glob(pattern)


async def _list_files(root: Path, pattern: str) -> list[str]:
    rg = ripgrep()
    if rg is None:
        matches = glob_matcher(pattern)
        return await _in_thread(lambda stop: [rel for rel in _walk(root, stop) if matches(rel)])
    args = [*_RG_COMMON, "--files", f"--glob={pattern}", *_skip_globs()]
    code, out, err = await _rg(rg, args, root)
    if code == 2 and not out:
        raise _rg_failure(err, pattern)
    return out.splitlines()


def _modified(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


# --------------------------------------------------------------------------------------------
# grep
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Search:
    """One grep request, resolved."""

    #: The directory searched. Results are paths relative to it.
    root: Path
    #: The file's name within `root` when the request named a single file, else None.
    only: str | None
    regex: str
    glob: str | None
    ignore_case: bool
    context: int
    #: Clankit's own config file as a path relative to `root`, when the search can reach it.
    config: str | None

    def shows(self, rel: str) -> bool:
        """May results from `rel` be shown? Not when it looks like it holds secrets, unless
        it is the one file the request named (which the permission check has cleared)."""
        if self.only is not None:
            return True
        return rel != self.config and not secret_name(rel.rpartition("/")[2])


@dataclass(frozen=True, slots=True)
class _Line:
    path: str
    number: int
    text: str
    #: False for a context line around a match.
    match: bool


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search the contents of files for a regular expression. output_mode chooses the "
        "result: files_with_matches (default) lists the files that contain a match, most "
        "recently modified first; content prints each matching line as path:line:text, "
        "with `context` lines around it if asked; count gives the number of matching "
        "lines per file. Narrow a search with path (a directory or one file) and glob "
        "(`*.py`, `src/**/*.ts`, `*.{js,ts}`). The pattern is matched line by line, is "
        "case-sensitive unless case_insensitive is set, and uses ripgrep's regex syntax: "
        "write `\\(` or `\\.` for a literal parenthesis or dot. Results are capped at "
        f"head_limit (default {DEFAULT_HEAD_LIMIT}). Binary files, files ignored by "
        ".gitignore, dependency and build directories, and files that look like secrets "
        "(.env, private keys) are skipped; a secret file is searched only when path names "
        "it. Prefer this tool to running grep or rg through bash."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The regular expression to find."},
            "path": {
                "type": "string",
                "description": "File or directory to search. Default: the workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Only search files whose path matches this glob.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "description": "What to return. Default: files_with_matches.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Ignore letter case. Default false.",
            },
            "context": {
                "type": "integer",
                "description": "Lines to show before and after each match (content mode).",
            },
            "head_limit": {
                "type": "integer",
                "description": (
                    f"Most lines (content) or files (other modes) to return. Default "
                    f"{DEFAULT_HEAD_LIMIT}, at most {MAX_HEAD_LIMIT}."
                ),
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        where = f" in {tool_input['path']}" if tool_input.get("path") else ""
        only = f" ({tool_input['glob']})" if tool_input.get("glob") else ""
        return f"grep {tool_input['pattern']!r}{where}{only}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        return read_permission(ctx, ctx.resolve(tool_input.get("path") or "."), verb="Search")

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        mode = tool_input.get("output_mode") or "files_with_matches"
        limit = _head_limit(tool_input.get("head_limit"))
        try:
            search = _search_for(tool_input, ctx)
            if mode == "content":
                lines, more = await _matching_lines(search, limit)
                return _content_result(ctx, search, lines, more)
            counts = await _matching_counts(search)
        except _SearchFailed as exc:
            return ToolOutput.error(str(exc))
        if not counts:
            return ToolOutput(_no_matches(ctx, search))
        if mode == "count":
            return _count_result(ctx, search, counts, limit)
        return _files_result(ctx, search, counts, limit)


def _head_limit(requested: Any) -> int:
    if not requested or requested < 1:
        return DEFAULT_HEAD_LIMIT
    return min(int(requested), MAX_HEAD_LIMIT)


def _search_for(tool_input: dict[str, Any], ctx: ToolContext) -> _Search:
    target = ctx.resolve(tool_input.get("path") or ".")
    if not target.exists():
        raise _SearchFailed(
            f"Cannot search {ctx.relative(target)}: it does not exist. Use glob to find the "
            "right path, or leave path out to search the whole workspace."
        )
    is_file = not target.is_dir()
    root = target.parent if is_file else target
    glob = _normalise_glob(tool_input.get("glob") or "")
    config = Path(os.path.realpath(paths.config_file()))
    real_root = os.path.realpath(root)
    return _Search(
        root=root,
        only=target.name if is_file else None,
        regex=tool_input["pattern"],
        glob=glob or None,
        ignore_case=bool(tool_input.get("case_insensitive", False)),
        context=max(0, min(int(tool_input.get("context") or 0), MAX_CONTEXT)),
        config=config.relative_to(real_root).as_posix()
        if config.is_relative_to(real_root)
        else None,
    )


async def _matching_counts(search: _Search) -> dict[str, int]:
    """Matching-line counts for every file with at least one match."""
    rg = ripgrep()
    if rg is None:
        counts = await _in_thread(lambda stop: _py_counts(search, stop))
    else:
        counts = await _rg_counts(rg, search)
    return {rel: count for rel, count in counts.items() if search.shows(rel)}


async def _matching_lines(search: _Search, limit: int) -> tuple[list[_Line], bool]:
    """The first `limit` matching (and context) lines in path order, and whether the search
    had more to give."""
    rg = ripgrep()
    if rg is None:
        return await _in_thread(lambda stop: _py_lines(search, limit, stop))
    return await _rg_lines(rg, search, limit)


# -- results ----------------------------------------------------------------------------------


def _shown(ctx: ToolContext, search: _Search, rel: str) -> str:
    return ctx.relative(search.root / rel)


def _scope(ctx: ToolContext, search: _Search) -> str:
    where = _place(ctx, search.root / search.only if search.only else search.root)
    return f"{where} (files matching {search.glob})" if search.glob else where


def _no_matches(ctx: ToolContext, search: _Search) -> str:
    return f"No matches for {search.regex!r} in {_scope(ctx, search)}."


def _files_result(
    ctx: ToolContext, search: _Search, counts: dict[str, int], limit: int
) -> ToolOutput:
    newest = sorted(counts, key=lambda rel: (-_modified(search.root / rel), rel))
    lines = [_shown(ctx, search, rel) for rel in newest[:limit]]
    if len(newest) > limit:
        lines.append(
            f"[Showing {limit} of {len(newest):,} files with matches, most recently modified "
            "first. Narrow the search with path or glob, or raise head_limit.]"
        )
    return ToolOutput("\n".join(lines))


def _count_result(
    ctx: ToolContext, search: _Search, counts: dict[str, int], limit: int
) -> ToolOutput:
    ordered = sorted(counts, key=_path_order)
    lines = [f"{_shown(ctx, search, rel)}:{counts[rel]}" for rel in ordered[:limit]]
    total = sum(counts.values())
    summary = f"{total:,} matching line(s) in {len(ordered):,} file(s)"
    if len(ordered) > limit:
        summary += f"; showing the first {limit} files. Raise head_limit or narrow the search"
    lines.append(f"[{summary}.]")
    return ToolOutput("\n".join(lines))


def _content_result(
    ctx: ToolContext, search: _Search, lines: list[_Line], more: bool
) -> ToolOutput:
    if not lines:
        return ToolOutput(_no_matches(ctx, search))
    budget = output_budget(ctx) - 200  # room for the closing note
    out: list[str] = []
    used = 0
    previous: _Line | None = None
    for line in lines:
        mark = ":" if line.match else "-"
        text = line.text
        if len(text) > MAX_LINE_CHARS:
            text = f"{text[:MAX_LINE_CHARS]} [... {len(text) - MAX_LINE_CHARS:,} more characters]"
        entry = f"{_shown(ctx, search, line.path)}{mark}{line.number}{mark}{text}"
        if out and used + len(entry) + 4 > budget:
            more = True
            break
        adjacent = previous is not None and (
            previous.path == line.path and previous.number + 1 == line.number
        )
        if search.context and previous is not None and not adjacent:
            out.append("--")
        out.append(entry)
        used += len(entry) + 4
        previous = line
    if more:
        out.append(
            "[There are more matches than shown. Narrow the search with a more specific "
            "pattern, a path or a glob, or raise head_limit.]"
        )
    return ToolOutput("\n".join(out))


# -- ripgrep ----------------------------------------------------------------------------------


def _rg_args(search: _Search) -> list[str]:
    args = [*_RG_COMMON, "--sort", "path", "--crlf"]
    if search.ignore_case:
        args.append("--ignore-case")
    if search.glob:
        args.append(f"--glob={search.glob}")
    args += [*_skip_globs(), f"--regexp={search.regex}"]
    if search.only is not None:
        args += ["--", search.only]
    return args


def _with_mode(search: _Search, *mode: str) -> list[str]:
    """ripgrep arguments for `search` with output options, which must precede a ``--``."""
    return [*mode, *_rg_args(search)]


async def _rg_counts(rg: str, search: _Search) -> dict[str, int]:
    code, out, err = await _rg(
        rg, _with_mode(search, "--count", "--with-filename", "--null"), search.root
    )
    counts: dict[str, int] = {}
    for record in out.splitlines():
        rel, _, count = record.rpartition("\0")
        if rel and count.isdigit():
            counts[rel] = int(count)
    if code == 2 and not counts:
        raise _rg_failure(err, search.regex)
    return counts


async def _rg_lines(rg: str, search: _Search, limit: int) -> tuple[list[_Line], bool]:
    """Read ripgrep's JSON stream until `limit` lines are in hand, then stop ripgrep: a
    pattern that matches half the repository must not be read to the end."""
    process = await asyncio.create_subprocess_exec(
        rg,
        *_with_mode(search, "--json", f"--context={search.context}"),
        cwd=search.root,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_STREAM_LIMIT,
    )
    assert process.stdout is not None and process.stderr is not None
    lines: list[_Line] = []
    try:
        async with asyncio.timeout(SEARCH_TIMEOUT):
            async with contextlib.aclosing(_rg_records(process.stdout, search)) as records:
                async for line in records:
                    if len(lines) == limit:
                        return lines, True
                    lines.append(line)
            stderr = (await process.stderr.read()).decode(errors="replace")
            if await process.wait() == 2 and not lines:
                raise _rg_failure(stderr, search.regex)
    except TimeoutError:
        raise _SearchFailed(_TIMED_OUT) from None
    finally:
        await _reap(process)
    return lines, False


async def _rg_records(stream: asyncio.StreamReader, search: _Search) -> AsyncIterator[_Line]:
    """The match and context lines in ripgrep's JSON output that may be shown."""
    while True:
        try:
            raw = await stream.readline()
        except ValueError:  # one record longer than _STREAM_LIMIT: skip it
            continue
        if not raw:
            return
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if record.get("type") not in ("match", "context"):
            continue
        data = record["data"]
        rel = _rg_text(data["path"])
        if not search.shows(rel):
            continue
        text = _rg_text(data["lines"]).rstrip("\r\n")
        yield _Line(rel, data["line_number"], text, record["type"] == "match")


def _rg_text(value: dict[str, str]) -> str:
    """ripgrep gives text as ``{"text": ...}``, or ``{"bytes": base64}`` if it is not UTF-8."""
    if "text" in value:
        return value["text"]
    return base64.b64decode(value["bytes"]).decode(errors="replace")


# -- pure Python ------------------------------------------------------------------------------


def _py_regex(search: _Search) -> re.Pattern[str]:
    try:
        return re.compile(search.regex, re.IGNORECASE if search.ignore_case else 0)
    except re.error as exc:
        where = f" at position {exc.pos}" if exc.pos is not None else ""
        raise _SearchFailed(_bad_regex(search.regex, f"{exc.msg}{where}")) from None


def _py_hits(search: _Search, stop: threading.Event) -> Iterator[tuple[str, list[str], list[int]]]:
    """Each searched file with a match, in path order: its lines and which of them match."""
    regex = _py_regex(search)
    if search.only is not None:
        candidates = [search.only]
    else:
        matches = glob_matcher(search.glob) if search.glob else None
        candidates = sorted(
            (rel for rel in _walk(search.root, stop) if matches is None or matches(rel)),
            key=_path_order,
        )
    for rel in candidates:
        if stop.is_set():
            return
        if not search.shows(rel):
            continue
        try:
            data = (search.root / rel).read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        lines = data.decode(errors="replace").split("\n")
        if lines[-1] == "":
            lines.pop()
        lines = [line.removesuffix("\r") for line in lines]
        hits = [number for number, line in enumerate(lines) if regex.search(line)]
        if hits:
            yield rel, lines, hits


def _py_counts(search: _Search, stop: threading.Event) -> dict[str, int]:
    return {rel: len(hits) for rel, _, hits in _py_hits(search, stop)}


def _py_lines(search: _Search, limit: int, stop: threading.Event) -> tuple[list[_Line], bool]:
    found: list[_Line] = []
    for rel, lines, hits in _py_hits(search, stop):
        matching = set(hits)
        wanted = sorted(
            {
                number
                for hit in hits
                for number in range(hit - search.context, hit + search.context + 1)
                if 0 <= number < len(lines)
            }
        )
        for number in wanted:
            if len(found) == limit:
                return found, True
            found.append(_Line(rel, number + 1, lines[number], number in matching))
    return found, False

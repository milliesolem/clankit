"""The file tools: read, write and edit.

Many agents work in one tree at the same time, so these tools are stricter than a text editor:

* an agent may only change a file it has read, and only while its picture of the file is
  current (the `FileLedger` keeps a stamp per agent and file);
* a write happens under the file's lock, is checkpointed first (which is what undo restores)
  and replaces the file atomically;
* a path outside the agent's write scope is refused, with advice to report the need to the
  parent task instead, and nothing is ever written into ``.git`` or Clankit's own
  configuration directory.

`edit` is exact string replacement, the edit format models handle most reliably, hardened the
way production agents harden it: a short ladder of safe near-matches before giving up, and
error messages that tell the model how to recover without another round trip.
"""

from __future__ import annotations

import contextlib
import difflib
import fnmatch
import glob
import os
import re
import secrets
import stat
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from clankit import paths
from clankit.tools.base import PermissionAsk, Tool, ToolContext, ToolOutput, scope_matches

__all__ = [
    "EditTool",
    "ReadTool",
    "WriteTool",
    "inside_workspace",
    "looks_secret",
    "output_budget",
    "permission_target",
    "read_permission",
    "secret_name",
    "unified_diff",
]

#: Lines one read call returns unless `limit` says otherwise.
READ_LINES = 2000
#: Characters of a single line that read shows; the rest of a longer line is cut.
READ_LINE_CHARS = 2000
#: Largest file these tools read or edit. Bigger files are data, not source.
MAX_FILE_BYTES = 10 * 1024 * 1024
#: Characters of unified diff kept for UIs and permission previews.
MAX_DIFF_CHARS = 16_000
#: Above this many lines on both sides, a changed region is shown as one replaced block
#: rather than aligned line by line (alignment cost grows with the product of the two).
MAX_ALIGNED_LINES = 4000
#: Result size assumed when no configuration is attached (`AgentConfig.tool_output_limit`).
DEFAULT_OUTPUT_LIMIT = 30_000

_DIFF_CONTEXT = 3
_HINT_LINES = 12
_HINT_LINE_CHARS = 200
#: Files with more lines than this get no similarity search for a missed edit (too slow).
_FUZZY_SEARCH_LINES = 10_000
_BINARY_SNIFF_BYTES = 8192

_SECRET_NAMES = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "credentials",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "credentials.toml",
    "credentials.ini",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
)
#: Templates that exist to be read and committed; asking about them only teaches the user
#: to wave secret prompts through.
_SECRET_TEMPLATES = (".env.example", ".env.sample", ".env.template", ".env.dist")

#: Characters models substitute for (or with) their plain equivalents, by code point. Every
#: mapping is one character to one character, so offsets in translated text equal offsets in
#: the original.
_TYPOGRAPHIC: dict[int, str] = {
    **dict.fromkeys(range(0x2018, 0x201C), "'"),  # curly and low single quotation marks
    **dict.fromkeys(range(0x201C, 0x2020), '"'),  # curly and low double quotation marks
    **dict.fromkeys((*range(0x2010, 0x2016), 0x2212), "-"),  # hyphens, dashes, minus sign
    # no-break, en, em, thin and the other fixed-width spaces
    **dict.fromkeys((0x00A0, *range(0x2002, 0x200B), 0x202F, 0x205F, 0x3000), " "),
}
_BYTE_ORDER_MARK = chr(0xFEFF)
_NUMBERED_LINE = re.compile(r"\s*\d+\t")

Tier = Literal["exact", "typographic", "whitespace"]


class _Refused(Exception):
    """A call that cannot go ahead. The message is the tool result, written for the model."""


# --------------------------------------------------------------------------------------------
# Shared by every tool that touches paths
# --------------------------------------------------------------------------------------------


def output_budget(ctx: ToolContext) -> int:
    """Characters a tool can return before the agent loop would cut the middle out of it.

    Tools that can page or summarise stay under this, so what the model sees ends at a clean
    boundary with a note on how to get the rest.
    """
    limit = ctx.config.agent.tool_output_limit if ctx.config is not None else DEFAULT_OUTPUT_LIMIT
    return max(1000, int(limit) - 500)


def inside_workspace(ctx: ToolContext, path: Path) -> bool:
    """`ctx.in_workspace`, counting a path that cannot be resolved (a symlink loop) as outside."""
    try:
        return ctx.in_workspace(path)
    except RuntimeError:
        return False


def permission_target(ctx: ToolContext, path: Path) -> tuple[str, bool]:
    """What permission rules are matched against for `path`, and whether it is in the workspace.

    The target is the file's real location (workspace-relative inside the workspace, absolute
    outside), so a symlink can neither carry a path out of the workspace unnoticed nor slip it
    past a deny rule.
    """
    real = Path(os.path.realpath(path))
    if inside_workspace(ctx, path):
        return real.relative_to(os.path.realpath(ctx.workspace)).as_posix(), True
    return str(real), False


def looks_secret(path: Path) -> bool:
    """Is `path` likely to hold credentials: by its name, or by being Clankit's own config?"""
    real = Path(os.path.realpath(path))
    if real == Path(os.path.realpath(paths.config_file())):
        return True
    return secret_name(path.name) or secret_name(real.name)


def secret_name(name: str) -> bool:
    """Does a file of this name usually hold credentials (``.env``, a private key, ...)?"""
    name = name.lower()
    if name in _SECRET_TEMPLATES:
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _SECRET_NAMES)


def read_permission(ctx: ToolContext, path: Path, verb: str = "Read") -> PermissionAsk | None:
    """What to ask before reading `path` (a file, or a directory about to be searched).

    Nothing inside the workspace, except for files that look like secrets. Anything outside
    asks, and offers the containing directory as the rule to remember, so that exploring one
    outside directory costs one question rather than one per file.
    """
    target, inside = permission_target(ctx, path)
    is_dir = path.is_dir()
    if not inside:
        folder = (target if is_dir else os.path.dirname(target)).rstrip("/")
        shown = f"{folder}/" if is_dir else target
        return PermissionAsk(
            "read",
            f"{verb} {shown} (outside the workspace)",
            shown,
            session_rule=f"{glob.escape(folder)}/*",
        )
    if not is_dir and looks_secret(path):
        return PermissionAsk(
            "read",
            f"{verb} {target} (it may contain secrets)",
            target,
            session_rule=glob.escape(target),
        )
    return None


def unified_diff(path: str, before: str | None, after: str, limit: int = MAX_DIFF_CHARS) -> str:
    """A unified diff of two versions of one file, for display. `before` None means "new file".

    Only the changed region is aligned (the common head and tail are set aside first), so the
    cost follows the size of the change, not of the file. Lines are compared without their
    line endings; a change that is only about those is reported in words.
    """
    old, new = _display_lines(before or ""), _display_lines(after)
    head = _common_prefix(old, new)
    tail = _common_prefix(old[head:][::-1], new[head:][::-1])
    start = max(0, head - _DIFF_CONTEXT)
    spare = max(0, tail - _DIFF_CONTEXT)
    a, b = old[start : len(old) - spare], new[start : len(new) - spare]

    label = path.lstrip("/")
    out = ["--- /dev/null" if before is None else f"--- a/{label}", f"+++ b/{label}"]
    for group in _opcode_groups(a, b):
        first, last = group[0], group[-1]
        out.append(
            f"@@ -{_hunk_range(start + first[1], last[2] - first[1])} "
            f"+{_hunk_range(start + first[3], last[4] - first[3])} @@"
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out.extend(f" {line}" for line in a[i1:i2])
                continue
            out.extend(f"-{line}" for line in a[i1:i2])
            out.extend(f"+{line}" for line in b[j1:j2])
    if len(out) == 2 and (before or "") != after:
        out.append("[only line endings or the final newline differ]")
    return _cap_lines(out, limit)


def _opcode_groups(a: list[str], b: list[str]) -> list[list[tuple[str, int, int, int, int]]]:
    if min(len(a), len(b)) > MAX_ALIGNED_LINES:
        return [[("replace", 0, len(a), 0, len(b))]]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return list(matcher.get_grouped_opcodes(_DIFF_CONTEXT))


def _hunk_range(start: int, length: int) -> str:
    if length == 1:
        return str(start + 1)
    return f"{start + 1 if length else start},{length}"


def _common_prefix(a: list[str], b: list[str]) -> int:
    count = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        count += 1
    return count


def _cap_lines(lines: list[str], limit: int) -> str:
    kept: list[str] = []
    used = 0
    for line in lines:
        used += len(line) + 1
        if used > limit:
            kept.append(f"[diff truncated: {len(lines) - len(kept):,} more lines]")
            break
        kept.append(line)
    return "\n".join(kept)


def _display_lines(text: str) -> list[str]:
    """A file's lines as they are shown: without line endings or a byte-order mark."""
    lines = text.removeprefix(_BYTE_ORDER_MARK).split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def _numbered(number: int, line: str) -> str:
    return f"{number:>6}\t{line}"


# --------------------------------------------------------------------------------------------
# Loading text
# --------------------------------------------------------------------------------------------


def _decode(data: bytes) -> tuple[str, str]:
    """Text, and the codec that turns it back into the same bytes.

    UTF-8 when the bytes are valid UTF-8; otherwise Latin-1, which maps every byte to one
    character, so the parts of a legacy-encoded file that an edit does not touch survive
    byte for byte.
    """
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _load(ctx: ToolContext, path: Path) -> tuple[str, str]:
    """`path` as (text, codec), or `_Refused` saying why it cannot be handled as text."""
    rel = ctx.relative(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        raise _Refused(_missing(ctx, path)) from None
    except OSError as exc:
        raise _Refused(f"{rel} cannot be read: {exc.strerror or exc}.") from None
    if stat.S_ISDIR(info.st_mode):
        raise _Refused(
            f'{rel} is a directory, not a file. List it with glob (pattern "*", path "{rel}").'
        )
    if not stat.S_ISREG(info.st_mode):
        raise _Refused(f"{rel} is not a regular file (it is a device, pipe or socket).")
    if info.st_size > MAX_FILE_BYTES:
        raise _Refused(
            f"{rel} is {info.st_size / 1_048_576:.1f} MB; this tool handles files up to "
            f"{MAX_FILE_BYTES // 1_048_576} MB. Use grep to find the part you need, or bash "
            "(sed -n '100,200p', head, tail) to view a slice."
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _Refused(f"{rel} cannot be read: {exc.strerror or exc}.") from None
    if b"\0" in data[:_BINARY_SNIFF_BYTES]:
        raise _Refused(
            f"{rel} is a binary file; the file tools work on text only. Inspect it with bash "
            "(file, xxd, strings) if you need to."
        )
    return _decode(data)


def _missing(ctx: ToolContext, path: Path) -> str:
    rel = ctx.relative(path)
    search = f'Use glob to find it by name (pattern "**/{path.name}").'
    if not path.parent.is_dir():
        return f"{rel} does not exist, and neither does the directory it would be in. {search}"
    try:
        names = os.listdir(path.parent)
    except OSError:
        names = []
    close = (
        difflib.get_close_matches(path.name, names, n=3, cutoff=0.6) if len(names) < 5000 else []
    )
    if not close:
        return f"{rel} does not exist. {search}"
    suggestions = ", ".join(ctx.relative(path.parent / name) for name in close)
    return f"{rel} does not exist. Did you mean {suggestions}? {search}"


# --------------------------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------------------------


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a text file. The result shows each line as its line number, a tab, then the "
        f"text. Without arguments it returns the file from the top, up to {READ_LINES} lines; "
        "when a file is longer the result ends with a note giving the offset to continue "
        "from. Read a file before changing it: edit and write refuse files you have not "
        "read. To read several files, ask for all of them in one turn; the calls run in "
        "parallel. This tool cannot list directories (use glob) or show binary files, and "
        "to find where something is, grep is cheaper than reading whole files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path of the file, relative to the workspace or absolute.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start at (the first line is 1). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": f"How many lines to return. Default {READ_LINES}.",
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }
    read_only = True

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"read {tool_input['file_path']}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        return read_permission(ctx, ctx.resolve(tool_input["file_path"]))

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(tool_input["file_path"])
        offset = max(1, int(tool_input.get("offset") or 1))
        limit = int(tool_input.get("limit") or READ_LINES)
        if limit < 1:
            return ToolOutput.error("limit must be at least 1 (it is the number of lines to show).")
        try:
            text, _ = _load(ctx, path)
        except _Refused as exc:
            return ToolOutput.error(str(exc))
        ctx.files.note_read(ctx.node_id, path)
        return _render(ctx, path, text, offset, limit)


def _render(ctx: ToolContext, path: Path, text: str, offset: int, limit: int) -> ToolOutput:
    """Numbered lines from `offset`, as many as `limit` and the output budget allow."""
    rel = ctx.relative(path)
    lines = _display_lines(text)
    total = len(lines)
    if total == 0:
        return ToolOutput(f"[{rel} is empty.]")
    if offset > total:
        return ToolOutput.error(
            f"{rel} has {total} lines, so there is nothing at line {offset}. "
            f"Use an offset between 1 and {total}."
        )

    budget = output_budget(ctx) - 300  # room for the closing note
    shown: list[str] = []
    used = cut = 0
    index = offset - 1
    stop = min(total, index + limit)
    while index < stop:
        line = lines[index]
        if len(line) > READ_LINE_CHARS:
            extra = len(line) - READ_LINE_CHARS
            line = f"{line[:READ_LINE_CHARS]} [... {extra:,} more characters on this line]"
            cut += 1
        entry = _numbered(index + 1, line)
        if shown and used + len(entry) + 1 > budget:
            break
        shown.append(entry)
        used += len(entry) + 1
        index += 1

    notes: list[str] = []
    if offset > 1 or index < total:
        notes.append(f"Showing lines {offset}-{index} of {total}.")
    if index < total:
        notes.append(f"To continue, call read with offset={index + 1}.")
    if cut:
        notes.append(
            f"{cut} long line(s) were cut at {READ_LINE_CHARS:,} characters; use grep with a "
            "pattern, or bash (cut -c), to see the rest of such a line."
        )
    body = "\n".join(shown)
    return ToolOutput(f"{body}\n\n[{' '.join(notes)}]" if notes else body)


# --------------------------------------------------------------------------------------------
# Writing: the rules edit and write share
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Change:
    """A validated change to one file, ready to be shown for approval and then committed."""

    path: Path
    #: The file's current text, or None when the file does not exist yet.
    before: str | None
    after: str
    #: `after` in the file's encoding: what is written.
    data: bytes
    #: "Edit", "Create" or "Overwrite", for the permission summary.
    verb: str
    #: The confirmation the model reads once the change is on disk.
    message: str

    @property
    def unchanged(self) -> bool:
        return self.before == self.after


def _check_writable(ctx: ToolContext, path: Path) -> None:
    """Refuse paths that this agent may not create or modify at all.

    The first two rules hold whatever the user has allowed: a write into ``.git`` can corrupt
    the repository or plant a hook, and a write into Clankit's configuration could grant the
    agent permissions or rewrite its own skills.
    """
    rel = ctx.relative(path)
    real = Path(os.path.realpath(path))
    if ".git" in path.parts or ".git" in real.parts:
        raise _Refused(
            f"Refusing to write to {rel}: it is inside a .git directory. Git's own files are "
            "changed by running git commands with bash, never by editing them."
        )
    config = Path(os.path.realpath(paths.config_dir()))
    if real == config or real.is_relative_to(config):
        raise _Refused(
            f"Refusing to write to {rel}: it is inside Clankit's own configuration directory "
            f"({config}), which agents may not change. If a setting or a skill should change, "
            "tell the user what to change."
        )
    if not ctx.in_write_scope(path):
        raise _Refused(
            f"{rel} is outside this task's write scope ({_violated_scope(ctx, rel)}). That "
            "file belongs to another task: do not change it, and do not work around this "
            "with bash. If your task needs a change there, describe the exact change in "
            "your report to your parent, who can arrange it."
        )


def _violated_scope(ctx: ToolContext, rel: str) -> str:
    """The scope `rel` falls outside of, as text: the agent's own if that one fails."""
    scopes = list(ctx.ancestor_scopes)
    if ctx.write_scope is not None:
        scopes.append(ctx.write_scope)
    for scope in reversed(scopes):
        if not any(scope_matches(pattern, rel) for pattern in scope):
            owner = "" if scope is ctx.write_scope else "inherited from a parent task: "
            return owner + (", ".join(scope) or "nothing")
    return "restricted"


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace `path` with `data` so that readers see the old file or the new one, never a
    partial one. An existing file keeps its permission bits; a symlink keeps pointing at its
    target, which is what gets replaced."""
    real = Path(os.path.realpath(path))
    real.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode: int | None = stat.S_IMODE(real.stat().st_mode)
    except FileNotFoundError:
        mode = None
    temporary = real.with_name(f".{real.name}.{secrets.token_hex(6)}.tmp")
    # Opened with 0o666 so that a new file gets the permissions the umask allows, as if it
    # had been created directly.
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, real)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _ask(ctx: ToolContext, plan: Callable[[], _Change]) -> PermissionAsk | None:
    """The permission question for a planned change, with the diff as its detail.

    A call that is going to be refused, or that changes nothing, asks nothing: the user
    should not be shown a prompt for something that will not happen. `_apply` plans again
    under the file's lock, so what is finally written never rests on this first look.
    """
    try:
        change = plan()
    except _Refused:
        return None
    if change.unchanged:
        return None
    target, inside = permission_target(ctx, change.path)
    return PermissionAsk(
        "edit",
        f"{change.verb} {target}",
        target,
        detail=unified_diff(target, change.before, change.after),
        session_rule="*" if inside else glob.escape(target),
    )


async def _apply(ctx: ToolContext, path: Path, plan: Callable[[], _Change]) -> ToolOutput:
    """Plan and commit a change while holding the file's lock.

    Nothing between reading the file and replacing it awaits, so no other agent's work can
    land in between; the lock states that intent and keeps it true if that ever changes.
    """
    async with ctx.files.lock(path):
        try:
            change = plan()
        except _Refused as exc:
            return ToolOutput.error(str(exc))
        rel = ctx.relative(path)
        if change.unchanged:
            return ToolOutput(f"{rel} already has exactly this content; nothing was written.")
        ctx.files.checkpoint(path)
        try:
            _write_atomically(path, change.data)
        except OSError as exc:
            return ToolOutput.error(f"Could not write {rel}: {exc.strerror or exc}.")
        ctx.files.note_write(ctx.node_id, path)
        diff = unified_diff(rel, change.before, change.after)
        return ToolOutput(change.message, display={"type": "diff", "path": rel, "diff": diff})


def _encode(text: str, codec: str, rel: str) -> bytes:
    try:
        return text.encode(codec)
    except UnicodeEncodeError:
        if codec == "utf-8":
            problem = "The new text contains characters that are not valid Unicode text"
            remedy = "Remove them and retry."
        else:
            problem = (
                f"{rel} is not UTF-8 (it is handled as Latin-1), and the new text contains "
                "characters that encoding cannot hold"
            )
            remedy = "Convert the file to UTF-8 first (iconv, through bash), read it, then retry."
        raise _Refused(f"{problem}, so nothing was written. {remedy}") from None


def _changed_elsewhere(rel: str, then: str) -> str:
    return (
        f"{rel} has changed since you last read it (another agent, the user or a command "
        f"modified it). Read it again, then {then}."
    )


# --------------------------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------------------------


class WriteTool(Tool):
    name = "write"
    description = (
        "Create a file, or replace the whole contents of an existing one. Missing parent "
        "directories are created. content must be the complete text of the file, exactly as "
        "it should be on disk. An existing file must have been read with read first, so "
        "that nothing is overwritten blindly; to change part of a file, edit is safer and "
        "much cheaper than rewriting it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path of the file, relative to the workspace or absolute.",
            },
            "content": {"type": "string", "description": "The complete new file contents."},
        },
        "required": ["file_path", "content"],
        "additionalProperties": False,
    }

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"write {tool_input['file_path']}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        return _ask(ctx, lambda: _plan_write(ctx, tool_input))

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(tool_input["file_path"])
        return await _apply(ctx, path, lambda: _plan_write(ctx, tool_input))


def _plan_write(ctx: ToolContext, tool_input: dict[str, Any]) -> _Change:
    path = ctx.resolve(tool_input["file_path"])
    content: str = tool_input["content"]
    rel = ctx.relative(path)
    _check_writable(ctx, path)
    if path.is_dir():
        raise _Refused(f"{rel} is a directory. Give the path of a file inside it.")
    size = _line_count(content)
    data = _encode(content, "utf-8", rel)
    if not path.exists():
        return _Change(path, None, content, data, "Create", f"Created {rel} ({size}).")
    if not ctx.files.has_read(ctx.node_id, path):
        raise _Refused(
            f"{rel} already exists and you have not read it. Read it first, so that nothing "
            "is overwritten blindly, or use edit to change part of it."
        )
    if ctx.files.is_stale(ctx.node_id, path):
        raise _Refused(_changed_elsewhere(rel, "decide whether it should still be overwritten"))
    before, _ = _load(ctx, path)
    message = f"Overwrote {rel} ({size}; it had {_line_count(before)})."
    return _Change(path, before, content, data, "Overwrite", message)


def _line_count(text: str) -> str:
    count = len(_display_lines(text))
    return "1 line" if count == 1 else f"{count:,} lines"


# --------------------------------------------------------------------------------------------
# edit
# --------------------------------------------------------------------------------------------


class EditTool(Tool):
    name = "edit"
    description = (
        "Change a file by replacing old_string with new_string. Read the file first. "
        "old_string must be the file's text exactly, indentation included, and must pick out "
        "one place: add neighbouring lines until it is unique, or set replace_all to change "
        "every occurrence (renaming something, say). Copy text from read output without the "
        "line number and tab in front. The file's line endings and final newline are kept. "
        "Prefer a few small edits to one sweeping one; to create a file or replace all of "
        "one, use write."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path of the file, relative to the workspace or absolute.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to replace. Must differ from new_string.",
            },
            "new_string": {"type": "string", "description": "The text to put in its place."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence of old_string. Default false.",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def summarize(self, tool_input: dict[str, Any]) -> str:
        return f"edit {tool_input['file_path']}"

    def permission(self, tool_input: dict[str, Any], ctx: ToolContext) -> PermissionAsk | None:
        return _ask(ctx, lambda: _plan_edit(ctx, tool_input))

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(tool_input["file_path"])
        return await _apply(ctx, path, lambda: _plan_edit(ctx, tool_input))


def _plan_edit(ctx: ToolContext, tool_input: dict[str, Any]) -> _Change:
    path = ctx.resolve(tool_input["file_path"])
    rel = ctx.relative(path)
    # Matching happens in the file's "LF view", which is what read shows the model; the
    # offsets found there are mapped back so that untouched bytes are never rewritten.
    old = tool_input["old_string"].replace("\r\n", "\n")
    new = tool_input["new_string"].replace("\r\n", "\n")
    replace_all = bool(tool_input.get("replace_all", False))

    if not tool_input["old_string"]:
        raise _Refused(
            "old_string is empty. To create a file or replace all of one, use write; to "
            "insert text, put a neighbouring line in old_string and repeat it in new_string."
        )
    if tool_input["old_string"] == tool_input["new_string"]:
        raise _Refused("old_string and new_string are identical, so there is nothing to change.")
    if old == new:
        raise _Refused(
            "old_string and new_string differ only in line endings, which edit always keeps "
            "as the file has them. To convert a file's line endings, use bash."
        )
    _check_writable(ctx, path)
    if not ctx.files.has_read(ctx.node_id, path):
        raise _Refused(
            f"You have not read {rel} yet. Read it first: an edit has to be based on the "
            "file's current contents. (If the file does not exist yet, create it with write.)"
        )
    text, codec = _load(ctx, path)
    view, joins = _lf_view(text)
    spans, tier = _locate(view, old)

    stale = ctx.files.is_stale(ctx.node_id, path)
    if stale and not (tier == "exact" and len(spans) == 1):
        raise _Refused(_changed_elsewhere(rel, "redo the edit against its current contents"))
    if not spans:
        raise _Refused(f"old_string was not found in {rel}. {_miss_hint(view, old, new)}")
    places = _line_numbers(view, spans)
    if len(spans) > 1 and not replace_all:
        raise _Refused(
            f"old_string matches {len(spans)} places in {rel} ({places}), and an edit must "
            "pick out one. Add neighbouring lines to old_string until it is unique, or set "
            f"replace_all to true to change all {len(spans)}."
        )

    newline = _newline_of(text)
    after = _replace(text, joins, spans, new.replace("\n", newline))
    after = _keep_final_newline(text, after, newline)
    if after == text:
        raise _Refused(f"This edit would leave {rel} exactly as it is, so nothing was changed.")
    data = _encode(after, codec, rel)

    count = "1 replacement" if len(spans) == 1 else f"{len(spans)} replacements"
    notes = [f"Edited {rel}: {count} at {places}."]
    if tier == "typographic":
        notes.append(
            "old_string matched only after treating typographic quotes, dashes and spaces "
            "as their plain equivalents; the file's text there differs from what you sent."
        )
    elif tier == "whitespace":
        notes.append("old_string matched only after ignoring whitespace at the ends of lines.")
    if stale:
        notes.append(
            "The file had changed since you last read it; the edit was applied to its "
            "current contents, which you may want to read again."
        )
    return _Change(path, text, after, data, "Edit", " ".join(notes))


def _lf_view(text: str) -> tuple[str, list[int]]:
    """`text` with CRLF line endings shown as LF, plus what is needed to map offsets back:
    the view offset of every ``\\n`` that stands for a ``\\r\\n`` in `text`, in order."""
    joins = [match.start() - index for index, match in enumerate(re.finditer("\r\n", text))]
    return text.replace("\r\n", "\n"), joins


def _locate(view: str, old: str) -> tuple[list[tuple[int, int]], Tier]:
    """Where `old` occurs in `view`, by the strictest reading that finds it at all.

    The ladder stops at differences a model cannot see or routinely gets wrong (typographic
    punctuation, whitespace at line ends). Anything looser would risk editing the wrong text.
    """
    spans = _occurrences(view, old)
    if spans:
        return spans, "exact"
    spans = _occurrences(view.translate(_TYPOGRAPHIC), old.translate(_TYPOGRAPHIC))
    if spans:
        return spans, "typographic"
    return _occurrences_ignoring_line_end_space(view, old), "whitespace"


def _occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = haystack.find(needle)
    while position != -1:
        spans.append((position, position + len(needle)))
        position = haystack.find(needle, position + len(needle))
    return spans


def _occurrences_ignoring_line_end_space(view: str, old: str) -> list[tuple[int, int]]:
    """Occurrences of `old` when spaces and tabs in front of a line break (and typographic
    variants) are disregarded on both sides.

    Editors strip that whitespace and models drop or invent it, and neither can see it. What
    follows `old`'s last line break is matched as written, unless it runs up to the end of a
    line in the file, where its trailing whitespace is just as invisible.
    """
    if not old.strip():
        return []
    *full_lines, rest = old.translate(_TYPOGRAPHIC).split("\n")
    pattern = "".join(re.escape(line.rstrip(" \t")) + r"[ \t]*\n" for line in full_lines)
    if rest:
        at_line_end = re.escape(rest.rstrip(" \t")) + r"[ \t]*(?=\n|\Z)"
        pattern += f"(?:{re.escape(rest)}|{at_line_end})"
    return [match.span() for match in re.finditer(pattern, view.translate(_TYPOGRAPHIC))]


def _replace(text: str, joins: list[int], spans: list[tuple[int, int]], new: str) -> str:
    """`text` with each span (given in LF-view offsets) replaced by `new`."""
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor : start + bisect_left(joins, start)])
        pieces.append(new)
        cursor = end + bisect_left(joins, end)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _newline_of(text: str) -> str:
    """The line ending most of the file uses; new lines an edit brings in get the same."""
    crlf = text.count("\r\n")
    return "\r\n" if crlf > text.count("\n") - crlf else "\n"


def _keep_final_newline(before: str, after: str, newline: str) -> str:
    """Whether a file ends with a newline is its author's choice; models drop or add one at
    the end of new_string by accident far more often than on purpose."""
    if not after or before.endswith("\n") == after.endswith("\n"):
        return after
    if before.endswith("\n"):
        return after + newline
    return after.rstrip("\r\n")


def _line_numbers(view: str, spans: list[tuple[int, int]]) -> str:
    numbers = [str(view.count("\n", 0, start) + 1) for start, _ in spans[:8]]
    listed = ", ".join(numbers) + (", ..." if len(spans) > 8 else "")
    return f"line {listed}" if len(spans) == 1 else f"lines {listed}"


def _miss_hint(view: str, old: str, new: str) -> str:
    """Why `old` may have missed, said so that the next attempt can succeed without the model
    having to read the whole file again."""
    if len(new.strip()) >= 20 and new.translate(_TYPOGRAPHIC) in view.translate(_TYPOGRAPHIC):
        return (
            "new_string is already present in the file, so this edit may already have been "
            "applied. Read the file to check before trying again."
        )
    old_lines = old.split("\n")
    filled = [line for line in old_lines if line.strip()]
    if filled and all(_NUMBERED_LINE.match(line) for line in filled):
        return (
            "old_string seems to include the line numbers that read prints in front of each "
            "line. Leave those out: pass only the file's own text."
        )
    lines = view.split("\n")
    block = _closest_block(lines, old_lines)
    if block is None:
        return (
            "Nothing in the file resembles it. Read the file again and copy the text to "
            "replace exactly as it appears."
        )
    first, last, whitespace_only = block
    last = min(last, first + _HINT_LINES - 1)
    shown = "\n".join(
        _numbered(number + 1, lines[number][:_HINT_LINE_CHARS]) for number in range(first, last + 1)
    )
    difference = (
        "They differ from old_string only in whitespace (indentation, spacing or blank lines)"
        if whitespace_only
        else "Compare them with old_string"
    )
    return (
        f"The closest lines in the file are:\n{shown}\n{difference}; copy the text exactly "
        "as it is in the file, without the line numbers."
    )


def _closest_block(lines: list[str], wanted: list[str]) -> tuple[int, int, bool] | None:
    """The stretch of `lines` that `wanted` most plausibly meant: (first, last, and whether
    the two differ only in whitespace).

    Each distinctive line of `wanted` that occurs in the file once whitespace is ignored
    votes for the place the block would have to start; a start backed by at least half of
    them wins. Failing that, the single line most similar to `wanted`'s longest line anchors
    the block.
    """
    positions: dict[str, list[int]] = {}
    for number, line in enumerate(lines):
        positions.setdefault(_squash(line), []).append(number)
    votes: Counter[int] = Counter()
    distinctive = 0
    for offset, line in enumerate(wanted):
        key = _squash(line)
        if not any(ch.isalnum() for ch in key):
            continue
        distinctive += 1
        votes.update(number - offset for number in positions.get(key, ()))
    if votes:
        start, count = min(votes.items(), key=lambda vote: (-vote[1], vote[0]))
        if count * 2 >= distinctive:
            return _clamp(start, len(wanted), len(lines), count == distinctive)

    anchor = max(wanted, key=lambda line: len(line.strip()))
    if not anchor.strip() or len(lines) > _FUZZY_SEARCH_LINES:
        return None
    stripped = [line.strip() for line in lines]
    close = difflib.get_close_matches(anchor.strip(), stripped, n=1, cutoff=0.6)
    if not close:
        return None
    start = stripped.index(close[0]) - wanted.index(anchor)
    return _clamp(start, len(wanted), len(lines), False)


def _clamp(start: int, length: int, total: int, flag: bool) -> tuple[int, int, bool]:
    first = max(0, start)
    return first, max(first, min(total, start + length) - 1), flag


def _squash(line: str) -> str:
    return "".join(line.translate(_TYPOGRAPHIC).split())

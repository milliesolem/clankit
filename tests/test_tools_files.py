"""The file tools: read, write and edit, and the write rules they share."""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from clankit import paths
from clankit.agent.events import PermissionRequested, ToolFinished, ToolStarted
from clankit.agent.session import Session
from clankit.config.schema import Config
from clankit.llm.scripted import ScriptedProvider, call
from clankit.tools import files
from clankit.tools.base import AllowAll, FileLedger, Tool, ToolContext, ToolOutput
from clankit.tools.files import EditTool, ReadTool, WriteTool, unified_diff
from clankit.tools.search import GlobTool, GrepTool
from clankit.tools.shell import BashTool
from clankit.tools.web import WebFetchTool

LDQUO, RDQUO = "\N{LEFT DOUBLE QUOTATION MARK}", "\N{RIGHT DOUBLE QUOTATION MARK}"
EM_DASH = "\N{EM DASH}"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        node_id="n1",
        files=FileLedger(),
        permissions=AllowAll(),
        emit=lambda event: None,
    )


async def run(tool: Tool, ctx: ToolContext, **tool_input: Any) -> ToolOutput:
    return await tool.run(tool_input, ctx)


async def read(ctx: ToolContext, file_path: str, **more: Any) -> ToolOutput:
    return await run(ReadTool(), ctx, file_path=file_path, **more)


async def edit(ctx: ToolContext, file_path: str, old: str, new: str, **more: Any) -> ToolOutput:
    return await run(EditTool(), ctx, file_path=file_path, old_string=old, new_string=new, **more)


async def write(ctx: ToolContext, file_path: str, content: str) -> ToolOutput:
    return await run(WriteTool(), ctx, file_path=file_path, content=content)


def put(workspace: Path, name: str, content: str | bytes) -> Path:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_bytes(content.encode())
    return path


def names_in(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def change_behind_the_agents_back(path: Path, content: str) -> None:
    """Rewrite `path` the way another program would, making sure its stamp moves."""
    before = path.stat()
    path.write_bytes(content.encode())
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


# --------------------------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------------------------

ALL_TOOLS = [ReadTool, WriteTool, EditTool, GlobTool, GrepTool, BashTool, WebFetchTool]
#: The only schema keywords weak models and strict backends can all be trusted with.
SAFE_KEYWORDS = {"type", "properties", "required", "additionalProperties", "description", "enum"}


def schema_keywords(schema: dict[str, Any]) -> set[str]:
    found = set(schema)
    for prop in schema.get("properties", {}).values():
        found |= schema_keywords(prop)
    return found


@pytest.mark.parametrize("tool_class", ALL_TOOLS, ids=lambda cls: cls.name)
def test_every_tool_declares_a_strict_conservative_schema(tool_class: type[Tool]) -> None:
    tool = tool_class()
    schema = tool.input_schema
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(schema["properties"])
    assert schema_keywords(schema) <= SAFE_KEYWORDS
    assert all("description" in prop for prop in schema["properties"].values())
    assert len(tool.description) > 100 and tool.spec().name == tool.name


def test_names_and_read_only_flags() -> None:
    flags = {cls.name: cls.read_only for cls in ALL_TOOLS}
    assert flags == {
        "read": True,
        "write": False,
        "edit": False,
        "glob": True,
        "grep": True,
        "bash": False,
        "web_fetch": True,
    }


def test_summaries_are_one_line_labels() -> None:
    assert ReadTool().summarize({"file_path": "src/app.py"}) == "read src/app.py"
    assert WriteTool().summarize({"file_path": "src/app.py", "content": "x"}) == "write src/app.py"
    summary = EditTool().summarize(
        {"file_path": "src/app.py", "old_string": "a", "new_string": "b"}
    )
    assert summary == "edit src/app.py"


# --------------------------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------------------------


async def test_read_numbers_lines_and_records_the_read(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.txt", "alpha\nbeta\n")
    out = await read(ctx, "a.txt")
    assert not out.is_error
    assert out.content == "     1\talpha\n     2\tbeta"
    assert ctx.files.has_read("n1", path) and not ctx.files.is_stale("n1", path)


async def test_read_accepts_an_absolute_path_inside_the_workspace(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.txt", "alpha\n")
    assert (await read(ctx, str(path))).content == "     1\talpha"


async def test_read_pages_with_offset_and_limit(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "ten.txt", "".join(f"line {n}\n" for n in range(1, 11)))
    out = await read(ctx, "ten.txt", offset=3, limit=2)
    assert out.content.startswith("     3\tline 3\n     4\tline 4\n\n[")
    assert "Showing lines 3-4 of 10." in out.content
    assert "offset=5" in out.content
    tail = await read(ctx, "ten.txt", offset=9)
    assert "line 10" in tail.content and "Showing lines 9-10 of 10." in tail.content
    assert "To continue" not in tail.content


async def test_read_caps_the_number_of_lines_and_says_how_to_continue(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "long.txt", "x\n" * (files.READ_LINES + 5))
    out = await read(ctx, "long.txt")
    lines = out.content.split("\n")
    assert lines[files.READ_LINES - 1] == f"{files.READ_LINES:>6}\tx"
    assert f"Showing lines 1-{files.READ_LINES} of {files.READ_LINES + 5}." in out.content
    assert f"offset={files.READ_LINES + 1}" in out.content


async def test_read_stops_at_the_output_budget_on_a_line_boundary(
    ctx: ToolContext, workspace: Path
) -> None:
    ctx.config = Config()
    ctx.config.agent.tool_output_limit = 3000
    put(workspace, "wide.txt", ("y" * 99 + "\n") * 200)
    out = await read(ctx, "wide.txt")
    assert len(out.content) <= 3000
    shown = [line for line in out.content.split("\n") if line.endswith("y")]
    assert 10 < len(shown) < 30 and all(len(line) == 106 for line in shown)
    assert f"To continue, call read with offset={len(shown) + 1}." in out.content


async def test_read_cuts_very_long_lines(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "min.js", "a" * (files.READ_LINE_CHARS + 3000) + "\nshort\n")
    out = await read(ctx, "min.js")
    first = out.content.split("\n")[0]
    assert first.endswith("[... 3,000 more characters on this line]")
    assert len(first) < files.READ_LINE_CHARS + 100
    assert "1 long line(s) were cut" in out.content and "grep" in out.content


async def test_read_shows_crlf_and_bom_files_cleanly(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "win.txt", b"\xef\xbb\xbfone\r\ntwo\r\n")
    assert (await read(ctx, "win.txt")).content == "     1\tone\n     2\ttwo"


async def test_read_decodes_legacy_encodings_instead_of_failing(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "latin.txt", b"caf\xe9\n")
    assert (await read(ctx, "latin.txt")).content == f"     1\tcaf{E_ACUTE}"


async def test_read_of_an_empty_file_says_so_and_counts_as_read(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "empty.py", "")
    out = await read(ctx, "empty.py")
    assert out.content == "[empty.py is empty.]" and not out.is_error
    assert ctx.files.has_read("n1", path)


async def test_read_missing_file_suggests_near_misses(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "src/app.py", "x\n")
    put(workspace, "src/api.py", "x\n")
    out = await read(ctx, "src/ap.py")
    assert out.is_error
    assert "src/ap.py does not exist" in out.content
    assert "Did you mean" in out.content and "src/app.py" in out.content
    assert "glob" in out.content
    assert not ctx.files.has_read("n1", workspace / "src/ap.py")


async def test_read_missing_file_in_missing_directory(ctx: ToolContext) -> None:
    out = await read(ctx, "nowhere/app.py")
    assert out.is_error and "neither does the directory" in out.content
    assert '"**/app.py"' in out.content


async def test_read_of_a_directory_points_to_glob(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "src").mkdir()
    out = await read(ctx, "src")
    assert out.is_error and "src is a directory" in out.content and "glob" in out.content


async def test_read_refuses_binary_files(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "logo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    out = await read(ctx, "logo.png")
    assert out.is_error and "binary file" in out.content
    assert not ctx.files.has_read("n1", path)


async def test_read_refuses_files_that_are_too_large(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files, "MAX_FILE_BYTES", 1024 * 1024)
    put(workspace, "dump.sql", "x" * (2 * 1024 * 1024))
    out = await read(ctx, "dump.sql")
    assert out.is_error and "dump.sql is 2.0 MB" in out.content
    assert "grep" in out.content and "sed -n" in out.content


async def test_read_rejects_an_offset_past_the_end_and_a_negative_limit(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "a.txt", "1\n2\n3\n")
    past = await read(ctx, "a.txt", offset=9)
    assert past.is_error and "has 3 lines" in past.content and "between 1 and 3" in past.content
    negative = await read(ctx, "a.txt", limit=-1)
    assert negative.is_error and "limit must be at least 1" in negative.content
    # A zero offset is what a model counting from 0 sends; it means "from the top".
    assert (await read(ctx, "a.txt", offset=0, limit=1)).content.startswith("     1\t1")


# -- read permission --------------------------------------------------------------------------


def test_reading_inside_the_workspace_needs_no_permission(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "src/app.py", "x\n")
    assert ReadTool().permission({"file_path": "src/app.py"}, ctx) is None
    assert ReadTool().permission({"file_path": "src"}, ctx) is None
    assert ReadTool().permission({"file_path": "not/there.py"}, ctx) is None


SECRET_LOOKING = (
    ".env .env.local prod.env server.pem id_rsa id_ed25519.pub .netrc credentials.json "
    "sub/dir/.ENV certs/site.key"
)


@pytest.mark.parametrize("name", SECRET_LOOKING.split())
def test_reading_a_file_that_looks_like_a_secret_asks(
    ctx: ToolContext, workspace: Path, name: str
) -> None:
    put(workspace, name, "TOKEN=1\n")
    ask = ReadTool().permission({"file_path": name}, ctx)
    assert ask is not None
    assert (ask.category, ask.target) == ("read", name)
    assert "secrets" in ask.summary
    # "allow for this session" covers this file and nothing else
    assert fnmatch.fnmatchcase(name, ask.session_rule)
    assert not fnmatch.fnmatchcase("src/app.py", ask.session_rule)


@pytest.mark.parametrize("name", [".env.example", ".env.sample", "environment.py", "keys.md"])
def test_templates_and_ordinary_files_are_not_treated_as_secrets(
    ctx: ToolContext, workspace: Path, name: str
) -> None:
    put(workspace, name, "TOKEN=\n")
    assert ReadTool().permission({"file_path": name}, ctx) is None


def test_reading_clankits_own_config_asks_even_inside_the_workspace(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLANKIT_CONFIG_DIR", str(workspace / "conf"))
    put(workspace, "conf/config.kdl", 'providers { anthropic api-key="sk-123" }\n')
    ask = ReadTool().permission({"file_path": "conf/config.kdl"}, ctx)
    assert ask is not None and ask.category == "read" and ask.target == "conf/config.kdl"


def test_a_symlink_to_a_secret_is_a_secret(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, ".env", "TOKEN=1\n")
    (workspace / "notes.txt").symlink_to(workspace / ".env")
    ask = ReadTool().permission({"file_path": "notes.txt"}, ctx)
    assert ask is not None and ask.target == ".env"


def test_reading_outside_the_workspace_asks_and_offers_the_directory(
    ctx: ToolContext, tmp_path: Path
) -> None:
    outside = put(tmp_path, "elsewhere/notes.txt", "x\n")
    ask = ReadTool().permission({"file_path": str(outside)}, ctx)
    assert ask is not None
    assert (ask.category, ask.target) == ("read", str(outside))
    assert "outside the workspace" in ask.summary
    assert ask.session_rule == f"{tmp_path}/elsewhere/*"
    assert fnmatch.fnmatchcase(ask.target, ask.session_rule)
    assert fnmatch.fnmatchcase(f"{tmp_path}/elsewhere/other.txt", ask.session_rule)
    # a relative path that climbs out is just as far outside
    climbing = ReadTool().permission({"file_path": "../elsewhere/notes.txt"}, ctx)
    assert climbing is not None and climbing.target == str(outside)


def test_a_symlink_that_leaves_the_workspace_is_outside_it(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    outside = put(tmp_path, "elsewhere/notes.txt", "x\n")
    (workspace / "shortcut").symlink_to(tmp_path / "elsewhere")
    ask = ReadTool().permission({"file_path": "shortcut/notes.txt"}, ctx)
    assert ask is not None and ask.category == "read"
    assert ask.target == str(outside)  # the real location, where rules can see it


def test_a_symlink_loop_is_treated_as_outside(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a").symlink_to(workspace / "b")
    (workspace / "b").symlink_to(workspace / "a")
    ask = ReadTool().permission({"file_path": "a/x.txt"}, ctx)
    assert ask is not None and ask.category == "read"


# --------------------------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------------------------


async def test_write_creates_a_file_and_its_directories(ctx: ToolContext, workspace: Path) -> None:
    out = await write(ctx, "pkg/sub/new.py", "print('hi')\nprint('there')\n")
    assert not out.is_error and out.content == "Created pkg/sub/new.py (2 lines)."
    assert (workspace / "pkg/sub/new.py").read_text() == "print('hi')\nprint('there')\n"
    assert out.display == {
        "type": "diff",
        "path": "pkg/sub/new.py",
        "diff": "--- /dev/null\n+++ b/pkg/sub/new.py\n@@ -0,0 +1,2 @@\n+print('hi')\n+print('there')",
    }
    assert ctx.files.changed_by("n1") == [workspace / "pkg/sub/new.py"]


async def test_write_will_not_overwrite_a_file_the_agent_has_not_read(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.txt", "precious\n")
    out = await write(ctx, "a.txt", "clobbered\n")
    assert out.is_error and "already exists and you have not read it" in out.content
    assert path.read_text() == "precious\n"
    assert WriteTool().permission({"file_path": "a.txt", "content": "clobbered\n"}, ctx) is None

    await read(ctx, "a.txt")
    out = await write(ctx, "a.txt", "replaced\nfully\n")
    assert out.content == "Overwrote a.txt (2 lines; it had 1 line)."
    assert path.read_text() == "replaced\nfully\n"
    assert out.display is not None and "-precious\n+replaced\n+fully" in out.display["diff"]


async def test_write_will_not_overwrite_a_file_that_changed_since_it_was_read(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.txt", "one\n")
    await read(ctx, "a.txt")
    change_behind_the_agents_back(path, "one\nsomebody else's line\n")
    out = await write(ctx, "a.txt", "mine\n")
    assert out.is_error and "has changed since you last read it" in out.content
    assert "Read it again" in out.content
    assert path.read_text() == "one\nsomebody else's line\n"


async def test_write_will_not_clobber_another_agents_work(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.txt", "shared\n")
    other = ctx.for_child("n2")
    await read(ctx, "a.txt")
    await read(other, "a.txt")
    assert not (await edit(other, "a.txt", "shared", "changed by n2")).is_error

    out = await write(ctx, "a.txt", "changed by n1\n")
    assert out.is_error and "has changed since you last read it" in out.content
    assert path.read_text() == "changed by n2\n"


async def test_write_of_identical_content_touches_nothing(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.txt", "same\n")
    await read(ctx, "a.txt")
    before = path.stat()
    out = await write(ctx, "a.txt", "same\n")
    assert not out.is_error and "already has exactly this content" in out.content
    assert (path.stat().st_mtime_ns, path.stat().st_ino) == (before.st_mtime_ns, before.st_ino)
    assert ctx.files.changed_by("n1") == []
    assert WriteTool().permission({"file_path": "a.txt", "content": "same\n"}, ctx) is None


async def test_write_to_a_directory_is_refused(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "src").mkdir()
    out = await write(ctx, "src", "x")
    assert out.is_error and "src is a directory" in out.content


async def test_write_reports_a_parent_that_is_a_file(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "a.txt", "x\n")
    out = await write(ctx, "a.txt/inside.txt", "x")
    assert out.is_error and "Could not write a.txt/inside.txt" in out.content


async def test_writes_are_atomic_and_keep_the_file_mode(ctx: ToolContext, workspace: Path) -> None:
    script = put(workspace, "run.sh", "#!/bin/sh\necho one\n")
    script.chmod(0o750)
    await read(ctx, "run.sh")
    assert not (await edit(ctx, "run.sh", "one", "two")).is_error
    assert stat.S_IMODE(script.stat().st_mode) == 0o750
    await read(ctx, "run.sh")
    assert not (await write(ctx, "run.sh", "#!/bin/sh\necho three\n")).is_error
    assert stat.S_IMODE(script.stat().st_mode) == 0o750

    umask = os.umask(0)
    os.umask(umask)
    await write(ctx, "fresh.txt", "x\n")
    assert stat.S_IMODE((workspace / "fresh.txt").stat().st_mode) == 0o666 & ~umask
    # nothing but the files themselves is left behind
    assert names_in(workspace) == ["fresh.txt", "run.sh"]


async def test_a_failed_write_leaves_no_temporary_file(
    ctx: ToolContext, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = put(workspace, "a.txt", "before\n")
    await read(ctx, "a.txt")

    def refuse(*args: Any) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", refuse)
    out = await edit(ctx, "a.txt", "before", "after")
    assert out.is_error and "Could not write a.txt: Permission denied" in out.content
    assert path.read_text() == "before\n"
    assert names_in(workspace) == ["a.txt"]


async def test_writing_through_a_symlink_changes_the_target_and_keeps_the_link(
    ctx: ToolContext, workspace: Path
) -> None:
    target = put(workspace, "real.txt", "one\n")
    link = workspace / "link.txt"
    link.symlink_to(target)
    await read(ctx, "link.txt")
    assert not (await edit(ctx, "link.txt", "one", "two")).is_error
    assert link.is_symlink() and target.read_text() == "two\n"


# --------------------------------------------------------------------------------------------
# edit
# --------------------------------------------------------------------------------------------


async def test_edit_replaces_exact_text_and_reports_a_diff(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "src/app.py", "def main():\n    return 1\n\nmain()\n")
    await read(ctx, "src/app.py")
    out = await edit(ctx, "src/app.py", "return 1", "return 2")
    assert not out.is_error and out.content == "Edited src/app.py: 1 replacement at line 2."
    assert path.read_text() == "def main():\n    return 2\n\nmain()\n"
    assert out.display == {
        "type": "diff",
        "path": "src/app.py",
        "diff": (
            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1,4 +1,4 @@\n"
            " def main():\n-    return 1\n+    return 2\n \n main()"
        ),
    }
    # the agent's picture is current after its own write: it can go straight on
    assert not (await edit(ctx, "src/app.py", "main()\n", "main()  # entry\n")).is_error
    assert path.read_text() == "def main():\n    return 2\n\nmain()  # entry\n"
    assert ctx.files.changed_by("n1") == [path]


async def test_edit_requires_the_file_to_have_been_read(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.txt", "one\n")
    out = await edit(ctx, "a.txt", "one", "two")
    assert out.is_error and "You have not read a.txt yet" in out.content
    assert path.read_text() == "one\n"
    # one agent's read does not count for another
    await read(ctx.for_child("n2"), "a.txt")
    assert (await edit(ctx, "a.txt", "one", "two")).is_error
    missing = await edit(ctx, "new.txt", "one", "two")
    assert missing.is_error and "create it with write" in missing.content


async def test_edit_refuses_calls_that_can_change_nothing(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "a.txt", "one\r\ntwo\r\n")
    await read(ctx, "a.txt")
    same = await edit(ctx, "a.txt", "one", "one")
    assert same.is_error and "identical" in same.content
    empty = await edit(ctx, "a.txt", "", "zero\n")
    assert empty.is_error and "old_string is empty" in empty.content and "write" in empty.content
    endings = await edit(ctx, "a.txt", "one\r\ntwo", "one\ntwo")
    assert endings.is_error and "differ only in line endings" in endings.content


async def test_edit_reports_how_many_places_match_and_where(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.py", "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "x = 1", "x = 9")
    assert out.is_error
    assert "matches 3 places in a.py (lines 1, 3, 5)" in out.content
    assert "replace_all" in out.content and "neighbouring lines" in out.content
    assert path.read_text().count("x = 1") == 3

    # more context makes it unique ...
    assert not (await edit(ctx, "a.py", "y = 2\nx = 1", "y = 2\nx = 7")).is_error
    # ... and replace_all takes the rest
    out = await edit(ctx, "a.py", "x = 1", "x = 9", replace_all=True)
    assert out.content == "Edited a.py: 2 replacements at lines 1, 5."
    assert path.read_text() == "x = 9\ny = 2\nx = 7\nz = 3\nx = 9\n"


async def test_edit_matches_across_line_ending_styles_and_preserves_them(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "win.txt", b"alpha\r\nbeta\r\ngamma\r\n")
    await read(ctx, "win.txt")
    out = await edit(ctx, "win.txt", "alpha\nbeta", "alpha\nBETA\ninserted")
    assert not out.is_error and "replacement at line 1" in out.content
    assert path.read_bytes() == b"alpha\r\nBETA\r\ninserted\r\ngamma\r\n"

    unix = put(workspace, "unix.txt", b"alpha\nbeta\n")
    await read(ctx, "unix.txt")
    assert not (await edit(ctx, "unix.txt", "alpha\r\nbeta", "one\r\ntwo")).is_error
    assert unix.read_bytes() == b"one\ntwo\n"


async def test_edit_leaves_untouched_lines_of_a_mixed_file_alone(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "mixed.txt", b"a\r\nb\nc\r\nd\r\n")
    await read(ctx, "mixed.txt")
    assert not (await edit(ctx, "mixed.txt", "c", "C\nc2")).is_error
    assert path.read_bytes() == b"a\r\nb\nC\r\nc2\r\nd\r\n"


async def test_edit_falls_back_to_typographic_equivalents(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "doc.md", f"She said {LDQUO}hello{RDQUO} {EM_DASH} twice.\n")
    await read(ctx, "doc.md")
    out = await edit(ctx, "doc.md", 'said "hello" - twice', "said nothing")
    assert not out.is_error and "typographic" in out.content
    assert path.read_text() == "She said nothing.\n"

    plain = put(workspace, "plain.md", 'print("hi")\n')
    await read(ctx, "plain.md")
    out = await edit(ctx, "plain.md", f"print({LDQUO}hi{RDQUO})", 'print("bye")')
    assert not out.is_error and plain.read_text() == 'print("bye")\n'


async def test_edit_falls_back_to_ignoring_whitespace_at_line_ends(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.py", "def f():   \n    return 1\t\n\nrest = 0\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
    assert not out.is_error and "ignoring whitespace at the ends of lines" in out.content
    assert path.read_text() == "def f():\n    return 2\n\nrest = 0\n"

    # the other way round: the model sent trailing blanks the file does not have
    other = put(workspace, "b.py", "one\ntwo\nthree\n")
    await read(ctx, "b.py")
    assert not (await edit(ctx, "b.py", "one  \ntwo", "1\n2")).is_error
    assert other.read_text() == "1\n2\nthree\n"


async def test_edit_never_matches_on_different_indentation(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.py", "class A:\n    def f(self):\n        return 1\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "def f(self):\n    return 1", "def f(self):\n    return 2")
    assert out.is_error and path.read_text().endswith("return 1\n")
    assert "old_string was not found in a.py" in out.content
    assert "     2\t    def f(self):\n     3\t        return 1" in out.content
    assert "differ from old_string only in whitespace" in out.content


async def test_edit_miss_shows_the_most_similar_line(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "a.py", "import os\n\ndef compute_total(items):\n    return sum(items)\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "def compute_totals(item):", "def total(items):")
    assert out.is_error and "The closest lines in the file are" in out.content
    assert "     3\tdef compute_total(items):" in out.content
    assert "without the line numbers" in out.content


async def test_edit_miss_notices_an_edit_that_was_already_applied(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "a.py", "def handler(request, response):\n    pass\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "def handler(request):", "def handler(request, response):")
    assert out.is_error and "new_string is already present" in out.content
    assert "may already have been applied" in out.content


async def test_edit_miss_notices_line_numbers_copied_from_read(
    ctx: ToolContext, workspace: Path
) -> None:
    put(workspace, "a.py", "alpha = 1\nbeta = 2\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "     1\talpha = 1\n     2\tbeta = 2", "alpha = 3")
    assert out.is_error and "line numbers that read prints" in out.content


async def test_edit_miss_with_nothing_similar_says_so(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "a.py", "alpha = 1\n")
    await read(ctx, "a.py")
    out = await edit(ctx, "a.py", "completely_unrelated(call)", "x")
    assert out.is_error and "Nothing in the file resembles it" in out.content


async def test_edit_tolerates_a_stale_file_when_the_match_is_exact_and_unique(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.py", "alpha = 1\nbeta = 2\n")
    await read(ctx, "a.py")
    change_behind_the_agents_back(path, "alpha = 1\nbeta = 2\ngamma = 3\n")
    out = await edit(ctx, "a.py", "beta = 2", "beta = 20")
    assert not out.is_error and "had changed since you last read it" in out.content
    assert path.read_text() == "alpha = 1\nbeta = 20\ngamma = 3\n"


async def test_edit_refuses_a_stale_file_when_the_match_is_not_exact_and_unique(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "a.py", "alpha = 1\nbeta = 2\n")
    await read(ctx, "a.py")
    changed = "alpha = 1  \nbeta = 2\nbeta = 2\n"
    change_behind_the_agents_back(path, changed)

    for old, extra in [
        ("beta = 2", {}),  # now ambiguous
        ("beta = 2", {"replace_all": True}),  # sweeping change on unseen content
        ("alpha = 1\nbeta", {}),  # only a near match
        ("delta = 4", {}),  # gone
    ]:
        out = await edit(ctx, "a.py", old, "x", **extra)
        assert out.is_error and "has changed since you last read it" in out.content
        assert "Read it again" in out.content
    assert path.read_text() == changed

    await read(ctx, "a.py")
    assert not (await edit(ctx, "a.py", "beta = 2", "x", replace_all=True)).is_error


async def test_edit_after_another_agents_write(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.py", "alpha = 1\nbeta = 2\n")
    other = ctx.for_child("n2")
    await read(ctx, "a.py")
    await read(other, "a.py")
    assert not (await edit(other, "a.py", "beta = 2", "beta = 2\nbeta_copy = 2")).is_error
    assert ctx.files.is_stale("n1", path) and not ctx.files.is_stale("n2", path)

    still_unique = await edit(ctx, "a.py", "alpha = 1", "alpha = 10")
    assert not still_unique.is_error and "had changed" in still_unique.content
    assert path.read_text() == "alpha = 10\nbeta = 2\nbeta_copy = 2\n"
    # ... which in turn makes n2's picture stale
    gone = await edit(other, "a.py", "alpha = 1\n", "")
    assert gone.is_error and "has changed since you last read it" in gone.content


async def test_edit_of_a_file_deleted_after_it_was_read(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.py", "alpha = 1\n")
    await read(ctx, "a.py")
    path.unlink()
    out = await edit(ctx, "a.py", "alpha", "beta")
    assert out.is_error and "a.py does not exist" in out.content


@pytest.mark.parametrize(
    ("content", "old", "new", "expected"),
    [
        ("a\nb\n", "b\n", "c", "a\nc\n"),  # the model dropped the final newline
        ("a\nb", "b", "c\n", "a\nc"),  # the model added one
        ("a\nb\n", "b", "c", "a\nc\n"),
        ("a\nb", "a\n", "", "b"),
        ("only\n", "only\n", "", ""),  # deleting everything leaves an empty file
    ],
)
async def test_edit_preserves_whether_the_file_ends_with_a_newline(
    ctx: ToolContext, workspace: Path, content: str, old: str, new: str, expected: str
) -> None:
    path = put(workspace, "a.txt", content)
    await read(ctx, "a.txt")
    assert not (await edit(ctx, "a.txt", old, new)).is_error
    assert path.read_text() == expected


async def test_edit_keeps_the_bytes_of_a_legacy_encoded_file(
    ctx: ToolContext, workspace: Path
) -> None:
    path = put(workspace, "latin.txt", b"caf\xe9 au lait\nprice = 1\n")
    await read(ctx, "latin.txt")
    assert not (await edit(ctx, "latin.txt", "price = 1", "price = 2")).is_error
    assert path.read_bytes() == b"caf\xe9 au lait\nprice = 2\n"
    out = await edit(ctx, "latin.txt", "price = 2", f"price = 2 {EM_DASH} sale")
    assert out.is_error and "not UTF-8" in out.content and "iconv" in out.content
    assert path.read_bytes() == b"caf\xe9 au lait\nprice = 2\n"


async def test_edit_waits_for_the_files_lock(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.txt", "one\n")
    await read(ctx, "a.txt")
    lock = ctx.files.lock(path)
    await lock.acquire()
    task = asyncio.create_task(edit(ctx, "a.txt", "one", "two"))
    await asyncio.sleep(0.02)
    assert not task.done() and path.read_text() == "one\n"
    lock.release()
    assert not (await task).is_error and path.read_text() == "two\n"


# --------------------------------------------------------------------------------------------
# Rules shared by edit and write
# --------------------------------------------------------------------------------------------


async def test_a_path_outside_the_write_scope_is_refused(ctx: ToolContext, workspace: Path) -> None:
    ctx.write_scope = ("src/api/", "tests/test_api_*.py")
    theirs = put(workspace, "src/ui/view.py", "x = 1\n")
    await read(ctx, "src/ui/view.py")

    out = await edit(ctx, "src/ui/view.py", "x = 1", "x = 2")
    assert out.is_error and theirs.read_text() == "x = 1\n"
    assert "src/ui/view.py is outside this task's write scope" in out.content
    assert "src/api/, tests/test_api_*.py" in out.content
    assert "belongs to another task" in out.content
    assert "report to your parent" in out.content
    created = await write(ctx, "docs/notes.md", "x")
    assert created.is_error and "write scope" in created.content
    assert not (workspace / "docs").exists()
    # nobody is asked to approve something that will not happen
    refused = {"file_path": "src/ui/view.py", "old_string": "x = 1", "new_string": "x = 2"}
    assert EditTool().permission(refused, ctx) is None

    assert not (await write(ctx, "src/api/routes.py", "x")).is_error
    assert not (await write(ctx, "tests/test_api_routes.py", "x")).is_error


async def test_a_child_cannot_write_more_than_its_ancestors(
    ctx: ToolContext, workspace: Path
) -> None:
    ctx.write_scope = ("src/",)
    child = ctx.for_child("n2", ("src/api/", "docs/"))
    grandchild = child.for_child("n3")  # unrestricted itself, still bound by both above

    for agent in (child, grandchild):
        escaped = await write(agent, "docs/guide.md", "x")
        assert escaped.is_error and "inherited from a parent task: src/" in escaped.content
    sibling = await write(child, "src/ui/view.py", "x")
    assert sibling.is_error and "(src/api/, docs/)" in sibling.content
    inherited = await write(grandchild, "src/ui/view.py", "x")
    assert (
        inherited.is_error and "inherited from a parent task: src/api/, docs/" in inherited.content
    )

    assert not (await write(child, "src/api/a.py", "x")).is_error
    assert not (await write(grandchild, "src/api/b.py", "x")).is_error
    assert not (workspace / "docs").exists() and not (workspace / "src/ui").exists()


@pytest.mark.parametrize("name", [".git/config", ".git/hooks/pre-commit", "vendor/lib/.git/HEAD"])
async def test_nothing_is_written_inside_a_git_directory(
    ctx: ToolContext, workspace: Path, name: str
) -> None:
    existing = put(workspace, ".git/config", "[core]\n")
    await read(ctx, ".git/config")
    out = await write(ctx, name, "[core]\n\thooksPath = /tmp/evil\n")
    assert (
        out.is_error and "inside a .git directory" in out.content and "git commands" in out.content
    )
    assert existing.read_text() == "[core]\n"
    assert WriteTool().permission({"file_path": name, "content": "x"}, ctx) is None
    changed = await edit(ctx, ".git/config", "[core]", "[evil]")
    assert changed.is_error and existing.read_text() == "[core]\n"


async def test_a_symlink_does_not_lead_into_a_git_directory(
    ctx: ToolContext, workspace: Path
) -> None:
    (workspace / ".git/hooks").mkdir(parents=True)
    (workspace / "scripts").symlink_to(workspace / ".git/hooks")
    out = await write(ctx, "scripts/pre-commit", "#!/bin/sh\n")
    assert out.is_error and "inside a .git directory" in out.content
    assert names_in(workspace / ".git/hooks") == []


async def test_files_named_like_git_are_ordinary(ctx: ToolContext, workspace: Path) -> None:
    assert not (await write(ctx, ".gitignore", "dist/\n")).is_error
    assert not (await write(ctx, ".github/workflows/ci.yml", "on: push\n")).is_error


async def test_clankits_own_configuration_is_never_written(
    ctx: ToolContext, tmp_path: Path
) -> None:
    config = paths.config_file()
    config.parent.mkdir(parents=True)
    config.write_text('permissions { mode "ask" }\n')
    await read(ctx, str(config))

    for target in (config, paths.skills_dir() / "helper" / "SKILL.md"):
        out = await write(ctx, str(target), 'permissions { mode "auto" }\n')
        assert out.is_error and "Clankit's own configuration directory" in out.content
        assert "tell the user" in out.content
    out = await edit(ctx, str(config), '"ask"', '"auto"')
    assert out.is_error and config.read_text() == 'permissions { mode "ask" }\n'
    assert not paths.skills_dir().exists()


async def test_undo_restores_edits_and_removes_created_files(
    ctx: ToolContext, workspace: Path
) -> None:
    kept = put(workspace, "kept.txt", "original\n")
    ctx.files.begin_run()
    await read(ctx, "kept.txt")
    await edit(ctx, "kept.txt", "original", "first change")
    await edit(ctx, "kept.txt", "first change", "second change")
    await write(ctx, "made/new.txt", "brand new\n")
    assert kept.read_text() == "second change\n" and (workspace / "made/new.txt").exists()

    restored = ctx.files.undo()
    assert sorted(restored) == sorted([kept, workspace / "made/new.txt"])
    assert kept.read_text() == "original\n"  # the state before the run, not before the last edit
    assert not (workspace / "made/new.txt").exists()
    # undo put different content on disk: the agent has to look again before editing
    out = await edit(ctx, "kept.txt", "original", "changed again")
    assert out.is_error and "You have not read kept.txt yet" in out.content
    assert ctx.files.undo() == []


async def test_undo_is_per_run(ctx: ToolContext, workspace: Path) -> None:
    path = put(workspace, "a.txt", "v1\n")
    await read(ctx, "a.txt")
    ctx.files.begin_run()
    await edit(ctx, "a.txt", "v1", "v2")
    ctx.files.begin_run()
    await edit(ctx, "a.txt", "v2", "v3")
    ctx.files.undo()
    assert path.read_text() == "v2\n"
    ctx.files.undo()
    assert path.read_text() == "v1\n"


# -- edit and write permission ----------------------------------------------------------------


async def test_edit_permission_inside_the_workspace(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "src/app.py", "def main():\n    return 1\n")
    await read(ctx, "src/app.py")
    request = {"file_path": "src/app.py", "old_string": "return 1", "new_string": "return 2"}
    ask = EditTool().permission(request, ctx)
    assert ask is not None
    assert (ask.category, ask.summary, ask.target) == ("edit", "Edit src/app.py", "src/app.py")
    assert ask.session_rule == "*"  # "allow for this session" means every edit in the workspace
    assert "-    return 1\n+    return 2" in ask.detail and ask.detail.startswith(
        "--- a/src/app.py"
    )
    # asking changes nothing
    assert (workspace / "src/app.py").read_text() == "def main():\n    return 1\n"


async def test_write_permission_says_create_or_overwrite(ctx: ToolContext, workspace: Path) -> None:
    create = WriteTool().permission({"file_path": "docs/new.md", "content": "# New\n"}, ctx)
    assert create is not None
    assert (create.category, create.summary, create.target) == (
        "edit",
        "Create docs/new.md",
        "docs/new.md",
    )
    assert create.session_rule == "*" and "+# New" in create.detail
    assert not (workspace / "docs").exists()

    put(workspace, "docs/old.md", "# Old\n")
    await read(ctx, "docs/old.md")
    overwrite = WriteTool().permission({"file_path": "docs/old.md", "content": "# Newer\n"}, ctx)
    assert overwrite is not None and overwrite.summary == "Overwrite docs/old.md"
    assert "-# Old\n+# Newer" in overwrite.detail


async def test_edit_permission_outside_the_workspace_names_the_exact_file(
    ctx: ToolContext, tmp_path: Path
) -> None:
    outside = put(tmp_path, "else[1]/notes.txt", "one\n")
    create = WriteTool().permission(
        {"file_path": str(outside.with_name("new.txt")), "content": "x"}, ctx
    )
    assert create is not None
    assert (create.category, create.target) == ("edit", str(outside.with_name("new.txt")))
    assert create.summary == f"Create {outside.with_name('new.txt')}"
    # the remembered rule is that one file, even with wildcard characters in its name
    assert create.session_rule != "*"
    assert fnmatch.fnmatchcase(create.target, create.session_rule)
    assert not fnmatch.fnmatchcase(str(outside), create.session_rule)

    await read(ctx, str(outside))
    request = {"file_path": str(outside), "old_string": "one", "new_string": "two"}
    ask = EditTool().permission(request, ctx)
    assert ask is not None and ask.target == str(outside) and ask.target.startswith("/")
    assert not (await edit(ctx, str(outside), "one", "two")).is_error
    assert outside.read_text() == "two\n"


async def test_editing_through_an_escaping_symlink_is_an_outside_edit(
    ctx: ToolContext, workspace: Path, tmp_path: Path
) -> None:
    outside = put(tmp_path, "elsewhere/notes.txt", "one\n")
    (workspace / "shortcut").symlink_to(tmp_path / "elsewhere")
    await read(ctx, "shortcut/notes.txt")
    request = {"file_path": "shortcut/notes.txt", "old_string": "one", "new_string": "two"}
    ask = EditTool().permission(request, ctx)
    assert ask is not None and ask.target == str(outside) and ask.session_rule != "*"


async def test_calls_that_will_be_refused_ask_nothing(ctx: ToolContext, workspace: Path) -> None:
    put(workspace, "a.txt", "one\n")
    unread = {"file_path": "a.txt", "old_string": "one", "new_string": "two"}
    assert EditTool().permission(unread, ctx) is None
    await read(ctx, "a.txt")
    assert EditTool().permission(unread, ctx) is not None
    for doomed in (
        {"file_path": "a.txt", "old_string": "absent", "new_string": "two"},
        {"file_path": "a.txt", "old_string": "one", "new_string": "one"},
        {"file_path": "missing.txt", "old_string": "one", "new_string": "two"},
    ):
        assert EditTool().permission(doomed, ctx) is None


# -- diffs ------------------------------------------------------------------------------------


def reference_diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), "a/f.py", "b/f.py", lineterm=""
    )
    return "\n".join(lines)


@pytest.mark.parametrize(
    "change",
    [
        lambda lines: [*lines[:40], "changed", *lines[41:]],
        lambda lines: [*lines[:10], *lines[12:]],
        lambda lines: [*lines[:99], "inserted a", "inserted b", *lines[99:]],
        lambda lines: ["new first", *lines],
        lambda lines: [*lines[:5], "x", *lines[6:50], "y", *lines[51:]],
        lambda lines: lines[:-1],
    ],
)
def test_unified_diff_agrees_with_difflib(change: Any) -> None:
    lines = [f"line {n}" for n in range(1, 101)]
    before, after = "\n".join(lines) + "\n", "\n".join(change(lines)) + "\n"
    assert unified_diff("f.py", before, after) == reference_diff(before, after)


def test_unified_diff_is_capped_and_says_so() -> None:
    before = "".join(f"old line {n}\n" for n in range(3000))
    after = "".join(f"new line {n}\n" for n in range(3000))
    diff = unified_diff("big.txt", before, after)
    assert len(diff) <= files.MAX_DIFF_CHARS + 100
    assert diff.splitlines()[-1].startswith("[diff truncated: ") and "more lines]" in diff


def test_unified_diff_of_a_huge_rewrite_does_not_align_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files, "MAX_ALIGNED_LINES", 5)
    before = "".join(f"a{n}\n" for n in range(10))
    after = "".join(f"b{n}\n" for n in range(12))
    diff = unified_diff("f", before, after, limit=10_000).splitlines()
    assert diff[2] == "@@ -1,10 +1,12 @@"
    assert diff[3:13] == [f"-a{n}" for n in range(10)] and diff[13:] == [
        f"+b{n}" for n in range(12)
    ]


def test_unified_diff_names_changes_it_cannot_show() -> None:
    assert unified_diff("f", "a\r\nb\r\n", "a\nb\n").splitlines()[2:] == [
        "[only line endings or the final newline differ]"
    ]
    assert unified_diff("f", "same\n", "same\n").splitlines() == ["--- a/f", "+++ b/f"]


# --------------------------------------------------------------------------------------------
# Through the agent loop
# --------------------------------------------------------------------------------------------


def make_agent(session: Session, *tools: Tool) -> Any:
    node = session.new_node("test", role="root")
    return session.agent(node, system="You are a test agent.", tools=list(tools))


async def test_file_tools_work_through_the_agent_loop(
    session: Session, scripted: ScriptedProvider, events: list, workspace: Path
) -> None:
    put(workspace, "a.txt", "one\n")
    scripted.push(
        call("edit", file_path="a.txt", old_string="one", new_string="two"),  # too early
        call("read", file_path="a.txt"),
        call("edit", file_path="a.txt", old_string="one", new_string="two"),
        call("write", file_path="b.txt", content="new\n"),
        "done",
    )
    agent = make_agent(session, ReadTool(), EditTool(), WriteTool())
    assert (await agent.run("change it")).text == "done"

    assert (workspace / "a.txt").read_text() == "two\n" and (
        workspace / "b.txt"
    ).read_text() == "new\n"
    assert [e.summary for e in events if isinstance(e, ToolStarted)] == [
        "edit a.txt",
        "read a.txt",
        "edit a.txt",
        "write b.txt",
    ]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert [e.is_error for e in finished] == [True, False, False, False]
    assert "You have not read a.txt yet" in finished[0].output
    assert finished[2].display is not None and finished[2].display["type"] == "diff"
    assert finished[3].display is not None and finished[3].display["path"] == "b.txt"
    assert session.undo() and (workspace / "a.txt").read_text() == "one\n"
    assert not (workspace / "b.txt").exists()


async def test_the_loop_rejects_arguments_the_schema_does_not_allow(
    session: Session, scripted: ScriptedProvider, workspace: Path
) -> None:
    put(workspace, "a.txt", "one\n")
    scripted.push(
        [call("read", path="a.txt"), call("read", file_path="a.txt", limit="ten")],
        "ok",
    )
    agent = make_agent(session, ReadTool())
    await agent.run("go")
    results = agent.messages[2].tool_result_blocks
    assert all(r.is_error and "Invalid input for read" in r.content for r in results)


async def test_an_edit_asks_the_user_with_a_diff_and_a_workspace_wide_rule(
    session: Session, scripted: ScriptedProvider, events: list, workspace: Path
) -> None:
    session.config.permissions.mode = "ask"
    put(workspace, "a.txt", "one\n")
    scripted.push(
        call("read", file_path="a.txt"),
        call("edit", file_path="a.txt", old_string="one", new_string="two"),
        call("write", file_path="b.txt", content="new\n"),
        "done",
    )
    agent = make_agent(session, ReadTool(), EditTool(), WriteTool())
    task = asyncio.create_task(agent.run("change it"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if session.view.permissions:
            break
    (request,) = [e for e in events if isinstance(e, PermissionRequested)]
    assert (request.tool, request.category, request.summary) == ("edit", "edit", "Edit a.txt")
    assert request.session_rule == "edit:*" and "-one\n+two" in request.detail
    assert (workspace / "a.txt").read_text() == "one\n"  # nothing happens before the answer

    assert session.resolve_permission(request.request_id, "allow_session")
    assert (await task).text == "done"
    assert (workspace / "a.txt").read_text() == "two\n" and (workspace / "b.txt").exists()
    # the write that followed was covered by the session rule
    assert sum(isinstance(e, PermissionRequested) for e in events) == 1


# -- windows -----------------------------------------------------------------------------------


def test_windows_command_prefers_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    from clankit.tools import shell as shellmod

    monkeypatch.setattr(
        shellmod.shutil,
        "which",
        lambda name: r"C:\Program Files\Git\bin\bash.exe" if name == "bash" else None,
    )
    assert shellmod._windows_command("ls -la") == [
        r"C:\Program Files\Git\bin\bash.exe",
        "-c",
        "ls -la",
    ]


def test_windows_command_falls_back_to_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    from clankit.tools import shell as shellmod

    monkeypatch.setattr(shellmod.shutil, "which", lambda name: None)
    monkeypatch.setenv("ComSpec", r"C:\Windows\System32\cmd.exe")
    assert shellmod._windows_command("dir") == [r"C:\Windows\System32\cmd.exe", "/d", "/c", "dir"]


class _FakeStream:
    """A subprocess stdout stand-in: fixed chunks, then end-of-file."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


async def test_windows_pump_reads_everything() -> None:
    from clankit.tools import shell as shellmod

    capture = shellmod._Capture(1000)
    await shellmod._pump(_FakeStream([b"abc", b"def", b"", b"never"]), capture)  # type: ignore[arg-type]
    assert capture.text() == "abcdef"


async def test_windows_drain_cancels_a_stuck_pump() -> None:
    from clankit.tools import shell as shellmod

    async def stuck() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(stuck())
    await shellmod._drain(task, 0.05)
    await asyncio.sleep(0)  # let the cancellation land
    assert task.cancelled()


def test_ctrl_break_is_a_noop_on_posix() -> None:
    from clankit.tools import shell as shellmod

    class Process:
        pid = 1

    shellmod._ctrl_break(Process())  # type: ignore[arg-type]

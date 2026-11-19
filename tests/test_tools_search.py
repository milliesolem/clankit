"""The search tools: glob and grep, with ripgrep and with the pure-Python fallback."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from clankit.config.schema import Config
from clankit.tools import search
from clankit.tools.base import AllowAll, FileLedger, ToolContext, ToolOutput
from clankit.tools.search import GlobTool, GrepTool, glob_matcher

HAVE_RIPGREP = shutil.which("rg") is not None
needs_ripgrep = pytest.mark.skipif(not HAVE_RIPGREP, reason="ripgrep is not installed")

#: The sample project, oldest file first (`tree` gives them increasing modification times).
PROJECT = {
    "README.md": "# Demo\nfoo is documented here\n",
    "a.py": "import os\nfoo = 1\n",
    "A.py": "FOO = 2\n",
    "a-b.py": "nothing here\n",
    "a/b.py": "def foo():\n    return 'two'\n",
    "src/main.py": "from pkg import mod\n\n\ndef main():\n    foo()\n    bar()\n    foo()\n",
    "src/pkg/mod.py": "foo = 'mod'\n",
    "src/app.tsx": "export const foo = 1\n",
    "src/app.test.ts": "test('foo', () => {})\n",
    ".github/workflows/ci.yml": "run: foo\n",
    "notes/multi.txt": "one\ntwo foo\nthree\nfour\nfive\nsix foo\nseven\n",
    "notes/win.txt": "alpha\r\nfoo\r\nomega\r\n",
    # None of these may ever show up:
    ".git/config": "foo\n",
    "node_modules/lib/index.js": "foo\n",
    ".venv/lib/site.py": "foo\n",
    "__pycache__/a.cpython-312.pyc": "foo\n",
    "build/out.py": "foo\n",
    "src/pkg/dist/bundle.js": "foo\n",
}
HIDDEN = {name for name in PROJECT if name.split("/")[0] in search.SKIP_DIRS or "/dist/" in name}
VISIBLE = [name for name in PROJECT if name not in HIDDEN]


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        node_id="n1",
        files=FileLedger(),
        permissions=AllowAll(),
        emit=lambda event: None,
    )


@pytest.fixture
def tree(workspace: Path, tmp_path: Path) -> Path:
    for age, (name, content) in enumerate(PROJECT.items()):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
        os.utime(path, (1_700_000_000 + age, 1_700_000_000 + age))
    (workspace / "blob.bin").write_bytes(b"\x00\x01foo\x00 foo\n")
    (workspace / ".env").write_text("FOO_TOKEN=foo\n")
    # Symbolic links are never followed: not to a file, and not out of the workspace.
    (workspace / "alias.py").symlink_to(workspace / "a.py")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.py").write_text("foo = 'outside'\n")
    (workspace / "shortcut").symlink_to(outside)
    return workspace


@pytest.fixture(params=["ripgrep", "python"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a test once per search implementation."""
    if request.param == "python":
        monkeypatch.setattr(search, "ripgrep", lambda: None)
    elif not HAVE_RIPGREP:
        pytest.skip("ripgrep is not installed")
    return request.param


async def glob(ctx: ToolContext, pattern: str, **more: Any) -> ToolOutput:
    return await GlobTool().run({"pattern": pattern, **more}, ctx)


async def grep(ctx: ToolContext, pattern: str, **more: Any) -> ToolOutput:
    return await GrepTool().run({"pattern": pattern, **more}, ctx)


def listed(output: ToolOutput) -> list[str]:
    assert not output.is_error, output.content
    return [
        line
        for line in output.content.split("\n")
        if not line.startswith("[") and not line.startswith("No ")
    ]


# --------------------------------------------------------------------------------------------
# glob
# --------------------------------------------------------------------------------------------


def test_summaries() -> None:
    assert GlobTool().summarize({"pattern": "**/*.py"}) == "glob **/*.py"
    assert GlobTool().summarize({"pattern": "*.py", "path": "src"}) == "glob *.py in src"
    assert GrepTool().summarize({"pattern": "foo", "path": "src/"}) == "grep 'foo' in src/"
    assert GrepTool().summarize({"pattern": "foo", "glob": "*.py"}) == "grep 'foo' (*.py)"
    assert GrepTool().summarize({"pattern": "foo"}) == "grep 'foo'"


async def test_glob_lists_matching_files_newest_first(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    out = await glob(ctx, "*.py")
    assert listed(out) == ["src/pkg/mod.py", "src/main.py", "a/b.py", "a-b.py", "A.py", "a.py"]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("src/*.py", ["src/main.py"]),
        ("src/**/*.py", ["src/main.py", "src/pkg/mod.py"]),
        ("**/pkg/*.py", ["src/pkg/mod.py"]),
        ("pkg/*.py", []),  # a pattern with a slash is relative to the directory searched
        ("*.{ts,tsx}", ["src/app.test.ts", "src/app.tsx"]),
        ("src/", ["src/app.test.ts", "src/app.tsx", "src/main.py", "src/pkg/mod.py"]),
        ("./src/app.*", ["src/app.test.ts", "src/app.tsx"]),
        ("[aA].py", ["A.py", "a.py"]),
        ("ci.yml", [".github/workflows/ci.yml"]),  # hidden directories are searched
        ("?.py", ["A.py", "a.py", "a/b.py"]),
    ],
)
async def test_glob_pattern_semantics(
    ctx: ToolContext, tree: Path, backend: str, pattern: str, expected: list[str]
) -> None:
    out = await glob(ctx, pattern)
    if not expected:
        assert out.content == f"No files match {pattern!r} in the workspace."
    else:
        assert sorted(listed(out)) == expected


async def test_glob_skips_heavy_directories_links_and_git(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    everything = listed(await glob(ctx, "**"))
    assert sorted(everything) == sorted([*VISIBLE, ".env", "blob.bin"])
    assert listed(await glob(ctx, "*", path="shortcut")) == ["shortcut/secret.py"]


async def test_glob_in_a_directory_named_by_path(ctx: ToolContext, tree: Path, backend: str) -> None:
    assert listed(await glob(ctx, "*.py", path="src")) == ["src/pkg/mod.py", "src/main.py"]
    # naming a directory that searches normally skip is a way to look inside it
    assert listed(await glob(ctx, "*", path="build")) == ["build/out.py"]
    assert listed(await glob(ctx, "*.js", path="node_modules/lib")) == ["node_modules/lib/index.js"]


async def test_glob_accepts_an_absolute_pattern(ctx: ToolContext, tree: Path, backend: str) -> None:
    out = await glob(ctx, f"{tree}/src/**/*.py")
    assert listed(out) == ["src/pkg/mod.py", "src/main.py"]
    assert listed(await glob(ctx, f"{tree}/src/main.py")) == ["src/main.py"]
    assert listed(await glob(ctx, f"{tree}/a/")) == ["a/b.py"]


async def test_glob_caps_the_listing_and_says_so(
    ctx: ToolContext, tree: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "GLOB_LIMIT", 3)
    out = await glob(ctx, "*.py")
    assert listed(out) == ["src/pkg/mod.py", "src/main.py", "a/b.py"]
    assert "[Showing the 3 most recently modified of 6 matching files." in out.content
    assert "Narrow the pattern" in out.content


async def test_glob_failures_explain_themselves(ctx: ToolContext, tree: Path, backend: str) -> None:
    missing = await glob(ctx, "*.py", path="nowhere")
    assert missing.is_error and "Cannot search nowhere: it does not exist" in missing.content
    a_file = await glob(ctx, "*.py", path="a.py")
    assert a_file.is_error and "a.py: it is a file, not a directory" in a_file.content
    empty = await glob(ctx, "  ")
    assert empty.is_error and "pattern names no files" in empty.content
    nothing = await glob(ctx, "*.rs", path="src")
    assert not nothing.is_error and nothing.content == "No files match '*.rs' in src."


@needs_ripgrep
async def test_glob_and_grep_respect_gitignore_with_ripgrep(ctx: ToolContext, tree: Path) -> None:
    (tree / ".gitignore").write_text("notes/\n*.tsx\n")
    assert sorted(listed(await glob(ctx, "src/*"))) == ["src/app.test.ts", "src/main.py"]
    assert listed(await glob(ctx, "*.txt")) == []
    found = listed(await grep(ctx, "foo"))
    assert "notes/multi.txt" not in found and "src/app.tsx" not in found and "a.py" in found


@needs_ripgrep
async def test_an_invalid_glob_is_reported(ctx: ToolContext, tree: Path) -> None:
    out = await glob(ctx, "src/[")
    assert out.is_error and out.content.startswith("Invalid glob: ")
    out = await grep(ctx, "foo", glob="{a,b")
    assert out.is_error and out.content.startswith("Invalid glob: ")


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("*.py", "a.py", True),
        ("*.py", "deep/down/a.py", True),
        ("*.py", "a.pyc", False),
        ("src/*.py", "src/a.py", True),
        ("src/*.py", "src/pkg/a.py", False),
        ("src/*.py", "lib/src/a.py", False),
        ("/a.py", "a.py", True),
        ("/a.py", "src/a.py", False),
        ("src/**", "src/pkg/a.py", True),
        ("src/**/a.py", "src/a.py", True),
        ("src/**/a.py", "src/x/y/a.py", True),
        ("**/test_*.py", "tests/unit/test_api.py", True),
        ("**/test_*.py", "test_api.py", True),
        ("*.{js,ts}", "web/app.ts", True),
        ("*.{js,ts}", "web/app.tsx", False),
        ("{src,lib}/**/*.{c,h}", "lib/core/x.h", True),
        ("{src,lib}/**/*.{c,h}", "test/core/x.h", False),
        ("file?.txt", "file1.txt", True),
        ("file?.txt", "file10.txt", False),
        ("[a-c]*.md", "b.md", True),
        ("[!a-c]*.md", "b.md", False),
        ("[!a-c]*.md", "z.md", True),
        ("a[.md", "a[.md", True),  # an unclosed bracket is an ordinary character
        ("{a}.md", "{a}.md", True),  # and so are braces without alternatives
        ("!*.py", "a.py", False),
        ("!*.py", "a.md", True),
        ("Makefile", "sub/Makefile", True),
        ("a+b (1).txt", "a+b (1).txt", True),
    ],
)
def test_glob_matcher(pattern: str, path: str, expected: bool) -> None:
    assert glob_matcher(pattern)(path) is expected


# -- glob permission --------------------------------------------------------------------------


def test_searching_inside_the_workspace_needs_no_permission(ctx: ToolContext, tree: Path) -> None:
    assert GlobTool().permission({"pattern": "**/*.py"}, ctx) is None
    assert GlobTool().permission({"pattern": "*.py", "path": "src"}, ctx) is None
    assert GlobTool().permission({"pattern": f"{tree}/src/*.py"}, ctx) is None
    assert GrepTool().permission({"pattern": "foo"}, ctx) is None
    assert GrepTool().permission({"pattern": "foo", "path": "src/main.py"}, ctx) is None


def test_searching_outside_the_workspace_asks_for_that_directory(
    ctx: ToolContext, tree: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    for ask in (
        GlobTool().permission({"pattern": "*.py", "path": str(outside)}, ctx),
        GlobTool().permission({"pattern": f"{outside}/**/*.py"}, ctx),
        GlobTool().permission({"pattern": "*.py", "path": "shortcut"}, ctx),  # a symlink out
        GrepTool().permission({"pattern": "foo", "path": "../elsewhere"}, ctx),
    ):
        assert ask is not None
        assert (ask.category, ask.target) == ("read", f"{outside}/")
        assert ask.session_rule == f"{outside}/*"
        # one answer covers the directory itself and every file read from it afterwards
        assert fnmatch.fnmatchcase(ask.target, ask.session_rule)
        assert fnmatch.fnmatchcase(f"{outside}/secret.py", ask.session_rule)
    one_file = GrepTool().permission({"pattern": "foo", "path": str(outside / "secret.py")}, ctx)
    assert one_file is not None and one_file.target == f"{outside}/secret.py"
    assert one_file.summary.startswith("Search ")


def test_grep_of_a_secret_file_asks_like_reading_it(ctx: ToolContext, tree: Path) -> None:
    ask = GrepTool().permission({"pattern": "TOKEN", "path": ".env"}, ctx)
    assert ask is not None and (ask.category, ask.target) == ("read", ".env")


# --------------------------------------------------------------------------------------------
# grep
# --------------------------------------------------------------------------------------------


async def test_grep_lists_files_with_matches_newest_first(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    out = await grep(ctx, "foo")
    assert listed(out) == [
        "notes/win.txt",
        "notes/multi.txt",
        ".github/workflows/ci.yml",
        "src/app.test.ts",
        "src/app.tsx",
        "src/pkg/mod.py",
        "src/main.py",
        "a/b.py",
        "a.py",
        "README.md",
    ]


async def test_grep_content_mode_prints_path_line_and_text(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    out = await grep(ctx, r"foo\(", output_mode="content")
    assert out.content == "a/b.py:1:def foo():\nsrc/main.py:5:    foo()\nsrc/main.py:7:    foo()"


async def test_grep_context_lines_and_group_separators(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    out = await grep(ctx, "foo", path="notes/multi.txt", output_mode="content", context=1)
    assert out.content.split("\n") == [
        "notes/multi.txt-1-one",
        "notes/multi.txt:2:two foo",
        "notes/multi.txt-3-three",
        "--",
        "notes/multi.txt-5-five",
        "notes/multi.txt:6:six foo",
        "notes/multi.txt-7-seven",
    ]
    overlapping = await grep(ctx, "foo", path="src/main.py", output_mode="content", context=1)
    assert overlapping.content.split("\n") == [
        "src/main.py-4-def main():",
        "src/main.py:5:    foo()",
        "src/main.py-6-    bar()",
        "src/main.py:7:    foo()",
    ]


async def test_grep_count_mode(ctx: ToolContext, tree: Path, backend: str) -> None:
    out = await grep(ctx, "foo", path="src", output_mode="count")
    assert out.content.split("\n") == [
        "src/app.test.ts:1",
        "src/app.tsx:1",
        "src/main.py:2",
        "src/pkg/mod.py:1",
        "[5 matching line(s) in 4 file(s).]",
    ]


async def test_grep_options_narrow_the_search(ctx: ToolContext, tree: Path, backend: str) -> None:
    assert "A.py" not in listed(await grep(ctx, "foo"))
    assert "A.py" in listed(await grep(ctx, "foo", case_insensitive=True))
    assert sorted(listed(await grep(ctx, "foo", glob="*.py"))) == [
        "a.py",
        "a/b.py",
        "src/main.py",
        "src/pkg/mod.py",
    ]
    assert sorted(listed(await grep(ctx, "foo", glob="src/*.{ts,tsx}"))) == [
        "src/app.test.ts",
        "src/app.tsx",
    ]
    assert listed(await grep(ctx, "foo", path="src/pkg")) == ["src/pkg/mod.py"]
    assert listed(await grep(ctx, "foo", path="a.py")) == ["a.py"]
    # a glob does not hide the one file that path names
    assert listed(await grep(ctx, "foo", path="a.py", glob="*.md")) == ["a.py"]
    assert listed(await grep(ctx, "^foo = ", glob="!*.py")) == []


async def test_grep_regex_features_work_on_both_backends(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    out = await grep(ctx, r"^(import|from)\s+\w+", output_mode="content")
    assert out.content == "a.py:1:import os\nsrc/main.py:1:from pkg import mod"
    # line ends are line ends, whatever the file's line-ending style
    crlf = await grep(ctx, "^foo$", path="notes", output_mode="content")
    assert crlf.content == "notes/win.txt:2:foo"


async def test_grep_skips_what_it_should(ctx: ToolContext, tree: Path, backend: str) -> None:
    found = listed(await grep(ctx, "foo", case_insensitive=True))
    assert set(found) <= set(VISIBLE)  # no skipped directory, no binary file, no .env
    assert "alias.py" not in found and not any(name.startswith("shortcut") for name in found)
    assert listed(await grep(ctx, "outside")) == []


async def test_grep_shows_a_secret_file_only_when_asked_for_it_by_name(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    quiet = await grep(ctx, "FOO_TOKEN", output_mode="content")
    assert quiet.content == "No matches for 'FOO_TOKEN' in the workspace."
    counted = await grep(ctx, "FOO_TOKEN", output_mode="count")
    assert counted.content == "No matches for 'FOO_TOKEN' in the workspace."
    direct = await grep(ctx, "FOO_TOKEN", path=".env", output_mode="content")
    assert direct.content == ".env:1:FOO_TOKEN=foo"


async def test_grep_never_prints_from_clankits_own_config(
    ctx: ToolContext, tree: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLANKIT_CONFIG_DIR", str(tree / "conf"))
    (tree / "conf").mkdir()
    (tree / "conf/config.kdl").write_text('api-key "sk-foo"\n')
    (tree / "conf/other.kdl").write_text('theme "foo"\n')
    out = await grep(ctx, "foo", path="conf", output_mode="content")
    assert out.content == 'conf/other.kdl:1:theme "foo"'


async def test_grep_head_limit_in_every_mode(ctx: ToolContext, tree: Path, backend: str) -> None:
    content = await grep(ctx, "foo", output_mode="content", head_limit=3)
    assert listed(content) == [
        ".github/workflows/ci.yml:1:run: foo",
        "README.md:2:foo is documented here",
        "a/b.py:1:def foo():",
    ]
    assert "[There are more matches than shown." in content.content
    assert "head_limit" in content.content

    files = await grep(ctx, "foo", head_limit=2)
    assert listed(files) == ["notes/win.txt", "notes/multi.txt"]
    assert "[Showing 2 of 10 files with matches" in files.content

    counts = await grep(ctx, "foo", output_mode="count", head_limit=2)
    assert listed(counts) == [".github/workflows/ci.yml:1", "README.md:1"]
    assert "[12 matching line(s) in 10 file(s); showing the first 2 files." in counts.content

    exact = await grep(ctx, "foo", path="src/main.py", output_mode="content", head_limit=2)
    assert "more matches" not in exact.content  # two matches, limit two: nothing was left out


async def test_grep_default_and_maximum_head_limit(
    ctx: ToolContext, tree: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "DEFAULT_HEAD_LIMIT", 4)
    monkeypatch.setattr(search, "MAX_HEAD_LIMIT", 5)
    assert len(listed(await grep(ctx, "foo"))) == 4
    assert len(listed(await grep(ctx, "foo", head_limit=0))) == 4
    assert len(listed(await grep(ctx, "foo", head_limit=1000))) == 5


async def test_grep_clips_long_lines_and_stays_within_the_output_budget(
    ctx: ToolContext, tree: Path, backend: str
) -> None:
    (tree / "min.js").write_text("x" * 5000 + " needle\n")
    long_line = await grep(ctx, "needle", output_mode="content")
    assert long_line.content.startswith("min.js:1:xxx")
    assert long_line.content.endswith("[... 4,607 more characters]")

    ctx.config = Config()
    ctx.config.agent.tool_output_limit = 2000
    (tree / "many.txt").write_text("".join(f"needle number {n:04d} {'pad' * 20}\n" for n in range(300)))
    out = await grep(ctx, "needle number", output_mode="content", head_limit=1000)
    assert len(out.content) <= 2000
    assert 5 < len(listed(out)) < 25 and "[There are more matches than shown." in out.content


async def test_grep_reports_an_invalid_regex(ctx: ToolContext, tree: Path, backend: str) -> None:
    for mode in ("files_with_matches", "content", "count"):
        out = await grep(ctx, "foo(", output_mode=mode)
        assert out.is_error
        assert out.content.startswith("Invalid regular expression 'foo(': ")
        assert "backslash" in out.content


async def test_grep_no_matches_and_missing_path(ctx: ToolContext, tree: Path, backend: str) -> None:
    nothing = await grep(ctx, "zebra", path="src", glob="*.py")
    assert not nothing.is_error
    assert nothing.content == "No matches for 'zebra' in src (files matching *.py)."
    assert (await grep(ctx, "zebra", output_mode="content")).content == (
        "No matches for 'zebra' in the workspace."
    )
    missing = await grep(ctx, "foo", path="nowhere")
    assert missing.is_error and "Cannot search nowhere: it does not exist" in missing.content


# --------------------------------------------------------------------------------------------
# The two implementations agree
# --------------------------------------------------------------------------------------------

SAME_ANSWER_CASES: list[tuple[type, dict[str, Any]]] = [
    (GlobTool, {"pattern": "**"}),
    (GlobTool, {"pattern": "*.py"}),
    (GlobTool, {"pattern": "src/**/*.{py,ts,tsx}"}),
    (GlobTool, {"pattern": "!*.py"}),
    (GlobTool, {"pattern": "*", "path": "notes"}),
    (GlobTool, {"pattern": "*.rs"}),
    (GrepTool, {"pattern": "foo"}),
    (GrepTool, {"pattern": "FOO", "case_insensitive": True, "output_mode": "count"}),
    (GrepTool, {"pattern": "foo", "output_mode": "content", "context": 2}),
    (GrepTool, {"pattern": "foo", "output_mode": "content", "head_limit": 4}),
    (GrepTool, {"pattern": r"\bfoo\b", "glob": "*.py", "output_mode": "content"}),
    (GrepTool, {"pattern": "o$", "path": "notes", "output_mode": "content", "context": 1}),
    (GrepTool, {"pattern": "", "path": "a.py", "output_mode": "content"}),
    (GrepTool, {"pattern": "foo", "path": ".env", "output_mode": "content"}),
    (GrepTool, {"pattern": "zebra"}),
]


@needs_ripgrep
@pytest.mark.parametrize(("tool", "tool_input"), SAME_ANSWER_CASES, ids=repr)
async def test_ripgrep_and_the_fallback_give_the_same_answer(
    ctx: ToolContext, tree: Path, monkeypatch: pytest.MonkeyPatch, tool: type, tool_input: dict
) -> None:
    with_ripgrep = await tool().run(tool_input, ctx)
    monkeypatch.setattr(search, "ripgrep", lambda: None)
    without = await tool().run(tool_input, ctx)
    assert (without.content, without.is_error) == (with_ripgrep.content, with_ripgrep.is_error)
    assert not with_ripgrep.is_error


# --------------------------------------------------------------------------------------------
# Runaway searches
# --------------------------------------------------------------------------------------------


def fake_ripgrep(directory: Path, script: str) -> str:
    """An executable standing in for rg, to provoke what a real one rarely does."""
    path = directory / "fake-rg"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return str(path)


def process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:  # a zombie waiting to be reaped is as good as gone
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


async def wait_until_gone(pid: int) -> bool:
    for _ in range(100):
        if process_is_gone(pid):
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.parametrize("mode", ["files_with_matches", "content"])
async def test_a_ripgrep_that_hangs_is_stopped_and_killed(
    ctx: ToolContext, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    pid_file = tmp_path / "rg.pid"
    hanging = fake_ripgrep(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30")
    monkeypatch.setattr(search, "ripgrep", lambda: hanging)
    monkeypatch.setattr(search, "SEARCH_TIMEOUT", 0.3)
    out = await grep(ctx, "foo", output_mode=mode)
    assert out.is_error and "The search was stopped after" in out.content
    assert "Narrow it" in out.content
    assert await wait_until_gone(int(pid_file.read_text()))


async def test_a_cancelled_search_kills_ripgrep(
    ctx: ToolContext, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "rg.pid"
    hanging = fake_ripgrep(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30")
    monkeypatch.setattr(search, "ripgrep", lambda: hanging)
    task = asyncio.create_task(grep(ctx, "foo", output_mode="content"))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if pid_file.exists() and pid_file.read_text().strip():
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await wait_until_gone(int(pid_file.read_text()))


async def test_ripgrep_is_stopped_once_enough_lines_are_in_hand(
    ctx: ToolContext, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = '{"type":"match","data":{"path":{"text":"a.py"},"lines":{"text":"foo\\\\n"},"line_number":1}}'
    pid_file = tmp_path / "rg.pid"
    endless = fake_ripgrep(
        tmp_path, f"echo $$ > {pid_file}\nwhile true; do echo '{record}'; done"
    )
    monkeypatch.setattr(search, "ripgrep", lambda: endless)
    out = await grep(ctx, "foo", output_mode="content", head_limit=5)
    assert listed(out) == ["a.py:1:foo"] * 5 and "more matches" in out.content
    assert await wait_until_gone(int(pid_file.read_text()))


async def test_other_ripgrep_failures_are_passed_on(
    ctx: ToolContext, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = fake_ripgrep(tmp_path, "echo 'rg: something odd happened' >&2\nexit 2")
    monkeypatch.setattr(search, "ripgrep", lambda: broken)
    for out in (await grep(ctx, "foo"), await grep(ctx, "foo", output_mode="content")):
        assert out.is_error and out.content == "The search failed: rg: something odd happened"
    listing = await glob(ctx, "*.py")
    assert listing.is_error and "something odd happened" in listing.content


async def test_unreadable_ripgrep_records_are_skipped(
    ctx: ToolContext, tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = "\n".join(
        [
            "echo 'not json at all'",
            """echo '{"type":"begin","data":{"path":{"text":"a.py"}}}'""",
            # a file name and a line that are not valid UTF-8 arrive base64-encoded
            """echo '{"type":"match","data":{"path":{"bytes":"YcOpLnB5"},"lines":{"bytes":"Zm9v/wo="},"line_number":3}}'""",
            """echo '{"type":"match","data":{"path":{"text":".env"},"lines":{"text":"foo\\n"},"line_number":1}}'""",
        ]
    )
    monkeypatch.setattr(search, "ripgrep", lambda: fake_ripgrep(tmp_path, script))
    out = await grep(ctx, "foo", output_mode="content")
    assert out.content == "a\N{LATIN SMALL LETTER E WITH ACUTE}.py:3:foo\N{REPLACEMENT CHARACTER}"


async def test_a_fallback_search_that_takes_too_long_is_told_to_stop(
    ctx: ToolContext, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    told_to_stop = threading.Event()

    def endless_counts(search_request: Any, stop: threading.Event) -> dict[str, int]:
        if stop.wait(10):
            told_to_stop.set()
        return {}

    monkeypatch.setattr(search, "ripgrep", lambda: None)
    monkeypatch.setattr(search, "_py_counts", endless_counts)
    monkeypatch.setattr(search, "SEARCH_TIMEOUT", 0.1)
    out = await grep(ctx, "foo")
    assert out.is_error and "The search was stopped after" in out.content
    assert await asyncio.to_thread(told_to_stop.wait, 5)

"""The worktree helper: snapshots, isolated checkouts, patches, cleanup, and the ledger fallback."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from clankit import paths
from clankit.agent.session import Session
from clankit.orchestration import worktree
from clankit.orchestration.worktree import Snapshot, WorktreeError, WorktreeSet
from clankit.tools.base import FileLedger


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture(autouse=True)
def plain_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own git configuration (signing, hooks, identity) out of the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(workspace: Path) -> Path:
    git(workspace, "init", "-q")
    git(workspace, "config", "user.email", "test@example.com")
    git(workspace, "config", "user.name", "Test")
    (workspace / ".gitignore").write_text("*.log\n")
    (workspace / "tracked.txt").write_text("one\n")
    (workspace / "old name.txt").write_text("moving\n")
    (workspace / "doomed.txt").write_text("bye\n")
    (workspace / "image.bin").write_bytes(b"\x89PNG\0\x01\x02")
    git(workspace, "add", "-A")
    git(workspace, "commit", "-qm", "initial")
    return workspace


def state(repo: Path) -> tuple[str, str, str, str]:
    """What the user's repository looks like from git's side: status, HEAD, branches, stash."""
    return (
        git(repo, "status", "--porcelain"),
        git(repo, "rev-parse", "HEAD"),
        git(repo, "branch", "--list"),
        git(repo, "stash", "list"),
    )


async def test_a_snapshot_holds_uncommitted_and_untracked_files_and_touches_nothing(
    repo: Path,
) -> None:
    (repo / "tracked.txt").write_text("one\ntwo\n")
    (repo / "staged.txt").write_text("staged\n")
    git(repo, "add", "staged.txt")
    (repo / "untracked dir").mkdir()
    (repo / "untracked dir" / "new file.txt").write_text("new\n")
    (repo / "debug.log").write_text("ignored\n")
    (repo / "doomed.txt").unlink()
    before = state(repo)

    snapshot = await worktree.take_snapshot(repo)
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("Candidate A")
        copy = tree.workspace
        assert (copy / "tracked.txt").read_text() == "one\ntwo\n"
        assert (copy / "staged.txt").read_text() == "staged\n"
        assert (copy / "untracked dir" / "new file.txt").read_text() == "new\n"
        assert not (copy / "debug.log").exists() and not (copy / "doomed.txt").exists()
        assert tree.root.parent == paths.state_dir() / "worktrees"
        assert tree.root.name.startswith("candidate-a-")
        assert await worktree.diff(tree) == ""

    assert state(repo) == before
    assert snapshot.prefix == "" and snapshot.repo == repo


async def test_changes_travel_from_a_worktree_to_the_workspace_and_undo_reverts_them(
    repo: Path,
) -> None:
    snapshot = await worktree.take_snapshot(repo)
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("a")
        copy = tree.workspace
        (copy / "tracked.txt").write_text("one\nchanged\n")
        (copy / "doomed.txt").unlink()
        (copy / "old name.txt").rename(copy / "new name.txt")
        (copy / "image.bin").write_bytes(b"\x89PNG\0\xff\xfe")
        (copy / "sub dir").mkdir()
        (copy / "sub dir" / "data.bin").write_bytes(bytes(range(256)))
        patch = await worktree.diff(tree)

    assert "GIT binary patch" in patch and "rename from" not in patch
    assert sorted(await worktree.patch_stat(patch, snapshot)) == [
        "doomed.txt  +0 -1",
        "image.bin  (binary)",
        "new name.txt  +1 -0",
        "old name.txt  +0 -1",
        "sub dir/data.bin  (binary)",
        "tracked.txt  +1 -0",
    ]

    ledger = FileLedger()
    ledger.begin_run()
    touched = await worktree.apply_patch(patch, repo, files=ledger, node_id="n7")

    assert {p.relative_to(repo).as_posix() for p in touched} == {
        "tracked.txt",
        "doomed.txt",
        "old name.txt",
        "new name.txt",
        "image.bin",
        "sub dir/data.bin",
    }
    assert (repo / "tracked.txt").read_text() == "one\nchanged\n"
    assert (repo / "new name.txt").read_text() == "moving\n"
    assert not (repo / "old name.txt").exists() and not (repo / "doomed.txt").exists()
    assert (repo / "image.bin").read_bytes() == b"\x89PNG\0\xff\xfe"
    assert (repo / "sub dir" / "data.bin").read_bytes() == bytes(range(256))
    assert set(ledger.changed_by("n7")) == set(touched)
    assert "M tracked.txt" in git(repo, "status", "--porcelain")  # the index is untouched

    ledger.undo()
    assert (repo / "tracked.txt").read_text() == "one\n"
    assert (repo / "old name.txt").read_text() == "moving\n"
    assert (repo / "doomed.txt").read_text() == "bye\n"
    assert (repo / "image.bin").read_bytes() == b"\x89PNG\0\x01\x02"
    assert not (repo / "new name.txt").exists() and not (repo / "sub dir" / "data.bin").exists()


async def test_a_patch_that_no_longer_applies_changes_nothing(repo: Path) -> None:
    snapshot = await worktree.take_snapshot(repo)
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("a")
        (tree.workspace / "tracked.txt").write_text("from the candidate\n")
        (tree.workspace / "brand new.txt").write_text("new\n")
        patch = await worktree.diff(tree)
    (repo / "tracked.txt").write_text("edited by the user meanwhile\n")

    with pytest.raises(WorktreeError, match="git apply failed"):
        await worktree.apply_patch(patch, repo, files=FileLedger(), node_id="n1")
    assert (repo / "tracked.txt").read_text() == "edited by the user meanwhile\n"
    assert not (repo / "brand new.txt").exists()
    with pytest.raises(WorktreeError):
        await worktree.apply_patch("this is not a patch\n", repo)


async def test_a_workspace_below_the_repository_root(repo: Path) -> None:
    (repo / "pkg").mkdir()
    (repo / "pkg" / "module.py").write_text("x = 1\n")
    snapshot = await worktree.take_snapshot(repo / "pkg")
    assert snapshot.prefix == "pkg"
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("a")
        assert tree.workspace == tree.root / "pkg"
        (tree.workspace / "module.py").write_text("x = 2\n")
        (tree.root / "tracked.txt").write_text("outside the workspace\n")
        patch = await worktree.diff(tree)
    assert "pkg/module.py" in patch and "tracked.txt" not in patch
    assert await worktree.patch_stat(patch, snapshot) == ["module.py  +1 -1"]
    await worktree.apply_patch(patch, snapshot.repo)
    assert (repo / "pkg" / "module.py").read_text() == "x = 2\n"


async def test_a_repository_without_commits_and_without_an_identity(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EMAIL", raising=False)
    git(workspace, "init", "-q")
    (workspace / "first.txt").write_text("hello\n")
    snapshot = await worktree.take_snapshot(workspace)
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("a")
        assert (tree.workspace / "first.txt").read_text() == "hello\n"


async def test_outside_a_repository_the_helper_refuses(workspace: Path) -> None:
    assert await worktree.repository_root(workspace) is None
    assert await worktree.repository_root(workspace / "missing") is None
    with pytest.raises(WorktreeError, match="is not inside a git repository"):
        await worktree.take_snapshot(workspace)


async def test_inherited_git_variables_and_hooks_have_no_effect(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "hook-ran"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "other-index"))
    snapshot = await worktree.take_snapshot(repo)
    async with WorktreeSet(snapshot) as trees:
        await trees.add("a")
    assert not marker.exists()
    assert not (tmp_path / "other-index").exists()


async def test_worktrees_are_removed_after_errors_and_cancellation(repo: Path) -> None:
    snapshot = await worktree.take_snapshot(repo)
    directory = paths.state_dir() / "worktrees"

    with pytest.raises(RuntimeError):
        async with WorktreeSet(snapshot) as trees:
            await trees.add("a")
            await trees.add("b")
            raise RuntimeError("boom")
    assert list(directory.iterdir()) == []

    created = asyncio.Event()

    async def work() -> None:
        async with WorktreeSet(snapshot) as trees:
            await trees.add("c")
            created.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(work())
    await created.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(directory.iterdir()) == []
    assert len(git(repo, "worktree", "list").splitlines()) == 1


async def test_run_to_completion_survives_repeated_cancellation() -> None:
    steps: list[str] = []

    async def cleanup() -> str:
        for name in ("one", "two", "three"):
            await asyncio.sleep(0.01)
            steps.append(name)
        return "clean"

    async def caller() -> str:
        return await worktree.run_to_completion(cleanup())

    task = asyncio.create_task(caller())
    for _ in range(2):
        await asyncio.sleep(0.005)
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert steps == ["one", "two", "three"]
    assert await worktree.run_to_completion(cleanup()) == "clean"


def test_change_keys_ignore_spacing_but_not_content() -> None:
    def patch(line: str, context: str = " context") -> str:
        return (
            "diff --git a/f.py b/f.py\nindex 1111111..2222222 100644\n--- a/f.py\n+++ b/f.py\n"
            f"@@ -1,2 +1,2 @@\n{context}\n-    old()\n+{line}\n"
        )

    key = worktree.change_key(patch("    new(a, b)"))
    assert worktree.change_key(patch("    new(a,  b)   ")) == key
    assert worktree.change_key(patch("    new(a, b)").replace("1111111", "3333333")) == key
    assert worktree.change_key(patch("    new(a, c)")) != key
    assert worktree.change_key(patch("new(a, b)")) != key  # indentation can carry meaning
    assert worktree.change_key(patch("    new(a, b)", " other")) != key
    binary = "diff --git a/i.bin b/i.bin\nGIT binary patch\nliteral 3\nKcmZQzWC8#H2LJ>B\n\n"
    assert worktree.change_key(binary) != worktree.change_key(binary.replace("Kcm", "Lcm"))


async def test_agents_in_a_worktree_see_it_as_their_working_directory(
    session: Session, repo: Path
) -> None:
    snapshot = await worktree.take_snapshot(repo)
    async with WorktreeSet(snapshot) as trees:
        tree = await trees.add("a")
        node = session.new_node("Candidate A", role="candidate")
        ctx = worktree.tool_context(session, node, tree)
        assert ctx.workspace == tree.workspace and ctx.node_id == node.id
        assert ctx.files is tree.files and ctx.files is not session.files
        assert ctx.permissions is session.permissions and ctx.config is session.config
        assert ctx.write_scope is None

        prompt = worktree.system_prompt(session, tree, "# Your role\n\nBuild it.")
        assert f"- Working directory: {tree.workspace}\n" in prompt
        assert f"- Working directory: {repo}\n" not in prompt
        assert prompt.endswith("# Your role\n\nBuild it.")
        assert "detached HEAD" in prompt

        result = await worktree.run_check_in(session, ctx, tree, "pwd && test -f tracked.txt")
        assert result.passed and result.output.strip() == str(tree.workspace)


def test_without_git_files_are_restored_from_the_run_checkpoints(workspace: Path) -> None:
    ledger = FileLedger()
    ledger.begin_run()
    existing, created, untracked = (
        workspace / "existing.txt",
        workspace / "new" / "created.txt",
        workspace / "untracked.txt",
    )
    existing.write_text("original\n")
    existing.chmod(0o750)
    for path, content in ((existing, "changed\n"), (created, "fresh\n")):
        ledger.checkpoint(path)
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
        ledger.note_write("n1", path)
    untracked.write_text("made by a shell command\n")

    captured = worktree.capture_files([existing, created])
    restored, missing = worktree.restore_files(ledger, [existing, created, untracked])

    assert restored == [existing, created] and missing == [untracked]
    assert existing.read_text() == "original\n" and not created.exists()
    assert existing.stat().st_mode & 0o777 == 0o750

    worktree.write_files(ledger, captured, "n2")
    assert existing.read_text() == "changed\n" and created.read_text() == "fresh\n"
    assert ledger.changed_by("n2") == [existing, created]
    ledger.undo()
    assert existing.read_text() == "original\n" and not created.exists()
    assert worktree.capture_files([created]) == {created: None}


def test_a_snapshot_is_a_plain_value() -> None:
    snapshot = Snapshot(Path("/repo"), "abc123", "pkg")
    tree = worktree.Worktree(Path("/copies/a"), snapshot)
    assert tree.workspace == Path("/copies/a/pkg")
    assert worktree.Worktree(Path("/copies/b"), Snapshot(Path("/repo"), "abc")).workspace == Path(
        "/copies/b"
    )

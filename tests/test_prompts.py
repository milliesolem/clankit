"""System prompt assembly and instruction-file discovery."""

from __future__ import annotations

from pathlib import Path

from clankit import paths
from clankit.agent import prompts
from clankit.agent.session import Session


def test_instruction_files_from_repo_root_down_with_agents_preferred(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    work = repo / "pkg" / "sub"
    work.mkdir(parents=True)
    (repo / "AGENTS.md").write_text("root rules")
    (repo / "CLAUDE.md").write_text("ignored because AGENTS.md exists")
    (repo / "pkg" / "CLAUDE.md").write_text("pkg rules")
    (work / "AGENTS.md").write_text("sub rules")
    (tmp_path / "AGENTS.md").write_text("outside the repository")
    paths.config_dir().mkdir(parents=True)
    (paths.config_dir() / "AGENTS.md").write_text("global rules")

    found = prompts.find_instruction_files(work)
    assert [p.read_text() for p in found] == [
        "global rules",
        "root rules",
        "pkg rules",
        "sub rules",
    ]

    section = prompts.instructions_section(work)
    assert section.index("global rules") < section.index("root rules") < section.index("sub rules")
    assert "outside the repository" not in section and "ignored because" not in section


def test_no_repository_means_only_the_workspace_applies(tmp_path: Path) -> None:
    work = tmp_path / "a" / "b"
    work.mkdir(parents=True)
    (tmp_path / "a" / "AGENTS.md").write_text("parent")
    (work / "CLAUDE.md").write_text("mine")
    assert [p.read_text() for p in prompts.find_instruction_files(work)] == ["mine"]
    assert prompts.instructions_section(tmp_path / "a" / "b" / "..") != ""


def test_oversized_instructions_are_truncated(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * (prompts.INSTRUCTION_LIMIT + 500))
    section = prompts.instructions_section(tmp_path)
    assert "truncated" in section and len(section) < prompts.INSTRUCTION_LIMIT + 1000


def test_system_prompt_shares_a_stable_base(session: Session, workspace: Path) -> None:
    (workspace / "AGENTS.md").write_text("Use tabs.")
    prompts.forget(session)
    one = prompts.system_prompt(session, "# Your role\nA")
    (workspace / "AGENTS.md").write_text("Changed mid-session.")
    two = prompts.system_prompt(session, "# Your role\nB")
    base = prompts.system_prompt(session)
    assert one.startswith(base) and two.startswith(base)  # identical prefix for every agent
    assert "Use tabs." in base and "Changed mid-session." not in base
    assert one.endswith("A") and two.endswith("B")
    assert str(workspace) in base and "not a git repository" in base
    prompts.forget(session)
    assert "Changed mid-session." in prompts.system_prompt(session)


def test_skill_catalog_appears_only_when_skills_exist(session: Session) -> None:
    assert "# Skills" not in prompts.base_prompt(session)
    skill = paths.skills_dir() / "release-notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: release-notes\ndescription: Draft release notes.\n---\nBody\n"
    )
    session.runtime.skills.refresh()
    assert "release-notes: Draft release notes." in prompts.base_prompt(session)


def test_new_skills_reach_new_agents_but_not_existing_ones(session: Session) -> None:
    before = prompts.system_prompt(session, "role")
    skill = paths.skills_dir() / "deploy-steps"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: deploy-steps\ndescription: How to deploy.\n---\nBody\n"
    )
    session.runtime.skills.refresh()
    after = prompts.system_prompt(session, "role")
    assert "deploy-steps" not in before and "deploy-steps: How to deploy." in after
    assert prompts.system_prompt(session, "other").startswith(after[: -len("role")])

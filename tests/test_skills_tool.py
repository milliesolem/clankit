"""The `skill` tool: loading instructions, refusing what agents may not load, useful errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from clankit import paths
from clankit.agent.events import Event, ToolFinished, ToolStarted
from clankit.agent.session import Session
from clankit.llm.scripted import ScriptedProvider, call
from clankit.skills import document
from clankit.skills.library import SkillLibrary
from clankit.skills.tool import BODY_LIMIT, FILE_LIMIT, SkillTool
from clankit.tools.base import AllowAll, FileLedger, ToolContext, ToolOutput


def write_skill(
    root: Path, name: str, frontmatter: str = "", body: str = "Do the thing.\n"
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    extra = f"{frontmatter}\n" if frontmatter else ""
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: About {name}.\n{extra}---\n\n{body}", encoding="utf-8"
    )
    return folder


@pytest.fixture
def user(tmp_path: Path) -> Path:
    return tmp_path / "skills"


@pytest.fixture
def library(user: Path) -> SkillLibrary:
    return SkillLibrary([user])


def context(workspace: Path, skills: SkillLibrary | None) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        node_id="n1",
        files=FileLedger(),
        permissions=AllowAll(),
        emit=lambda event: event,
        skills=skills,
    )


async def load(name: str, workspace: Path, skills: SkillLibrary | None) -> ToolOutput:
    return await SkillTool().run({"name": name}, context(workspace, skills))


def test_tool_contract_is_static() -> None:
    tool = SkillTool()  # no constructor arguments
    spec = tool.spec()
    assert (spec.name, tool.read_only) == ("skill", True)
    assert spec.input_schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The skill's name, exactly as listed."}
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert tool.permission({"name": "x"}, None) is None  # type: ignore[arg-type]
    assert tool.summarize({"name": "release-notes"}) == "skill release-notes"


async def test_schema_does_not_change_when_skills_do(library: SkillLibrary, user: Path) -> None:
    before = SkillTool().spec()
    write_skill(user, "added-later")
    library.refresh()
    assert SkillTool().spec() == before


async def test_returns_body_directory_and_bundled_files(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    folder = write_skill(user, "release-notes", body="# Release notes\n\nRun scripts/collect.py.\n")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "collect.py").write_text("print('x')\n")
    (folder / "references").mkdir()
    (folder / "references" / "style.md").write_text("Be brief.\n")

    output = await load("release-notes", workspace, library)
    assert not output.is_error and output.finish is None
    assert output.content == (
        '<skill_content name="release-notes">\n'
        "# Release notes\n\nRun scripts/collect.py.\n"
        "</skill_content>\n"
        "\n"
        f"Skill directory: {folder}\n"
        "Paths in this skill are relative to that directory. Use absolute paths when you "
        "read its files or run its scripts.\n"
        "Bundled files:\n"
        "- references/style.md\n"
        "- scripts/collect.py"
    )
    assert "description:" not in output.content and Path(str(folder)).is_absolute()


async def test_skill_without_bundled_files_lists_none(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    write_skill(user, "bare")
    output = await load("  bare ", workspace, library)  # stray whitespace is forgiven
    assert not output.is_error and "Bundled files" not in output.content


async def test_file_listing_and_long_instructions_are_capped(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    folder = write_skill(user, "big", body="step\n" * (BODY_LIMIT // 5 + 500))
    for index in range(FILE_LIMIT + 7):
        (folder / f"asset-{index:02}.txt").write_text("x")

    output = await load("big", workspace, library)
    listed = [line for line in output.content.splitlines() if line.startswith("- asset-")]
    assert len(listed) == FILE_LIMIT and listed[0] == "- asset-00.txt"
    assert "(7 more files not listed)" in output.content
    assert output.content.count("step\n") <= BODY_LIMIT // 5
    assert "continue for another 2,5" in output.content
    assert f"Read the rest of {folder / 'SKILL.md'}" in output.content


async def test_skill_directory_variables_are_resolved(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    folder = write_skill(user, "portable", body="Run ${CLAUDE_SKILL_DIR}/scripts/a.sh\n")
    output = await load("portable", workspace, library)
    assert f"Run {folder}/scripts/a.sh" in output.content


async def test_unknown_name_lists_available_skills_closest_first(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    for name in ("alpha", "release-notes", "zebra"):
        write_skill(user, name)
    write_skill(user, "release-private", "disable-model-invocation: true")

    output = await load("relese-notes", workspace, library)
    assert output.is_error
    assert output.content == (
        "There is no skill named 'relese-notes'. Available skills, closest match first: "
        "release-notes, alpha, zebra."
    )


async def test_unknown_name_with_an_empty_library(library: SkillLibrary, workspace: Path) -> None:
    output = await load("anything", workspace, library)
    assert output.is_error and "no skills are installed" in output.content


async def test_missing_library_is_a_clear_error(workspace: Path) -> None:
    output = await load("anything", workspace, None)
    assert output.is_error and "no skill library is attached" in output.content


async def test_user_only_skill_is_refused(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    write_skill(user, "deploy", "disable-model-invocation: true", body="SECRET STEPS\n")
    output = await load("deploy", workspace, library)
    assert output.is_error and "can only be started by the user" in output.content
    assert "SECRET STEPS" not in output.content


async def test_pending_proposal_is_refused(library: SkillLibrary, workspace: Path) -> None:
    text = document.render("draft", "A draft. Not ready.", "PROPOSED STEPS")
    library.save("draft", {"SKILL.md": text}, pending=True)
    output = await load("draft", workspace, library)
    assert output.is_error and "waiting for the user's approval" in output.content
    assert "PROPOSED STEPS" not in output.content

    library.approve("draft")
    approved = await load("draft", workspace, library)
    assert not approved.is_error and "PROPOSED STEPS" in approved.content


async def test_project_skill_needs_trust(library: SkillLibrary, workspace: Path) -> None:
    write_skill(workspace / ".clankit" / "skills", "build", body="REPO STEPS\n")

    hidden = await load("build", workspace, library)
    assert hidden.is_error and "no skill named 'build'" in hidden.content

    untrusted = await load("build", workspace, library.with_project(workspace))
    assert untrusted.is_error and "has not trusted this project's skills" in untrusted.content
    assert "REPO STEPS" not in untrusted.content

    trusted = await load("build", workspace, library.with_project(workspace, trusted=True))
    assert not trusted.is_error and "REPO STEPS" in trusted.content


async def test_unreadable_skill_is_an_error_not_a_crash(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    folder = write_skill(user, "vanishing")
    assert library.get("vanishing") is not None
    (folder / "SKILL.md").unlink()  # removed after the scan
    output = await load("vanishing", workspace, library)
    assert output.is_error and "could not be read" in output.content


async def test_agent_loads_a_skill_through_the_session(
    session: Session, scripted: ScriptedProvider, events: list[Event]
) -> None:
    write_skill(paths.skills_dir(), "release-notes", body="Collect merged pull requests.\n")
    session.runtime.skills.refresh()
    node = session.new_node("test", role="root")
    agent = session.agent(node, system="test", tools=[SkillTool()])
    scripted.push(call("skill", name="release-notes"), call("skill"), "done")

    result = await agent.run("write the release notes")
    assert result.text == "done"
    (loaded,) = agent.messages[2].tool_result_blocks
    assert not loaded.is_error and "Collect merged pull requests." in loaded.content
    assert str(paths.skills_dir() / "release-notes") in loaded.content
    (invalid,) = agent.messages[4].tool_result_blocks  # the missing name never reaches the tool
    assert invalid.is_error and "Invalid input for skill" in invalid.content
    assert next(e.summary for e in events if isinstance(e, ToolStarted)) == "skill release-notes"
    assert [e.is_error for e in events if isinstance(e, ToolFinished)] == [False, True]

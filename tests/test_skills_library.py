"""The skill library: lenient loading, the catalog, project skills, proposals, strict writing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clankit.skills import document
from clankit.skills.library import PENDING_DIR, Skill, SkillError, SkillLibrary

GOOD = "---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nDo the thing.\n"


def write_skill(
    root: Path, name: str, *, description: str = "Does a thing.", text: str | bytes | None = None
) -> Path:
    """Create the skill folder `root/name`; `text` replaces the default SKILL.md."""
    folder = root / name
    folder.mkdir(parents=True)
    content = GOOD.format(name=name, description=description) if text is None else text
    if isinstance(content, bytes):
        (folder / "SKILL.md").write_bytes(content)
    else:
        (folder / "SKILL.md").write_text(content, encoding="utf-8")
    return folder


def rendered(name: str, description: str = "Does a thing. Not for other things.") -> str:
    return document.render(name, description, f"# {name}\n\nDo the thing.")


@pytest.fixture
def user(tmp_path: Path) -> Path:
    return tmp_path / "skills"


@pytest.fixture
def library(user: Path) -> SkillLibrary:
    return SkillLibrary([user])


# -- loading -------------------------------------------------------------------------------------


def test_missing_library_directory_is_an_empty_library(library: SkillLibrary) -> None:
    assert library.skills == [] and library.pending() == [] and library.diagnostics == []
    assert library.catalog() == "" and library.get("anything") is None


def test_indexes_metadata_and_reads_body_and_files_on_demand(
    library: SkillLibrary, user: Path
) -> None:
    folder = write_skill(user, "release-notes", description="Drafts release notes.")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "collect.py").write_text("print('x')\n")
    (folder / "references").mkdir()
    (folder / "references" / "style.md").write_text("Be brief.\n")
    (folder / ".DS_Store").write_text("junk")
    write_skill(user, "alpha")

    assert [s.name for s in library.skills] == ["alpha", "release-notes"]
    skill = library.get("release-notes")
    assert skill is not None
    assert (skill.description, skill.source, skill.pending) == (
        "Drafts release notes.",
        "user",
        False,
    )
    assert skill.path == folder and skill.path.is_absolute()
    assert skill.body() == "# release-notes\n\nDo the thing.\n"
    assert skill.files() == ["references/style.md", "scripts/collect.py"]
    data = skill.to_dict()
    assert data["name"] == "release-notes" and data["path"] == str(folder)
    assert data["model_invocable"] is True and data["generated"] is False
    assert library.diagnostics == []


def test_body_is_read_from_disk_at_call_time(library: SkillLibrary, user: Path) -> None:
    folder = write_skill(user, "live")
    skill = library.get("live")
    assert skill is not None
    (folder / "SKILL.md").write_text("---\nname: live\ndescription: x\n---\nEdited later.\n")
    assert skill.body() == "Edited later.\n"


def test_bom_and_crlf_files_load(library: SkillLibrary, user: Path) -> None:
    raw = "﻿---\r\nname: windows\r\ndescription: Made on Windows.\r\n---\r\nLine one\r\nLine two\r\n"
    write_skill(user, "windows", text=raw.encode("utf-8"))
    skill = library.get("windows")
    assert skill is not None and skill.description == "Made on Windows."
    assert skill.body() == "Line one\nLine two\n"
    assert library.diagnostics == []


def test_prose_that_is_not_valid_yaml_is_read_as_written(library: SkillLibrary, user: Path) -> None:
    write_skill(
        user,
        "prose",
        text=(
            "---\n"
            "name: prose\n"
            "description: Use this skill when: the user works on C# projects [legacy] #42\n"
            "argument-hint: [issue-number] [format]\n"
            "compatibility: Requires: git\n"
            "---\nBody\n"
        ),
    )
    skill = library.get("prose")
    assert skill is not None
    assert skill.description == "Use this skill when: the user works on C# projects [legacy] #42"
    assert skill.argument_hint == "[issue-number] [format]"
    assert library.diagnostics == []


def test_block_scalars_and_quoted_values_are_left_to_yaml(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(
        user,
        "folded",
        text=(
            "---\nname: folded\ndescription: >\n  Folded text\n  on two lines. Use when: needed.\n"
            'when_to_use: "Quoted: fine"\n---\nBody with --- inside\n\n---\n\nmore\n'
        ),
    )
    skill = library.get("folded")
    assert skill is not None
    assert skill.description == "Folded text on two lines. Use when: needed."
    assert skill.when_to_use == "Quoted: fine"
    assert skill.body() == "Body with --- inside\n\n---\n\nmore\n"


def test_scalars_stay_strings(library: SkillLibrary, user: Path) -> None:
    write_skill(
        user,
        "typed",
        text=(
            "---\nname: typed\ndescription: 2024-01-01\n"
            "metadata:\n  version: 1.0\n  enabled: yes\n  nested:\n    a: b\n---\nBody\n"
        ),
    )
    skill = library.get("typed")
    assert skill is not None and skill.description == "2024-01-01"
    assert skill.metadata == {"version": "1.0", "enabled": "yes"}  # nested values are dropped


def test_files_without_usable_frontmatter_are_skipped_and_reported(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(user, "plain", text="# Just markdown\n\nNo frontmatter here.\n")
    write_skill(user, "unclosed", text="---\nname: unclosed\ndescription: never closed\n")
    write_skill(user, "broken", text="---\nname: broken\n  description: [unbalanced\n---\nBody\n")
    write_skill(user, "listy", text="---\n- a\n- b\n---\nBody\n")
    write_skill(user, "fine")

    assert [s.name for s in library.skills] == ["fine"]
    report = "\n".join(library.diagnostics)
    for name in ("plain", "unclosed", "broken", "listy"):
        assert f"{user / name}: skipped" in report
    assert "no frontmatter" in report and "not valid YAML" in report


def test_missing_description_loads_but_is_not_offered_to_agents(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(user, "quiet", text="---\nname: quiet\n---\nBody\n")
    write_skill(user, "empty", text="---\n---\nBody\n")
    write_skill(user, "loud", description="Speaks up.")

    assert [s.name for s in library.skills] == ["empty", "loud", "quiet"]
    assert library.catalog() == "- loud: Speaks up."
    assert sum("no description" in d for d in library.diagnostics) == 2


def test_folder_name_is_the_identity_and_mismatches_warn(library: SkillLibrary, user: Path) -> None:
    write_skill(user, "actual-folder", text="---\nname: Something Else\ndescription: x\n---\nB\n")
    write_skill(user, "Upper_Case", text="---\nname: Upper_Case\ndescription: x\n---\nB\n")
    write_skill(user, "x" * 70, text=f"---\nname: {'x' * 70}\ndescription: x\n---\nB\n")

    assert library.get("Something Else") is None
    skill = library.get("actual-folder")
    assert skill is not None and skill.name == "actual-folder"
    assert "'Something Else'" in skill.warnings[0] and "folder name is used" in skill.warnings[0]
    assert library.get("Upper_Case") is not None and library.get("x" * 70) is not None
    report = "\n".join(library.diagnostics)
    assert "'Upper_Case' is not valid" in report and "70 characters long" in report


def test_long_description_warns_and_is_shortened_only_in_the_catalog(
    library: SkillLibrary, user: Path
) -> None:
    long = "word " * 300
    write_skill(user, "wordy", description=long)
    skill = library.get("wordy")
    assert skill is not None and len(skill.description) == len(long.strip())
    (entry,) = library.catalog().splitlines()
    assert len(entry) < 1100 and entry.endswith("...")
    assert any("1499 characters long" in d for d in library.diagnostics)


def test_shared_extension_fields_are_honoured_and_unknown_ones_ignored(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(
        user,
        "deploy",
        text=(
            "---\nname: deploy\ndescription: Deploys the app.\n"
            "when_to_use: Use when the user says ship it.\n"
            "argument-hint: [environment]\n"
            "disable-model-invocation: TRUE\n"
            "user-invocable: no\n"
            "allowed-tools: Bash(git:*) Read\n"
            "model: opus\nhooks:\n  PreToolUse: []\n---\nBody\n"
        ),
    )
    write_skill(
        user,
        "helper",
        text="---\nname: helper\ndescription: Helps.\nwhen_to_use: When stuck.\ndisable-model-invocation: maybe\n---\nB\n",
    )
    deploy, helper = library.get("deploy"), library.get("helper")
    assert deploy is not None and helper is not None
    assert deploy.disable_model_invocation and not deploy.user_invocable
    assert not deploy.model_invocable and deploy.argument_hint == "[environment]"
    assert helper.model_invocable and helper.user_invocable
    assert helper.warnings == ["disable-model-invocation must be true or false; using false"]
    # only the invocable skill is listed, with when_to_use appended to its description
    assert library.catalog() == "- helper: Helps. When stuck."


def test_lowercase_skill_md_is_accepted(library: SkillLibrary, user: Path) -> None:
    folder = user / "lower"
    folder.mkdir(parents=True)
    (folder / "skill.md").write_text("---\nname: lower\ndescription: Lowercase file.\n---\nBody\n")
    skill = library.get("lower")
    assert skill is not None and skill.body() == "Body\n" and skill.files() == []


def test_large_body_does_not_stop_indexing(library: SkillLibrary, user: Path) -> None:
    body = "line\n" * 40_000  # well past what a scan reads
    write_skill(user, "huge", text=f"---\nname: huge\ndescription: Big.\n---\n{body}")
    skill = library.get("huge")
    assert skill is not None and skill.description == "Big." and skill.body() == body


def test_instructions_fill_in_arguments_and_the_skill_directory(
    library: SkillLibrary, user: Path
) -> None:
    folder = write_skill(
        user,
        "vars",
        text=(
            "---\nname: vars\ndescription: x\n---\n"
            "Run ${CLAUDE_SKILL_DIR}/scripts/a.py and ${CLANKIT_SKILL_DIR}/b.sh on $ARGUMENTS.\n"
        ),
    )
    skill = library.get("vars")
    assert skill is not None
    assert (
        skill.instructions("issue 7")
        == f"Run {folder}/scripts/a.py and {folder}/b.sh on issue 7.\n"
    )
    assert skill.instructions().endswith("on .\n")


def test_yaml_alias_bombs_are_not_expanded(library: SkillLibrary, user: Path) -> None:
    levels = "\n".join(
        f"  l{n}: &l{n} [{', '.join(f'*l{n - 1}' for _ in range(9))}]" for n in range(1, 12)
    )
    bomb = (
        "---\nname: bomb\ndescription: Looks harmless.\n"
        f"metadata:\n  l0: &l0 [x, x, x, x, x, x, x, x, x]\n{levels}\n  author: someone\n"
        "disable-model-invocation: *l11\nuser-invocable: *l11\n---\nBody\n"
    )
    write_skill(user, "bomb", text=bomb)
    skill = library.get("bomb")  # returns promptly: 9**12 list items are never written out
    assert skill is not None and skill.metadata == {"author": "someone"}
    assert skill.model_invocable and len(skill.warnings) == 2
    assert all(len(line) < 300 for line in library.diagnostics)


# -- discovery -----------------------------------------------------------------------------------


def test_first_directory_wins_a_name_clash_and_the_loser_is_reported(tmp_path: Path) -> None:
    user, extra = tmp_path / "user", tmp_path / "extra"
    write_skill(user, "shared", description="From the user.")
    loser = write_skill(extra, "shared", description="From elsewhere.")
    write_skill(extra, "only-extra")
    library = SkillLibrary([user, extra])

    assert [(s.name, s.source) for s in library.skills] == [
        ("only-extra", "extra"),
        ("shared", "user"),
    ]
    assert library.get("shared").description == "From the user."  # type: ignore[union-attr]
    assert library.diagnostics == [
        f"{loser}: shadowed by the skill of the same name in {user / 'shared'}"
    ]


def test_category_folders_are_searched_and_hidden_ones_are_not(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(user / "team" / "docs", "nested")
    write_skill(user, "top")
    write_skill(user / "top", "inside-a-skill")  # a skill folder is not searched further
    write_skill(user / ".git", "hidden")
    write_skill(user / PENDING_DIR, "proposal")
    write_skill(user / "node_modules", "vendored")
    assert [s.name for s in library.skills] == ["nested", "top"]


def test_symlink_loops_do_not_hang_the_scan(library: SkillLibrary, user: Path) -> None:
    write_skill(user / "group", "real")
    (user / "group" / "loop").symlink_to(user, target_is_directory=True)
    assert [s.name for s in library.skills] == ["real"]


def test_refresh_and_set_directories_pick_up_changes(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    write_skill(second, "elsewhere")
    library = SkillLibrary([first])
    assert library.skills == [] and library.directories == [first]
    write_skill(first, "late")
    assert library.skills == []  # scans are cached until told otherwise
    library.refresh()
    assert [s.name for s in library.skills] == ["late"]
    library.set_directories([second])
    assert [s.name for s in library.skills] == ["elsewhere"] and library.directories == [second]


# -- catalog -------------------------------------------------------------------------------------


def test_catalog_is_sorted_compact_and_cut_on_whole_entries(
    library: SkillLibrary, user: Path
) -> None:
    for name in ("charlie", "alpha", "bravo", "delta"):
        write_skill(user, name, description=f"Skill {name}.")
    full = library.catalog()
    assert full.splitlines() == [
        "- alpha: Skill alpha.",
        "- bravo: Skill bravo.",
        "- charlie: Skill charlie.",
        "- delta: Skill delta.",
    ]
    assert library.catalog(len(full)) == full

    cut = library.catalog(len(full) - 1)
    assert cut.splitlines() == [
        "- alpha: Skill alpha.",
        "- bravo: Skill bravo.",
        "(2 more skills not shown)",
    ]
    assert len(cut) <= len(full) - 1
    assert library.catalog(75).splitlines()[-1] == "(2 more skills not shown)"
    assert library.catalog(50) == "- alpha: Skill alpha.\n(3 more skills not shown)"
    assert library.catalog(10) == "(4 skills not shown)"


def test_catalog_names_a_single_hidden_skill_in_the_singular(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(user, "alpha", description="A.")
    write_skill(user, "bravo", description="B." + " long" * 30)
    assert library.catalog(60) == "- alpha: A.\n(1 more skill not shown)"


# -- project skills ------------------------------------------------------------------------------


def test_project_skills_are_listed_but_untrusted_by_default(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    write_skill(user, "mine", description="Mine.")
    write_skill(workspace / ".clankit" / "skills", "build", description="Builds the project.")
    write_skill(workspace / ".agents" / "skills", "lint", description="Lints the project.")

    assert [s.name for s in library.skills] == ["mine"]  # the base library never looks
    view = library.with_project(workspace)
    assert [(s.name, s.source, s.trusted) for s in view.skills] == [
        ("build", "project", False),
        ("lint", "project", False),
        ("mine", "user", True),
    ]
    assert view.catalog() == "- mine: Mine."
    assert not view.get("build").model_invocable  # type: ignore[union-attr]

    trusted = library.with_project(workspace, trusted=True)
    assert trusted.catalog().splitlines() == [
        "- build: Builds the project.",
        "- lint: Lints the project.",
        "- mine: Mine.",
    ]


def test_user_skill_beats_project_skill_and_clankit_beats_agents(
    library: SkillLibrary, user: Path, workspace: Path
) -> None:
    write_skill(user, "deploy", description="The user's.")
    hostile = write_skill(workspace / ".clankit" / "skills", "deploy", description="The repo's.")
    write_skill(workspace / ".clankit" / "skills", "fmt", description="Native.")
    write_skill(workspace / ".agents" / "skills", "fmt", description="Generic.")
    view = library.with_project(workspace, trusted=True)

    assert view.get("deploy").description == "The user's."  # type: ignore[union-attr]
    assert view.get("fmt").description == "Native."  # type: ignore[union-attr]
    assert f"{hostile}: shadowed by the skill of the same name in {user / 'deploy'}" in (
        view.diagnostics
    )


def test_project_view_follows_the_library_it_came_from(
    library: SkillLibrary, user: Path, workspace: Path, tmp_path: Path
) -> None:
    view = library.with_project(workspace)
    assert view.skills == []
    library.save("fresh", {"SKILL.md": rendered("fresh")})
    assert [s.name for s in view.skills] == ["fresh"]  # a write through one is seen by the other
    other = tmp_path / "other"
    write_skill(other, "moved")
    library.set_directories([other])
    assert [s.name for s in view.skills] == ["moved"] and view.directories == [other]


# -- proposals -----------------------------------------------------------------------------------


def test_proposals_are_staged_listed_and_approved(library: SkillLibrary, user: Path) -> None:
    staged = library.save("fix-flaky", {"SKILL.md": rendered("fix-flaky")}, pending=True)
    assert staged.pending and staged.source == "pending"
    assert staged.path == user / PENDING_DIR / "fix-flaky"
    assert [p.name for p in library.pending()] == ["fix-flaky"]
    assert library.skills == [] and library.get("fix-flaky") is None and library.catalog() == ""

    approved = library.approve("fix-flaky")
    assert (approved.pending, approved.source, approved.path) == (False, "user", user / "fix-flaky")
    assert library.pending() == [] and [s.name for s in library.skills] == ["fix-flaky"]
    assert not (user / PENDING_DIR / "fix-flaky").exists()


def test_approving_replaces_the_older_version_whole(library: SkillLibrary, user: Path) -> None:
    old = write_skill(user, "tool", description="Old version.")
    (old / "stale.txt").write_text("left over from the old version")
    library.save(
        "tool",
        {
            "SKILL.md": rendered("tool", "New version. Not the old one."),
            "scripts/run.sh": "#!/bin/sh\n",
        },
        pending=True,
    )
    assert library.get("tool").description == "Old version."  # type: ignore[union-attr]

    approved = library.approve("tool")
    assert approved.description == "New version. Not the old one."
    assert approved.files() == ["scripts/run.sh"]  # nothing of the old folder survives
    assert sorted(p.name for p in user.iterdir()) == [PENDING_DIR, "tool"]
    assert list((user / PENDING_DIR).iterdir()) == []  # no temporary folders left behind


def test_reject_deletes_the_proposal_only(library: SkillLibrary, user: Path) -> None:
    write_skill(user, "keep")
    library.save("keep", {"SKILL.md": rendered("keep")}, pending=True)
    library.reject("keep")
    assert library.pending() == [] and library.get("keep") is not None


@pytest.mark.parametrize("name", ["nope", "", ".", "..", "../keep", "a/b", ".hidden"])
def test_unknown_or_unsafe_proposal_names_are_key_errors(
    library: SkillLibrary, user: Path, name: str
) -> None:
    write_skill(user, "keep")
    write_skill(user / PENDING_DIR / ".hidden", "x")
    with pytest.raises(KeyError):
        library.approve(name)
    with pytest.raises(KeyError):
        library.reject(name)
    assert (user / "keep" / "SKILL.md").exists()


def test_a_proposal_that_is_not_a_skill_cannot_be_approved_but_can_be_rejected(
    library: SkillLibrary, user: Path
) -> None:
    write_skill(user / PENDING_DIR, "junk", text="not a skill")
    assert library.pending() == []
    with pytest.raises(SkillError, match="cannot be approved"):
        library.approve("junk")
    library.reject("junk")
    assert not (user / PENDING_DIR / "junk").exists()


# -- deleting ------------------------------------------------------------------------------------


def test_delete_removes_a_user_skill(library: SkillLibrary, user: Path) -> None:
    write_skill(user, "gone")
    write_skill(user, "stays")
    library.delete("gone")
    assert [s.name for s in library.skills] == ["stays"] and not (user / "gone").exists()
    with pytest.raises(KeyError):
        library.delete("gone")


def test_delete_refuses_anything_outside_the_user_library(tmp_path: Path, workspace: Path) -> None:
    user, extra = tmp_path / "user", tmp_path / "extra"
    write_skill(user, "mine")
    shared = write_skill(extra, "shared")
    project = write_skill(workspace / ".clankit" / "skills", "build")
    library = SkillLibrary([user, extra])

    with pytest.raises(SkillError, match="only skills in the user library"):
        library.delete("shared")
    with pytest.raises(SkillError, match="project skill"):
        library.with_project(workspace, trusted=True).delete("build")
    with pytest.raises(KeyError):
        library.delete("build")  # the base library does not even see project skills
    library.save("waiting", {"SKILL.md": rendered("waiting")}, pending=True)
    with pytest.raises(KeyError):
        library.delete("waiting")  # proposals are rejected, not deleted
    assert shared.exists() and project.exists() and (user / PENDING_DIR / "waiting").exists()


def test_deleting_a_symlinked_skill_removes_only_the_link(
    library: SkillLibrary, user: Path, tmp_path: Path
) -> None:
    real = write_skill(tmp_path / "elsewhere", "linked")
    user.mkdir()
    (user / "linked").symlink_to(real, target_is_directory=True)
    assert library.get("linked") is not None
    library.delete("linked")
    assert not (user / "linked").exists() and (real / "SKILL.md").exists()


# -- strict writing ------------------------------------------------------------------------------


def test_save_writes_a_complete_folder_with_runnable_scripts(
    library: SkillLibrary, user: Path
) -> None:
    skill = library.save(
        "bundle",
        {
            "SKILL.md": rendered("bundle"),
            "scripts/run.py": "print('hi')\n",
            "references/notes.md": "Notes.\n",
        },
    )
    assert isinstance(skill, Skill) and skill.path == user / "bundle" and skill.warnings == []
    assert skill.files() == ["references/notes.md", "scripts/run.py"]
    assert os.access(user / "bundle" / "scripts" / "run.py", os.X_OK)
    assert not os.access(user / "bundle" / "references" / "notes.md", os.X_OK)
    assert library.get("bundle") is not None and library.diagnostics == []


@pytest.mark.parametrize(
    ("name", "files", "message"),
    [
        ("Bad Name", {"SKILL.md": "---\nname: Bad Name\ndescription: x\n---\nB\n"}, "not valid"),
        ("no-file", {"notes.md": "x"}, "needs a SKILL.md"),
        ("no-front", {"SKILL.md": "just text"}, "no frontmatter"),
        (
            "renamed",
            {"SKILL.md": "---\nname: other\ndescription: x\n---\nB\n"},
            "says name: 'other'",
        ),
        ("no-desc", {"SKILL.md": "---\nname: no-desc\n---\nB\n"}, "description is empty"),
        ("no-body", {"SKILL.md": "---\nname: no-body\ndescription: x\n---\n"}, "body is empty"),
        (
            "tagged",
            {"SKILL.md": "---\nname: tagged\ndescription: Use <b>this</b>.\n---\nB\n"},
            "'<' or '>'",
        ),
        (
            "escape",
            {"SKILL.md": "---\nname: escape\ndescription: x\n---\nB\n", "../outside.txt": "x"},
            "not a path inside",
        ),
        (
            "absolute",
            {"SKILL.md": "---\nname: absolute\ndescription: x\n---\nB\n", "/tmp/abs.txt": "x"},
            "not a path inside",
        ),
    ],
)
def test_save_refuses_invalid_skills_and_writes_nothing(
    library: SkillLibrary,
    user: Path,
    tmp_path: Path,
    name: str,
    files: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(SkillError, match=message):
        library.save(name, files)
    assert library.skills == [] and library.pending() == []
    assert not (tmp_path / "outside.txt").exists()
    leftovers = [p for p in user.rglob("*") if p.name != PENDING_DIR] if user.exists() else []
    assert leftovers == []


def test_save_never_overwrites_an_active_skill_unless_told_which(
    library: SkillLibrary, user: Path
) -> None:
    original = write_skill(user, "mine", description="Hand-written.")
    (original / "notes.md").write_text("keep me")
    with pytest.raises(SkillError, match="already exists"):
        library.save("mine", {"SKILL.md": rendered("mine")})
    assert library.get("mine").description == "Hand-written."  # type: ignore[union-attr]

    base = library.get("mine")
    revised = library.save(
        "mine", {"SKILL.md": rendered("mine", "Revised. Not the original.")}, base=base
    )
    assert revised.description == "Revised. Not the original."
    assert revised.files() == ["notes.md"]  # files of the revised version are carried over


def test_save_refuses_a_base_it_may_not_revise(tmp_path: Path) -> None:
    user, extra = tmp_path / "user", tmp_path / "extra"
    write_skill(extra, "shared")
    library = SkillLibrary([user, extra])
    with pytest.raises(SkillError, match="cannot be revised"):
        library.save("shared", {"SKILL.md": rendered("shared")}, base=library.get("shared"))
    with pytest.raises(SkillError, match="no user skill library"):
        SkillLibrary([]).save("orphan", {"SKILL.md": rendered("orphan")})
    assert not user.exists()


def test_revising_a_skill_with_symlinked_files_stays_inside_the_library(
    library: SkillLibrary, user: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: linked\ndescription: Original.\n---\nOriginal body.\n")
    folder = user / "linked"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").symlink_to(outside)
    base = library.get("linked")
    assert base is not None and base.description == "Original."

    library.save("linked", {"SKILL.md": rendered("linked", "Rewritten. Not linked.")}, base=base)
    assert "Original body." in outside.read_text()  # the link's target was not written through
    assert not (user / "linked" / "SKILL.md").is_symlink()


# -- the strict emitter --------------------------------------------------------------------------


def test_render_output_loads_back_cleanly_whatever_the_description_contains() -> None:
    description = (
        'Use when: the user says "ship it" for C# #1 [beta]. Not for hotfixes: those differ.'
    )
    text = document.render(
        "release-flow",
        description,
        "# Steps\n\n1. Tag.\n",
        {"clankit.origin": "generated", "clankit.session": "12e4", "clankit.date": "2026-01-02"},
    )
    parsed = document.parse(text)
    assert parsed.meta == {
        "name": "release-flow",
        "description": description,
        "metadata": {
            "clankit.origin": "generated",
            "clankit.session": "12e4",  # still a string, not the number 120000.0
            "clankit.date": "2026-01-02",
        },
    }
    assert parsed.body == "# Steps\n\n1. Tag.\n"
    assert text.startswith("---\nname: release-flow\ndescription: ") and text.endswith("\n")


@pytest.mark.parametrize(
    ("name", "description", "body", "message"),
    [
        ("Has Space", "d", "b", "not valid"),
        ("double--hyphen", "d", "b", "not valid"),
        ("-leading", "d", "b", "not valid"),
        ("n" * 65, "d", "b", "65 characters long"),
        ("ok", "  ", "b", "description is empty"),
        ("ok", "d" * 1025, "b", "1025 characters long"),
        ("ok", "see <tag>", "b", "'<' or '>'"),
        ("ok", "before --- after", "b", "'---'"),
        ("ok", "d", "\n\n", "body is empty"),
        ("ok", "d", "line\n" * 501, "502 lines long"),
    ],
)
def test_render_refuses_what_other_agents_would_reject(
    name: str, description: str, body: str, message: str
) -> None:
    with pytest.raises(SkillError, match=message):
        document.render(name, description, body)


def test_render_refuses_metadata_keys_that_need_quoting() -> None:
    with pytest.raises(SkillError, match="metadata key"):
        document.render("ok", "d", "b", {"bad key: x": "v"})

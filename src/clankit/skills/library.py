"""The skill library: finding skill folders, indexing them, and managing proposals.

A skill is a folder with a ``SKILL.md`` (see `clankit.skills.document`) and optional bundled
scripts and reference files. The library follows the progressive-disclosure model the format
was designed for: a scan keeps only each skill's metadata, `catalog()` turns that into the
short index agents see in their system prompt, and a skill's instructions and file listing
are read from disk only when somebody asks for them.

Where skills come from, in order of precedence (the first one found wins a name clash, and
the loser is reported in `diagnostics`):

1. ``directories[0]``, the user's own library (``~/.config/clankit/skills``). It is the only
   place the library ever writes to or deletes from.
2. Any further directories passed in, which are read-only ("extra").
3. For a view made with `with_project()`: ``<workspace>/.clankit/skills`` and
   ``<workspace>/.agents/skills``. A repository is somebody else's text, so its skills are
   listed but agents may not load them unless the caller says the project is trusted, and a
   project skill can never replace a user skill of the same name.

Skills written by the generator wait in ``<user library>/.pending`` until the user approves
them. That folder is never scanned for active skills.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from clankit.skills import document
from clankit.skills.document import DESCRIPTION_MAX, SKILL_FILE, SkillError

__all__ = [
    "GENERATED",
    "ORIGIN_KEY",
    "PENDING_DIR",
    "PROJECT_DIRS",
    "Skill",
    "SkillError",
    "SkillLibrary",
]

PENDING_DIR = ".pending"
#: Where a project keeps its skills, relative to the workspace, in order of precedence.
PROJECT_DIRS = (".clankit/skills", ".agents/skills")
#: Bounds on a scan, so a skills directory that is really a checkout of something huge
#: cannot stall start-up.
MAX_DEPTH = 6
MAX_DIRECTORIES = 2000
#: Characters of one skill's description shown in the catalog.
CATALOG_ENTRY_CHARS = 1024
#: Metadata written by the skill generator.
ORIGIN_KEY = "clankit.origin"
GENERATED = "generated"

_IGNORED_DIRECTORIES = frozenset({"node_modules", "__pycache__"})
_SKILL_DIR_VARIABLES = ("${CLANKIT_SKILL_DIR}", "${CLAUDE_SKILL_DIR}")
_WRITABLE_SOURCES = ("user", "pending")


@dataclass(slots=True)
class Skill:
    """One skill folder as the library indexes it: metadata only.

    `name` is always the folder's name, which is what makes it unique within a directory and
    safe to use as an identifier. `source` says where the skill was found: "user", "extra",
    "project" or "pending". `warnings` lists what is wrong with the file but did not stop it
    from loading.
    """

    name: str
    description: str
    path: Path
    source: str = "user"
    pending: bool = False
    when_to_use: str = ""
    argument_hint: str = ""
    disable_model_invocation: bool = False
    user_invocable: bool = True
    #: False for skills of a project the user has not trusted.
    trusted: bool = True
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def file(self) -> Path:
        return _skill_file(self.path) or self.path / SKILL_FILE

    @property
    def generated(self) -> bool:
        """Was this skill written by the skill generator (rather than by hand)?"""
        return self.metadata.get(ORIGIN_KEY) == GENERATED

    @property
    def model_invocable(self) -> bool:
        """May an agent load this skill on its own initiative?"""
        return self.trusted and not self.pending and not self.disable_model_invocation

    @property
    def listing(self) -> str:
        """The text the catalog shows for this skill: what it does and when to use it."""
        text = f"{self.description} {self.when_to_use}".strip()
        if len(text) <= CATALOG_ENTRY_CHARS:
            return text
        return text[: CATALOG_ENTRY_CHARS - 3].rstrip() + "..."

    def body(self) -> str:
        """SKILL.md without its frontmatter, read from disk now."""
        text = document.decode(self.file.read_bytes())
        try:
            return document.parse(text).body
        except SkillError:  # the file changed since the scan; show what is there
            return text

    def instructions(self, arguments: str = "") -> str:
        """The body as an agent should receive it.

        ``$ARGUMENTS`` becomes `arguments` (what the user typed after ``/name``), and the
        skill-directory variable (also under the name skills written for Claude Code use)
        becomes this skill's absolute path, so bundled scripts can be run from anywhere.
        """
        text = self.body().replace("$ARGUMENTS", arguments)
        for variable in _SKILL_DIR_VARIABLES:
            text = text.replace(variable, str(self.path))
        return text

    def files(self) -> list[str]:
        """Bundled files (everything except SKILL.md and hidden files), relative to `path`."""
        found: list[str] = []
        for directory, subdirectories, names in os.walk(self.path):
            subdirectories[:] = [d for d in subdirectories if not _is_ignored(d)]
            for name in names:
                relative = (Path(directory) / name).relative_to(self.path).as_posix()
                if not name.startswith(".") and relative.lower() != SKILL_FILE.lower():
                    found.append(relative)
        return sorted(found)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "source": self.source,
            "pending": self.pending,
            "when_to_use": self.when_to_use,
            "argument_hint": self.argument_hint,
            "model_invocable": self.model_invocable,
            "user_invocable": self.user_invocable,
            "trusted": self.trusted,
            "generated": self.generated,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class _Shared:
    """What a library and its project views have in common.

    Views read the directories live, so a configuration change reaches all of them, and
    `epoch` is bumped whenever skills may have changed, which invalidates every view's scan.
    """

    directories: list[Path]
    epoch: int = 0


@dataclass(slots=True)
class _Scan:
    skills: dict[str, Skill]
    diagnostics: list[str]


class SkillLibrary:
    """The skills found in a list of directories, the first of which is the user's own.

    Scans are cached; call `refresh()` after changing skill files by other means than this
    class. Raises `KeyError` for names it does not know and `SkillError` for operations it
    refuses.
    """

    def __init__(self, directories: Sequence[Path]) -> None:
        self._shared = _Shared([Path(d) for d in directories])
        self._workspace: Path | None = None
        self._project_trusted = False
        self._cache: tuple[int, _Scan] | None = None

    # -- where skills come from -----------------------------------------------------------------

    def set_directories(self, directories: Sequence[Path]) -> None:
        self._shared.directories = [Path(d) for d in directories]
        self.refresh()

    @property
    def directories(self) -> list[Path]:
        return list(self._shared.directories)

    def with_project(self, workspace: Path, *, trusted: bool = False) -> SkillLibrary:
        """A view of this library that also lists the skills kept in `workspace`.

        Project skills get source "project". Unless `trusted` is true they are shown but not
        offered to agents: cloning a repository must not be enough to put its instructions in
        front of a model. Pass ``trusted=True`` only after the user has agreed to that for
        this workspace. The view shares this library's directories and proposals.
        """
        view = SkillLibrary(())
        view._shared = self._shared
        view._workspace = Path(workspace)
        view._project_trusted = trusted
        return view

    def refresh(self) -> None:
        """Forget cached scan results (here and in every view) after skills changed on disk."""
        self._shared.epoch += 1

    # -- reading --------------------------------------------------------------------------------

    @property
    def skills(self) -> list[Skill]:
        """Active skills from every source, sorted by name."""
        return sorted(self._scanned().skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        return self._scanned().skills.get(name)

    @property
    def diagnostics(self) -> list[str]:
        """Problems found by the latest scan: skipped files, format warnings, shadowed skills."""
        return list(self._scanned().diagnostics)

    def catalog(self, max_chars: int | None = None) -> str:
        """The skill index for system prompts: one ``- name: description`` line per skill.

        Lists only skills an agent may load. The order is stable (sorted by name) so the
        text is a cacheable prompt prefix. When the index would exceed `max_chars`, whole
        entries are left out from the end and a final line says how many; "" when there is
        nothing to list.
        """
        entries = [
            f"- {skill.name}: {skill.listing}"
            for skill in self.skills
            if skill.model_invocable and skill.description
        ]
        text = "\n".join(entries)
        if max_chars is None or len(text) <= max_chars:
            return text
        for shown in range(len(entries) - 1, 0, -1):
            hidden = len(entries) - shown
            noun = "skill" if hidden == 1 else "skills"
            text = "\n".join([*entries[:shown], f"({hidden} more {noun} not shown)"])
            if len(text) <= max_chars:
                return text
        return f"({len(entries)} skills not shown)"

    # -- proposals ------------------------------------------------------------------------------

    def pending(self) -> list[Skill]:
        """Proposals staged by the skill generator, sorted by name."""
        directory = self._pending_dir()
        if directory is None or not directory.is_dir():
            return []
        proposals: list[Skill] = []
        for child in sorted(directory.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            with contextlib.suppress(OSError, SkillError):
                proposals.append(_read_skill(child, source="pending", pending=True))
        return proposals

    def approve(self, name: str) -> Skill:
        """Make the proposal `name` an active skill, replacing an older version of it.

        Raises `KeyError` if there is no such proposal and `SkillError` if it is not a
        readable skill.
        """
        proposal = self._proposal(name)
        try:
            _read_skill(proposal, source="pending", pending=True)
        except (OSError, SkillError) as exc:
            raise SkillError(f"the proposal {name!r} cannot be approved: {exc}") from exc
        target = proposal.parent.parent / name
        _swap_in(proposal, target, graveyard=proposal.parent)
        self.refresh()
        return _read_skill(target, source="user")

    def reject(self, name: str) -> None:
        """Delete the proposal `name`. Raises `KeyError` if there is none."""
        _remove(self._proposal(name))

    def delete(self, name: str) -> None:
        """Delete the active skill `name` from the user library.

        Raises `KeyError` if no such skill is active and `SkillError` if it lives anywhere
        else (a project, an extra directory): those are not the library's to remove.
        """
        skill = self.get(name)
        if skill is None:
            raise KeyError(name)
        if skill.source != "user":
            raise SkillError(
                f"{name!r} is a {skill.source} skill in {skill.path}; only skills in the "
                f"user library can be deleted from here"
            )
        _remove(skill.path)
        self.refresh()

    def save(
        self,
        name: str,
        files: Mapping[str, str],
        *,
        pending: bool = False,
        base: Skill | None = None,
    ) -> Skill:
        """Write a skill folder into the user library in one atomic step.

        `files` maps paths inside the folder to their text and must contain a strictly valid
        ``SKILL.md`` for `name`. With `pending` the folder is staged as a proposal instead of
        becoming active. `base` is the existing version being revised: its files are carried
        over and overlaid with `files`, and it is the only way to replace an active skill, so
        nothing is overwritten by accident. Raises `SkillError` when the request is not safe
        or not valid; nothing is written in that case.
        """
        root = self._user_root()
        if root is None:
            raise SkillError("there is no user skill library to write to")
        _check_writable(name, files)
        if base is not None and (base.name != name or base.source not in _WRITABLE_SOURCES):
            raise SkillError(f"{base.name!r} ({base.source}) cannot be revised as {name!r}")
        staging_root = root / PENDING_DIR
        target = staging_root / name if pending else root / name
        if not pending and base is None and (target.exists() or target.is_symlink()):
            raise SkillError(f"{target} already exists; revise it by passing it as `base`")

        staging_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=staging_root, prefix=".incoming-"))
        try:
            staged = workdir / name
            if base is None:
                staged.mkdir()
            else:
                # Links are copied as what they point to, so nothing staged here can lead a
                # later write out of the library.
                shutil.copytree(base.path, staged, ignore_dangling_symlinks=True)
            for relative, text in files.items():
                _write_file(staged, relative, text)
            _swap_in(staged, target, graveyard=staging_root)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        self.refresh()
        return _read_skill(target, source="pending" if pending else "user", pending=pending)

    # -- internals ------------------------------------------------------------------------------

    def _user_root(self) -> Path | None:
        directories = self._shared.directories
        return directories[0].expanduser().absolute() if directories else None

    def _pending_dir(self) -> Path | None:
        root = self._user_root()
        return root / PENDING_DIR if root is not None else None

    def _proposal(self, name: str) -> Path:
        """The folder of the proposal `name`. A name that is not a plain folder name cannot
        be a proposal, which also keeps user-supplied names from reaching outside `.pending`."""
        directory = self._pending_dir()
        plain = bool(name) and name == Path(name).name and not name.startswith(".")
        if directory is None or not plain or not (directory / name).is_dir():
            raise KeyError(name)
        return directory / name

    def _roots(self) -> list[tuple[Path, str, bool]]:
        roots = [
            (directory.expanduser().absolute(), "user" if index == 0 else "extra", True)
            for index, directory in enumerate(self._shared.directories)
        ]
        if self._workspace is not None:
            roots.extend(
                (self._workspace.absolute() / relative, "project", self._project_trusted)
                for relative in PROJECT_DIRS
            )
        return roots

    def _scanned(self) -> _Scan:
        if self._cache is None or self._cache[0] != self._shared.epoch:
            self._cache = (self._shared.epoch, self._scan())
        return self._cache[1]

    def _scan(self) -> _Scan:
        skills: dict[str, Skill] = {}
        diagnostics: list[str] = []
        for root, source, trusted in self._roots():
            directories, complete = _skill_directories(root)
            if not complete:
                diagnostics.append(
                    f"{root}: more than {MAX_DIRECTORIES} folders; the rest were not searched"
                )
            for directory in directories:
                try:
                    skill = _read_skill(directory, source=source, trusted=trusted)
                except (OSError, SkillError) as exc:
                    diagnostics.append(f"{directory}: skipped - {exc}")
                    continue
                diagnostics.extend(f"{skill.file}: {warning}" for warning in skill.warnings)
                winner = skills.setdefault(skill.name, skill)
                if winner is not skill:
                    diagnostics.append(
                        f"{skill.path}: shadowed by the skill of the same name in {winner.path}"
                    )
        return _Scan(skills, diagnostics)


# --------------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------------


def _is_ignored(directory_name: str) -> bool:
    return directory_name.startswith(".") or directory_name in _IGNORED_DIRECTORIES


def _skill_file(directory: Path) -> Path | None:
    for name in (SKILL_FILE, SKILL_FILE.lower()):
        if (directory / name).is_file():
            return directory / name
    return None


def _skill_directories(root: Path) -> tuple[list[Path], bool]:
    """Skill folders under `root`, nearest first, and whether the search was complete.

    Skills may be grouped in category folders, so the search goes deeper wherever a folder is
    not itself a skill. Hidden folders (which covers `.pending` and `.git`) are skipped, and
    folders reached twice through symlinks are visited once. The search gives up after
    looking at `MAX_DIRECTORIES` folders.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    examined = 0
    while queue:
        directory, depth = queue.popleft()
        try:
            real = directory.resolve()
            children = sorted(child for child in directory.iterdir() if child.is_dir())
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        for child in children:
            if _is_ignored(child.name):
                continue
            examined += 1
            if examined > MAX_DIRECTORIES:
                return found, False
            if _skill_file(child) is not None:
                found.append(child)
            elif depth + 1 < MAX_DEPTH:
                queue.append((child, depth + 1))
    return found, True


def _read_skill(
    directory: Path, *, source: str, pending: bool = False, trusted: bool = True
) -> Skill:
    """Index the skill in `directory`, leniently: only a file without usable frontmatter is
    refused (`SkillError`); every other defect becomes a warning on the returned skill."""
    skill_file = _skill_file(directory)
    if skill_file is None:
        raise SkillError(f"there is no {SKILL_FILE}")
    meta = document.read(skill_file, head_only=True).meta
    name = directory.name
    warnings: list[str] = []

    declared = document.text_field(meta.get("name"))
    if declared and declared != name:
        warnings.append(
            f"the frontmatter says name: {declared!r} but the folder is {name!r}; "
            "the folder name is used"
        )
    if problem := document.name_problem(name):
        warnings.append(f"{problem} (other agents may refuse to load this skill)")

    description = document.text_field(meta.get("description"))
    if not description:
        warnings.append(
            "there is no description, so agents cannot discover this skill; add a "
            "'description:' line saying what it does and when to use it"
        )
    elif len(description) > DESCRIPTION_MAX:
        warnings.append(
            f"the description is {len(description)} characters long (the format allows "
            f"{DESCRIPTION_MAX}); it is shortened in the skill index"
        )

    return Skill(
        name=name,
        description=description,
        path=directory,
        source=source,
        pending=pending,
        when_to_use=document.text_field(meta.get("when_to_use")),
        argument_hint=document.text_field(meta.get("argument-hint")),
        disable_model_invocation=_flag(meta, "disable-model-invocation", False, warnings),
        user_invocable=_flag(meta, "user-invocable", True, warnings),
        trusted=trusted,
        metadata=document.metadata_field(meta.get("metadata")),
        warnings=warnings,
    )


def _flag(meta: Mapping[str, Any], key: str, default: bool, warnings: list[str]) -> bool:
    if key not in meta:
        return default
    value = document.flag_field(meta[key])
    if value is None:
        warnings.append(f"{key} must be true or false; using {str(default).lower()}")
        return default
    return value


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------


def _check_writable(name: str, files: Mapping[str, str]) -> None:
    """Refuse anything that would not load back as a clean skill named `name`."""
    if problem := document.name_problem(name):
        raise SkillError(problem)
    if SKILL_FILE not in files:
        raise SkillError(f"a skill needs a {SKILL_FILE}")
    parsed = document.parse(files[SKILL_FILE])
    declared = document.text_field(parsed.meta.get("name"))
    if declared != name:
        raise SkillError(f"{SKILL_FILE} says name: {declared!r} but the skill is {name!r}")
    problems = (
        document.description_problem(document.text_field(parsed.meta.get("description"))),
        document.body_problem(parsed.body),
    )
    if any(problems):
        raise SkillError("; ".join(p for p in problems if p))


def _write_file(folder: Path, relative: str, text: str) -> None:
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise SkillError(f"{relative!r} is not a path inside the skill folder")
    path = folder.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if parts[0] == "scripts":
        path.chmod(0o755)


def _swap_in(source: Path, target: Path, *, graveyard: Path) -> None:
    """Move the folder `source` to `target`, replacing what is there.

    A folder cannot be renamed over a non-empty one, so the old version is first renamed out
    of the way (into `graveyard`, which must be on the same filesystem and is never scanned)
    and put back if the new one cannot take its place. Readers therefore see the old skill
    or the new one, never a mixture of their files.
    """
    retired: Path | None = None
    if target.exists() or target.is_symlink():
        retired = Path(tempfile.mkdtemp(dir=graveyard, prefix=".retired-")) / target.name
        os.rename(target, retired)
    try:
        os.rename(source, target)
    except OSError:
        if retired is not None:
            os.rename(retired, target)
            retired.parent.rmdir()
        raise
    if retired is not None:
        shutil.rmtree(retired.parent, ignore_errors=True)


def _remove(folder: Path) -> None:
    """Delete a skill folder. A symlinked skill loses only the link, not what it points to."""
    if folder.is_symlink():
        folder.unlink()
    else:
        shutil.rmtree(folder)

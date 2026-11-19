"""Saved workflows: named scripts that ship with Clankit or that the user keeps.

Built-in workflows live in this package's ``builtin`` directory; the user's own live in
``<config dir>/workflows`` (``~/.config/clankit/workflows/<name>.py``). A user workflow with
the name of a built-in one replaces it, so a built-in can be customised by copying it there.
Files are read and checked on every lookup, so a new or edited workflow is picked up without
a restart, and a broken file is reported instead of hiding the others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from clankit import paths
from clankit.orchestration.workflow.script import Meta, Script, ScriptError, compile_script

BUILTIN_DIR = Path(__file__).parent / "builtin"
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


@dataclass(frozen=True, slots=True)
class SavedWorkflow:
    """A saved script, addressed by its file name without ``.py``."""

    name: str
    source: str  # "built-in" or "user"
    path: Path
    script: Script

    @property
    def meta(self) -> Meta:
        return self.script.meta

    def describe(self) -> str:
        """One list entry for an agent's instructions: name, purpose, when, arguments."""
        text = f"- `{self.name}`: {self.meta.description}"
        if self.meta.when_to_use:
            text += f" When: {self.meta.when_to_use}"
        if self.meta.args:
            text += " Args: " + "; ".join(f"`{k}`: {v}" for k, v in self.meta.args) + "."
        return text


@dataclass(slots=True)
class Catalog:
    workflows: list[SavedWorkflow] = field(default_factory=list)
    #: Files that could not be used, each with the reason.
    problems: list[str] = field(default_factory=list)


def user_workflow_dir() -> Path:
    return paths.config_dir() / "workflows"


def load_saved() -> Catalog:
    """Every usable saved workflow (built-ins first replaced by user ones), sorted by name."""
    found: dict[str, SavedWorkflow] = {}
    problems: list[str] = []
    for source, directory in (("built-in", BUILTIN_DIR), ("user", user_workflow_dir())):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            workflow = _load(path, source, problems)
            if workflow is not None:
                found[workflow.name] = workflow
    return Catalog([found[name] for name in sorted(found)], problems)


def find_saved(name: str) -> SavedWorkflow | None:
    return next((w for w in load_saved().workflows if w.name == name), None)


def saved_names() -> list[str]:
    return [w.name for w in load_saved().workflows]


def _load(path: Path, source: str, problems: list[str]) -> SavedWorkflow | None:
    name = path.stem
    if not NAME.match(name):
        problems.append(
            f"{path}: the file name must be lowercase letters, digits, '-' or '_' (it is the "
            "workflow's name)"
        )
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{path}: cannot be read: {exc}")
        return None
    try:
        script = compile_script(text, filename=str(path))
    except ScriptError as exc:
        problems.append(f"{path}:\n{exc}")
        return None
    return SavedWorkflow(name, source, path, script)

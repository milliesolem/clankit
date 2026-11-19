"""The JSON the browser receives: application state, session state, setting rows.

Every function here reads engine objects and runs on the runtime's event loop (through
`Engine.submit`); what it returns is plain data that the request thread serialises. Nothing
returned here ever contains an API key: providers are described by
`clankit.config.actions.provider_rows`, which masks them.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from clankit import __version__
from clankit.commands import COMMANDS
from clankit.config import fields
from clankit.config.actions import provider_rows
from clankit.config.schema import Config, ConfigError, Setting
from clankit.llm.provider import ProviderError
from clankit.orchestration import all_orchestrators, get_orchestrator
from clankit.skills.library import Skill
from clankit.themes import all_themes, custom_theme_problems, get_theme
from clankit.themes.model import Theme

if TYPE_CHECKING:
    from clankit.agent.session import Session
    from clankit.orchestration.base import Orchestrator
    from clankit.runtime import Runtime
    from clankit.web.sessions import SessionIndex

#: The theme tokens a theme picker shows as swatches, in order.
SWATCH_TOKENS = ("bg", "bg_alt", "bg_raised", "accent", "success", "error")


def app_state(runtime: Runtime, index: SessionIndex) -> dict[str, Any]:
    """Everything the settings view, the pickers and the status row need."""
    config = runtime.config
    return {
        "version": __version__,
        "workspace": str(index.workspace),
        "config": config_summary(runtime),
        "sessions": index.listing(),
        "providers": [row.to_dict() for row in provider_rows(config)],
        "models": [model.to_dict() for model in runtime.providers.known_models()],
        "methods": [method_row(cls, config) for cls in all_orchestrators()],
        "themes": [theme_row(theme) for theme in all_themes(config).values()],
        "theme_css": theme_css_path(config),
        "commands": [
            {"name": c.name, "summary": c.summary, "argument": c.argument} for c in COMMANDS
        ],
        "skills": skills_state(runtime),
        "sections": [section_row(section, config) for section in fields.SECTIONS],
    }


def config_summary(runtime: Runtime) -> dict[str, Any]:
    config = runtime.config
    store = runtime.store
    return {
        "path": str(store.path) if store.path is not None else "",
        "theme": get_theme(config.ui.theme, config).id,
        "show_thinking": config.ui.show_thinking,
        "show_tree": config.ui.show_tree,
        "permission_mode": config.permissions.mode,
        "method": config.orchestration.method,
        "default_model": config.models.default,
        "usable": bool(runtime.providers.provider_ids()),
        "warnings": [*store.warnings, *custom_theme_problems(config)],
    }


def theme_row(theme: Theme) -> dict[str, Any]:
    colours = theme.colours()
    return {
        "id": theme.id,
        "name": theme.name,
        "dark": theme.dark,
        "swatches": [colours[token] for token in SWATCH_TOKENS],
    }


def theme_css_path(config: Config) -> str:
    """The stylesheet URL, versioned by the themes it contains so browsers refetch on change."""
    themes = all_themes(config)
    digest = hashlib.sha256(
        repr([(t.id, t.colours()) for t in themes.values()]).encode("utf-8")
    ).hexdigest()[:12]
    return f"/theme.css?v={digest}"


def method_row(cls: type[Orchestrator], config: Config) -> dict[str, Any]:
    return {
        "name": cls.name,
        "title": cls.title or cls.name,
        "summary": cls.summary,
        "guidance": cls.guidance,
        "needs_model": cls.needs_model,
        "settings": [
            setting_row(spec, config.orchestration.option(cls.name, spec.key))
            for spec in cls.settings
        ],
    }


def setting_row(spec: Setting, raw: Any) -> dict[str, Any]:
    """A method setting with its current value (the default when unset or invalid)."""
    try:
        value = spec.coerce(raw)
    except ConfigError:
        value = spec.default
    return {
        "key": spec.key,
        "kind": spec.kind,
        "default": spec.default,
        "help": spec.help,
        "choices": list(spec.choices),
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "value": value,
    }


def section_row(section: fields.Section, config: Config) -> dict[str, Any]:
    return {
        "id": section.id,
        "title": section.title,
        "fields": [
            {
                "path": f.path,
                "label": f.label,
                "help": f.help,
                "kind": f.kind,
                "choices": list(f.choices),
                "minimum": f.minimum,
                "maximum": f.maximum,
                "empty": f.empty,
                "value": fields.get_value(config, f.path),
            }
            for f in section.fields
        ],
    }


def skills_state(runtime: Runtime) -> dict[str, Any]:
    library = runtime.skills
    return {
        "directory": str(library.directories[0]) if library.directories else "",
        "active": [skill_row(skill) for skill in library.skills],
        "pending": [skill_row(skill) for skill in library.pending()],
        "diagnostics": library.diagnostics,
    }


def skill_row(skill: Skill) -> dict[str, Any]:
    row = skill.to_dict()
    row["reason"] = skill.metadata.get("clankit.reason", "")
    return row


# -- sessions -----------------------------------------------------------------------------------


def session_meta(session: Session, index: SessionIndex) -> dict[str, Any]:
    """The facts the status row and the crumb show about a session."""
    try:
        model = str(session.model)
    except ProviderError:
        model = ""
    context = session.context_usage()
    try:
        method_title = get_orchestrator(session.method).title or session.method
    except KeyError:
        method_title = session.method
    return {
        "id": session.id,
        "workspace": str(session.workspace),
        "title": index.title(session),
        "method": session.method,
        "method_title": method_title,
        "demo": session.method == "demo",
        "model": model,
        "busy": session.busy,
        "usage": session.usage.to_dict(),
        "cost": session.cost(),
        "context": {"used": context[0], "window": context[1]} if context else None,
    }


def session_state(session: Session, index: SessionIndex) -> dict[str, Any]:
    """A session as a browser loads it: its facts, its folded view and its runs."""
    return {
        "meta": session_meta(session, index),
        "snapshot": session.view.snapshot(),
        "runs": [run.to_dict() for run in index.runs(session)],
    }

"""The general settings, declared once so both user interfaces render and edit them alike.

Each `Field` names a value in the `Config` tree by dotted path, with the label, help text
and validation a settings form needs. The terminal UI and the web UI build their settings
screens from `SECTIONS`, read with `get_value` and write with `set_value` inside
``runtime.update_config(...)``, so a setting behaves identically wherever it is changed.

Settings that need more than a single value have their own helpers: providers and API keys
in `clankit.config.actions`, orchestration-method settings on each `Orchestrator`, custom
themes in `clankit.themes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clankit.config.schema import (
    MODEL_ROLES,
    PERMISSION_MODES,
    SKILL_GENERATION_MODES,
    Config,
    ConfigError,
    Setting,
)
from clankit.llm.types import EFFORT_LEVELS, ModelRef


@dataclass(frozen=True, slots=True)
class Field:
    """One editable setting.

    `path` is a dotted attribute path into `Config`; a path under ``models.roles`` addresses
    a dictionary entry. `kind` is one of "int", "float", "bool", "choice", "string", "model"
    (a ``provider/model`` reference; empty means "not set"), or "theme" (a theme id).
    """

    path: str
    label: str
    help: str
    kind: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    #: Shown instead of an empty value, e.g. "model default".
    empty: str = ""


@dataclass(frozen=True, slots=True)
class Section:
    id: str
    title: str
    fields: tuple[Field, ...]


SECTIONS: tuple[Section, ...] = (
    Section(
        "models",
        "Models",
        (
            Field(
                "models.default",
                "Default model",
                "Used by every agent unless a role below says otherwise.",
                "model",
            ),
            *(
                Field(
                    f"models.roles.{role}",
                    f"{role.capitalize()} model",
                    description + ".",
                    "model",
                    empty="default model",
                )
                for role, description in MODEL_ROLES
            ),
            Field(
                "agent.effort",
                "Reasoning effort",
                "How hard models that support it should think. Higher is slower and costs more.",
                "choice",
                choices=("", *EFFORT_LEVELS),
                empty="model default",
            ),
        ),
    ),
    Section(
        "agents",
        "Agents",
        (
            Field(
                "agent.max_turns",
                "Turn limit",
                "Model calls one agent may make for one task.",
                "int",
                minimum=1,
                maximum=10_000,
            ),
            Field(
                "agent.run_budget",
                "Run budget",
                "Output tokens one prompt may generate across all its agents before Clankit asks whether to continue. 0 turns the check off.",
                "int",
                minimum=0,
            ),
            Field(
                "agent.compact_at",
                "Compact at",
                "Fraction of the context window at which an agent summarises its conversation and starts fresh.",
                "float",
                minimum=0.3,
                maximum=0.95,
            ),
            Field(
                "agent.context_window",
                "Assumed context window",
                "Tokens assumed for models whose real context window is unknown.",
                "int",
                minimum=4_000,
            ),
            Field(
                "agent.tool_output_limit",
                "Tool output limit",
                "Characters of one tool result passed to the model; longer output loses its middle.",
                "int",
                minimum=1_000,
            ),
            Field(
                "agent.shell_timeout",
                "Shell timeout",
                "Seconds a shell command may run by default.",
                "int",
                minimum=1,
                maximum=600,
            ),
        ),
    ),
    Section(
        "permissions",
        "Permissions",
        (
            Field(
                "permissions.mode",
                "Mode",
                "ask: confirm edits, commands and network access. accept-edits: stop asking about edits in the workspace. auto: ask nothing. read-only: refuse every change.",
                "choice",
                choices=PERMISSION_MODES,
            ),
        ),
    ),
    Section(
        "skills",
        "Skills",
        (
            Field(
                "skills.generate",
                "Generate skills",
                "After a run, distil reusable skills from the work. off: never. ask: propose and wait for approval. auto: save without asking.",
                "choice",
                choices=SKILL_GENERATION_MODES,
            ),
            Field(
                "skills.directory",
                "Skills directory",
                "Where skills are stored.",
                "string",
                empty="~/.config/clankit/skills",
            ),
        ),
    ),
    Section(
        "appearance",
        "Appearance",
        (
            Field(
                "ui.theme", "Theme", "Colour theme for the terminal and web interfaces.", "theme"
            ),
            Field(
                "ui.show_thinking",
                "Show reasoning",
                "Show models' reasoning summaries in the transcript.",
                "bool",
            ),
            Field("ui.show_tree", "Show task tree", "Keep the task tree panel open.", "bool"),
        ),
    ),
    Section(
        "web",
        "Web interface",
        (
            Field(
                "web.host",
                "Host",
                "Address the web interface binds to. Keep 127.0.0.1 unless you know why not.",
                "string",
            ),
            Field(
                "web.port",
                "Port",
                "Port the web interface listens on.",
                "int",
                minimum=1,
                maximum=65_535,
            ),
            Field(
                "web.open_browser",
                "Open browser",
                "Open a browser window when the web interface starts.",
                "bool",
            ),
        ),
    ),
)


def all_fields() -> list[Field]:
    return [f for section in SECTIONS for f in section.fields]


def find_field(path: str) -> Field | None:
    return next((f for f in all_fields() if f.path == path), None)


def _walk(config: Config, path: str) -> tuple[Any, str]:
    """The object holding the final path component, and that component."""
    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    return target, parts[-1]


def get_value(config: Config, path: str) -> Any:
    target, leaf = _walk(config, path)
    if isinstance(target, dict):
        return target.get(leaf, "")
    return getattr(target, leaf)


def coerce(field: Field, value: Any) -> Any:
    """Validate a value entered by the user; raises `ConfigError` with a readable message.

    Theme ids are checked in `set_value`, which knows the user's custom themes.
    """
    if field.kind in ("int", "bool", "choice"):
        spec = Setting(
            field.label,
            field.kind,  # type: ignore[arg-type]
            None,
            field.help,
            choices=field.choices,
            minimum=int(field.minimum) if field.minimum is not None else None,
            maximum=int(field.maximum) if field.maximum is not None else None,
        )
        if value is None or (field.kind == "int" and str(value).strip() == ""):
            raise ConfigError(f"{field.label}: a value is required")
        return spec.coerce(value)
    if field.kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{field.label}: expected a number, got {value!r}") from exc
        if field.minimum is not None and number < field.minimum:
            raise ConfigError(f"{field.label}: must be at least {field.minimum:g}")
        if field.maximum is not None and number > field.maximum:
            raise ConfigError(f"{field.label}: must be at most {field.maximum:g}")
        return number
    text = "" if value is None else str(value).strip()
    if field.kind == "model" and text:
        try:
            ModelRef.parse(text)
        except ValueError as exc:
            raise ConfigError(f"{field.label}: {exc}") from exc
    if field.kind == "model" and not text and field.path == "models.default":
        raise ConfigError("Default model: a model is required")
    return text


def set_value(config: Config, path: str, value: Any) -> None:
    """Validate and store a setting on `config`. Call inside ``runtime.update_config``.

    For role models an empty value removes the override.
    """
    field = find_field(path)
    if field is None:
        raise ConfigError(f"unknown setting {path!r}")
    if field.kind == "theme":
        from clankit.themes import all_themes

        text = "" if value is None else str(value).strip()
        if text not in all_themes(config):
            raise ConfigError(f"{field.label}: unknown theme {text!r}")
        cleaned: Any = text
    else:
        cleaned = coerce(field, value)
    target, leaf = _walk(config, path)
    if isinstance(target, dict):
        if cleaned == "":
            target.pop(leaf, None)
        else:
            target[leaf] = cleaned
    else:
        setattr(target, leaf, cleaned)

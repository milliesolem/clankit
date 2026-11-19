"""The configuration model.

Plain dataclasses with defaults; no I/O here. `clankit.config.kdl` maps this to and from
``~/.config/clankit/config.kdl`` and `clankit.config.store.ConfigStore` owns loading, saving
and change notification.

Naming rule for the KDL mapping: every attribute ``some_name`` is written ``some-name``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

PERMISSION_MODES: tuple[str, ...] = ("ask", "accept-edits", "auto", "read-only")
SKILL_GENERATION_MODES: tuple[str, ...] = ("off", "ask", "auto")
PROVIDER_KINDS: tuple[str, ...] = ("anthropic", "openai-compatible")

#: Model roles an orchestrator may ask for. A role without an override uses the default model.
MODEL_ROLES: tuple[tuple[str, str], ...] = (
    ("subagent", "Subtasks, unless the delegating agent picks a model itself"),
    ("planner", "Planning and architecture in methods that separate thinking from typing"),
    ("reviewer", "Independent review and verification of finished work"),
    ("judge", "Choosing between candidate solutions"),
    ("utility", "Cheap housekeeping: summaries, titles, skill distillation"),
)


class ConfigError(Exception):
    """The configuration file is malformed or contains an invalid value."""


@dataclass(slots=True)
class ModelEntry:
    """Optional facts about one model, declared by the user or shipped in the catalog.

    `note` is shown to agents when they choose a model for a subtask, so write it as advice:
    "cheap and fast; good for mechanical edits".
    """

    id: str
    name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    tools: bool | None = None
    free: bool | None = None
    note: str = ""
    #: US dollars per million tokens, when known. Used only to show cost estimates.
    input_price: float | None = None
    output_price: float | None = None


@dataclass(slots=True)
class ProviderConfig:
    """One model backend.

    Built-in providers (see `clankit.config.catalog`) need only an API key from the user; a
    custom provider needs `kind` and `base_url` as well. `api_key_env` names an environment
    variable used when `api_key` is empty, so keys can stay out of the file if preferred.
    `keyless` marks local servers (Ollama, LM Studio, llama.cpp) that need no key at all.
    `options` holds adapter-specific switches and is passed through untouched.
    """

    id: str
    kind: str = "openai-compatible"
    name: str = ""
    api_key: str = ""
    api_key_env: str = ""
    base_url: str = ""
    keyless: bool = False
    enabled: bool = True
    #: Model calls allowed in flight at once. 0 means the built-in default (8). Free tiers
    #: with tight rate limits should be 1 or 2.
    concurrency: int = 0
    models: list[ModelEntry] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    @property
    def usable(self) -> bool:
        """Can requests be sent to this provider right now?"""
        if not (self.enabled and (self.keyless or bool(self.resolved_api_key))):
            return False
        return not self.base_url_problem()

    def base_url_problem(self) -> str:
        """Why this provider's base URL cannot work, or "" when it is fine.

        `usable` and the settings screens both use this, so a misconfigured address is never
        shown with a "ready" checkmark.
        """
        url = self.base_url.strip()
        if not url:
            return "" if self.kind == "anthropic" else "no base-url; set the address of its API"
        parts = urlsplit(url)
        if parts.scheme in ("http", "https") and parts.hostname:
            return ""
        return "base-url is not a valid address"

    def model(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)


@dataclass(slots=True)
class ModelsConfig:
    """Which model does what. Values are ``provider/model`` references."""

    default: str = "anthropic/claude-opus-5-5"
    roles: dict[str, str] = field(default_factory=dict)

    def for_role(self, role: str | None) -> str:
        return (self.roles.get(role) if role else None) or self.default


@dataclass(slots=True)
class OrchestrationConfig:
    """`method` picks the orchestrator; `options[method]` holds that method's settings.

    Each orchestrator declares its settings (name, type, default, help) in code, so this stays
    a plain mapping and new methods need no schema change. In KDL::

        orchestration {
            method "task-tree"
            task-tree {
                max-depth 10
            }
        }
    """

    method: str = "task-tree"
    options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def option(self, method: str, key: str, default: Any = None) -> Any:
        return self.options.get(method, {}).get(key, default)

    def set_option(self, method: str, key: str, value: Any) -> None:
        self.options.setdefault(method, {})[key] = value


@dataclass(slots=True)
class AgentConfig:
    #: Reasoning effort for models that support it: low | medium | high | xhigh | max.
    #: Empty leaves each model on its own default.
    effort: str = ""
    #: Model calls one agent may make for one task before it is stopped.
    max_turns: int = 200
    #: Context window assumed for models whose real window is unknown, in tokens.
    context_window: int = 128_000
    #: Fraction of the context window at which an agent summarises and starts fresh.
    compact_at: float = 0.8
    #: Characters of a single tool result passed to the model; the middle is elided beyond it.
    tool_output_limit: int = 30_000
    #: Default timeout for shell commands, in seconds.
    shell_timeout: int = 120
    #: Output tokens one run (one user prompt, all its agents) may generate before Clankit
    #: pauses and asks whether to continue. 0 disables the check.
    run_budget: int = 1_000_000


@dataclass(slots=True)
class PermissionsConfig:
    """What agents may do without asking.

    Modes: "ask" confirms edits, shell commands and network access; "accept-edits" stops
    asking about edits inside the workspace; "auto" asks nothing; "read-only" refuses
    everything that changes state. Rules refine the mode and are written
    ``category:pattern`` with shell-style wildcards, e.g. ``shell:git status*``,
    ``edit:docs/**``, ``network:https://docs.python.org/*``. Deny rules win.
    """

    mode: str = "ask"
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillsConfig:
    #: Directory of skills. Empty means ``<config dir>/skills``.
    directory: str = ""
    #: After a run: "off" never proposes skills, "ask" proposes and waits for approval,
    #: "auto" saves without asking.
    generate: str = "ask"


@dataclass(slots=True)
class UIConfig:
    theme: str = "ayu-dark"
    show_thinking: bool = True
    show_tree: bool = True


@dataclass(slots=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    open_browser: bool = True


@dataclass(slots=True)
class UpdatesConfig:
    """Self-updates from PyPI: `check` looks for a newer release at most once a day and,
    when the running installation is a regular pip one, installs it quietly."""

    check: bool = True


@dataclass(slots=True)
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    web: WebConfig = field(default_factory=WebConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)
    #: User-defined colour themes: theme id -> {token name: "#rrggbb"}. A theme may set
    #: ``base`` to the id of a built-in theme and override only some tokens.
    themes: dict[str, dict[str, str]] = field(default_factory=dict)

    def copy(self) -> Config:
        return copy.deepcopy(self)

    def usable_providers(self) -> list[ProviderConfig]:
        return [p for p in self.providers.values() if p.usable]


SettingKind = Literal["int", "bool", "choice", "model", "string"]


@dataclass(frozen=True, slots=True)
class Setting:
    """A user-adjustable option, declared next to the code that uses it.

    Orchestrators list their settings; both UIs render settings forms from these declarations
    and the values are stored under ``orchestration { <method> { <key> <value> } }``.
    `kind` "model" is a ``provider/model`` reference where the empty string means "use the
    role default".
    """

    key: str
    kind: SettingKind
    default: Any
    help: str
    choices: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    def coerce(self, value: Any) -> Any:
        """Validate and normalise a raw value; raises ConfigError with a readable message."""
        if value is None:
            return self.default
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ConfigError(f"{self.key}: expected a whole number, got {value!r}")
            try:
                number = int(value)
            except ValueError as exc:
                raise ConfigError(f"{self.key}: expected a whole number, got {value!r}") from exc
            if self.minimum is not None and number < self.minimum:
                raise ConfigError(f"{self.key}: must be at least {self.minimum}")
            if self.maximum is not None and number > self.maximum:
                raise ConfigError(f"{self.key}: must be at most {self.maximum}")
            return number
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in (
                "true",
                "false",
                "yes",
                "no",
                "on",
                "off",
            ):
                return value.lower() in ("true", "yes", "on")
            raise ConfigError(f"{self.key}: expected true or false, got {value!r}")
        if self.kind == "choice":
            if value not in self.choices:
                raise ConfigError(f"{self.key}: must be one of {', '.join(self.choices)}")
            return value
        return str(value)

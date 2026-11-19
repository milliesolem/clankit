"""The layout of config.kdl: which node holds which setting, and what its value must be.

`clankit.config.kdl` reads and writes the file by walking `ROOT`, a tree of four parts:

* `Single`: one node with one value, ``max-turns 200``; when absent, the default applies;
* `Repeated`: a list written as one node per element, ``allow "shell:git status*"``;
* `Keyed`: a list of blocks told apart by their argument, ``model "qwen3-coder" { ... }``;
* `Block`: a node with children. Beyond its named parts a block may be a map, in which
  every other child is an entry whose name is the key (a provider id, a role, a theme token).

Between the file and `Config` sits "plain data": nested dicts and lists shaped exactly like
the file, e.g. ``{"model": "...", "agent": {"max-turns": 200, ...}, "providers": {...}}``.
`plain_from_config` and `config_from_plain` translate between the two, and every value is
vetted by the same `Kind` whether it comes from the file or from a `Config`.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from clankit.config.kdlsyntax import Scalar, is_writable
from clankit.config.schema import (
    PERMISSION_MODES,
    PROVIDER_KINDS,
    SKILL_GENERATION_MODES,
    AgentConfig,
    Config,
    ConfigError,
    ModelEntry,
    ModelsConfig,
    OrchestrationConfig,
    PermissionsConfig,
    ProviderConfig,
    SkillsConfig,
    UIConfig,
    WebConfig,
)
from clankit.llm.types import EFFORT_LEVELS, ModelRef

Plain = dict[str, Any]

#: Values longer than this are never quoted back in a message: text of that length in the wrong
#: place may well be a pasted API key. The same goes for names.
_ECHO_LIMIT = 24
_NAME_LIMIT = 32

_NUMBER_TEXT = re.compile(r"-?\d+(\.\d+)?")
_COLOUR_TEXT = re.compile(r"#[0-9a-fA-F]{6}")
_URL_TEXT = re.compile(r"https?://[^\s/]")


# --------------------------------------------------------------------------------------------
# What values must be
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Kind:
    """What a value must be: an exact type, and optionally a rule for values of that type.

    `expects` completes the sentence "<setting> must be ...". Types are compared exactly, so
    that "10" is not a number and #true is neither a number nor text.
    """

    expects: str
    types: tuple[type, ...]
    rule: Callable[[Any], str | None] | None = None

    def problem(self, value: object) -> str | None:
        """Why `value` is unacceptable, as a clause that follows the setting's name."""
        if type(value) not in self.types:
            return f"must be {self.expects}, not {describe(value)}{self._hint(value)}"
        if isinstance(value, float) and not math.isfinite(value):
            return "must be a finite number"
        if isinstance(value, str) and not is_writable(value):
            return "contains a character that cannot be saved to a file"
        return self.rule(value) if self.rule else None

    def normalised(self, value: Any) -> Any:
        """A whole number given where a fraction is allowed is stored as a float."""
        return float(value) if self.types == (int, float) else value

    def _hint(self, value: object) -> str:
        numeric = int in self.types and str not in self.types
        if isinstance(value, str) and numeric and _NUMBER_TEXT.fullmatch(value):
            return " (write it without the quotes)"
        if isinstance(value, str) and self.types == (bool,) and value in ("true", "false"):
            return f" (write #{value}, without quotes)"
        if isinstance(value, float) and self.types == (int,) and value.is_integer():
            return " (write it without the decimal point)"
        return ""


def describe(value: object) -> str:
    """The type of a wrong value in the file's own terms. Never the value: it may be secret."""
    if value is None:
        return "#null"
    if type(value) is bool:
        return "#true" if value else "#false"
    if type(value) is int:
        return "a whole number"
    if type(value) is float:
        return "a number with a decimal point" if math.isfinite(value) else "infinity"
    if type(value) is str:
        return "text"
    return f"a {type(value).__name__}"


def named(name: object) -> str:
    """A name for use in a message, unless it is too long to be one."""
    text = str(name)
    return text if len(text) <= _NAME_LIMIT and text.isprintable() else "a very long name"


def _but(value: object) -> str:
    """``, not "x"`` for a wrong value that is short enough to be safe to show."""
    shown = f'"{value}"' if isinstance(value, str) else str(value)
    return f", not {shown}" if len(shown) <= _ECHO_LIMIT and shown.isprintable() else ""


def _within(minimum: int, maximum: int | None) -> Callable[[float], str | None]:
    def rule(value: float) -> str | None:
        if value < minimum:
            return f"must be at least {minimum}{_but(value)}"
        if maximum is not None and value > maximum:
            return f"must be at most {maximum}{_but(value)}"
        return None

    return rule


def _whole(minimum: int, maximum: int | None = None) -> Kind:
    return Kind("a whole number", (int,), _within(minimum, maximum))


def _number(minimum: int, maximum: int | None = None) -> Kind:
    return Kind("a number", (int, float), _within(minimum, maximum))


def _one_of(choices: Sequence[str], *, or_empty: str = "") -> Kind:
    """One word from a list. `or_empty` says what leaving the value empty means, if allowed."""
    listed = ", ".join(f'"{choice}"' for choice in choices)
    if or_empty:
        listed += f', or "" {or_empty}'

    def rule(value: str) -> str | None:
        if value in choices or (or_empty and not value):
            return None
        return f"must be one of {listed}{_but(value)}"

    return Kind(_TEXT.expects, (str,), rule)


def _model_reference(value: str) -> str | None:
    try:
        ModelRef.parse(value)
    except ValueError:
        example = ModelsConfig().default
        return f'must be written "provider/model", for example "{example}"{_but(value)}'
    return None


def _role_model(value: str) -> str | None:
    # An empty role is allowed: `ModelsConfig.for_role` reads it as "use the default model".
    return _model_reference(value) if value else None


def _colour(value: str) -> str | None:
    if _COLOUR_TEXT.fullmatch(value):
        return None
    return f'must be a colour written "#rrggbb", for example "#c08040"{_but(value)}'


def _url(value: str) -> str | None:
    # The value is not shown: a URL can carry credentials.
    if not value or _URL_TEXT.match(value):
        return None
    return 'must start with "http://" or "https://"'


def _not_empty(value: str) -> str | None:
    return None if value.strip() else "must not be empty"


def _name_problem(name: str) -> str | None:
    return None if name.strip() else "a name cannot be empty"


def _provider_id_problem(name: str) -> str | None:
    if "/" in name:
        return 'a provider id cannot contain "/", which separates provider from model'
    return _name_problem(name)


_TEXT = Kind("text in double quotes", (str,))
_NAME = Kind(_TEXT.expects, (str,), _not_empty)
_FLAG = Kind("#true or #false", (bool,))
_SCALAR = Kind("text in double quotes, a number, #true or #false", (str, int, float, bool))


# --------------------------------------------------------------------------------------------
# The parts of the file
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Single:
    """One node with one value. `default` applies when the node is absent; a default of None
    means the setting has no value unless written."""

    kind: Kind
    default: Scalar = None


@dataclass(frozen=True, slots=True)
class Repeated:
    """A list of values, one node each, in file order."""

    kind: Kind


@dataclass(frozen=True, slots=True)
class Keyed:
    """A list of blocks, one node each, told apart by the node's argument. In plain data each
    is a dict that holds the argument under `key`."""

    key: str
    kind: Kind
    block: Block


@dataclass(frozen=True, slots=True)
class Block:
    """The children of a node, or the top level of the document.

    `parts` are the children with fixed names, in the order they are written to a new file.
    With `entries`, every other child is a map entry named by its key and `key_problem` vets
    the keys; without, other children are unknown and skipped. An `ordered` block gets new
    children next to their neighbours in `parts` rather than at the end. `where` locates the
    block in messages.
    """

    where: str
    parts: Mapping[str, Single | Repeated | Keyed | Block] = field(default_factory=dict)
    entries: Kind | Block | None = None
    key_problem: Callable[[str], str | None] = _name_problem
    ordered: bool = False


def _attribute(name: str) -> str:
    return name.replace("-", "_")


def _settings(
    defaults: Any, kinds: Mapping[str, Kind], *, apart: Sequence[str] = ()
) -> dict[str, Single]:
    """`Single` parts for the attributes of a config dataclass, defaults read from an instance.

    Fails at import if the dataclass has an attribute that is neither in `kinds` nor listed in
    `apart` (handled separately): a setting added to the schema must be given a place in the
    file, or it would silently never be saved.
    """
    attributes = {f.name for f in dataclasses.fields(defaults)} - set(apart)
    described = {_attribute(name) for name in kinds}
    if attributes != described:
        stray = ", ".join(sorted(attributes ^ described))
        raise TypeError(f"the KDL layout of {type(defaults).__name__} is out of date: {stray}")
    return {name: Single(kind, getattr(defaults, _attribute(name))) for name, kind in kinds.items()}


_MODEL_ENTRY = Block(
    "in a model entry",
    _settings(
        ModelEntry(id=""),
        {
            "name": _TEXT,
            "context-window": _whole(1),
            "max-output": _whole(1),
            "tools": _FLAG,
            "free": _FLAG,
            "note": _TEXT,
            "input-price": _number(0),
            "output-price": _number(0),
        },
        apart=("id",),
    ),
)

_PROVIDER = Block(
    "in a provider",
    {
        **_settings(
            ProviderConfig(id=""),
            {
                "kind": _one_of(PROVIDER_KINDS),
                "name": _TEXT,
                "api-key": _TEXT,
                "api-key-env": _TEXT,
                "base-url": Kind(_TEXT.expects, (str,), _url),
                "keyless": _FLAG,
                "enabled": _FLAG,
                "concurrency": _whole(0),
            },
            apart=("id", "models", "options"),
        ),
        "model": Keyed("id", _NAME, _MODEL_ENTRY),
        "options": Block("in options", entries=_SCALAR),
    },
)

ROOT = Block(
    "at the top level",
    {
        "model": Single(Kind(_TEXT.expects, (str,), _model_reference), ModelsConfig().default),
        "roles": Block("in roles", entries=Kind(_TEXT.expects, (str,), _role_model)),
        "providers": Block("in providers", entries=_PROVIDER, key_problem=_provider_id_problem),
        "orchestration": Block(
            "in orchestration",
            {"method": Single(_NAME, OrchestrationConfig().method)},
            entries=Block("in a method's settings", entries=_SCALAR),
        ),
        "agent": Block(
            "in agent",
            _settings(
                AgentConfig(),
                {
                    "effort": _one_of(EFFORT_LEVELS, or_empty="to leave it to each model"),
                    "max-turns": _whole(1),
                    "context-window": _whole(1),
                    "compact-at": _number(0, 1),
                    "tool-output-limit": _whole(0),
                    "shell-timeout": _whole(0),
                    "run-budget": _whole(0),
                },
            ),
        ),
        "permissions": Block(
            "in permissions",
            {
                **_settings(
                    PermissionsConfig(),
                    {"mode": _one_of(PERMISSION_MODES)},
                    apart=("allow", "deny"),
                ),
                "allow": Repeated(_NAME),
                "deny": Repeated(_NAME),
            },
        ),
        "skills": Block(
            "in skills",
            _settings(
                SkillsConfig(), {"directory": _TEXT, "generate": _one_of(SKILL_GENERATION_MODES)}
            ),
        ),
        "ui": Block(
            "in ui",
            _settings(UIConfig(), {"theme": _TEXT, "show-thinking": _FLAG, "show-tree": _FLAG}),
        ),
        "web": Block(
            "in web",
            _settings(
                WebConfig(), {"host": _TEXT, "port": _whole(0, 65535), "open-browser": _FLAG}
            ),
        ),
        "themes": Block(
            "in themes",
            entries=Block(
                "in a theme",
                {"base": Single(_NAME)},
                entries=Kind(_TEXT.expects, (str,), _colour),
            ),
        ),
    },
    ordered=True,
)


# --------------------------------------------------------------------------------------------
# Config <-> plain data
# --------------------------------------------------------------------------------------------


def same(first: object, second: object) -> bool:
    """Equality of plain data that tells 1 from #true and 1 from 1.0, which `==` does not."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(same(first[k], second[k]) for k in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(map(same, first, second))
    return type(first) is type(second) and first == second


def config_from_plain(plain: Plain) -> Config:
    """Build a `Config` from plain data in the shape a reader of `ROOT` produces."""
    orchestration = dict(plain["orchestration"])
    providers = {}
    for key, provider in plain["providers"].items():
        models = [_instance(ModelEntry, entry) for entry in provider["model"]]
        rest = {name: value for name, value in provider.items() if name != "model"}
        providers[key] = _instance(ProviderConfig, rest, id=key, models=models)
    return Config(
        models=ModelsConfig(default=plain["model"], roles=plain["roles"]),
        providers=providers,
        orchestration=OrchestrationConfig(orchestration.pop("method"), orchestration),
        agent=_instance(AgentConfig, plain["agent"]),
        permissions=_instance(PermissionsConfig, plain["permissions"]),
        skills=_instance(SkillsConfig, plain["skills"]),
        ui=_instance(UIConfig, plain["ui"]),
        web=_instance(WebConfig, plain["web"]),
        themes={
            key: {token: colour for token, colour in theme.items() if colour is not None}
            for key, theme in plain["themes"].items()
        },
    )


def plain_from_config(config: Config) -> Plain:
    """`config` as plain data, vetted and in the same shape a reader of `ROOT` produces.

    Raises `ConfigError` listing every value the file cannot hold, each named by a dotted path
    such as ``web.port``.
    """
    problems: list[str] = []
    plain = _checked(ROOT, _laid_out(config), "", problems)
    if problems:
        raise ConfigError("\n".join(problems))
    return plain


def _instance[T](cls: type[T], plain: Mapping[str, Any], **more: Any) -> T:
    return cls(**{_attribute(name): value for name, value in plain.items()}, **more)


def _attributes(instance: Any) -> Plain:
    """The attributes of a config dataclass under the names they have in the file."""
    return {
        f.name.replace("_", "-"): getattr(instance, f.name) for f in dataclasses.fields(instance)
    }


def _laid_out(config: Config) -> Plain:
    """The values of `config`, unvetted, arranged like the file."""
    methods = config.orchestration.options
    if "method" in methods:
        raise ConfigError(
            'orchestration: a method cannot be called "method"; that name selects the method'
        )
    providers: Plain = {}
    for key, provider in config.providers.items():
        if provider.id != key:
            raise ConfigError(f"providers.{named(key)}: the provider's id must equal its key")
        plain = _attributes(provider)
        del plain["id"]
        plain["model"] = [_attributes(entry) for entry in plain.pop("models")]
        providers[key] = plain
    return {
        "model": config.models.default,
        "roles": config.models.roles,
        "providers": providers,
        "orchestration": {"method": config.orchestration.method, **methods},
        "agent": _attributes(config.agent),
        "permissions": _attributes(config.permissions),
        "skills": _attributes(config.skills),
        "ui": _attributes(config.ui),
        "web": _attributes(config.web),
        "themes": config.themes,
    }


def _checked(schema: Block, plain: object, path: str, problems: list[str]) -> Plain:
    """Vet the plain data of one block and fill in what is absent. `path` ends in "." or is
    empty."""
    here = path.rstrip(".") or "the configuration"
    if not isinstance(plain, Mapping):
        problems.append(f"{here} must be a group of settings, not {describe(plain)}")
        plain = {}
    result: Plain = {}
    for name, part in schema.parts.items():
        where = path + name
        if isinstance(part, Single):
            value = plain.get(name, part.default)
            if value is not None or part.default is not None:
                value = _vetted(part.kind, value, where, problems)
            result[name] = value
        elif isinstance(part, Repeated):
            values = _list(plain.get(name, []), where, problems)
            result[name] = [_vetted(part.kind, value, where, problems) for value in values]
        elif isinstance(part, Keyed):
            result[name] = _checked_items(part, plain.get(name, []), where, problems)
        else:
            result[name] = _checked(part, plain.get(name, {}), where + ".", problems)
    for key, value in plain.items():
        if key in schema.parts:
            continue
        if problem := _entry_key_problem(schema, key):
            problems.append(f"{here}: {problem}")
        elif isinstance(schema.entries, Kind):
            result[key] = _vetted(schema.entries, value, path + key, problems)
        elif schema.entries is not None:
            result[key] = _checked(schema.entries, value, f"{path}{key}.", problems)
    return result


def _checked_items(part: Keyed, items: object, where: str, problems: list[str]) -> list[Plain]:
    result: list[Plain] = []
    seen: set[Any] = set()
    for item in _list(items, where, problems):
        key = _vetted(part.kind, item.get(part.key), f"{where}.{part.key}", problems)
        if key in seen:
            problems.append(f"{where} lists {named(key)} more than once")
        seen.add(key)
        rest = {name: value for name, value in item.items() if name != part.key}
        result.append({part.key: key, **_checked(part.block, rest, f"{where}.{key}.", problems)})
    return result


def _entry_key_problem(schema: Block, key: object) -> str | None:
    if schema.entries is None:
        return f"{named(key)} is not a setting"
    if type(key) is not str or not is_writable(key):
        return "a name must be text that can be saved to a file"
    return schema.key_problem(key)


def _vetted(kind: Kind, value: Any, where: str, problems: list[str]) -> Any:
    problem = kind.problem(value)
    if problem:
        problems.append(f"{where} {problem}")
        return value
    return kind.normalised(value)


def _list(value: object, where: str, problems: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    problems.append(f"{where} must be a list, not {describe(value)}")
    return []

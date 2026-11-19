"""Provider and key management, shared by both settings screens.

The functions that change things take a `Config` and mutate it; call them inside
``runtime.update_config(...)`` so the change is written to config.kdl:

    runtime.update_config(lambda c: set_api_key(c, "anthropic", key))

`provider_rows` is the read side: everything a settings screen shows about each provider,
with secrets already masked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from clankit.config.catalog import CATALOG, effective_providers, key_source
from clankit.config.schema import PROVIDER_KINDS, Config, ConfigError, ProviderConfig

_ID = re.compile(r"^[a-z][a-z0-9-]{0,39}$")


def mask_key(key: str) -> str:
    """A key reduced to something recognisable but useless: ``sk-ant-…a91f``."""
    if not key:
        return ""
    if len(key) <= 10:
        return "…" + key[-2:]
    prefix = re.match(r"^[A-Za-z]{2,6}[-_](?:[A-Za-z]{2,6}[-_])?", key)
    head = prefix.group(0) if prefix else key[:3]
    return f"{head}…{key[-4:]}"


@dataclass(frozen=True, slots=True)
class ProviderRow:
    """What a settings screen shows for one provider. Never contains a full key."""

    id: str
    name: str
    kind: str
    builtin: bool
    #: "config" (key in the file), "env" (key from the environment), "keyless", or "none".
    key_source: str
    masked_key: str
    api_key_env: str
    base_url: str
    enabled: bool
    usable: bool
    #: Why the base URL cannot work, or "" when it is fine.
    url_problem: str
    #: "no" | "partly" | "yes" | "local" for built-in providers, "" for custom ones.
    free: str
    note: str
    signup_url: str
    models: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


def provider_rows(config: Config) -> list[ProviderRow]:
    rows = []
    for provider_id, provider in effective_providers(config).items():
        cat = CATALOG.get(provider_id)
        rows.append(
            ProviderRow(
                id=provider_id,
                name=provider.name or provider_id,
                kind=provider.kind,
                builtin=cat is not None,
                key_source=key_source(config, provider_id),
                masked_key=mask_key(provider.resolved_api_key),
                api_key_env=provider.api_key_env,
                base_url=provider.base_url,
                enabled=provider.enabled,
                usable=provider.usable,
                url_problem=provider.base_url_problem(),
                free=cat.free if cat else "",
                note=cat.note if cat else "",
                signup_url=cat.signup_url if cat else "",
                models=tuple(m.id for m in provider.models),
            )
        )
    return rows


def _entry(config: Config, provider_id: str) -> ProviderConfig:
    """The user's own entry for a provider, created if absent."""
    entry = config.providers.get(provider_id)
    if entry is None:
        cat = CATALOG.get(provider_id)
        if cat is None:
            raise ConfigError(
                f"unknown provider {provider_id!r}; add it as a custom provider first"
            )
        entry = config.providers[provider_id] = ProviderConfig(id=provider_id, kind=cat.kind)
    return entry


def _prune(config: Config, provider_id: str) -> None:
    """Drop a built-in provider's user entry once it says nothing the catalog does not."""
    entry = config.providers.get(provider_id)
    cat = CATALOG.get(provider_id)
    if entry is None or cat is None or cat.keyless:
        return
    blank = ProviderConfig(id=provider_id, kind=entry.kind)
    if entry == blank:
        del config.providers[provider_id]


def set_api_key(config: Config, provider_id: str, key: str) -> None:
    """Store (or, with an empty key, remove) a provider's API key."""
    key = key.strip()
    if any(ch.isspace() for ch in key):
        raise ConfigError("an API key cannot contain spaces or line breaks")
    entry = _entry(config, provider_id)
    entry.api_key = key
    if key:
        entry.enabled = True
    _prune(config, provider_id)


def set_enabled(config: Config, provider_id: str, enabled: bool) -> None:
    """Switch a provider on or off. Switching a keyless local provider on is what makes
    Clankit use it."""
    _entry(config, provider_id).enabled = enabled
    if enabled:
        _prune(config, provider_id)


def set_base_url(config: Config, provider_id: str, base_url: str) -> None:
    base_url = base_url.strip()
    if base_url and not re.match(r"^https?://[^\s/]+", base_url):
        raise ConfigError("a base URL must start with http:// or https://")
    _entry(config, provider_id).base_url = base_url
    _prune(config, provider_id)


def add_custom_provider(
    config: Config,
    provider_id: str,
    *,
    base_url: str,
    kind: str = "openai-compatible",
    api_key: str = "",
    keyless: bool = False,
    name: str = "",
) -> None:
    """Add a provider that is not in the catalog, e.g. a self-hosted OpenAI-compatible server."""
    provider_id = provider_id.strip().lower()
    if not _ID.match(provider_id):
        raise ConfigError(
            "a provider id starts with a letter and contains only lowercase letters, digits and hyphens"
        )
    if provider_id in CATALOG or provider_id in config.providers:
        raise ConfigError(f"a provider named {provider_id!r} already exists")
    if kind not in PROVIDER_KINDS:
        raise ConfigError(f"provider kind must be one of: {', '.join(PROVIDER_KINDS)}")
    if not re.match(r"^https?://[^\s/]+", base_url.strip()):
        raise ConfigError("a base URL must start with http:// or https://")
    config.providers[provider_id] = ProviderConfig(
        id=provider_id,
        kind=kind,
        name=name.strip(),
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        keyless=keyless,
    )


def remove_provider(config: Config, provider_id: str) -> None:
    """Remove the user's entry: a custom provider disappears, a built-in one loses its key and
    overrides. Model settings that pointed at it are left for the user to change."""
    config.providers.pop(provider_id, None)


def models_using(config: Config, provider_id: str) -> list[str]:
    """Settings that reference a model of `provider_id` (to warn before removing it)."""
    prefix = f"{provider_id}/"
    used = ["default model"] if config.models.default.startswith(prefix) else []
    used.extend(
        f"{role} model" for role, ref in config.models.roles.items() if ref.startswith(prefix)
    )
    return used

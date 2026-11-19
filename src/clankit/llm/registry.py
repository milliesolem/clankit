"""Turns configuration into live providers and answers "which models can I use?"."""

from __future__ import annotations

import logging
from collections.abc import Callable

from clankit.config.catalog import effective_providers
from clankit.config.schema import Config, ProviderConfig
from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import ModelInfo, ModelRef

log = logging.getLogger(__name__)

ProviderFactory = Callable[[ProviderConfig], Provider]

DEFAULT_CONCURRENCY = 8


def build_provider(config: ProviderConfig) -> Provider:
    """Construct the adapter for a provider configuration."""
    if config.kind == "anthropic":
        from clankit.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    if config.kind == "openai-compatible":
        from clankit.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(config)
    raise ProviderError(
        f"provider {config.id!r} has unknown kind {config.kind!r} "
        "(expected 'anthropic' or 'openai-compatible')",
        kind="bad_request",
        provider=config.id,
    )


class ProviderRegistry:
    """Providers by id, built lazily from the configuration.

    `register()` injects a ready-made provider under an id; injected providers always win and
    survive `reconfigure()`. Tests use this to install a `ScriptedProvider`.
    """

    def __init__(self, config: Config, *, factory: ProviderFactory = build_provider) -> None:
        self._factory = factory
        self._configs: dict[str, ProviderConfig] = {}
        self._default = ""
        self._live: dict[str, tuple[ProviderConfig, Provider]] = {}
        self._injected: dict[str, Provider] = {}
        self._discovered: dict[str, list[ModelInfo]] = {}
        self._stale: list[Provider] = []
        self.reconfigure(config)

    # -- configuration --------------------------------------------------------------------------

    def reconfigure(self, config: Config) -> None:
        """Adopt a new configuration. Clients whose settings changed are rebuilt on next use."""
        self._configs = effective_providers(config)
        self._default = config.models.default
        for provider_id, (old, provider) in list(self._live.items()):
            if self._configs.get(provider_id) != old:
                del self._live[provider_id]
                self._discovered.pop(provider_id, None)
                self._stale.append(provider)

    def register(self, provider: Provider) -> None:
        self._injected[provider.id] = provider

    # -- lookup ---------------------------------------------------------------------------------

    def provider_ids(self, *, usable_only: bool = True) -> list[str]:
        ids = list(self._injected)
        for provider_id, cfg in self._configs.items():
            if provider_id not in ids and (cfg.usable or not usable_only):
                ids.append(provider_id)
        return ids

    def config_for(self, provider_id: str) -> ProviderConfig | None:
        return self._configs.get(provider_id)

    def get(self, provider_id: str) -> Provider:
        if provider_id in self._injected:
            return self._injected[provider_id]
        cfg = self._configs.get(provider_id)
        if cfg is None:
            known = ", ".join(sorted(self._configs)) or "none"
            raise ProviderError(
                f"unknown provider {provider_id!r} (known providers: {known})",
                kind="not_found",
                provider=provider_id,
            )
        if not cfg.enabled:
            raise ProviderError(
                f"provider {provider_id!r} is disabled in the configuration",
                kind="auth",
                provider=provider_id,
            )
        if not cfg.usable:
            hint = f" or set ${cfg.api_key_env}" if cfg.api_key_env else ""
            raise ProviderError(
                f"no API key for {cfg.name or provider_id}: add one in settings{hint}",
                kind="auth",
                provider=provider_id,
            )
        live = self._live.get(provider_id)
        if live is None:
            live = (cfg, self._factory(cfg))
            self._live[provider_id] = live
        return live[1]

    def resolve(self, ref: ModelRef) -> tuple[Provider, str]:
        return self.get(ref.provider), ref.model

    def concurrency(self, provider_id: str) -> int:
        """How many model calls may be in flight at once for a provider."""
        cfg = self._configs.get(provider_id)
        return cfg.concurrency if cfg is not None and cfg.concurrency > 0 else DEFAULT_CONCURRENCY

    # -- models ---------------------------------------------------------------------------------

    def known_models(self) -> list[ModelInfo]:
        """Models of every usable provider, without touching the network.

        Combines models declared in the configuration and catalog with whatever `discover()`
        has fetched so far. Declared facts win over discovered ones.
        """
        models: list[ModelInfo] = []
        for provider_id in self.provider_ids():
            seen: dict[str, ModelInfo] = {}
            cfg = self._configs.get(provider_id)
            for entry in cfg.models if cfg else ():
                seen[entry.id] = ModelInfo(
                    provider=provider_id,
                    id=entry.id,
                    name=entry.name or entry.id,
                    context_window=entry.context_window,
                    max_output=entry.max_output,
                    supports_tools=entry.tools,
                    free=entry.free,
                    note=entry.note,
                    input_price=entry.input_price,
                    output_price=entry.output_price,
                )
            for info in self._discovered.get(provider_id, ()):
                declared = seen.get(info.id)
                if declared is None:
                    seen[info.id] = info
                else:
                    declared.context_window = declared.context_window or info.context_window
                    declared.max_output = declared.max_output or info.max_output
                    if declared.supports_tools is None:
                        declared.supports_tools = info.supports_tools
                    if declared.free is None:
                        declared.free = info.free
                    if declared.input_price is None:
                        declared.input_price = info.input_price
                        declared.output_price = info.output_price
            models.extend(seen.values())
        return models

    def model_info(self, ref: ModelRef) -> ModelInfo | None:
        return next(
            (m for m in self.known_models() if m.provider == ref.provider and m.id == ref.model),
            None,
        )

    async def discover(self, provider_id: str | None = None) -> list[ModelInfo]:
        """Ask providers which models they offer (network). Failures are logged, not raised."""
        targets = [provider_id] if provider_id else self.provider_ids()
        for pid in targets:
            try:
                self._discovered[pid] = await self.get(pid).list_models()
            except ProviderError as exc:
                log.warning("could not list models for %s: %s", pid, exc)
        return self.known_models()

    def default_model(self) -> ModelRef | None:
        """The configured default model if its provider is usable, else the first usable
        provider's preferred model, else None (nothing is configured yet)."""
        usable = self.provider_ids()
        try:
            ref = ModelRef.parse(self._default)
            if ref.provider in usable:
                return ref
        except ValueError:
            pass
        for provider_id in usable:
            cfg = self._configs.get(provider_id)
            preferred = cfg.options.get("default-model") if cfg else None
            if preferred:
                return ModelRef(provider_id, str(preferred))
            if cfg and cfg.models:
                return ModelRef(provider_id, cfg.models[0].id)
            discovered = self._discovered.get(provider_id)
            if discovered:
                return ModelRef(provider_id, discovered[0].id)
        return None

    # -- lifecycle ------------------------------------------------------------------------------

    async def aclose(self) -> None:
        providers = (
            [p for _, p in self._live.values()] + self._stale + list(self._injected.values())
        )
        self._live.clear()
        self._stale.clear()
        for provider in providers:
            try:
                await provider.aclose()
            except Exception:
                log.exception("error closing provider %s", provider.id)

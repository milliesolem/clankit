"""The provider registry's view of models: catalog guesses vs. what servers actually offer."""

from __future__ import annotations

from clankit.config.schema import Config, ModelEntry, ProviderConfig
from clankit.config.store import ConfigStore
from clankit.llm.registry import ProviderRegistry
from clankit.llm.types import ModelInfo
from clankit.runtime import Runtime

OLLAMA_MODELS = {"qwen3.8:27b", "gemma4:12b", "gpt-oss:20b"}


def registry(config: Config) -> ProviderRegistry:
    return ProviderRegistry(config)


def ollama_config(base_url: str = "") -> Config:
    config = Config()
    config.models.default = ""
    entry = ProviderConfig(id="ollama")
    if base_url:
        entry.base_url = base_url
    config.providers["ollama"] = entry
    return config


def ids(registry: ProviderRegistry, provider_id: str) -> set[str]:
    return {model.id for model in registry.known_models_for(provider_id)}


def test_catalog_models_are_suggested_for_the_default_address() -> None:
    models = ids(registry(ollama_config()), "ollama")
    assert models == OLLAMA_MODELS


def test_catalog_models_are_dropped_when_the_base_url_moved() -> None:
    config = ollama_config("http://192.168.137.1:11434/v1")
    assert ids(registry(config), "ollama") == set()


def test_catalog_models_are_dropped_once_the_server_was_asked() -> None:
    reg = registry(ollama_config())
    reg._discovered["ollama"] = [
        ModelInfo(provider="ollama", id="qwen3-coder:30b", name="Qwen3 Coder")
    ]
    assert ids(reg, "ollama") == {"qwen3-coder:30b"}


def test_models_the_user_declared_survive_both_cases() -> None:
    config = ollama_config("http://192.168.137.1:11434/v1")
    config.providers["ollama"].models.append(ModelEntry(id="mine:1", name="Mine"))
    reg = registry(config)
    assert ids(reg, "ollama") == {"mine:1"}
    reg._discovered["ollama"] = [
        ModelInfo(provider="ollama", id="qwen3-coder:30b", name="Qwen3 Coder")
    ]
    assert ids(reg, "ollama") == {"mine:1", "qwen3-coder:30b"}


def test_default_model_prefers_what_the_server_reports() -> None:
    config = ollama_config("http://192.168.137.1:11434/v1")
    reg = registry(config)
    assert reg.default_model() is None
    reg._discovered["ollama"] = [
        ModelInfo(provider="ollama", id="qwen3-coder:30b", name="Qwen3 Coder")
    ]
    assert str(reg.default_model()) == "ollama/qwen3-coder:30b"


def test_default_model_keeps_the_catalog_suggestion_at_the_default_address() -> None:
    reg = registry(ollama_config())
    assert str(reg.default_model()) == "ollama/qwen3.8:27b"


def test_usable_requires_a_working_address() -> None:
    assert ProviderConfig(id="x", keyless=True, base_url="http://localhost:11434/v1").usable
    assert not ProviderConfig(id="x", keyless=True, base_url="not a url").usable
    assert not ProviderConfig(id="x", keyless=True, base_url="").usable
    assert ProviderConfig(id="x", kind="anthropic", api_key="k").usable
    assert ProviderConfig(id="x", keyless=True, base_url="http://localhost:11434/v1").base_url_problem() == ""
    assert "base-url" in ProviderConfig(id="x", keyless=True, base_url="not a url").base_url_problem()


def test_runtime_model_view_excludes_guesses_for_a_custom_ollama() -> None:
    runtime = Runtime(ConfigStore.in_memory(ollama_config("http://192.168.137.1:11434/v1")))
    assert runtime.providers.known_models() == []

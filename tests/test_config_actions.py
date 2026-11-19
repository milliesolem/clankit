"""Provider/key helpers and the declarative settings fields shared by both UIs."""

from __future__ import annotations

import pytest

from clankit.config import actions, fields
from clankit.config.schema import Config, ConfigError


def rows(config: Config) -> dict[str, actions.ProviderRow]:
    return {row.id: row for row in actions.provider_rows(config)}


def test_keys_are_masked_everywhere() -> None:
    key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz-a91f"
    config = Config()
    actions.set_api_key(config, "anthropic", key)
    row = rows(config)["anthropic"]
    assert row.masked_key == "sk-ant-…a91f" and row.key_source == "config" and row.usable
    assert key not in repr(row) and key not in str(row.to_dict())
    assert actions.mask_key("") == "" and actions.mask_key("short") == "…rt"
    assert actions.mask_key("gsk_1234567890abcdef") == "gsk_…cdef"
    assert actions.mask_key("0123456789abcdef") == "012…cdef"


def test_removing_a_key_drops_the_empty_entry() -> None:
    config = Config()
    actions.set_api_key(config, "anthropic", "sk-ant-xxxxxxxxxxxxxxxx")
    actions.set_api_key(config, "anthropic", "")
    assert "anthropic" not in config.providers
    assert rows(config)["anthropic"].key_source == "none"
    # an entry that carries other settings survives
    actions.set_api_key(config, "deepseek", "sk-xxxxxxxxxxxxxxxxxxxx")
    actions.set_base_url(config, "deepseek", "https://proxy.example.com/v1")
    actions.set_api_key(config, "deepseek", "")
    assert config.providers["deepseek"].base_url == "https://proxy.example.com/v1"


def test_key_from_environment_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-0123456789abcdef")
    row = rows(Config())["deepseek"]
    assert row.key_source == "env" and row.usable and row.masked_key.endswith("cdef")


@pytest.mark.parametrize("bad", ["has space", "line\nbreak"])
def test_malformed_keys_are_rejected(bad: str) -> None:
    with pytest.raises(ConfigError):
        actions.set_api_key(Config(), "anthropic", bad)


def test_custom_providers() -> None:
    config = Config()
    actions.add_custom_provider(config, "My-Box", base_url="http://localhost:8000/v1", keyless=True)
    row = rows(config)["my-box"]
    assert (row.builtin, row.key_source, row.usable, row.kind) == (
        False,
        "keyless",
        True,
        "openai-compatible",
    )
    for kwargs in (
        {"provider_id": "my-box", "base_url": "http://x"},  # duplicate
        {"provider_id": "anthropic", "base_url": "http://x"},  # shadows a built-in
        {"provider_id": "9lives", "base_url": "http://x"},  # bad id
        {"provider_id": "ok", "base_url": "localhost:8000"},  # bad url
        {"provider_id": "ok", "base_url": "http://x", "kind": "telepathy"},
    ):
        with pytest.raises(ConfigError):
            actions.add_custom_provider(config, **kwargs)  # type: ignore[arg-type]
    config.models.default = "my-box/some-model"
    config.models.roles["reviewer"] = "my-box/other"
    assert actions.models_using(config, "my-box") == ["default model", "reviewer model"]
    actions.remove_provider(config, "my-box")
    assert "my-box" not in rows(config)


def test_unknown_provider_cannot_get_a_key() -> None:
    with pytest.raises(ConfigError):
        actions.set_api_key(Config(), "nonexistent", "k" * 20)


def test_every_field_resolves_and_round_trips_its_current_value() -> None:
    config = Config()
    paths = [f.path for f in fields.all_fields()]
    assert len(paths) == len(set(paths))
    for field in fields.all_fields():
        if field.kind == "theme":
            continue  # validated against the theme registry, covered by the theme tests
        value = fields.get_value(config, field.path)
        if field.path.startswith("models.roles."):
            assert value == ""
            continue
        fields.set_value(config, field.path, value)
        assert fields.get_value(config, field.path) == value
    assert config == Config()


def test_fields_validate_and_coerce() -> None:
    config = Config()
    fields.set_value(config, "agent.max_turns", "50")
    fields.set_value(config, "ui.show_thinking", "false")
    fields.set_value(config, "agent.compact_at", "0.5")
    fields.set_value(config, "agent.effort", "high")
    fields.set_value(config, "models.roles.reviewer", " anthropic/claude-fable-5-2 ")
    assert (
        config.agent.max_turns,
        config.ui.show_thinking,
        config.agent.compact_at,
        config.agent.effort,
    ) == (
        50,
        False,
        0.5,
        "high",
    )
    assert config.models.roles == {"reviewer": "anthropic/claude-fable-5-2"}
    fields.set_value(config, "models.roles.reviewer", "")
    fields.set_value(config, "agent.effort", "")
    assert config.models.roles == {} and config.agent.effort == ""

    for path, bad in (
        ("agent.max_turns", "abc"),
        ("agent.max_turns", 0),
        ("agent.max_turns", ""),
        ("agent.compact_at", 2),
        ("permissions.mode", "yolo"),
        ("models.default", "no-slash"),
        ("models.default", ""),
        ("web.port", 70_000),
        ("no.such.setting", 1),
    ):
        with pytest.raises(ConfigError):
            fields.set_value(config, path, bad)

"""Reading ``GET /models`` payloads: record what a backend reports, guess nothing."""

from __future__ import annotations

from typing import Any

import pytest

from clankit.llm.compat.models import parse_models
from clankit.llm.provider import ProviderError


def one(entry: dict[str, Any], **kwargs: Any) -> Any:
    (model,) = parse_models({"data": [entry]}, provider="acme", **kwargs)
    return model


def test_openrouter_entry_is_read_in_full() -> None:
    model = one(
        {
            "id": "qwen/qwen3-coder:free",
            "name": "Qwen: Qwen3 Coder (free)",
            "context_length": 262144,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "top_provider": {"context_length": 262144, "max_completion_tokens": 65536},
            "supported_parameters": ["max_tokens", "tools", "tool_choice"],
            "expiration_date": "2026-12-31",
        }
    )
    assert (model.provider, model.id, model.name) == (
        "acme",
        "qwen/qwen3-coder:free",
        "Qwen: Qwen3 Coder (free)",
    )
    assert (model.context_window, model.max_output) == (262144, 65536)
    assert (model.free, model.input_price, model.output_price) == (True, 0.0, 0.0)
    assert model.supports_tools is True and model.note == "expires 2026-12-31"
    assert str(model.ref) == "acme/qwen/qwen3-coder:free"


@pytest.mark.parametrize(
    ("pricing", "free", "prices"),
    [
        pytest.param({"prompt": "0", "completion": "0"}, True, (0.0, 0.0), id="free"),
        pytest.param(
            {"prompt": "0.00000015", "completion": "0.0000006"}, False, (0.15, 0.6), id="paid"
        ),
        pytest.param({"prompt": "0", "completion": "0.000002"}, False, (0.0, 2.0), id="half-free"),
        pytest.param({"prompt": "-1", "completion": "-1"}, None, (None, None), id="router"),
        pytest.param({"prompt": "n/a"}, None, (None, None), id="unreadable"),
        pytest.param(None, None, (None, None), id="absent"),
    ],
)
def test_prices_are_per_million_tokens_and_free_means_both_are_zero(
    pricing: Any, free: bool | None, prices: tuple[float | None, float | None]
) -> None:
    model = one({"id": "m", "pricing": pricing})
    assert model.free is free
    assert (model.input_price, model.output_price) == prices


@pytest.mark.parametrize(
    ("entry", "context", "output"),
    [
        pytest.param({"context_length": 8192}, 8192, None, id="openrouter"),
        pytest.param(
            {"context_window": 1048576, "max_output_tokens": 393216}, 1048576, 393216, id="deepseek"
        ),
        pytest.param(
            {"context_window": 131072, "max_completion_tokens": 16384}, 131072, 16384, id="groq"
        ),
        pytest.param({"max_model_len": 65536}, 65536, None, id="vllm"),
        pytest.param({"context_length": "big", "max_output_tokens": 0}, None, None, id="junk"),
        pytest.param({"context_length": True}, None, None, id="boolean-is-not-a-size"),
        pytest.param({}, None, None, id="ids-only"),
    ],
)
def test_limits_are_read_from_the_fields_backends_use(
    entry: dict[str, Any], context: int | None, output: int | None
) -> None:
    model = one({"id": "m", **entry})
    assert (model.context_window, model.max_output) == (context, output)
    assert model.supports_tools is None and model.free is None and model.note == ""


def test_tool_support_is_known_only_when_parameters_are_listed() -> None:
    assert one({"id": "m", "supported_parameters": ["tools"]}).supports_tools is True
    assert one({"id": "m", "supported_parameters": ["temperature"]}).supports_tools is False
    assert one({"id": "m"}).supports_tools is None


def test_models_that_do_not_write_plain_text_are_left_out() -> None:
    payload = {
        "data": [
            {"id": "chat", "architecture": {"output_modalities": ["text"]}},
            {"id": "music", "architecture": {"output_modalities": ["text", "audio"]}},
            {"id": "image", "architecture": {"output_modalities": ["image"]}},
            {"id": "unspecified"},
        ]
    }
    assert [m.id for m in parse_models(payload, provider="acme")] == ["chat", "unspecified"]


def test_listed_prefix_is_removed_from_ids() -> None:
    payload = {"data": [{"id": "models/gemini-4-hydrogen"}, {"id": "tunedModels/mine"}]}
    models = parse_models(payload, provider="gemini", prefix="models/")
    assert [m.id for m in models] == ["gemini-4-hydrogen", "tunedModels/mine"]


def test_bare_lists_and_odd_entries_are_tolerated() -> None:
    models = parse_models(["a", {"id": "b"}, {"name": "no id"}, {"id": 7}, None], provider="acme")
    assert [m.id for m in models] == ["a", "b"]
    assert parse_models({"data": []}, provider="acme") == []


@pytest.mark.parametrize(
    "payload", [{"object": "error", "message": "nope"}, "<html></html>", None, {"data": {"id": "x"}}]
)
def test_payload_that_is_not_a_model_list_is_an_error(payload: Any) -> None:
    with pytest.raises(ProviderError) as caught:
        parse_models(payload, provider="acme")
    assert caught.value.provider == "acme" and "unexpected shape" in caught.value.message

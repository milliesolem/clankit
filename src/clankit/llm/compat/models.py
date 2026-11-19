"""``GET /models`` payloads to `ModelInfo`.

Every backend lists ids. A few say more, and only what a backend actually reports is
recorded: OpenRouter gives limits, prices, supported parameters and an expiry date for
models about to be withdrawn; DeepSeek, Groq and vLLM give context sizes. Unknown stays None.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clankit.llm.provider import ProviderError
from clankit.llm.types import ModelInfo

_CONTEXT_KEYS = ("context_length", "context_window", "max_model_len")
_OUTPUT_KEYS = ("max_completion_tokens", "max_output_tokens")


def parse_models(payload: object, *, provider: str, prefix: str = "") -> list[ModelInfo]:
    """Read a model listing. `prefix` is removed from ids that carry it (Gemini lists
    ``models/<id>`` but is called with ``<id>``)."""
    entries = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ProviderError(
            "the model list has an unexpected shape (expected a list of models under 'data')",
            kind="unknown",
            provider=provider,
        )
    models = (_model(entry, provider, prefix) for entry in entries)
    return [model for model in models if model is not None]


def _model(entry: object, provider: str, prefix: str) -> ModelInfo | None:
    if isinstance(entry, str):
        entry = {"id": entry}
    if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
        return None
    if not _writes_text(entry):
        return None
    top = entry.get("top_provider")
    input_price, output_price = _prices(entry.get("pricing"))
    parameters = entry.get("supported_parameters")
    expires = entry.get("expiration_date")
    return ModelInfo(
        provider=provider,
        id=entry["id"].removeprefix(prefix),
        name=str(entry.get("name") or entry.get("display_name") or ""),
        context_window=_first_int(entry, _CONTEXT_KEYS),
        max_output=_first_int(entry, _OUTPUT_KEYS)
        or (_first_int(top, _OUTPUT_KEYS) if isinstance(top, Mapping) else None),
        supports_tools="tools" in parameters if isinstance(parameters, list) else None,
        free=None
        if input_price is None or output_price is None
        else input_price + output_price == 0,
        note=f"expires {expires}" if isinstance(expires, str) and expires else "",
        input_price=input_price,
        output_price=output_price,
    )


def _writes_text(entry: Mapping[str, Any]) -> bool:
    """False for models that declare an output other than text (image and audio generators)."""
    architecture = entry.get("architecture")
    modalities = (
        architecture.get("output_modalities") if isinstance(architecture, Mapping) else None
    )
    return not isinstance(modalities, list) or modalities == ["text"]


def _prices(pricing: object) -> tuple[float | None, float | None]:
    """US dollars per million tokens from OpenRouter's per-token strings.

    Routers that choose a model per request report a negative price, meaning "it depends";
    that is unknown, not free.
    """
    if not isinstance(pricing, Mapping):
        return None, None
    prices: list[float | None] = []
    for key in ("prompt", "completion"):
        try:
            per_token = float(pricing.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            per_token = -1.0
        prices.append(round(per_token * 1_000_000, 6) if per_token >= 0 else None)
    return prices[0], prices[1]


def _first_int(entry: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None

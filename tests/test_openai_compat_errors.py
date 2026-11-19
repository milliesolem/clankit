"""Classifying backend failures: retry, shrink the conversation, or stop and tell the user."""

from __future__ import annotations

import time
from email.utils import formatdate
from typing import Any

import pytest

from clankit.llm.compat.errors import MAX_RETRY_WAIT, status_error, stream_error


def body(message: str, **fields: Any) -> dict[str, Any]:
    """An error the way the SDK hands it over: the inside of ``{"error": {...}}``."""
    return {"message": message, **fields}


# --------------------------------------------------------------------------------------------
# By status
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "bad_request"),
        (401, "auth"),
        (403, "auth"),
        (404, "not_found"),
        (408, "network"),
        (409, "bad_request"),
        (413, "context_overflow"),
        (422, "bad_request"),
        (498, "overloaded"),
        (500, "overloaded"),
        (502, "overloaded"),
        (503, "overloaded"),
        (504, "overloaded"),
        (529, "overloaded"),
    ],
)
def test_status_decides_when_the_body_says_nothing_special(status: int, kind: str) -> None:
    error = status_error("acme", status, body("Something happened."), {})
    assert (error.kind, error.status, error.provider) == (kind, status, "acme")
    assert "Something happened." in error.message and f"HTTP {status}" in error.message
    assert error.retryable is (kind in ("overloaded", "network"))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("<html><head><title>502 Bad Gateway</title></head></html>", id="html"),
        pytest.param(None, id="no-body"),
        pytest.param({"detail": "Not authenticated"}, id="fastapi-detail"),
        pytest.param({"errors": [{"code": 7003, "message": "No route"}]}, id="cloudflare"),
        pytest.param([{"error": {"code": 500, "message": "Internal"}}], id="gemini-array"),
        pytest.param("x" * 5000, id="huge"),
    ],
)
def test_any_error_body_yields_one_short_line(payload: Any) -> None:
    error = status_error("acme", 500, payload, {})
    assert error.kind == "overloaded"
    assert "\n" not in error.message and len(error.message) < 500


def test_upstream_explanation_behind_a_router_is_included() -> None:
    payload = body("Provider returned error", code=400, metadata={"raw": "maximum context length"})
    error = status_error("openrouter", 400, payload, {})
    assert error.kind == "context_overflow" and "maximum context length" in error.message


# --------------------------------------------------------------------------------------------
# The prompt is too long
# --------------------------------------------------------------------------------------------

OVERFLOW_MESSAGES = [
    "This model's maximum context length is 131072 tokens. However, you requested 140211 "
    "tokens (132019 in the messages, 8192 in the completion).",
    "This endpoint's maximum context length is 262144 tokens. However, you requested about "
    "300000 tokens.",
    "Please reduce the length of the messages or completion. Current length is 70000 while "
    "limit is 65536",
    "prompt is too long: 213462 tokens > 200000 maximum",
    "The input token count (1300000) exceeds the maximum number of tokens allowed (1048576).",
    "the request exceeds the available context size, try increasing it",
    "Trying to keep the first 9000 tokens when context the overflows. However, the model is "
    "loaded with context length of only 4096 tokens, which is not enough.",
    "Input is too long for requested model.",
    "too many tokens: the request has 9000 tokens",
    "Prompt contains 40000 tokens, too large for model with 32768 maximum context length",
    "The prompt is too long: 300000, model maximum context length: 131072",
    "inputs tokens + max_new_tokens must be <= 4097",
    "This model's maximum prompt length is 131072 but the request contains 140000 tokens.",
    "Your input exceeds the context window of this model.",
    "Request too large for model `qwen/qwen3.8-27b` on tokens per minute (TPM): Limit 8000, "
    "Requested 12000",
    "Prompt exceeds max length",
]


@pytest.mark.parametrize("message", OVERFLOW_MESSAGES)
@pytest.mark.parametrize("status", [400, 422])
def test_context_overflow_phrasings(message: str, status: int) -> None:
    error = status_error("acme", status, body(message), {})
    assert error.kind == "context_overflow" and not error.retryable


@pytest.mark.parametrize(
    "fields",
    [
        {"code": "context_length_exceeded"},
        {"type": "exceed_context_size_error"},
        {"type": "request_too_large"},
    ],
)
def test_context_overflow_is_recognised_by_code_or_type(fields: dict[str, str]) -> None:
    assert status_error("acme", 400, body("Rejected.", **fields), {}).kind == "context_overflow"


@pytest.mark.parametrize(
    "message",
    [
        "max_tokens is too large: 500000. This model supports at most 8192 completion tokens.",
        "Invalid value for 'tools[0].function.name'.",
        "Unsupported parameter: 'stream_options'.",
        "messages: roles must alternate between user and assistant",
    ],
)
def test_other_bad_requests_are_not_mistaken_for_overflow(message: str) -> None:
    assert status_error("acme", 400, body(message), {}).kind == "bad_request"


@pytest.mark.parametrize(
    "payload",
    [
        body("Model Not Exist"),
        body("The model `gpt-9` does not exist or you do not have access to it."),
        body("nope/model is not a valid model ID"),
        body("Invalid model specified"),
        body('model "qwen3.8:27b" not found, try pulling it first'),
        body("Rejected.", code="model_not_found"),
    ],
)
def test_unknown_model_reported_as_a_bad_request_is_not_found(payload: dict[str, Any]) -> None:
    error = status_error("acme", 400, payload, {})
    assert error.kind == "not_found" and "model id" in error.message


# --------------------------------------------------------------------------------------------
# Rate limits, quotas and balances
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "message", "expected"),
    [
        pytest.param({"retry-after": "7"}, "Slow down.", 7.0, id="seconds"),
        pytest.param({"retry-after": "0.5"}, "Slow down.", 0.5, id="fractional-seconds"),
        pytest.param({"retry-after-ms": "1500"}, "Slow down.", 1.5, id="milliseconds"),
        pytest.param(
            {"retry-after-ms": "250", "retry-after": "9"}, "Slow down.", 0.25, id="ms-wins"
        ),
        pytest.param({}, "Please try again in 7.66s.", 7.66, id="groq-wording"),
        pytest.param({}, "Rate limit hit. Try again in 1m2.5s", 62.5, id="compound-duration"),
        pytest.param({}, "Please retry in 850ms", 0.85, id="milliseconds-in-text"),
        pytest.param({}, "Please retry after 30 seconds", 30.0, id="words"),
        pytest.param({}, "Rate Limit Reached", None, id="nothing-to-go-on"),
        pytest.param({"retry-after": "soon"}, "Slow down.", None, id="unreadable-header"),
    ],
)
def test_wait_before_retrying_is_read_from_headers_or_text(
    headers: dict[str, str], message: str, expected: float | None
) -> None:
    error = status_error("acme", 429, body(message), headers)
    assert error.kind == "rate_limit" and error.retryable
    assert error.retry_after == pytest.approx(expected)


def test_wait_given_as_a_date_or_an_absolute_reset_time() -> None:
    soon = time.time() + 40
    by_date = status_error("acme", 429, body("x"), {"retry-after": formatdate(soon, usegmt=True)})
    by_reset = status_error("acme", 429, body("x"), {"x-ratelimit-reset": str(int(soon * 1000))})
    assert by_date.kind == by_reset.kind == "rate_limit"
    assert by_date.retry_after == pytest.approx(40, abs=2)
    assert by_reset.retry_after == pytest.approx(40, abs=2)


def test_gemini_retry_info_is_read_from_the_error_details() -> None:
    payload = [
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "You exceeded your current quota, please check your plan and billing "
                "details.",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
                ],
            }
        }
    ]
    error = status_error("gemini", 429, payload, {})
    assert error.kind == "rate_limit" and error.retry_after == 34.0


@pytest.mark.parametrize(
    ("payload", "headers"),
    [
        pytest.param(
            body(
                "You exceeded your current quota, please check your plan and billing details.",
                type="insufficient_quota",
                code="insufficient_quota",
            ),
            {},
            id="openai-insufficient-quota",
        ),
        pytest.param(
            body(
                "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free "
                "model requests per day",
                code=429,
            ),
            {},
            id="openrouter-daily-free-limit",
        ),
        pytest.param(
            body("Rate limit reached on tokens per day (TPD). Please try again in 2h30m."),
            {"retry-after": "9000"},
            id="groq-daily-with-long-wait",
        ),
        pytest.param(
            body("Rate limit exceeded"),
            {"x-ratelimit-reset": str(int((time.time() + 6 * 3600) * 1000))},
            id="reset-hours-away",
        ),
        pytest.param(body("You have reached your spending limit."), {}, id="spend-cap"),
        pytest.param(
            [{"error": {"code": 429, "message": "Quota exceeded for metric requests_per_day"}}],
            {},
            id="gemini-daily",
        ),
    ],
)
def test_quota_that_will_not_return_soon_is_not_retried(
    payload: Any, headers: dict[str, str]
) -> None:
    error = status_error("acme", 429, payload, headers)
    assert error.kind == "auth" and not error.retryable and error.status == 429
    assert error.retry_after is None
    assert "quota exhausted" in error.message and "Not retried" in error.message


def test_long_wait_is_reported_in_the_message() -> None:
    error = status_error("groq", 429, body("Daily limit."), {"retry-after": "9000"})
    assert "resets in about 2 h" in error.message
    shorter = status_error("groq", 429, body("x"), {"retry-after": str(MAX_RETRY_WAIT + 60)})
    assert "resets in about 6 min" in shorter.message


def test_wait_at_the_limit_is_still_a_rate_limit() -> None:
    error = status_error("acme", 429, body("x"), {"retry-after": str(MAX_RETRY_WAIT)})
    assert error.kind == "rate_limit" and error.retry_after == MAX_RETRY_WAIT


def test_quota_wording_with_a_short_wait_stays_retryable() -> None:
    """Gemini and Groq mention quotas and billing in ordinary per-minute rate limits."""
    message = (
        "Rate limit reached for model in organization on tokens per minute (TPM). Please try "
        "again in 7.66s. Need more tokens? Upgrade at https://console.example.test/billing"
    )
    error = status_error("groq", 429, body(message), {"retry-after": "8"})
    assert error.kind == "rate_limit" and error.retry_after == 8.0


def test_insufficient_balance_is_stated_plainly() -> None:
    error = status_error("deepseek", 402, body("Insufficient Balance"), {})
    assert error.kind == "auth" and not error.retryable and error.status == 402
    assert error.message == (
        "insufficient balance: Insufficient Balance. "
        "Add credit to the account; retrying will not help"
    )
    assert "insufficient balance" in status_error("acme", 402, None, {}).message


def test_spending_hold_that_clears_shortly_is_a_rate_limit() -> None:
    payload = body(
        "This request would exceed your available credits given your current in-flight requests.",
        code=402,
        metadata={"reason": "in_flight_budget_exhausted"},
    )
    held = status_error("openrouter", 402, payload, {"retry-after": "3"})
    assert held.kind == "rate_limit" and held.retry_after == 3.0
    # ...whereas a request that can never fit has no Retry-After and stays fatal.
    assert status_error("openrouter", 402, payload, {}).kind == "auth"


def test_groq_failed_generation_is_worth_another_attempt() -> None:
    payload = body(
        "Invalid tool call generated",
        type="invalid_request_error",
        failed_generation={"reason": "not valid JSON", "attempted_arguments": "{'a': 1}"},
    )
    error = status_error("groq", 400, payload, {})
    assert error.kind == "overloaded" and error.retryable
    assert "the model produced an invalid tool call" in error.message


# --------------------------------------------------------------------------------------------
# Errors inside a successful response
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "kind", "status"),
    [
        pytest.param(body("Provider disconnected", code=502), "overloaded", 502, id="5xx-code"),
        pytest.param(body("Rate limit exceeded", code=429), "rate_limit", 429, id="429-code"),
        pytest.param(body("No credits", code=402), "auth", 402, id="402-code"),
        pytest.param(body("timeout", type="server_error", code="x"), "overloaded", None, id="type"),
        pytest.param(body("rate_limit_exceeded"), "rate_limit", None, id="rate-limit-text"),
        pytest.param(body("Invalid API key provided"), "auth", None, id="auth-text"),
        pytest.param(body("bad", type="invalid_request_error"), "bad_request", None, id="invalid"),
        pytest.param(body("context_length_exceeded"), "context_overflow", None, id="overflow"),
        pytest.param("upstream connect error", "overloaded", None, id="bare-string"),
        pytest.param(None, "overloaded", None, id="nothing"),
        pytest.param({"error": {"message": "wrapped", "code": 503}}, "overloaded", 503, id="wrap"),
    ],
)
def test_in_stream_errors(payload: Any, kind: str, status: int | None) -> None:
    error = stream_error("acme", payload)
    assert (error.kind, error.status, error.provider) == (kind, status, "acme")

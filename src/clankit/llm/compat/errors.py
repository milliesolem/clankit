"""Failures of OpenAI-compatible backends, classified for the agent loop.

The loop only needs to know which of three things to do: wait and retry, shrink the
conversation, or stop and tell the user. Backends agree on little beyond the HTTP status, so
the status decides first and the error text refines it:

* 402, and a 429 whose quota will not return soon, are *not* rate limits. Retrying them
  spends the user's time for nothing, so they become "auth" (the account must be fixed) and
  say so plainly.
* "The prompt is too long" has a dozen phrasings and usually arrives as a plain 400.
* An error object inside a 200 stream has no status at all.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx2
import openai

from clankit.llm.provider import ErrorKind, ProviderError

#: A wait longer than this is treated as an exhausted quota rather than a rate limit: an
#: agent that sleeps for hours looks hung.
MAX_RETRY_WAIT = 300.0

_CONTEXT_OVERFLOW = re.compile(
    r"context[ _-]?(length|window|size|limit|overflow)"
    r"|maximum context|max(imum)? prompt length"
    r"|too many (input |prompt )?tokens"
    r"|(prompt|input|request|messages?) (is |are )?too (long|large)"
    r"|token count .{0,60}exceeds"
    r"|exceeds? .{0,40}(token limit|maximum number of tokens|max(imum)? (length|tokens))"
    r"|reduce the length|request_too_large|max_new_tokens",
    re.IGNORECASE,
)
_UNKNOWN_MODEL = re.compile(
    r"model[ _]not[ _](found|exist)"
    r"|model .{0,80}(does not exist|not found|is not available|is not a valid)"
    r"|(unknown|invalid|no such) model|not a valid model|no endpoints found",
    re.IGNORECASE,
)
_QUOTA = re.compile(
    r"insufficient[ _]quota|quota[ _]exceeded|exceeded your (current )?quota"
    r"|per[ _-]?day|daily (limit|quota)|spend(ing)? (limit|cap)|billing|credits?\b",
    re.IGNORECASE,
)
_WAIT_PHRASE = re.compile(
    r"(?:try again|retry) (?:in|after) ((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)\s*)+)"
    r'|retrydelay\\?"\s*:\s*\\?"(\d+(?:\.\d+)?s)',
    re.IGNORECASE,
)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_STREAM_KINDS: tuple[tuple[re.Pattern[str], ErrorKind], ...] = (
    (_CONTEXT_OVERFLOW, "context_overflow"),
    (re.compile(r"rate[ _]limit", re.IGNORECASE), "rate_limit"),
    (re.compile(r"authenticat|unauthori[sz]ed|invalid[ _]api[ _]key", re.IGNORECASE), "auth"),
    (re.compile(r"invalid[ _]request", re.IGNORECASE), "bad_request"),
)

_LABELS: dict[ErrorKind, str] = {
    "auth": "the API rejected this account or key (check the key and its permissions)",
    "rate_limit": "rate limited",
    "overloaded": "the API is temporarily unavailable",
    "context_overflow": "the conversation no longer fits in the model's context window",
    "bad_request": "the API rejected the request",
    "not_found": "not found (check the model id, and the provider's base-url)",
    "network": "could not reach the API",
    "unknown": "the API reported an error",
}


def translate(exc: Exception, *, provider: str, base_url: str) -> ProviderError:
    """Turn anything the SDK or the transport raised into a `ProviderError`."""
    if isinstance(exc, openai.APIStatusError):
        return status_error(provider, exc.status_code, exc.body, exc.response.headers)
    if isinstance(exc, (openai.APITimeoutError, httpx2.TimeoutException)):
        return ProviderError(
            f"{_LABELS['network']} at {base_url}: the request timed out",
            kind="network",
            provider=provider,
        )
    if isinstance(exc, (openai.APIConnectionError, httpx2.HTTPError, httpx2.StreamError)):
        # The SDK wraps failures to connect; a connection lost mid-stream can arrive raw.
        cause = exc.__cause__ if isinstance(exc, openai.APIConnectionError) else exc
        return ProviderError(
            f"{_LABELS['network']} at {base_url}: {_brief(cause or exc)}{_local_hint(base_url)}",
            kind="network",
            provider=provider,
        )
    if isinstance(exc, openai.APIError):
        return stream_error(provider, exc.body)  # an error object inside a 200 stream
    if isinstance(exc, ValueError):
        # A frame that is not JSON. Asking again usually yields a clean stream.
        return ProviderError(
            f"the response stream could not be decoded: {_brief(exc)}",
            kind="network",
            provider=provider,
        )
    return ProviderError(
        f"unexpected {type(exc).__name__}: {_brief(exc)}", kind="unknown", provider=provider
    )


def status_error(
    provider: str, status: int, body: object, headers: Mapping[str, str]
) -> ProviderError:
    """Classify an HTTP error response. `body` is the decoded error body, in any wrapping."""
    detail = _detail(body)
    haystack = _haystack(body)
    wait = _retry_wait(headers, haystack)
    kind = _kind_for_status(status)
    if status == 402 and wait is not None and wait <= MAX_RETRY_WAIT:
        # OpenRouter: spending is momentarily capped by requests still in flight.
        return _error(provider, "rate_limit", detail, status=status, retry_after=wait)
    if status == 402:
        return ProviderError(
            f"insufficient balance: {detail or 'the account has no credit left'}. "
            "Add credit to the account; retrying will not help",
            kind="auth",
            provider=provider,
            status=status,
        )
    if status == 429:
        return _rate_limit(provider, detail, haystack, wait)
    if kind == "bad_request":
        if _CONTEXT_OVERFLOW.search(haystack):
            kind = "context_overflow"
        elif _UNKNOWN_MODEL.search(haystack):
            kind = "not_found"
        elif "failed_generation" in haystack:
            # Groq: the model wrote a tool call the backend could not parse. The request
            # was fine; sampling again usually produces a valid call.
            return ProviderError(
                f"the model produced an invalid tool call: {detail}",
                kind="overloaded",
                provider=provider,
                status=status,
            )
    return _error(provider, kind, detail, status=status, retry_after=wait)


def stream_error(provider: str, body: object) -> ProviderError:
    """Classify an error object delivered inside a successful (HTTP 200) response."""
    error = _error_object(body)
    code = error.get("code")
    if isinstance(code, int) and 400 <= code < 600:
        return status_error(provider, code, body, {})  # OpenRouter passes the status along
    haystack = _haystack(body)
    # Without a status the text decides; failing that, the usual cause is a backend that
    # broke off part-way, which is worth another attempt.
    kind = next((kind for pattern, kind in _STREAM_KINDS if pattern.search(haystack)), "overloaded")
    return _error(provider, kind, _detail(body))


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------


def _kind_for_status(status: int) -> ErrorKind:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 413:
        return "context_overflow"  # includes a single request above a tokens-per-minute cap
    if status == 408:
        return "network"
    if status >= 498:  # 5xx, and Groq's 498 "capacity exceeded"
        return "overloaded"
    return "bad_request"


def _rate_limit(provider: str, detail: str, haystack: str, wait: float | None) -> ProviderError:
    """A 429 is a rate limit unless its quota will not come back soon."""
    exhausted = wait > MAX_RETRY_WAIT if wait is not None else bool(_QUOTA.search(haystack))
    if not exhausted:
        return _error(provider, "rate_limit", detail, status=429, retry_after=wait)
    resets = f" It resets in about {_duration(wait)}." if wait is not None else ""
    return ProviderError(
        f"quota exhausted: {detail or 'the provider reports no remaining quota'}.{resets} "
        "Not retried: wait for the quota to reset, add credit, or use another model",
        kind="auth",
        provider=provider,
        status=429,
    )


def _error(
    provider: str,
    kind: ErrorKind,
    detail: str,
    *,
    status: int | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    message = f"{_LABELS[kind]}: {detail}" if detail else _LABELS[kind]
    if status is not None:
        message += f" (HTTP {status})"
    return ProviderError(
        message, kind=kind, provider=provider, status=status, retry_after=retry_after
    )


# --------------------------------------------------------------------------------------------
# Reading error bodies
# --------------------------------------------------------------------------------------------


def _error_object(body: object) -> Mapping[str, Any]:
    """The innermost error description, whatever it is wrapped in: ``{"error": {...}}``,
    Gemini's ``[{"error": {...}}]``, Cloudflare's ``{"errors": [{...}]}``."""
    for _ in range(3):
        if isinstance(body, list) and body:
            body = body[0]
        elif isinstance(body, Mapping) and isinstance(body.get("error"), (Mapping, list)):
            body = body["error"]
        elif isinstance(body, Mapping) and isinstance(body.get("errors"), list) and body["errors"]:
            body = body["errors"][0]
        else:
            break
    return body if isinstance(body, Mapping) else {}


def _detail(body: object) -> str:
    """The backend's own explanation, as one short line."""
    error = _error_object(body)
    text = next(
        (
            error[key]
            for key in ("message", "detail", "error", "msg")
            if isinstance(error.get(key), str)
        ),
        body if isinstance(body, str) else "",
    )
    metadata = error.get("metadata")
    upstream = metadata.get("raw") if isinstance(metadata, Mapping) else None
    if isinstance(upstream, str) and upstream.strip():
        text = f"{text}: {upstream}"  # OpenRouter: what the provider behind it said
    return _brief(text)


def _haystack(body: object) -> str:
    """Everything in the body as searchable text (codes and types included)."""
    return body if isinstance(body, str) else json.dumps(body, default=str)


def _retry_wait(headers: Mapping[str, str], haystack: str) -> float | None:
    """Seconds until a retry can succeed, from the headers or failing that the error text."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name, "")
        try:
            return max(0.0, float(value) * scale)
        except ValueError:
            pass
        if name == "retry-after" and value:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    try:
        reset = float(headers.get("x-ratelimit-reset", ""))
    except ValueError:
        reset = 0.0
    if reset > 1e9:  # an absolute time: seconds, or milliseconds, since the epoch
        return max(0.0, (reset / 1000 if reset > 1e11 else reset) - time.time())
    phrase = _WAIT_PHRASE.search(haystack)
    if not phrase:
        return None
    parts = _DURATION_PART.findall(phrase.group(1) or phrase.group(2))
    return sum(float(amount) * _UNIT_SECONDS[unit.lower()] for amount, unit in parts)


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{max(minutes, 1)} min"
    return f"{round(minutes / 60)} h"


def _local_hint(base_url: str) -> str:
    host = urlsplit(base_url).hostname or ""
    return " - is the server running?" if host in ("localhost", "127.0.0.1", "::1") else ""


def _brief(value: object, limit: int = 300) -> str:
    """One short line: error bodies from gateways can be whole HTML pages."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}..."

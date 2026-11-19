"""Keeping credentials out of everything the skill generator sends to a model or writes.

A session transcript is full of places a secret can hide: a command line with a token in it,
an error message that echoes an environment variable, a script with a hard-coded key. The
generator passes all text through a `Redactor` before a model sees it, and refuses to save a
skill in which the redactor still finds something.

Two layers, exact before approximate:

1. every secret the process actually knows (provider API keys from the configuration and
   the environment, and environment variables whose names mark them as secret) is removed by
   exact match, whatever it looks like;
2. anything shaped like a well-known credential (provider key prefixes, tokens in
   ``Authorization`` headers, private key blocks, password assignments) is removed by pattern.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

from clankit.config.catalog import effective_providers
from clankit.config.schema import Config

REDACTED = "[REDACTED]"
#: Shorter values are too likely to be ordinary words to remove by exact match.
MIN_SECRET_CHARS = 8

_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL", re.IGNORECASE)

#: Patterns whose whole match is a credential.
_WHOLE = tuple(
    re.compile(pattern, flags)
    for pattern, flags in (
        (
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
        (r"\bsk-[A-Za-z0-9_-]{16,}", 0),  # Anthropic, OpenAI, DeepSeek, OpenRouter
        (r"\bgh[pousr]_[A-Za-z0-9]{20,}", 0),
        (r"\bgithub_pat_[A-Za-z0-9_]{20,}", 0),
        (r"\bglpat-[A-Za-z0-9_-]{16,}", 0),
        (r"\bhf_[A-Za-z0-9]{20,}", 0),
        (r"\bxox[abprs]-[A-Za-z0-9-]{10,}", 0),
        (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0),
        (r"\bAIza[0-9A-Za-z_-]{30,}", 0),
        (r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", 0),  # JWT
    )
)
#: Patterns where only group 1 is the credential and the text around it is kept.
_PARTIAL = tuple(
    re.compile(pattern, flags)
    for pattern, flags in (
        (r"\bBearer\s+([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE),
        (r"://[^/\s:@]+:([^/\s:@]{4,})@", 0),  # password in a URL
        # a quoted literal assigned to something named like a secret
        (
            r"""\b[\w-]*(?:api[_-]?key|secret|token|passw(?:or)?d)[\w-]*["']?\s*[:=]\s*"""
            r"""["']([^"'\s]{12,})["']""",
            re.IGNORECASE,
        ),
        # NAME_TOKEN=value on a command line or in a shell script (not a $VARIABLE reference)
        (
            r"\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*="
            r"(?![$\"'])((?=[^\s\"']*\d)(?=[^\s\"']*[A-Za-z])[^\s\"']{16,})",
            0,
        ),
    )
)


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        # Longest first, so a secret that contains another is removed whole.
        self._secrets = sorted(
            {secret for secret in secrets if len(secret) >= MIN_SECRET_CHARS},
            key=len,
            reverse=True,
        )

    @classmethod
    def for_config(cls, config: Config, environ: Mapping[str, str] | None = None) -> Redactor:
        """A redactor that knows every credential this process can see."""
        env = os.environ if environ is None else environ
        secrets = [provider.resolved_api_key for provider in effective_providers(config).values()]
        # A variable such as GOOGLE_APPLICATION_CREDENTIALS holds a path, which is not itself
        # a secret; removing every mention of that path would only blur the record.
        secrets.extend(
            value
            for name, value in env.items()
            if _SECRET_NAME_RE.search(name) and not os.path.exists(value)
        )
        return cls(secrets)

    def scrub(self, text: str) -> str:
        """`text` with every known or credential-shaped secret replaced by a marker."""
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for pattern in _WHOLE:
            text = pattern.sub(REDACTED, text)
        for pattern in _PARTIAL:
            text = pattern.sub(_mask_group, text)
        return text

    def finds_secret(self, text: str) -> bool:
        return self.scrub(text) != text


def _mask_group(match: re.Match[str]) -> str:
    start, end = match.span(1)
    whole = match.group(0)
    offset = match.start()
    return whole[: start - offset] + REDACTED + whole[end - offset :]

"""The adapter for every backend that speaks OpenAI's chat-completions protocol.

That covers DeepSeek, OpenRouter, Gemini's compatibility endpoint, Groq, NVIDIA, Z.ai,
Cloudflare, Cerebras and the local runtimes (Ollama, LM Studio, llama.cpp, vLLM), plus any
server a user adds by base URL. What differs between them lives in `clankit.llm.compat`;
this module is the pipeline: build the request, send it, fold the chunks into events.

Provider options read here (``ProviderConfig.options``):

* ``tool-mode``        - "native" (default), "sequential" (one tool call per turn) or "text"
  (tools offered through the prompt, for models without usable native tool calling)
* ``model-tool-modes`` - a mapping of model id to tool mode, overriding ``tool-mode``
* ``profile``          - treat this provider like a built-in one ("deepseek", "ollama", ...)
  when its id and host do not identify what is behind it
* a ``{placeholder}`` in the base URL is filled from the option of the same name written
  with dashes (``{account_id}`` from ``account-id``), else from the environment variable
  ``<PROVIDER ID>_<PLACEHOLDER>`` (``CLOUDFLARE_ACCOUNT_ID``)

The official ``openai`` SDK does the HTTP and the SSE framing, but it is driven below its
typed surface (``client.post(..., cast_to=object)``): backends add fields the typed models
would hide, and those fields must be stored and sent back. The SDK is also built for
OpenAI's own endpoint and by default attaches the user's ambient OpenAI identity
(``OPENAI_ORG_ID``, ``OPENAI_PROJECT_ID``, ``OPENAI_CUSTOM_HEADERS``) to every request; the
client is constructed so that none of it reaches a third party.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any

import httpx2
import openai
from openai import AsyncOpenAI, AsyncStream, Omit

from clankit.config.schema import ProviderConfig
from clankit.llm.compat import errors
from clankit.llm.compat.models import parse_models
from clankit.llm.compat.profiles import profile_for, tool_mode_for
from clankit.llm.compat.stream import StreamAccumulator
from clankit.llm.compat.wire import KIND, build_body
from clankit.llm.provider import Provider, ProviderError
from clankit.llm.types import ChatRequest, ModelInfo, StreamEvent

#: The SDK refuses to be constructed without a key. Requests to keyless servers carry no
#: Authorization header at all, so this value never leaves the process.
_NO_KEY = "unused"

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class OpenAICompatibleProvider(Provider):
    kind = KIND

    def __init__(
        self, config: ProviderConfig, *, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        super().__init__(config.id)
        self._config = config
        self._http_client = http_client
        self._client: AsyncOpenAI | None = None

    # -- Provider -------------------------------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        client = self._connect()
        profile = profile_for(self._config)
        mode = tool_mode_for(self._config, request.model)
        body = build_body(request, provider=self.id, profile=profile, mode=mode)
        accumulator = StreamAccumulator(
            provider=self.id, model=request.model, mode=mode, tools=request.tools
        )
        try:
            response = await client.post(
                "/chat/completions",
                body=body,
                cast_to=object,
                stream=True,
                stream_cls=AsyncStream[object],
                options=self._request_options(),
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        try:
            async with aclosing(self._chunks(response)) as chunks:
                async for chunk in chunks:
                    for event in accumulator.feed(chunk):
                        yield event
        finally:
            # Also reached when the consumer stops early or is cancelled: the connection
            # must not keep streaming (and billing) a response nobody reads.
            await response.close()
        for event in accumulator.finish():
            yield event

    async def list_models(self) -> list[ModelInfo]:
        client = self._connect()
        try:
            payload = await client.get("/models", cast_to=object, options=self._request_options())
        except Exception as exc:
            raise self._translate(exc) from exc
        prefix = profile_for(self._config).model_prefix
        return parse_models(payload, provider=self.id, prefix=prefix)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
        elif self._http_client is not None:
            await self._http_client.aclose()

    # -- internals ------------------------------------------------------------------------------

    def _connect(self) -> AsyncOpenAI:
        """The SDK client, built on first use so that a configuration problem surfaces as a
        `ProviderError` from the call that needed it."""
        if self._client is None:
            key = self._config.resolved_api_key
            if not key and not self._config.keyless:
                hint = f" or set ${self._config.api_key_env}" if self._config.api_key_env else ""
                raise ProviderError(
                    f"no API key for {self._config.name or self.id}: add one in settings{hint}",
                    kind="auth",
                    provider=self.id,
                )
            self._client = AsyncOpenAI(
                api_key=key or _NO_KEY,
                base_url=self._base_url(),
                # The agent loop owns retries; a second layer here would multiply the waits.
                max_retries=0,
                default_headers=_isolating_headers(key),
                http_client=self._http_client,
            )
        return self._client

    def _base_url(self) -> str:
        template = self._config.base_url.strip()
        if not template:
            raise ProviderError(
                f"provider {self.id!r} has no base-url; set the URL of its "
                "OpenAI-compatible endpoint (for example http://localhost:8000/v1)",
                kind="bad_request",
                provider=self.id,
            )
        return _PLACEHOLDER.sub(self._fill, template)

    def _fill(self, placeholder: re.Match[str]) -> str:
        name = placeholder.group(1)
        option = name.replace("_", "-")
        variable = f"{self.id}_{name}".replace("-", "_").upper()
        value = self._config.options.get(option) or os.environ.get(variable)
        if not value:
            raise ProviderError(
                f"the base-url of {self._config.name or self.id} needs a value for "
                f"{{{name}}}: set the option {option!r} on the provider, or ${variable}",
                kind="bad_request",
                provider=self.id,
            )
        return str(value)

    def _request_options(self) -> openai.RequestOptions:
        if self._config.resolved_api_key:
            return {}
        return {"headers": {"Authorization": Omit()}}

    async def _chunks(self, response: AsyncStream[object]) -> AsyncIterator[object]:
        """The chunks of a response - or the one whole completion a backend sent instead.

        Some servers and gateways ignore ``stream: true``, or report an error as a JSON
        document with status 200. Both are handled here so that neither looks like an empty
        stream. Failures of the transport and of the SDK's decoding leave as `ProviderError`.
        """
        try:
            if "json" not in response.response.headers.get("content-type", ""):
                async for chunk in response:
                    yield chunk
                return
            document = json.loads(await response.response.aread())
        except Exception as exc:
            raise self._translate(exc) from exc
        if isinstance(document, Mapping) and document.get("error"):
            raise errors.stream_error(self.id, document)
        yield document

    def _translate(self, exc: Exception) -> ProviderError:
        base_url = str(self._client.base_url) if self._client else self._config.base_url
        return errors.translate(exc, provider=self.id, base_url=base_url)


def _isolating_headers(key: str) -> dict[str, Any]:
    """Default headers that keep the user's OpenAI identity out of requests to others.

    Left alone, the SDK adds ``OpenAI-Organization`` and ``OpenAI-Project`` from the
    environment and every header named in ``OPENAI_CUSTOM_HEADERS`` - where an
    ``Authorization`` entry would even replace this provider's own key. Each is switched off
    by name; the key is stated explicitly so that nothing can stand in for it.
    """
    headers: dict[str, Any] = {
        "OpenAI-Organization": Omit(),
        "OpenAI-Project": Omit(),
        # Brotli is never offered: httpx2's decoder crashes on the older `brotli` bindings
        # some systems carry (no `output_buffer_limit` keyword), with no graceful fallback.
        # gzip and deflate decode through the standard library everywhere.
        "Accept-Encoding": "gzip, deflate",
    }
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").split("\n"):
        name, separator, _ = line.partition(":")
        if separator and name.strip():
            headers[name.strip()] = Omit()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers

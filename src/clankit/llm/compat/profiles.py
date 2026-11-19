"""What is known about each OpenAI-compatible backend's deviations from the plain protocol.

A `Profile` only records facts that change what the adapter sends. Everything the adapter
merely has to *tolerate* in responses (reasoning field names, usage placement, missing ids)
is handled for every backend alike in `clankit.llm.compat.stream` and needs no entry here.

The default profile is deliberately the most conservative one: nothing beyond the standard
request fields, and no reasoning passed back. A wrong guess there costs some quality; a
wrong guess the other way is an HTTP 400 on every call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

from clankit.config.schema import ProviderConfig
from clankit.llm.provider import ProviderError
from clankit.llm.types import Effort

ToolMode = Literal["native", "sequential", "text"]
TOOL_MODES: tuple[str, ...] = ("native", "sequential", "text")

Replay = Literal["drop", "verbatim", "required"]

#: Every Clankit effort level is understood as written.
_AS_WRITTEN: Mapping[str, str] = MappingProxyType(
    {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
)
#: The backend stops at "high"; stronger requests are clamped rather than rejected.
_UP_TO_HIGH: Mapping[str, str] = MappingProxyType(
    {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
)
#: DeepSeek's three levels, using the mapping its own documentation gives for the others.
_LOW_HIGH_MAX: Mapping[str, str] = MappingProxyType(
    {"low": "low", "medium": "high", "high": "high", "xhigh": "high", "max": "max"}
)


@dataclass(frozen=True, slots=True)
class Profile:
    """One backend's request-side behaviour.

    `replay` says what happens to an assistant turn this adapter stored when the *same*
    provider and model continue the conversation (another model always gets the turn rebuilt
    from its text and tool calls):

    * "drop"     - rebuild it as well, so only standard fields are ever sent. For backends
      that reject fields they do not know, and for anything unidentified.
    * "verbatim" - send the stored message back unchanged, including reasoning, reasoning
      details and thought signatures. For backends that document wanting exactly that.
    * "required" - as "verbatim", and when the request carries tools every assistant turn
      must have `reasoning_field`, so turns that have none get an empty one. This is
      DeepSeek's rule; it answers 400 otherwise.
    """

    #: Name of the output-token cap in the request body.
    max_tokens_key: str = "max_tokens"
    #: The backend understands ``parallel_tool_calls: false``.
    parallel_flag: bool = False
    replay: Replay = "drop"
    reasoning_field: str = "reasoning_content"
    #: Request field that takes the reasoning effort: "reasoning_effort", "reasoning" (an
    #: object with an ``effort`` member), or "" when the backend has no dependable knob.
    effort_key: str = ""
    #: Clankit effort level -> the backend's word for it. A level that is absent is not sent.
    effort_levels: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    #: Tool calls rebuilt from another model's turn need a placeholder thought signature.
    placeholder_signature: bool = False
    #: Prefix the backend puts on the ids it lists, which requests are written without.
    model_prefix: str = ""

    def effort_params(self, effort: Effort | None) -> dict[str, Any]:
        """Request fields that ask for `effort`, or nothing when this backend cannot be asked."""
        level = self.effort_levels.get(effort or "")
        if not level or not self.effort_key:
            return {}
        if self.effort_key == "reasoning":
            return {"reasoning": {"effort": level}}
        return {self.effort_key: level}


DEFAULT_PROFILE = Profile()

PROFILES: Mapping[str, Profile] = MappingProxyType(
    {
        "deepseek": Profile(
            replay="required", effort_key="reasoning_effort", effort_levels=_LOW_HIGH_MAX
        ),
        "openrouter": Profile(
            replay="verbatim",
            parallel_flag=True,
            effort_key="reasoning",
            effort_levels=_AS_WRITTEN,
        ),
        "gemini": Profile(
            replay="verbatim",
            effort_key="reasoning_effort",
            effort_levels=_UP_TO_HIGH,
            placeholder_signature=True,
            model_prefix="models/",
        ),
        # Groq answers 400 to message fields outside its list, and its models disagree about
        # which effort words they accept, so neither reasoning nor effort is sent.
        "groq": Profile(max_tokens_key="max_completion_tokens", parallel_flag=True),
        "nvidia": Profile(replay="verbatim"),
        "zai": Profile(replay="verbatim"),
        "cloudflare": Profile(),
        "cerebras": Profile(
            max_tokens_key="max_completion_tokens",
            parallel_flag=True,
            replay="verbatim",
            effort_key="reasoning_effort",
            effort_levels=_UP_TO_HIGH,
        ),
        "moonshot": Profile(max_tokens_key="max_completion_tokens", replay="verbatim"),
        "minimax": Profile(max_tokens_key="max_completion_tokens", replay="verbatim"),
        # Local runtimes ignore message fields their chat template does not use, and the
        # templates that do use earlier reasoning need it back.
        "ollama": Profile(replay="verbatim"),
        "lmstudio": Profile(replay="verbatim"),
        "llamacpp": Profile(replay="verbatim", parallel_flag=True),
        "vllm": Profile(replay="verbatim", parallel_flag=True),
    }
)

#: Hosts of the hosted backends, for providers configured under another id.
_HOSTS: Mapping[str, str] = MappingProxyType(
    {
        "api.deepseek.com": "deepseek",
        "openrouter.ai": "openrouter",
        "generativelanguage.googleapis.com": "gemini",
        "api.groq.com": "groq",
        "integrate.api.nvidia.com": "nvidia",
        "api.z.ai": "zai",
        "api.cloudflare.com": "cloudflare",
        "api.cerebras.ai": "cerebras",
        "api.moonshot.ai": "moonshot",
        "api.minimax.io": "minimax",
    }
)


def profile_for(config: ProviderConfig) -> Profile:
    """The profile for a provider: the ``profile`` option if set, else the provider id, else
    the host of its base URL, else the conservative default.

    The option exists for backends reached under another name or through a proxy, where
    neither the id nor the host says what is on the other end.
    """
    chosen = str(config.options.get("profile") or "")
    if chosen:
        if chosen not in PROFILES:
            raise ProviderError(
                f"unknown profile {chosen!r} for provider {config.id!r}; "
                f"known profiles: {', '.join(sorted(PROFILES))}",
                kind="bad_request",
                provider=config.id,
            )
        return PROFILES[chosen]
    if config.id in PROFILES:
        return PROFILES[config.id]
    host = (urlsplit(config.base_url).hostname or "").lower()
    return PROFILES.get(_HOSTS.get(host, ""), DEFAULT_PROFILE)


def tool_mode_for(config: ProviderConfig, model: str) -> ToolMode:
    """How tools are offered to `model`: the model's own setting, else the provider's.

    Read from ``options["model-tool-modes"][model]`` and ``options["tool-mode"]``.
    """
    per_model = config.options.get("model-tool-modes")
    mode = per_model.get(model) if isinstance(per_model, Mapping) else None
    mode = mode or config.options.get("tool-mode") or "native"
    if mode not in TOOL_MODES:
        raise ProviderError(
            f"tool-mode for {config.id}/{model} is {mode!r}; "
            f"it must be one of: {', '.join(TOOL_MODES)}",
            kind="bad_request",
            provider=config.id,
        )
    return mode

"""Built-in knowledge about model providers.

The catalog is what lets a user write just::

    providers {
        anthropic api-key="sk-ant-..."
    }

Everything else about a known provider (protocol, base URL, the environment variable its key
conventionally lives in, a few recommended models) comes from here. The user's file never
contains catalog data; `effective_providers()` overlays the two at runtime.

Facts in this file were checked against the providers' documentation on 2026-11-19. Where a
fact could not be established it is left unset (``None``) rather than guessed, and the
model notes - which agents read when they pick a model for a subtask - say what is not known.
Free offers change quickly: model lineups that rotate (OpenRouter's free models, whatever a
local server has loaded) are discovered at runtime instead of being listed here.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

from clankit.config.schema import Config, ModelEntry, ProviderConfig


@dataclass(frozen=True, slots=True)
class CatalogProvider:
    """Defaults for a known provider.

    `free` is "no", "partly" (a free tier or some free models), "yes", or "local" (runs on
    the user's machine). `keyless` providers need no API key but must be switched on by
    mentioning them in the configuration, since Clankit cannot know a local server is running.
    `default_model` is what Clankit uses when this is the only usable provider; it is empty
    for providers whose models are only known once the server has been asked.
    """

    id: str
    name: str
    kind: str
    base_url: str = ""
    api_key_env: str = ""
    keyless: bool = False
    free: str = "no"
    default_model: str = ""
    signup_url: str = ""
    note: str = ""
    #: Model calls in flight at once; 0 = default. Set 1-2 for rate-limited free tiers.
    concurrency: int = 0
    models: tuple[ModelEntry, ...] = ()
    options: dict[str, object] = field(default_factory=dict)
    #: The models worth using are found by asking the provider (``GET /models``); `models`
    #: is at most a starting point.
    discovers_models: bool = False


CATALOG: dict[str, CatalogProvider] = {}


def _add(provider: CatalogProvider) -> None:
    CATALOG[provider.id] = provider


def _model(
    model_id: str,
    name: str,
    note: str,
    *,
    context: int | None = None,
    output: int | None = None,
    tools: bool | None = True,
    free: bool | None = None,
    price: tuple[float, float] | None = None,
) -> ModelEntry:
    """A catalog model. `price` is US dollars per million (input, output) tokens."""
    return ModelEntry(
        id=model_id,
        name=name,
        context_window=context,
        max_output=output,
        tools=tools,
        free=free,
        note=note,
        input_price=price[0] if price else None,
        output_price=price[1] if price else None,
    )


# --------------------------------------------------------------------------------------------
# Paid
# --------------------------------------------------------------------------------------------

_add(
    CatalogProvider(
        id="anthropic",
        name="Anthropic",
        kind="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-opus-5-5",
        signup_url="https://platform.claude.com/settings/keys",
        note="Paid. Claude models, from very cheap (Haiku) to the most capable (Fable).",
        models=(
            _model(
                "claude-opus-5-5",
                "Claude Opus 5.5",
                "Strong all-rounder; the default for hard engineering work.",
                context=1_000_000,
                output=128_000,
                free=False,
                price=(4.0, 20.0),
            ),
            _model(
                "claude-fable-5-2",
                "Claude Fable 5.2",
                "Most capable and most expensive (twice Opus); for the hardest reasoning and "
                "long autonomous work.",
                context=1_000_000,
                output=128_000,
                free=False,
                price=(8.0, 40.0),
            ),
            _model(
                "claude-sonnet-5-5",
                "Claude Sonnet 5.5",
                "Fast and capable at half the price of Opus; good for well-specified subtasks.",
                context=1_000_000,
                output=128_000,
                free=False,
                price=(2.0, 10.0),
            ),
            _model(
                "claude-haiku-5-5",
                "Claude Haiku 5.5",
                "Very cheap and fast; good for mechanical edits, searches and summaries. Keep "
                "its prompts under 100K tokens: above that it is billed at five times the rate.",
                context=1_000_000,
                output=128_000,
                free=False,
                price=(0.10, 0.50),
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="deepseek",
        name="DeepSeek",
        kind="openai-compatible",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-flash",
        signup_url="https://platform.deepseek.com/api_keys",
        note="Paid, and very cheap. Prices shown are peak rates; outside 01:00-04:00 and "
        "06:00-10:00 UTC on weekdays everything costs half.",
        models=(
            _model(
                "deepseek-flash",
                "DeepSeek V4.1 Flash",
                "Very cheap, reasons before answering, 1M-token context; a good workhorse for "
                "subtasks and bulk edits.",
                context=1_048_576,
                output=393_216,
                free=False,
                price=(0.30, 1.20),
            ),
            _model(
                "deepseek-v4-pro",
                "DeepSeek V4 Pro",
                "DeepSeek's larger model at about four times the price of Flash; try it where "
                "Flash struggles with planning or debugging. Text only.",
                context=1_048_576,
                output=393_216,
                free=False,
                price=(1.32, 3.96),
            ),
        ),
    )
)

# --------------------------------------------------------------------------------------------
# Free tiers and free models
# --------------------------------------------------------------------------------------------

_add(
    CatalogProvider(
        id="openrouter",
        name="OpenRouter",
        kind="openai-compatible",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        free="partly",
        default_model="openrouter/free",
        signup_url="https://openrouter.ai/settings/keys",
        note="One key for hundreds of models, some free (ids ending in ':free'). Free models "
        "allow 20 requests a minute and 50 a day (1000 a day once 10 credits have been bought), "
        "and several log or train on prompts. The free lineup rotates within weeks, so it is "
        "discovered at runtime.",
        concurrency=2,
        discovers_models=True,
        models=(
            _model(
                "openrouter/free",
                "Free models router",
                "Sends each request to whichever free model is available, so quality and "
                "behaviour vary from call to call. Fine for trying things out; for real work "
                "pick one specific ':free' model and stay on it.",
                context=200_000,
                free=True,
                price=(0.0, 0.0),
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="gemini",
        name="Google Gemini",
        kind="openai-compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        free="partly",
        default_model="gemini-4-hydrogen",
        signup_url="https://aistudio.google.com/apikey",
        note="The Gemini API's free tier costs nothing for the models listed here, but Google "
        "uses free-tier prompts and responses to improve its products, and the rate limits are "
        "unpublished: expect to be slowed down. Keys with billing enabled are charged instead.",
        concurrency=2,
        models=(
            _model(
                "gemini-4-hydrogen",
                "Gemini 4 Hydrogen",
                "Probably the strongest model with a free tier; 1M-token context. Rate limits "
                "on the free tier are tight and unpublished.",
                context=1_048_576,
                output=65_536,
            ),
            _model(
                "gemini-3.8-flash",
                "Gemini 3.8 Flash",
                "Previous-generation Flash aimed at coding and agents; a fallback when "
                "Hydrogen is rate limited.",
            ),
            _model(
                "gemini-3.5-flash-lite",
                "Gemini 3.5 Flash-Lite",
                "Fastest and lightest; for simple mechanical subtasks and summaries only.",
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="groq",
        name="Groq",
        kind="openai-compatible",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        free="partly",
        default_model="qwen/qwen3.8-27b",
        signup_url="https://console.groq.com/keys",
        note="Very fast open-weight models. The free plan allows 30 requests a minute, 1000 a "
        "day and only 8K tokens a minute per model, so any prompt above roughly 8K tokens is "
        "refused: usable for small, self-contained subtasks. Prices shown apply to paid plans.",
        concurrency=1,
        models=(
            _model(
                "qwen/qwen3.8-27b",
                "Qwen 3.8 27B",
                "The most capable open-weight model here and the only one of the three that "
                "makes parallel tool calls. On the free plan keep prompts under 8K tokens.",
                context=131_072,
                output=16_384,
                price=(0.80, 4.00),
            ),
            _model(
                "openai/gpt-oss-120b",
                "GPT-OSS 120B",
                "Cheap and very fast, one tool call per turn; much weaker than the Qwen model "
                "at multi-step coding. For short mechanical subtasks.",
                context=131_072,
                output=65_536,
                price=(0.15, 0.60),
            ),
            _model(
                "openai/gpt-oss-20b",
                "GPT-OSS 20B",
                "Smallest and fastest; for classification, extraction and one-line edits.",
                context=131_072,
                output=65_536,
                price=(0.075, 0.30),
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="nvidia",
        name="NVIDIA API Catalog",
        kind="openai-compatible",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        free="yes",
        default_model="moonshotai/kimi-k3.1-pro",
        signup_url="https://build.nvidia.com",
        note="Free for development and evaluation, not for production use; roughly 40 "
        "requests a minute across the account (an observed figure, not a published one). "
        "Which models accept tool calls is not documented per model.",
        concurrency=2,
        discovers_models=True,
        models=(
            _model(
                "moonshotai/kimi-k3.1-pro",
                "Kimi K3.1 Pro",
                "A strong open-weight coding model, well ahead of the Qwen and GLM models in "
                "this catalog on public benchmarks. Tool calling on this endpoint is "
                "unconfirmed.",
                tools=None,
            ),
            _model(
                "z-ai/glm-5.3-flash",
                "GLM 5.3 Flash",
                "Mid-sized reasoning model; a second choice when Kimi is unavailable. Tool "
                "calling on this endpoint is unconfirmed.",
                tools=None,
            ),
            _model(
                "minimaxai/minimax-m3.1-flash",
                "MiniMax M3.1 Flash",
                "Comparable to GLM 5.3 Flash. Tool calling on this endpoint is unconfirmed.",
                tools=None,
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="zai",
        name="Z.ai GLM",
        kind="openai-compatible",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZAI_API_KEY",
        free="partly",
        default_model="glm-4.7-flash",
        signup_url="https://z.ai/manage-apikey/apikey-list",
        note="The Flash models are free of charge; their rate limits are shown only in the "
        "Z.ai console. The current GLM-5 models are paid.",
        concurrency=2,
        models=(
            _model(
                "glm-4.7-flash",
                "GLM 4.7 Flash",
                "Free. The lightweight tier of the previous GLM generation: use it for simple "
                "edits and summaries, not for planning or large changes.",
                free=True,
                price=(0.0, 0.0),
            ),
            _model(
                "glm-4.5-flash",
                "GLM 4.5 Flash",
                "Free and older still; a fallback for GLM 4.7 Flash.",
                free=True,
                price=(0.0, 0.0),
            ),
            _model(
                "glm-5.3",
                "GLM 5.3",
                "Z.ai's current flagship, paid; a solid mid-priced coding model that always "
                "reasons before answering.",
                free=False,
                price=(1.40, 4.40),
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="cloudflare",
        name="Cloudflare Workers AI",
        kind="openai-compatible",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api_key_env="CLOUDFLARE_API_KEY",
        free="partly",
        default_model="@cf/qwen/qwen3.8-27b",
        signup_url="https://dash.cloudflare.com/profile/api-tokens",
        note="Needs your account id as well as an API token: set the option account-id on "
        "this provider, or $CLOUDFLARE_ACCOUNT_ID. 10,000 'neurons' a day are free, which is "
        "little: about 34K output tokens of Qwen 3.8 27B.",
        concurrency=2,
        models=(
            _model(
                "@cf/qwen/qwen3.8-27b",
                "Qwen 3.8 27B",
                "The strongest model available on the free allocation, which it exhausts "
                "after about 34K output tokens a day.",
            ),
            _model(
                "@cf/qwen/qwen3-30b-a3b-fp8",
                "Qwen3 30B-A3B",
                "Weaker, but roughly nine times more tokens fit in the daily free allocation; "
                "for many small subtasks. Tool calling is unconfirmed.",
                tools=None,
            ),
            _model(
                "@cf/openai/gpt-oss-120b",
                "GPT-OSS 120B",
                "For short mechanical subtasks. Tool calling on this endpoint is unconfirmed.",
                tools=None,
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="cerebras",
        name="Cerebras",
        kind="openai-compatible",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        default_model="qwen-3.8-27b",
        signup_url="https://cloud.cerebras.ai",
        note="Not free: new accounts get a $5 credit that lasts 30 days and requires a "
        "payment method. Extremely fast inference.",
        models=(
            _model(
                "qwen-3.8-27b",
                "Qwen 3.8 27B",
                "Capable open-weight coding model at very high speed. The context here is "
                "65K tokens, far smaller than elsewhere: keep tasks compact.",
                context=65_536,
                output=32_768,
                free=False,
                price=(0.99, 1.49),
            ),
            _model(
                "gpt-oss-120b",
                "GPT-OSS 120B",
                "Cheaper and weaker; one tool call per turn. For short mechanical subtasks.",
                context=131_072,
                output=40_960,
                free=False,
                price=(0.35, 0.75),
            ),
        ),
    )
)

# --------------------------------------------------------------------------------------------
# Local (keyless; switched on by mentioning them in the configuration)
# --------------------------------------------------------------------------------------------

_add(
    CatalogProvider(
        id="ollama",
        name="Ollama",
        kind="openai-compatible",
        base_url="http://localhost:11434/v1",
        keyless=True,
        free="local",
        default_model="qwen3.8:27b",
        signup_url="https://ollama.com/download",
        note="Runs models on this machine. Ollama gives models only a 4K-token context on "
        "GPUs under 24 GiB (32K up to 48 GiB): set OLLAMA_CONTEXT_LENGTH=64000 or more, or "
        "agents will lose their instructions mid-task.",
        concurrency=1,
        discovers_models=True,
        models=(
            _model(
                "qwen3.8:27b",
                "Qwen 3.8 27B",
                "The best local coding model for a 24 GB GPU (an 18 GB download: 'ollama pull "
                "qwen3.8:27b'). Well below hosted frontier models: give it small, clear tasks.",
                free=True,
            ),
            _model(
                "gemma4:12b",
                "Gemma 4 12B",
                "Fits GPUs with 8-16 GB. Expect it to manage single-file edits, not multi-step "
                "work.",
                free=True,
            ),
            _model(
                "gpt-oss:20b",
                "GPT-OSS 20B",
                "Fits GPUs with 16 GB. For simple mechanical subtasks.",
                free=True,
            ),
        ),
    )
)

_add(
    CatalogProvider(
        id="lmstudio",
        name="LM Studio",
        kind="openai-compatible",
        base_url="http://localhost:1234/v1",
        keyless=True,
        free="local",
        signup_url="https://lmstudio.ai",
        note="Serves the models downloaded in LM Studio once its local server is started. "
        "Load a model with at least 64K tokens of context for agent work.",
        concurrency=1,
        discovers_models=True,
    )
)

_add(
    CatalogProvider(
        id="llamacpp",
        name="llama.cpp",
        kind="openai-compatible",
        base_url="http://localhost:9931/v1",
        api_key_env="LLAMA_API_KEY",
        keyless=True,
        free="local",
        signup_url="https://github.com/ggml-org/llama.cpp",
        note="llama-server with whatever model it was started with. Older builds listen on "
        "port 8080: change base-url if so. A key is only needed when the server was started "
        "with --api-key.",
        concurrency=1,
        discovers_models=True,
    )
)

_add(
    CatalogProvider(
        id="vllm",
        name="vLLM",
        kind="openai-compatible",
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        keyless=True,
        free="local",
        signup_url="https://docs.vllm.ai",
        note="Serves the model it was started with. Tool calls only work when the server "
        "runs with --enable-auto-tool-choice and the --tool-call-parser its model needs; "
        'without them set the option tool-mode to "text".',
        discovers_models=True,
    )
)


def effective_providers(config: Config) -> dict[str, ProviderConfig]:
    """Every provider Clankit knows about: catalog defaults overlaid with the user's entries.

    * A catalog provider the user did not mention is included and enabled; it is `usable`
      when its API key is in the environment. Keyless (local) catalog providers are included
      but disabled until the user mentions them.
    * A user entry for a catalog id overrides field by field: empty values fall back to the
      catalog, user-declared models are merged over catalog models by id.
    * A user entry with an unknown id is a custom provider and is used as written.

    Catalog order comes first, then custom providers in file order.
    """
    result: dict[str, ProviderConfig] = {}
    for provider_id, cat in CATALOG.items():
        user = config.providers.get(provider_id)
        models = {m.id: copy.copy(m) for m in cat.models}
        options: dict[str, object] = dict(cat.options)
        if cat.default_model:
            options.setdefault("default-model", cat.default_model)
        if user is None:
            result[provider_id] = ProviderConfig(
                id=provider_id,
                kind=cat.kind,
                name=cat.name,
                api_key_env=cat.api_key_env,
                base_url=cat.base_url,
                keyless=cat.keyless,
                enabled=not cat.keyless,
                concurrency=cat.concurrency,
                models=list(models.values()),
                options=options,
            )
            continue
        for entry in user.models:
            base = models.get(entry.id)
            if base is None:
                models[entry.id] = copy.copy(entry)
                continue
            base.name = entry.name or base.name
            base.context_window = entry.context_window or base.context_window
            base.max_output = entry.max_output or base.max_output
            base.tools = entry.tools if entry.tools is not None else base.tools
            base.free = entry.free if entry.free is not None else base.free
            base.note = entry.note or base.note
            if entry.input_price is not None:
                base.input_price = entry.input_price
            if entry.output_price is not None:
                base.output_price = entry.output_price
        options.update(user.options)
        result[provider_id] = ProviderConfig(
            id=provider_id,
            kind=cat.kind,
            name=user.name or cat.name,
            api_key=user.api_key,
            api_key_env=user.api_key_env or cat.api_key_env,
            base_url=user.base_url or cat.base_url,
            keyless=cat.keyless or user.keyless,
            enabled=user.enabled,
            concurrency=user.concurrency or cat.concurrency,
            models=list(models.values()),
            options=options,
        )
    for provider_id, user in config.providers.items():
        if provider_id not in result:
            result[provider_id] = copy.deepcopy(user)
    return result


def key_source(config: Config, provider_id: str) -> str:
    """Where a provider's API key comes from: "config", "env", "none", or "keyless"."""
    provider = effective_providers(config).get(provider_id)
    if provider is None:
        return "none"
    if provider.keyless:
        return "keyless"
    if provider.api_key:
        return "config"
    if provider.api_key_env and os.environ.get(provider.api_key_env):
        return "env"
    return "none"

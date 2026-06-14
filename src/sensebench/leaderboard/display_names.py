"""Presentation helpers for the leaderboard UI.

Produces a prettified model display label, a vendor logo slug, and a brand
family/initial used for the colored-initial fallback when no logo is available.

The family map MUST stay in sync with ``FAMILY_COLORS``/``FAMILY_PATTERNS`` in
``src/sensebench/site/static/site.js`` and the ``.fam-*`` classes in
``src/sensebench/site/static/site.css`` — the three are mirrors of one table.
"""

from __future__ import annotations

import re

OTHER_FAMILY: str = "Other"

# Raw model id (the submitted display_name) -> human label. Effort is appended
# by the UI, so it is intentionally excluded here.
_EXPLICIT_NAMES: dict[str, str] = {
    # OpenAI
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4-nano": "GPT-5.4 Nano",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4o-mini": "GPT-4o Mini",
    # Anthropic
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    # Google (AI Studio gemini/ prefix)
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini/gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    # xAI / DeepSeek / Chinese open models via OpenRouter
    "openrouter/x-ai/grok-4.3": "Grok 4.3",
    "openrouter/x-ai/grok-4.1-fast": "Grok 4.1 Fast",
    "openrouter/deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "openrouter/deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "openrouter/moonshotai/kimi-k2.5": "Kimi K2.5",
    "openrouter/z-ai/glm-5": "GLM-5",
    "openrouter/minimax/minimax-m3": "MiniMax M3",
    "openrouter/qwen/qwen3.7-max": "Qwen3.7-Max",
    "openrouter/qwen/qwen3.7-plus": "Qwen3.7-Plus",
    # Self-hosted (vLLM) open weights
    "Qwen/Qwen3.6-27B-FP8": "Qwen3.6 27B",
    "Qwen/Qwen3.6-35B-A3B-FP8": "Qwen3.6 35B-A3B",
    "google/gemma-4-26B-A4B-it": "Gemma 4 26B-A4B",
    "google/gemma-4-E4B-it": "Gemma 4 E4B",
    "google/gemma-4-E2B-it": "Gemma 4 E2B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "ibm-granite/granite-4.1-8b-fp8": "Granite 4.1 8B",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 3.2 24B",
}

# Exact vendor string -> bundled logo slug (src/sensebench/site/static/logos/<slug>.svg).
_VENDOR_LOGO_SLUGS: dict[str, str] = {
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "Google": "gemini",
    "xAI": "grok",
    "DeepSeek": "deepseek",
    "Moonshot": "moonshot",
    "Moonshot AI": "moonshot",
    "Z.ai": "zhipu",
    "Zhipu": "zhipu",
    "MiniMax": "minimax",
    "Alibaba": "qwen",
    "Meta": "meta",
    "IBM": "ibm",
    "Mistral AI": "mistral",
    "Mistral": "mistral",
}

# Tokens rendered fully upper-cased by the fallback.
_ACRONYMS: dict[str, str] = {
    "gpt": "GPT",
    "glm": "GLM",
    "moe": "MoE",
    "ibm": "IBM",
    "ai": "AI",
    "fp8": "FP8",
    "fp16": "FP16",
    "bf16": "BF16",
    "awq": "AWQ",
    "gptq": "GPTQ",
}

# Suffix tokens dropped from the fallback label (these are provenance, not identity).
_DROP_TOKENS: frozenset[str] = frozenset(
    {"it", "instruct", "fp8", "fp16", "bf16", "awq", "gptq", "int4", "int8"}
)
_DATE_RE = re.compile(r"^(?:20\d{6}|\d{4}-\d{2}-\d{2})$")

# Mirror of site.js FAMILY_PATTERNS (order matters; first match wins).
_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gemma"), "Gemma"),
    (re.compile(r"qwen"), "Qwen"),
    (re.compile(r"glm"), "GLM"),
    (re.compile(r"llama|maverick|scout"), "Llama"),
    (re.compile(r"mistral|mixtral|magistral|ministral|pixtral"), "Mistral"),
    (re.compile(r"nemotron"), "Nemotron"),
    (re.compile(r"deepseek"), "DeepSeek"),
    (re.compile(r"granite"), "Granite"),
    (re.compile(r"phi-?\d"), "Phi"),
    (re.compile(r"hunyuan"), "Hunyuan"),
    (re.compile(r"olmo"), "OLMo"),
    (re.compile(r"command|c4ai"), "Command"),
    (re.compile(r"gpt|davinci"), "GPT"),
    (re.compile(r"claude"), "Claude"),
    (re.compile(r"minimax"), "MiniMax"),
)


def prettify_model_name(model: str) -> str:
    """Return a human label for a raw model id, excluding reasoning effort."""
    if model in _EXPLICIT_NAMES:
        return _EXPLICIT_NAMES[model]
    tail = model.rsplit("/", 1)[-1]
    # Join a trailing two-part version, e.g. claude-opus-4-6 -> ...-4.6.
    tail = re.sub(r"-(\d)-(\d)$", r"-\1.\2", tail)
    rendered: list[str] = []
    for token in re.split(r"[-_]", tail):
        if not token or token.lower() in _DROP_TOKENS or _DATE_RE.match(token):
            continue
        low = token.lower()
        if low in _ACRONYMS:
            rendered.append(_ACRONYMS[low])
            continue
        normalized = token
        if re.match(r"^v\d", normalized):
            normalized = "V" + normalized[1:]
        normalized = re.sub(r"(\d)b$", r"\1B", normalized)
        if re.search(r"\d", normalized):
            rendered.append(normalized)
        else:
            rendered.append(normalized[:1].upper() + normalized[1:])
    label = " ".join(rendered).strip()
    return label or model


def vendor_logo_slug(llm_vendor: str | None) -> str | None:
    """Bundled logo slug for a vendor, or None to use the initial fallback."""
    if not llm_vendor:
        return None
    return _VENDOR_LOGO_SLUGS.get(llm_vendor)


def model_family(llm_vendor: str | None, model: str) -> str:
    """Brand family for the model (mirrors site.js familyOf)."""
    name = (model or "").lower()
    for pattern, label in _FAMILY_PATTERNS:
        if pattern.search(name):
            return label
    return llm_vendor or OTHER_FAMILY


def vendor_initial(llm_vendor: str | None, model: str) -> str:
    """Single uppercase character for the colored-initial fallback badge."""
    family = model_family(llm_vendor, model)
    source = family if family != OTHER_FAMILY else (llm_vendor or model or "?")
    for char in source:
        if char.isalnum():
            return char.upper()
    return "?"

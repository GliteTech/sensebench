"""Tests for leaderboard display-name and logo helpers."""

from __future__ import annotations

import pytest

from sensebench.leaderboard.display_names import (
    model_family,
    prettify_model_name,
    vendor_initial,
    vendor_logo_slug,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Explicit map.
        ("gpt-5.5", "GPT-5.5"),
        ("gpt-5.4-mini", "GPT-5.4 Mini"),
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("gemini/gemini-3.5-flash", "Gemini 3.5 Flash"),
        ("gemini/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite"),
        ("openrouter/deepseek/deepseek-v4-pro", "DeepSeek V4 Pro"),
        ("openrouter/moonshotai/kimi-k2.5", "Kimi K2.5"),
        ("Qwen/Qwen3.6-27B-FP8", "Qwen3.6 27B"),
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B"),
        ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral Small 3.2 24B"),
        # Fallback algorithm (ids not in the explicit map).
        ("fake-model", "Fake Model"),
        ("openrouter/acme/widget-9-mini", "Widget 9 Mini"),
        ("vendor/some-model-fp8-20260101", "Some Model"),
        ("openrouter/x-ai/grok-5-fast", "Grok 5 Fast"),
        ("zhipu/glm-6-air", "GLM 6 Air"),
    ],
)
def test_prettify_model_name(model: str, expected: str) -> None:
    assert prettify_model_name(model) == expected


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        ("OpenAI", "openai"),
        ("Anthropic", "anthropic"),
        ("Google", "gemini"),
        ("xAI", "grok"),
        ("DeepSeek", "deepseek"),
        ("Alibaba", "qwen"),
        ("Mistral AI", "mistral"),
        ("Z.ai", "zhipu"),
        ("Moonshot", "moonshot"),
        ("IBM", "ibm"),
        (None, None),
        ("Unknown Vendor", None),
    ],
)
def test_vendor_logo_slug(vendor: str | None, expected: str | None) -> None:
    assert vendor_logo_slug(vendor) == expected


@pytest.mark.parametrize(
    ("vendor", "model", "expected"),
    [
        ("OpenAI", "gpt-5.5", "GPT"),
        ("Anthropic", "claude-opus-4-8", "Claude"),
        ("DeepSeek", "openrouter/deepseek/deepseek-v4-pro", "DeepSeek"),
        ("Alibaba", "Qwen/Qwen3.6-27B-FP8", "Qwen"),
        ("Google", "gemini/gemini-3.5-flash", "Gemini"),
        # No family pattern -> falls back to the vendor string.
        ("TestVendor", "fake-model", "TestVendor"),
        (None, "fake-model", "Other"),
    ],
)
def test_model_family(vendor: str | None, model: str, expected: str) -> None:
    assert model_family(llm_vendor=vendor, model=model) == expected


@pytest.mark.parametrize(
    ("vendor", "model", "expected"),
    [
        ("OpenAI", "gpt-5.5", "G"),
        ("Anthropic", "claude-opus-4-8", "C"),
        ("Alibaba", "Qwen/Qwen3.6-27B-FP8", "Q"),
        ("Google", "gemini/gemini-3.5-flash", "G"),
        ("TestVendor", "fake-model", "T"),
        (None, "fake-model", "F"),
    ],
)
def test_vendor_initial(vendor: str | None, model: str, expected: str) -> None:
    assert vendor_initial(llm_vendor=vendor, model=model) == expected

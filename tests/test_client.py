from __future__ import annotations

from pytest import approx

from sensebench.runner.client import (
    COMPLETION_TOKENS_FIELD,
    INPUT_COST_PER_TOKEN_FIELD,
    OPENROUTER_COMPLETIONS_COST_FIELD,
    OPENROUTER_COST_DETAILS_FIELD,
    OPENROUTER_PROMPT_COST_FIELD,
    OPENROUTER_USAGE_COST_FIELD,
    OUTPUT_COST_PER_TOKEN_FIELD,
    PROMPT_TOKENS_FIELD,
    USAGE_FIELD,
    _cost_from_response,
)
from sensebench.runs.models import CostSourceKind, TokenUsage

PRICED_MODEL_ID: str = "test/model"
PROVIDER_MODEL_ID: str = "qwen/qwen3-235b-a22b-04-28"
PROVIDER_TOTAL_COST_USD: float = 0.123
PROVIDER_PROMPT_COST_USD: float = 0.023
PROVIDER_COMPLETION_COST_USD: float = 0.1
LITELLM_COMPLETION_COST_USD: float = 0.19
INPUT_UNIT_PRICE_USD: float = 0.01
OUTPUT_UNIT_PRICE_USD: float = 0.02


class _FailingCostLiteLlm:
    model_cost: dict[str, dict[str, float]] = {}

    def completion_cost(self, *, completion_response: object) -> float:
        raise AssertionError("provider-reported cost should bypass LiteLLM cost estimation")


class _PricedLiteLlm:
    model_cost: dict[str, dict[str, float]] = {
        PRICED_MODEL_ID: {
            INPUT_COST_PER_TOKEN_FIELD: INPUT_UNIT_PRICE_USD,
            OUTPUT_COST_PER_TOKEN_FIELD: OUTPUT_UNIT_PRICE_USD,
        }
    }

    def completion_cost(self, *, completion_response: object) -> float:
        return LITELLM_COMPLETION_COST_USD


def test_cost_from_response_prefers_provider_reported_cost() -> None:
    cost = _cost_from_response(
        litellm_module=_FailingCostLiteLlm(),
        payload={
            USAGE_FIELD: {
                OPENROUTER_USAGE_COST_FIELD: PROVIDER_TOTAL_COST_USD,
                OPENROUTER_COST_DETAILS_FIELD: {
                    OPENROUTER_PROMPT_COST_FIELD: PROVIDER_PROMPT_COST_USD,
                    OPENROUTER_COMPLETIONS_COST_FIELD: PROVIDER_COMPLETION_COST_USD,
                },
            },
        },
        response={},
        model=PROVIDER_MODEL_ID,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )

    assert cost.total_usd == PROVIDER_TOTAL_COST_USD
    assert cost.input_uncached_usd == PROVIDER_PROMPT_COST_USD
    assert cost.output_usd == PROVIDER_COMPLETION_COST_USD
    assert cost.source == CostSourceKind.PROVIDER_REPORTED


def test_cost_from_response_falls_back_to_litellm_estimate() -> None:
    cost = _cost_from_response(
        litellm_module=_PricedLiteLlm(),
        payload={USAGE_FIELD: {PROMPT_TOKENS_FIELD: 7, COMPLETION_TOKENS_FIELD: 6}},
        response={},
        model=PRICED_MODEL_ID,
        usage=TokenUsage(input_tokens=7, cached_input_tokens=0, output_tokens=6),
    )

    assert cost.total_usd == LITELLM_COMPLETION_COST_USD
    assert cost.input_uncached_usd == approx(0.07)
    assert cost.output_usd == approx(0.12)
    assert cost.input_uncached_unit_price_usd == INPUT_UNIT_PRICE_USD
    assert cost.output_unit_price_usd == OUTPUT_UNIT_PRICE_USD
    assert cost.source == CostSourceKind.LITELLM_ESTIMATE

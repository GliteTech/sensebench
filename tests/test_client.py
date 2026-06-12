from __future__ import annotations

from pytest import approx

from sensebench.runner.client import _cost_from_response
from sensebench.runs.models import CostSourceKind, TokenUsage


class _FailingCostLiteLlm:
    model_cost: dict[str, dict[str, float]] = {}

    def completion_cost(self, *, completion_response: object) -> float:
        raise AssertionError("provider-reported cost should bypass LiteLLM cost estimation")


class _PricedLiteLlm:
    model_cost: dict[str, dict[str, float]] = {
        "test/model": {
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
        }
    }

    def completion_cost(self, *, completion_response: object) -> float:
        return 0.19


def test_cost_from_response_prefers_provider_reported_cost() -> None:
    cost = _cost_from_response(
        litellm_module=_FailingCostLiteLlm(),
        payload={
            "usage": {
                "cost": 0.123,
                "cost_details": {
                    "upstream_inference_prompt_cost": 0.023,
                    "upstream_inference_completions_cost": 0.1,
                },
            },
        },
        response={},
        model="qwen/qwen3-235b-a22b-04-28",
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )

    assert cost.total_usd == 0.123
    assert cost.input_uncached_usd == 0.023
    assert cost.output_usd == 0.1
    assert cost.source == CostSourceKind.PROVIDER_REPORTED


def test_cost_from_response_falls_back_to_litellm_estimate() -> None:
    cost = _cost_from_response(
        litellm_module=_PricedLiteLlm(),
        payload={"usage": {"prompt_tokens": 7, "completion_tokens": 6}},
        response={},
        model="test/model",
        usage=TokenUsage(input_tokens=7, cached_input_tokens=0, output_tokens=6),
    )

    assert cost.total_usd == 0.19
    assert cost.input_uncached_usd == approx(0.07)
    assert cost.output_usd == approx(0.12)
    assert cost.input_uncached_unit_price_usd == 0.01
    assert cost.output_unit_price_usd == 0.02
    assert cost.source == CostSourceKind.LITELLM_ESTIMATE

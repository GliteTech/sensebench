from __future__ import annotations

import asyncio
import time

from pytest import approx

from sensebench.prompts.models import MessageRole
from sensebench.prompts.render import ChatMessage
from sensebench.runner.client import (
    RateLimitedCompletionClient,
    _cost_from_response,
    _gemini_model_name,
    _gemini_raw_output_from_payload,
    _gemini_request_payload,
    _gemini_usage_from_payload,
)
from sensebench.runner.costs import unavailable_cost
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runs.models import AttemptKind, CallRecord, CallStatus, CostSourceKind, TokenUsage

RATE_LIMIT_TEST_REQUESTS_PER_MINUTE: float = 1200.0
RATE_LIMIT_TEST_MINIMUM_INTERVAL_SECONDS: float = 0.04
GEMINI_MODEL: str = "gemini/gemma-4-26b-a4b-it"
GEMINI_NATIVE_MODEL: str = "gemma-4-26b-a4b-it"


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


class _RecordingClient:
    def __init__(self) -> None:
        self.started_at: list[float] = []

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        self.started_at.append(time.monotonic())
        return CompletionResult(
            call=CallRecord(
                call_id=request.call_id,
                item_id=request.item_id,
                vote_index=request.vote_index,
                attempt_index=request.attempt_index,
                attempt_kind=request.attempt_kind,
                transport_retry_count=0,
                status=CallStatus.SUCCESS,
                model=request.model,
                messages=[],
                usage=TokenUsage(),
                cost=unavailable_cost(),
            )
        )


def _request(*, call_id: str) -> CompletionRequest:
    return CompletionRequest(
        call_id=call_id,
        item_id=call_id,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        model="test/model",
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
    )


async def _rate_limited_start_times() -> list[float]:
    client = _RecordingClient()
    rate_limited = RateLimitedCompletionClient(
        client=client,
        requests_per_minute=RATE_LIMIT_TEST_REQUESTS_PER_MINUTE,
    )
    await asyncio.gather(
        rate_limited.complete(request=_request(call_id="call-1")),
        rate_limited.complete(request=_request(call_id="call-2")),
    )
    return client.started_at


def test_rate_limited_client_spaces_request_starts() -> None:
    started_at = asyncio.run(_rate_limited_start_times())

    assert len(started_at) == 2, "two requests are delegated"
    assert (
        started_at[1] - started_at[0] >= RATE_LIMIT_TEST_MINIMUM_INTERVAL_SECONDS
    ), "request starts are spaced by the limiter"


def test_gemini_model_name_strips_litellm_prefix() -> None:
    assert _gemini_model_name(model=GEMINI_MODEL) == GEMINI_NATIVE_MODEL


def test_gemini_request_payload_maps_disabled_thinking_to_minimal() -> None:
    request = CompletionRequest(
        call_id="call-1",
        item_id="item-1",
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        model=GEMINI_MODEL,
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content="system"),
            ChatMessage(role=MessageRole.USER, content="choose"),
        ],
        parameters={"max_tokens": 1024, "thinking": {"type": "disabled"}},
    )

    payload = _gemini_request_payload(request=request)

    assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "choose"}]}]
    assert payload["generationConfig"] == {
        "maxOutputTokens": 1024,
        "thinkingConfig": {"thinkingLevel": "MINIMAL"},
    }


def test_gemini_raw_output_excludes_thought_parts() -> None:
    raw_output = _gemini_raw_output_from_payload(
        payload={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hidden", "thought": True},
                            {"text": '{"sense_index": 2}'},
                        ]
                    }
                }
            ]
        }
    )

    assert raw_output == '{"sense_index": 2}'


def test_gemini_usage_maps_prompt_candidate_and_thought_tokens() -> None:
    usage = _gemini_usage_from_payload(
        payload={
            "usageMetadata": {
                "promptTokenCount": 7,
                "cachedContentTokenCount": 2,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 5,
            }
        }
    )

    assert usage.input_tokens == 7
    assert usage.cached_input_tokens == 2
    assert usage.output_tokens == 8
    assert usage.reasoning_output_tokens == 5


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

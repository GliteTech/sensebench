"""LiteLLM transport wrapper."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from sensebench.runner.costs import unavailable_cost
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runs.models import (
    CallRecord,
    CallStatus,
    CostBreakdown,
    CostSourceKind,
    MessageRecord,
    TokenUsage,
)

DEFAULT_TRANSPORT_RETRIES: int = 2
DEFAULT_RETRY_SLEEP_SECONDS: float = 1.0
ROLE_FIELD: str = "role"
CONTENT_FIELD: str = "content"
REPR_FIELD: str = "repr"
CHOICES_FIELD: str = "choices"
MESSAGE_FIELD: str = "message"
TEXT_FIELD: str = "text"
USAGE_FIELD: str = "usage"
PROMPT_TOKENS_FIELD: str = "prompt_tokens"
COMPLETION_TOKENS_FIELD: str = "completion_tokens"
PROMPT_TOKENS_DETAILS_FIELD: str = "prompt_tokens_details"
CACHED_TOKENS_FIELD: str = "cached_tokens"
COMPLETION_TOKENS_DETAILS_FIELD: str = "completion_tokens_details"
REASONING_TOKENS_FIELD: str = "reasoning_tokens"
MODEL_FIELD: str = "model"
INPUT_COST_PER_TOKEN_FIELD: str = "input_cost_per_token"
CACHE_READ_INPUT_TOKEN_COST_FIELD: str = "cache_read_input_token_cost"
OUTPUT_COST_PER_TOKEN_FIELD: str = "output_cost_per_token"
OPENROUTER_USAGE_COST_FIELD: str = "cost"
OPENROUTER_COST_DETAILS_FIELD: str = "cost_details"
OPENROUTER_PROMPT_COST_FIELD: str = "upstream_inference_prompt_cost"
OPENROUTER_COMPLETIONS_COST_FIELD: str = "upstream_inference_completions_cost"


class CompletionClient(Protocol):
    async def complete(self, *, request: CompletionRequest) -> CompletionResult: ...


def _messages_payload(*, request: CompletionRequest) -> list[dict[str, str]]:
    return [
        {
            ROLE_FIELD: message.role.value,
            CONTENT_FIELD: message.content,
        }
        for message in request.messages
    ]


def _response_to_dict(*, response: object) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    if isinstance(response, dict):
        return dict(response)
    return {REPR_FIELD: repr(response)}


def _raw_output_from_response(*, payload: dict[str, object]) -> str | None:
    choices = payload.get(CHOICES_FIELD)
    if not isinstance(choices, list) or len(choices) == 0:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get(MESSAGE_FIELD)
    if isinstance(message, dict):
        content = message.get(CONTENT_FIELD)
        if isinstance(content, str):
            return content
    text = first.get(TEXT_FIELD)
    if isinstance(text, str):
        return text
    return None


def _usage_from_payload(*, payload: dict[str, object]) -> TokenUsage:
    usage = payload.get(USAGE_FIELD)
    if not isinstance(usage, dict):
        return TokenUsage()
    input_tokens = usage.get(PROMPT_TOKENS_FIELD)
    output_tokens = usage.get(COMPLETION_TOKENS_FIELD)
    cached_input_tokens: int | None = None
    prompt_details = usage.get(PROMPT_TOKENS_DETAILS_FIELD)
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get(CACHED_TOKENS_FIELD)
        if isinstance(cached_tokens, int):
            cached_input_tokens = cached_tokens
    reasoning_output_tokens: int | None = None
    completion_details = usage.get(COMPLETION_TOKENS_DETAILS_FIELD)
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get(REASONING_TOKENS_FIELD)
        if isinstance(reasoning_tokens, int):
            reasoning_output_tokens = reasoning_tokens
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        reasoning_output_tokens=reasoning_output_tokens,
    )


def _model_from_payload(*, payload: dict[str, object], requested_model: str) -> str:
    raw_model = payload.get(MODEL_FIELD)
    if isinstance(raw_model, str) and len(raw_model) > 0:
        return raw_model
    return requested_model


def _unit_price(*, model_info: dict[str, object], key: str) -> float | None:
    value = model_info.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _component_cost(*, tokens: int | None, unit_price: float | None) -> float | None:
    if tokens is None:
        return None
    if tokens == 0:
        return 0.0
    if unit_price is None:
        return None
    return tokens * unit_price


def _completion_cost(
    *,
    litellm_module: Any,
    response: object,
) -> float | None:
    try:
        raw_cost = litellm_module.completion_cost(completion_response=response)
    except Exception:
        return None
    if isinstance(raw_cost, int | float):
        return float(raw_cost)
    return None


def _numeric_cost_field(*, payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def _provider_reported_cost(*, payload: dict[str, object]) -> CostBreakdown | None:
    usage = payload.get(USAGE_FIELD)
    if not isinstance(usage, dict):
        return None
    total_usd = _numeric_cost_field(payload=usage, key=OPENROUTER_USAGE_COST_FIELD)
    if total_usd is None:
        return None
    cost_details = usage.get(OPENROUTER_COST_DETAILS_FIELD)
    detail_payload = cost_details if isinstance(cost_details, dict) else {}
    return CostBreakdown(
        total_usd=total_usd,
        input_uncached_usd=_numeric_cost_field(
            payload=detail_payload,
            key=OPENROUTER_PROMPT_COST_FIELD,
        ),
        output_usd=_numeric_cost_field(
            payload=detail_payload,
            key=OPENROUTER_COMPLETIONS_COST_FIELD,
        ),
        source=CostSourceKind.PROVIDER_REPORTED,
    )


def _cost_from_response(
    *,
    litellm_module: Any,
    payload: dict[str, object],
    response: object,
    model: str,
    usage: TokenUsage,
) -> CostBreakdown:
    provider_cost = _provider_reported_cost(payload=payload)
    if provider_cost is not None:
        return provider_cost
    total_usd = _completion_cost(litellm_module=litellm_module, response=response)
    raw_model_info = litellm_module.model_cost.get(model)
    model_info: dict[str, object] = raw_model_info if isinstance(raw_model_info, dict) else {}
    input_unit_price = _unit_price(model_info=model_info, key=INPUT_COST_PER_TOKEN_FIELD)
    cached_input_unit_price = _unit_price(
        model_info=model_info,
        key=CACHE_READ_INPUT_TOKEN_COST_FIELD,
    )
    output_unit_price = _unit_price(model_info=model_info, key=OUTPUT_COST_PER_TOKEN_FIELD)

    cached_tokens = usage.cached_input_tokens if usage.cached_input_tokens is not None else 0
    uncached_tokens: int | None = None
    if usage.input_tokens is not None:
        uncached_tokens = max(usage.input_tokens - cached_tokens, 0)
    input_uncached_usd = _component_cost(
        tokens=uncached_tokens,
        unit_price=input_unit_price,
    )
    input_cached_usd = _component_cost(
        tokens=cached_tokens,
        unit_price=cached_input_unit_price,
    )
    output_usd = _component_cost(tokens=usage.output_tokens, unit_price=output_unit_price)
    component_total = sum(
        value for value in [input_uncached_usd, input_cached_usd, output_usd] if value is not None
    )
    if total_usd is None and component_total > 0:
        total_usd = component_total
    if total_usd is None and input_uncached_usd is None and output_usd is None:
        return unavailable_cost()
    return CostBreakdown(
        total_usd=total_usd,
        input_uncached_usd=input_uncached_usd,
        input_cached_usd=input_cached_usd,
        output_usd=output_usd,
        input_uncached_unit_price_usd=input_unit_price,
        input_cached_unit_price_usd=cached_input_unit_price,
        output_unit_price_usd=output_unit_price,
        source=CostSourceKind.LITELLM_ESTIMATE,
    )


class LiteLlmClient:
    def __init__(
        self,
        *,
        max_transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
        retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    ) -> None:
        self._max_transport_retries = max_transport_retries
        self._retry_sleep_seconds = retry_sleep_seconds

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        import litellm

        litellm.suppress_debug_info = True
        started = time.monotonic()
        retry_count = 0
        last_error: Exception | None = None
        for attempt_number in range(self._max_transport_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=request.model,
                    messages=_messages_payload(request=request),
                    **request.parameters,
                )
                payload = _response_to_dict(response=response)
                usage = _usage_from_payload(payload=payload)
                model = _model_from_payload(payload=payload, requested_model=request.model)
                return CompletionResult(
                    call=CallRecord(
                        call_id=request.call_id,
                        item_id=request.item_id,
                        vote_index=request.vote_index,
                        attempt_index=request.attempt_index,
                        attempt_kind=request.attempt_kind,
                        transport_retry_count=retry_count,
                        status=CallStatus.SUCCESS,
                        model=model,
                        messages=[
                            MessageRecord(role=message.role, content=message.content)
                            for message in request.messages
                        ],
                        raw_output=_raw_output_from_response(payload=payload),
                        raw_response=payload,
                        usage=usage,
                        cost=_cost_from_response(
                            litellm_module=litellm,
                            payload=payload,
                            response=response,
                            model=model,
                            usage=usage,
                        ),
                        latency_seconds=time.monotonic() - started,
                    )
                )
            except Exception as exc:
                last_error = exc
                if attempt_number >= self._max_transport_retries:
                    break
                retry_count += 1
                await asyncio.sleep(self._retry_sleep_seconds * retry_count)
        assert last_error is not None, "last_error is set when transport fails"
        return CompletionResult(
            call=CallRecord(
                call_id=request.call_id,
                item_id=request.item_id,
                vote_index=request.vote_index,
                attempt_index=request.attempt_index,
                attempt_kind=request.attempt_kind,
                transport_retry_count=retry_count,
                status=CallStatus.TRANSPORT_ERROR,
                model=request.model,
                messages=[
                    MessageRecord(role=message.role, content=message.content)
                    for message in request.messages
                ],
                usage=TokenUsage(),
                cost=unavailable_cost(),
                latency_seconds=time.monotonic() - started,
                error_kind=type(last_error).__name__,
                error_message=str(last_error),
            )
        )

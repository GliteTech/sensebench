"""LiteLLM transport wrapper."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from sensebench.prompts.models import MessageRole
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
API_BASE_PARAMETER: str = "api_base"
GEMINI_DEFAULT_API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL_PREFIX: str = "gemini/"
GEMINI_API_KEY_ENV_NAMES: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_GENERATION_CONFIG_FIELD: str = "generationConfig"
GEMINI_SYSTEM_INSTRUCTION_FIELD: str = "systemInstruction"
GEMINI_CONTENTS_FIELD: str = "contents"
GEMINI_PARTS_FIELD: str = "parts"
GEMINI_TEXT_FIELD: str = "text"
GEMINI_THOUGHT_FIELD: str = "thought"
GEMINI_ROLE_FIELD: str = "role"
GEMINI_USER_ROLE: str = "user"
GEMINI_MODEL_ROLE: str = "model"
GEMINI_MAX_OUTPUT_TOKENS_FIELD: str = "maxOutputTokens"
GEMINI_TEMPERATURE_FIELD: str = "temperature"
GEMINI_TOP_P_FIELD: str = "topP"
GEMINI_SEED_FIELD: str = "seed"
GEMINI_THINKING_CONFIG_FIELD: str = "thinkingConfig"
GEMINI_THINKING_LEVEL_FIELD: str = "thinkingLevel"
GEMINI_MINIMAL_THINKING_LEVEL: str = "MINIMAL"
GEMINI_CANDIDATES_FIELD: str = "candidates"
GEMINI_CONTENT_FIELD: str = "content"
GEMINI_USAGE_METADATA_FIELD: str = "usageMetadata"
GEMINI_PROMPT_TOKEN_COUNT_FIELD: str = "promptTokenCount"
GEMINI_CANDIDATES_TOKEN_COUNT_FIELD: str = "candidatesTokenCount"
GEMINI_CACHED_CONTENT_TOKEN_COUNT_FIELD: str = "cachedContentTokenCount"
GEMINI_THOUGHTS_TOKEN_COUNT_FIELD: str = "thoughtsTokenCount"
GEMINI_ERROR_FIELD: str = "error"
GEMINI_ERROR_STATUS_FIELD: str = "status"
GEMINI_ERROR_MESSAGE_FIELD: str = "message"
GEMINI_REQUEST_ID_HEADER: str = "x-request-id"
THINKING_PARAMETER: str = "thinking"
THINKING_TYPE_FIELD: str = "type"
THINKING_DISABLED_VALUE: str = "disabled"
LITELLM_MAX_TOKENS_PARAMETER: str = "max_tokens"
LITELLM_TEMPERATURE_PARAMETER: str = "temperature"
LITELLM_TOP_P_PARAMETER: str = "top_p"
LITELLM_SEED_PARAMETER: str = "seed"
HTTP_TOO_MANY_REQUESTS_STATUS: int = 429
HTTP_SERVER_ERROR_MIN_STATUS: int = 500


class CompletionClient(Protocol):
    async def complete(self, *, request: CompletionRequest) -> CompletionResult: ...


@dataclass(frozen=True, slots=True)
class GeminiError:
    error_kind: str
    error_message: str


class RateLimitedCompletionClient:
    def __init__(
        self,
        *,
        client: CompletionClient,
        requests_per_minute: float,
    ) -> None:
        assert requests_per_minute > 0, "requests_per_minute is positive"
        self._client = client
        self._interval_seconds = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_request_time = 0.0

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        async with self._lock:
            now = time.monotonic()
            wait_seconds = self._next_request_time - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._next_request_time = now + self._interval_seconds
        return await self._client.complete(request=request)


def _messages_payload(*, request: CompletionRequest) -> list[dict[str, str]]:
    return [
        {
            ROLE_FIELD: message.role.value,
            CONTENT_FIELD: message.content,
        }
        for message in request.messages
    ]


def _message_records(*, request: CompletionRequest) -> list[MessageRecord]:
    return [
        MessageRecord(role=message.role, content=message.content)
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


def _gemini_api_key() -> str | None:
    for env_name in GEMINI_API_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value is not None and len(value.strip()) > 0:
            return value
    return None


def _gemini_model_name(*, model: str) -> str:
    if model.startswith(GEMINI_MODEL_PREFIX):
        return model.removeprefix(GEMINI_MODEL_PREFIX)
    return model


def _gemini_base_url(*, parameters: dict[str, object]) -> str:
    value = parameters.get(API_BASE_PARAMETER)
    if isinstance(value, str) and len(value.strip()) > 0:
        return value.strip().rstrip("/")
    return GEMINI_DEFAULT_API_BASE_URL


def _gemini_text_part(*, text: str) -> dict[str, str]:
    return {GEMINI_TEXT_FIELD: text}


def _gemini_generation_config(*, parameters: dict[str, object]) -> dict[str, object]:
    config: dict[str, object] = {}
    max_tokens = parameters.get(LITELLM_MAX_TOKENS_PARAMETER)
    if isinstance(max_tokens, int):
        config[GEMINI_MAX_OUTPUT_TOKENS_FIELD] = max_tokens
    temperature = parameters.get(LITELLM_TEMPERATURE_PARAMETER)
    if isinstance(temperature, int | float):
        config[GEMINI_TEMPERATURE_FIELD] = float(temperature)
    top_p = parameters.get(LITELLM_TOP_P_PARAMETER)
    if isinstance(top_p, int | float):
        config[GEMINI_TOP_P_FIELD] = float(top_p)
    seed = parameters.get(LITELLM_SEED_PARAMETER)
    if isinstance(seed, int):
        config[GEMINI_SEED_FIELD] = seed
    thinking = parameters.get(THINKING_PARAMETER)
    if isinstance(thinking, dict) and thinking.get(THINKING_TYPE_FIELD) == THINKING_DISABLED_VALUE:
        config[GEMINI_THINKING_CONFIG_FIELD] = {
            GEMINI_THINKING_LEVEL_FIELD: GEMINI_MINIMAL_THINKING_LEVEL
        }
    return config


def _gemini_request_payload(*, request: CompletionRequest) -> dict[str, object]:
    contents: list[dict[str, object]] = []
    system_parts: list[dict[str, str]] = []
    for message in request.messages:
        if message.role == MessageRole.SYSTEM:
            system_parts.append(_gemini_text_part(text=message.content))
            continue
        role = GEMINI_MODEL_ROLE if message.role == MessageRole.ASSISTANT else GEMINI_USER_ROLE
        contents.append(
            {
                GEMINI_ROLE_FIELD: role,
                GEMINI_PARTS_FIELD: [_gemini_text_part(text=message.content)],
            }
        )
    payload: dict[str, object] = {GEMINI_CONTENTS_FIELD: contents}
    if len(system_parts) > 0:
        payload[GEMINI_SYSTEM_INSTRUCTION_FIELD] = {GEMINI_PARTS_FIELD: system_parts}
    generation_config = _gemini_generation_config(parameters=request.parameters)
    if len(generation_config) > 0:
        payload[GEMINI_GENERATION_CONFIG_FIELD] = generation_config
    return payload


def _gemini_response_to_dict(*, response: object) -> dict[str, object]:
    if isinstance(response, dict):
        return response
    return {REPR_FIELD: repr(response)}


def _gemini_first_candidate(*, payload: dict[str, object]) -> dict[str, object] | None:
    candidates = payload.get(GEMINI_CANDIDATES_FIELD)
    if not isinstance(candidates, list) or len(candidates) == 0:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    return first


def _gemini_parts(*, payload: dict[str, object]) -> list[dict[str, object]]:
    candidate = _gemini_first_candidate(payload=payload)
    if candidate is None:
        return []
    content = candidate.get(GEMINI_CONTENT_FIELD)
    if not isinstance(content, dict):
        return []
    parts = content.get(GEMINI_PARTS_FIELD)
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, dict)]


def _gemini_raw_output_from_payload(*, payload: dict[str, object]) -> str | None:
    chunks: list[str] = []
    for part in _gemini_parts(payload=payload):
        if part.get(GEMINI_THOUGHT_FIELD) is True:
            continue
        text = part.get(GEMINI_TEXT_FIELD)
        if isinstance(text, str):
            chunks.append(text)
    output = "".join(chunks)
    if len(output) == 0:
        return None
    return output


def _gemini_usage_from_payload(*, payload: dict[str, object]) -> TokenUsage:
    usage = payload.get(GEMINI_USAGE_METADATA_FIELD)
    if not isinstance(usage, dict):
        return TokenUsage()
    input_tokens = usage.get(GEMINI_PROMPT_TOKEN_COUNT_FIELD)
    output_tokens = usage.get(GEMINI_CANDIDATES_TOKEN_COUNT_FIELD)
    reasoning_tokens = usage.get(GEMINI_THOUGHTS_TOKEN_COUNT_FIELD)
    cached_input_tokens = usage.get(GEMINI_CACHED_CONTENT_TOKEN_COUNT_FIELD)
    if isinstance(output_tokens, int) and isinstance(reasoning_tokens, int):
        output_tokens += reasoning_tokens
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        cached_input_tokens=cached_input_tokens if isinstance(cached_input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        reasoning_output_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else None,
    )


def _gemini_error(*, payload: dict[str, object]) -> GeminiError:
    error = payload.get(GEMINI_ERROR_FIELD)
    if isinstance(error, dict):
        status = error.get(GEMINI_ERROR_STATUS_FIELD)
        message = error.get(GEMINI_ERROR_MESSAGE_FIELD)
        return GeminiError(
            error_kind=status if isinstance(status, str) else "GeminiApiError",
            error_message=message if isinstance(message, str) else str(error),
        )
    return GeminiError(error_kind="GeminiApiError", error_message=str(payload))


def _is_retryable_http_status(*, http_status: int) -> bool:
    return (
        http_status == HTTP_TOO_MANY_REQUESTS_STATUS
        or http_status >= HTTP_SERVER_ERROR_MIN_STATUS
    )


def _transport_error_result(
    *,
    request: CompletionRequest,
    retry_count: int,
    started: float,
    error_kind: str,
    error_message: str,
    http_status: int | None = None,
    provider_request_id: str | None = None,
    raw_response: dict[str, object] | None = None,
) -> CompletionResult:
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
            messages=_message_records(request=request),
            raw_response=raw_response,
            usage=TokenUsage(),
            cost=unavailable_cost(),
            latency_seconds=time.monotonic() - started,
            http_status=http_status,
            provider_request_id=provider_request_id,
            error_kind=error_kind,
            error_message=error_message,
        )
    )


class GeminiApiClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
        retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._max_transport_retries = max_transport_retries
        self._retry_sleep_seconds = retry_sleep_seconds
        self._timeout_seconds = timeout_seconds

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        started = time.monotonic()
        api_key = self._api_key if self._api_key is not None else _gemini_api_key()
        if api_key is None:
            return _transport_error_result(
                request=request,
                retry_count=0,
                started=started,
                error_kind="MissingApiKey",
                error_message="GEMINI_API_KEY or GOOGLE_API_KEY is required",
            )
        model = _gemini_model_name(model=request.model)
        base_url = _gemini_base_url(parameters=request.parameters)
        model_path = quote(model, safe="")
        url = f"{base_url}/models/{model_path}:generateContent"
        payload = _gemini_request_payload(request=request)
        retry_count = 0
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            for attempt_number in range(self._max_transport_retries + 1):
                try:
                    response = await client.post(url, params={"key": api_key}, json=payload)
                    raw_payload = response.json()
                    response_payload = _gemini_response_to_dict(response=raw_payload)
                    provider_request_id = response.headers.get(GEMINI_REQUEST_ID_HEADER)
                    if response.status_code == 200:
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
                                messages=_message_records(request=request),
                                raw_output=_gemini_raw_output_from_payload(
                                    payload=response_payload,
                                ),
                                raw_response=response_payload,
                                usage=_gemini_usage_from_payload(payload=response_payload),
                                cost=unavailable_cost(),
                                latency_seconds=time.monotonic() - started,
                                http_status=response.status_code,
                                provider_request_id=provider_request_id,
                            )
                        )
                    if (
                        _is_retryable_http_status(http_status=response.status_code)
                        and attempt_number < self._max_transport_retries
                    ):
                        retry_count += 1
                        await asyncio.sleep(self._retry_sleep_seconds * retry_count)
                        continue
                    gemini_error = _gemini_error(payload=response_payload)
                    return _transport_error_result(
                        request=request,
                        retry_count=retry_count,
                        started=started,
                        error_kind=gemini_error.error_kind,
                        error_message=gemini_error.error_message,
                        http_status=response.status_code,
                        provider_request_id=provider_request_id,
                        raw_response=response_payload,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt_number >= self._max_transport_retries:
                        break
                    retry_count += 1
                    await asyncio.sleep(self._retry_sleep_seconds * retry_count)
        assert last_error is not None, "last_error is set when transport fails"
        return _transport_error_result(
            request=request,
            retry_count=retry_count,
            started=started,
            error_kind=type(last_error).__name__,
            error_message=str(last_error),
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
                        messages=_message_records(request=request),
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
        return _transport_error_result(
            request=request,
            retry_count=retry_count,
            started=started,
            error_kind=type(last_error).__name__,
            error_message=str(last_error),
        )

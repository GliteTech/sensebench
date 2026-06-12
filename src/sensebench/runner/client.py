"""LiteLLM transport wrapper."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runs.models import CallRecord, CallStatus, MessageRecord, TokenUsage

DEFAULT_TRANSPORT_RETRIES: int = 2
DEFAULT_RETRY_SLEEP_SECONDS: float = 1.0


class CompletionClient(Protocol):
    async def complete(self, *, request: CompletionRequest) -> CompletionResult: ...


def _messages_payload(*, request: CompletionRequest) -> list[dict[str, str]]:
    return [
        {
            "role": message.role.value,
            "content": message.content,
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
    return {"repr": repr(response)}


def _raw_output_from_response(*, payload: dict[str, object]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    return None


def _usage_from_payload(*, payload: dict[str, object]) -> TokenUsage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    cached_input_tokens: int | None = None
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
        if isinstance(cached_tokens, int):
            cached_input_tokens = cached_tokens
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
    )


def _model_from_payload(*, payload: dict[str, object], requested_model: str) -> str:
    raw_model = payload.get("model")
    if isinstance(raw_model, str) and len(raw_model) > 0:
        return raw_model
    return requested_model


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
                cost_usd: float | None = None
                try:
                    raw_cost = litellm.completion_cost(completion_response=response)
                    if isinstance(raw_cost, int | float):
                        cost_usd = float(raw_cost)
                except Exception:
                    cost_usd = None
                return CompletionResult(
                    call=CallRecord(
                        call_id=request.call_id,
                        item_id=request.item_id,
                        vote_index=request.vote_index,
                        attempt_index=request.attempt_index,
                        attempt_kind=request.attempt_kind,
                        transport_retry_count=retry_count,
                        status=CallStatus.SUCCESS,
                        model=_model_from_payload(payload=payload, requested_model=request.model),
                        messages=[
                            MessageRecord(role=message.role, content=message.content)
                            for message in request.messages
                        ],
                        raw_output=_raw_output_from_response(payload=payload),
                        raw_response=payload,
                        usage=_usage_from_payload(payload=payload),
                        cost_usd=cost_usd,
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
                cost_usd=None,
                latency_seconds=time.monotonic() - started,
                error_kind=type(last_error).__name__,
                error_message=str(last_error),
            )
        )

"""Evaluate one WSD item."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sensebench.prompts.models import MessageRole, OutputMode
from sensebench.prompts.render import ChatMessage, RenderedTask
from sensebench.runner.client import CompletionClient
from sensebench.runner.extract import extract_sense_index
from sensebench.runner.models import CompletionRequest, ItemEvaluation
from sensebench.runs.models import (
    AttemptKind,
    CallRecord,
    CallStatus,
    CandidateRecord,
    PredictionRecord,
    PredictionStatus,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from sensebench.wordnet import sense_keys_match

DEFAULT_VOTES_PER_ITEM: int = 1
DEFAULT_SEMANTIC_REASKS: int = 1


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    model: str
    votes_per_item: int = DEFAULT_VOTES_PER_ITEM
    semantic_reasks_per_invalid_vote: int = DEFAULT_SEMANTIC_REASKS
    llm_parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoteOutcome:
    vote: VoteRecord
    calls: list[CallRecord]


def _candidate_records(*, rendered: RenderedTask) -> list[CandidateRecord]:
    return [
        CandidateRecord(
            index=candidate.index,
            sense_key=candidate.sense_key,
            synset_id=candidate.synset_id,
        )
        for candidate in rendered.candidates
    ]


def _zero_usage() -> TokenUsage:
    return TokenUsage(input_tokens=None, cached_input_tokens=None, output_tokens=None)


def _sum_optional_ints(*, values: list[int | None]) -> int | None:
    observed_values: list[int] = [value for value in values if value is not None]
    if len(observed_values) == 0:
        return None
    return sum(observed_values)


def _sum_usage(*, calls: list[CallRecord]) -> TokenUsage:
    return TokenUsage(
        input_tokens=_sum_optional_ints(
            values=[call.usage.input_tokens for call in calls],
        ),
        cached_input_tokens=_sum_optional_ints(
            values=[call.usage.cached_input_tokens for call in calls],
        ),
        output_tokens=_sum_optional_ints(
            values=[call.usage.output_tokens for call in calls],
        ),
    )


def _sum_cost(*, calls: list[CallRecord]) -> float | None:
    values: list[float] = [call.cost_usd for call in calls if call.cost_usd is not None]
    if len(values) == 0:
        return None
    return sum(values)


def _sum_latency(*, calls: list[CallRecord]) -> float | None:
    values: list[float] = [
        call.latency_seconds for call in calls if call.latency_seconds is not None
    ]
    if len(values) == 0:
        return None
    return sum(values)


def _is_correct(*, predicted_sense_key: str | None, gold_sense_keys: list[str]) -> bool | None:
    if predicted_sense_key is None:
        return None
    if predicted_sense_key in gold_sense_keys:
        return True
    try:
        return sense_keys_match(
            predicted_sense_key=predicted_sense_key,
            gold_sense_keys=gold_sense_keys,
        )
    except RuntimeError:
        return False


def _repair_instruction(*, output_mode: OutputMode, candidate_count: int) -> str:
    if output_mode == OutputMode.JSON_SENSE_INDEX:
        return (
            "Your previous answer was invalid. Choose one candidate index between "
            f"1 and {candidate_count}. Return only valid JSON exactly like "
            '{"sense_index": 1}.'
        )
    if output_mode == OutputMode.PLAIN_SENSE_INDEX:
        return (
            "Your previous answer was invalid. Choose one candidate index between "
            f"1 and {candidate_count}. Return only the integer."
        )
    raise ValueError(f"Unsupported output mode: {output_mode}")


def _repair_messages(*, rendered: RenderedTask, previous_output: str | None) -> list[ChatMessage]:
    messages = list(rendered.messages)
    if previous_output is not None:
        messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=previous_output,
            )
        )
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content=_repair_instruction(
                output_mode=rendered.output_mode,
                candidate_count=len(rendered.candidates),
            ),
        )
    )
    return messages


def _call_id(
    *,
    item_id: str,
    vote_index: int,
    attempt_index: int,
) -> str:
    return f"{item_id}__v{vote_index}__a{attempt_index}"


def _sense_key_for_index(*, rendered: RenderedTask, sense_index: int | None) -> str | None:
    if sense_index is None:
        return None
    for candidate in rendered.candidates:
        if candidate.index == sense_index:
            return candidate.sense_key
    return None


async def _run_one_vote(
    *,
    rendered: RenderedTask,
    client: CompletionClient,
    config: EvaluationConfig,
    vote_index: int,
) -> VoteOutcome:
    calls: list[CallRecord] = []
    messages: list[ChatMessage] = rendered.messages
    invalid_reason: str | None = None
    for attempt_index in range(1, config.semantic_reasks_per_invalid_vote + 2):
        attempt_kind = AttemptKind.INITIAL if attempt_index == 1 else AttemptKind.SEMANTIC_REASK
        request = CompletionRequest(
            call_id=_call_id(
                item_id=rendered.item_id,
                vote_index=vote_index,
                attempt_index=attempt_index,
            ),
            item_id=rendered.item_id,
            vote_index=vote_index,
            attempt_index=attempt_index,
            attempt_kind=attempt_kind,
            model=config.model,
            messages=messages,
            parameters=config.llm_parameters,
        )
        completion = await client.complete(request=request)
        calls.append(completion.call)
        if completion.call.status == CallStatus.TRANSPORT_ERROR:
            return VoteOutcome(
                vote=VoteRecord(
                    vote_index=vote_index,
                    status=VoteStatus.TRANSPORT_ERROR,
                    call_ids=[call.call_id for call in calls],
                    invalid_reason=completion.call.error_kind,
                ),
                calls=calls,
            )
        extracted = extract_sense_index(
            text=completion.call.raw_output,
            output_mode=rendered.output_mode,
            candidate_count=len(rendered.candidates),
        )
        if extracted.sense_index is not None:
            return VoteOutcome(
                vote=VoteRecord(
                    vote_index=vote_index,
                    status=VoteStatus.SUCCESS,
                    chosen_sense_index=extracted.sense_index,
                    chosen_sense_key=_sense_key_for_index(
                        rendered=rendered,
                        sense_index=extracted.sense_index,
                    ),
                    call_ids=[call.call_id for call in calls],
                ),
                calls=calls,
            )
        invalid_reason = (
            extracted.invalid_reason.value if extracted.invalid_reason is not None else None
        )
        messages = _repair_messages(rendered=rendered, previous_output=completion.call.raw_output)
    return VoteOutcome(
        vote=VoteRecord(
            vote_index=vote_index,
            status=VoteStatus.INVALID_OUTPUT,
            call_ids=[call.call_id for call in calls],
            invalid_reason=invalid_reason,
        ),
        calls=calls,
    )


def _choose_prediction(*, votes: list[VoteRecord]) -> int | None:
    valid_indexes: list[int] = [
        vote.chosen_sense_index
        for vote in votes
        if vote.status == VoteStatus.SUCCESS and vote.chosen_sense_index is not None
    ]
    if len(valid_indexes) == 0:
        return None
    counts = Counter(valid_indexes)
    highest_count = max(counts.values())
    tied: set[int] = {
        sense_index for sense_index, count in counts.items() if count == highest_count
    }
    for sense_index in valid_indexes:
        if sense_index in tied:
            return sense_index
    return None


def _monosemous_evaluation(*, rendered: RenderedTask, gold_sense_keys: list[str]) -> ItemEvaluation:
    candidate = rendered.candidates[0]
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=_candidate_records(rendered=rendered),
        votes=[],
        predicted_sense_index=candidate.index,
        predicted_sense_key=candidate.sense_key,
        is_correct=_is_correct(
            predicted_sense_key=candidate.sense_key,
            gold_sense_keys=gold_sense_keys,
        ),
        status=PredictionStatus.MONOSEMOUS,
        was_monosemous=True,
        usage=_zero_usage(),
    )
    return ItemEvaluation(prediction=prediction, calls=[], rendered=rendered)


def _no_candidates_evaluation(
    *, rendered: RenderedTask, gold_sense_keys: list[str]
) -> ItemEvaluation:
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=[],
        votes=[],
        is_correct=None,
        status=PredictionStatus.NO_CANDIDATES,
        was_monosemous=False,
        usage=_zero_usage(),
    )
    return ItemEvaluation(prediction=prediction, calls=[], rendered=rendered)


async def evaluate_item(
    *,
    rendered: RenderedTask,
    gold_sense_keys: list[str],
    client: CompletionClient,
    config: EvaluationConfig,
) -> ItemEvaluation:
    if len(rendered.candidates) == 0:
        return _no_candidates_evaluation(rendered=rendered, gold_sense_keys=gold_sense_keys)
    if len(rendered.candidates) == 1:
        return _monosemous_evaluation(rendered=rendered, gold_sense_keys=gold_sense_keys)

    outcomes: list[VoteOutcome] = []
    for vote_index in range(1, config.votes_per_item + 1):
        outcomes.append(
            await _run_one_vote(
                rendered=rendered,
                client=client,
                config=config,
                vote_index=vote_index,
            )
        )
    votes: list[VoteRecord] = [outcome.vote for outcome in outcomes]
    calls: list[CallRecord] = [call for outcome in outcomes for call in outcome.calls]
    chosen_index = _choose_prediction(votes=votes)
    chosen_key = _sense_key_for_index(rendered=rendered, sense_index=chosen_index)
    status = PredictionStatus.SUCCESS if chosen_key is not None else PredictionStatus.NO_VALID_VOTE
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=_candidate_records(rendered=rendered),
        votes=votes,
        predicted_sense_index=chosen_index,
        predicted_sense_key=chosen_key,
        is_correct=_is_correct(predicted_sense_key=chosen_key, gold_sense_keys=gold_sense_keys),
        status=status,
        was_monosemous=False,
        usage=_sum_usage(calls=calls),
        cost_usd=_sum_cost(calls=calls),
        latency_seconds=_sum_latency(calls=calls),
    )
    return ItemEvaluation(prediction=prediction, calls=calls, rendered=rendered)

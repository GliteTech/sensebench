"""Evaluate one WSD item."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import assert_never

from sensebench.datasets.models import ItemID, SenseKey
from sensebench.prompts.models import SENSE_INDEX_FIELD, MessageRole, OutputMode
from sensebench.prompts.render import ChatMessage, RenderedTask
from sensebench.runner.client import CompletionClient
from sensebench.runner.costs import no_call_cost, sum_costs
from sensebench.runner.extract import ValidSenseIndexExtraction, extract_sense_index
from sensebench.runner.models import CompletionRequest, ItemEvaluation
from sensebench.runs.models import (
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    CandidateRecord,
    InvalidOutputReason,
    ModelID,
    PredictionRecord,
    PredictionStatus,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from sensebench.wordnet import sense_keys_match

DEFAULT_VOTES_PER_ITEM: int = 1
DEFAULT_SEMANTIC_REASKS: int = 1
SENSE_INDEX_EXAMPLE_JSON: str = f'{{"{SENSE_INDEX_FIELD}": 1}}'


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    model: ModelID
    votes_per_item: int = DEFAULT_VOTES_PER_ITEM
    semantic_reasks_per_invalid_vote: int = DEFAULT_SEMANTIC_REASKS
    llm_parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.votes_per_item >= 1, "votes_per_item is at least 1"
        assert self.semantic_reasks_per_invalid_vote >= 0, (
            "semantic_reasks_per_invalid_vote is non-negative"
        )


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


def _no_call_usage() -> TokenUsage:
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
        reasoning_output_tokens=_sum_optional_ints(
            values=[call.usage.reasoning_output_tokens for call in calls],
        ),
    )


def _sum_latency(*, calls: list[CallRecord]) -> float | None:
    values: list[float] = [
        call.latency_seconds for call in calls if call.latency_seconds is not None
    ]
    if len(values) == 0:
        return None
    return sum(values)


def prediction_is_correct(
    *,
    predicted_sense_key: SenseKey | None,
    gold_sense_keys: list[SenseKey],
) -> bool | None:
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
            f"{SENSE_INDEX_EXAMPLE_JSON}."
        )
    if output_mode == OutputMode.PLAIN_SENSE_INDEX:
        return (
            "Your previous answer was invalid. Choose one candidate index between "
            f"1 and {candidate_count}. Return only the integer."
        )
    assert_never(output_mode)


def _repair_messages(*, rendered: RenderedTask, previous_output: str | None) -> list[ChatMessage]:
    messages: list[ChatMessage] = list(rendered.messages)
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
    item_id: ItemID,
    vote_index: int,
    attempt_index: int,
) -> CallID:
    return f"{item_id}__v{vote_index}__a{attempt_index}"


def _sense_key_for_index(
    *,
    rendered: RenderedTask,
    sense_index: int | None,
) -> SenseKey | None:
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
    invalid_reason: InvalidOutputReason | str | None = None
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
        if isinstance(extracted, ValidSenseIndexExtraction):
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
        invalid_reason = extracted.invalid_reason
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


def choose_prediction(*, votes: list[VoteRecord]) -> int | None:
    valid_indexes: list[int] = [
        vote.chosen_sense_index
        for vote in votes
        if vote.status == VoteStatus.SUCCESS and vote.chosen_sense_index is not None
    ]
    if len(valid_indexes) == 0:
        return None
    counts: Counter[int] = Counter(valid_indexes)
    highest_count = max(counts.values())
    tied: set[int] = {
        sense_index for sense_index, count in counts.items() if count == highest_count
    }
    for sense_index in valid_indexes:
        if sense_index in tied:
            return sense_index
    return None


def _index_for_sense_key(
    *,
    rendered: RenderedTask,
    sense_key: SenseKey | None,
) -> int | None:
    if sense_key is None:
        return None
    for candidate in rendered.candidates:
        if candidate.sense_key == sense_key:
            return candidate.index
    return None


def _choose_prediction_key(*, votes: list[VoteRecord]) -> SenseKey | None:
    """Majority over sense keys.

    Used when each vote shuffles the candidate order independently, so the
    per-vote indices are not comparable; the sense key is the stable identity.
    """
    valid_keys: list[SenseKey] = [
        vote.chosen_sense_key
        for vote in votes
        if vote.status == VoteStatus.SUCCESS and vote.chosen_sense_key is not None
    ]
    if len(valid_keys) == 0:
        return None
    counts: Counter[SenseKey] = Counter(valid_keys)
    highest_count = max(counts.values())
    tied: set[SenseKey] = {key for key, count in counts.items() if count == highest_count}
    for key in valid_keys:
        if key in tied:
            return key
    return None


def _monosemous_evaluation(
    *,
    rendered: RenderedTask,
    gold_sense_keys: list[SenseKey],
) -> ItemEvaluation:
    assert len(rendered.candidates) == 1, "rendered task has exactly one candidate"
    candidate = rendered.candidates[0]
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=_candidate_records(rendered=rendered),
        votes=[],
        predicted_sense_index=candidate.index,
        predicted_sense_key=candidate.sense_key,
        is_correct=prediction_is_correct(
            predicted_sense_key=candidate.sense_key,
            gold_sense_keys=gold_sense_keys,
        ),
        status=PredictionStatus.MONOSEMOUS,
        was_monosemous=True,
        usage=_no_call_usage(),
        cost=no_call_cost(),
    )
    return ItemEvaluation(prediction=prediction, calls=[], rendered=rendered)


def _no_candidates_evaluation(
    *,
    rendered: RenderedTask,
    gold_sense_keys: list[SenseKey],
) -> ItemEvaluation:
    assert len(rendered.candidates) == 0, "rendered task has no candidates"
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=[],
        votes=[],
        is_correct=None,
        status=PredictionStatus.NO_CANDIDATES,
        was_monosemous=False,
        usage=_no_call_usage(),
        cost=no_call_cost(),
    )
    return ItemEvaluation(prediction=prediction, calls=[], rendered=rendered)


async def evaluate_item(
    *,
    rendered: RenderedTask,
    gold_sense_keys: list[SenseKey],
    client: CompletionClient,
    config: EvaluationConfig,
    render_for_vote: Callable[[int], RenderedTask] | None = None,
) -> ItemEvaluation:
    if len(rendered.candidates) == 0:
        return _no_candidates_evaluation(rendered=rendered, gold_sense_keys=gold_sense_keys)
    if len(rendered.candidates) == 1:
        return _monosemous_evaluation(rendered=rendered, gold_sense_keys=gold_sense_keys)

    outcomes: list[VoteOutcome] = []
    for vote_index in range(1, config.votes_per_item + 1):
        vote_rendered = rendered if render_for_vote is None else render_for_vote(vote_index)
        outcomes.append(
            await _run_one_vote(
                rendered=vote_rendered,
                client=client,
                config=config,
                vote_index=vote_index,
            )
        )
    votes: list[VoteRecord] = [outcome.vote for outcome in outcomes]
    calls: list[CallRecord] = [call for outcome in outcomes for call in outcome.calls]
    if render_for_vote is None:
        # Default path: all votes share one render, so the indices are comparable.
        chosen_index = choose_prediction(votes=votes)
        chosen_key = _sense_key_for_index(rendered=rendered, sense_index=chosen_index)
    else:
        # Per-vote shuffle: each vote uses its own candidate order, so indices are
        # not comparable across votes; aggregate by sense key and report the index
        # in the canonical (base) candidate order.
        chosen_key = _choose_prediction_key(votes=votes)
        chosen_index = _index_for_sense_key(rendered=rendered, sense_key=chosen_key)
    status = PredictionStatus.SUCCESS if chosen_key is not None else PredictionStatus.NO_VALID_VOTE
    prediction = PredictionRecord(
        item_id=rendered.item_id,
        gold_sense_keys=gold_sense_keys,
        candidates=_candidate_records(rendered=rendered),
        votes=votes,
        predicted_sense_index=chosen_index,
        predicted_sense_key=chosen_key,
        is_correct=prediction_is_correct(
            predicted_sense_key=chosen_key,
            gold_sense_keys=gold_sense_keys,
        ),
        status=status,
        was_monosemous=False,
        usage=_sum_usage(calls=calls),
        cost=sum_costs(costs=[call.cost for call in calls]),
        latency_seconds=_sum_latency(calls=calls),
    )
    return ItemEvaluation(prediction=prediction, calls=calls, rendered=rendered)

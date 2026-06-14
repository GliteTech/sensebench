from __future__ import annotations

from asyncio import run as run_async
from json import dumps

from sensebench.datasets.context import ContextWindow
from sensebench.datasets.models import ItemID, SenseKey
from sensebench.prompts.models import SENSE_INDEX_FIELD, MessageRole, OutputMode, PromptID
from sensebench.prompts.render import CandidateChoice, ChatMessage, RenderedTask
from sensebench.runner.client import CompletionClient
from sensebench.runner.evaluate import EvaluationConfig, evaluate_item
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runs.models import (
    AttemptKind,
    CallRecord,
    CallStatus,
    CostBreakdown,
    CostSourceKind,
    MessageRecord,
    ModelID,
    TokenUsage,
)
from sensebench.wordnet import SynsetID

ITEM_ID: ItemID = "i1"
PROMPT_ID: PromptID = "p001"
FAKE_MODEL: ModelID = "fake"
FIRST_SENSE_KEY: SenseKey = "sense-1"
SECOND_SENSE_KEY: SenseKey = "sense-2"
FIRST_SYNSET_ID: SynsetID = "synset-1"
SECOND_SYNSET_ID: SynsetID = "synset-2"
SENSE_KEYS_BY_INDEX: dict[int, SenseKey] = {
    1: FIRST_SENSE_KEY,
    2: SECOND_SENSE_KEY,
}
SYNSET_IDS_BY_INDEX: dict[int, SynsetID] = {
    1: FIRST_SYNSET_ID,
    2: SECOND_SYNSET_ID,
}
USER_MESSAGE_CONTENT: str = "choose"
RENDER_HASH: str = "sha256:test"
INVALID_RAW_OUTPUT: str = "not json"
INPUT_TOKENS: int = 1
OUTPUT_TOKENS: int = 1
CALL_COST_USD: float = 0.01
CALL_LATENCY_SECONDS: float = 0.1


def raw_output_for_sense_index(*, sense_index: int) -> str:
    return dumps({SENSE_INDEX_FIELD: sense_index})


class FakeClient(CompletionClient):
    def __init__(self, *, outputs: list[str]) -> None:
        self.outputs: list[str] = list(outputs)

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        assert len(self.outputs) > 0, "outputs contains a completion response"
        output = self.outputs.pop(0)
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
                messages=[
                    MessageRecord(role=message.role, content=message.content)
                    for message in request.messages
                ],
                raw_output=output,
                usage=TokenUsage(
                    input_tokens=INPUT_TOKENS,
                    cached_input_tokens=None,
                    output_tokens=OUTPUT_TOKENS,
                ),
                cost=CostBreakdown(
                    total_usd=CALL_COST_USD,
                    source=CostSourceKind.LITELLM_ESTIMATE,
                ),
                latency_seconds=CALL_LATENCY_SECONDS,
            )
        )


def _rendered(*, candidate_count: int) -> RenderedTask:
    assert candidate_count >= 1, "candidate_count is at least one"
    assert candidate_count <= len(SENSE_KEYS_BY_INDEX), "candidate_count has fixture senses"
    candidates: list[CandidateChoice] = [
        CandidateChoice(
            index=index,
            sense_key=SENSE_KEYS_BY_INDEX[index],
            synset_id=SYNSET_IDS_BY_INDEX[index],
        )
        for index in range(1, candidate_count + 1)
    ]
    return RenderedTask(
        item_id=ITEM_ID,
        prompt_id=PROMPT_ID,
        messages=[ChatMessage(role=MessageRole.USER, content=USER_MESSAGE_CONTENT)],
        candidates=candidates,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        render_hash=RENDER_HASH,
        shuffle_seed=None,
        context=ContextWindow(
            text="x",
            target_start_char=0,
            target_end_char=1,
            sentences_before=0,
            sentences_after=0,
        ),
    )


def test_invalid_output_gets_one_semantic_reask() -> None:
    evaluation = run_async(
        evaluate_item(
            rendered=_rendered(candidate_count=2),
            gold_sense_keys=[SECOND_SENSE_KEY],
            client=FakeClient(
                outputs=[
                    INVALID_RAW_OUTPUT,
                    raw_output_for_sense_index(sense_index=2),
                ]
            ),
            config=EvaluationConfig(model=FAKE_MODEL),
        )
    )

    assert evaluation.prediction.predicted_sense_key == SECOND_SENSE_KEY
    assert len(evaluation.calls) == 2
    assert evaluation.calls[1].attempt_kind == AttemptKind.SEMANTIC_REASK


def test_majority_tie_uses_earliest_vote() -> None:
    evaluation = run_async(
        evaluate_item(
            rendered=_rendered(candidate_count=2),
            gold_sense_keys=[SECOND_SENSE_KEY],
            client=FakeClient(
                outputs=[
                    raw_output_for_sense_index(sense_index=2),
                    raw_output_for_sense_index(sense_index=1),
                ]
            ),
            config=EvaluationConfig(model=FAKE_MODEL, votes_per_item=2),
        )
    )

    assert evaluation.prediction.predicted_sense_key == SECOND_SENSE_KEY


def test_monosemous_item_short_circuits_without_calls() -> None:
    evaluation = run_async(
        evaluate_item(
            rendered=_rendered(candidate_count=1),
            gold_sense_keys=[FIRST_SENSE_KEY],
            client=FakeClient(outputs=[]),
            config=EvaluationConfig(model=FAKE_MODEL),
        )
    )

    assert evaluation.prediction.was_monosemous is True
    assert len(evaluation.calls) == 0


SYNSET_BY_KEY: dict[SenseKey, SynsetID] = {
    FIRST_SENSE_KEY: FIRST_SYNSET_ID,
    SECOND_SENSE_KEY: SECOND_SYNSET_ID,
}


def _rendered_ordered(*, ordered_keys: list[SenseKey]) -> RenderedTask:
    candidates: list[CandidateChoice] = [
        CandidateChoice(index=index, sense_key=key, synset_id=SYNSET_BY_KEY[key])
        for index, key in enumerate(ordered_keys, start=1)
    ]
    return RenderedTask(
        item_id=ITEM_ID,
        prompt_id=PROMPT_ID,
        messages=[ChatMessage(role=MessageRole.USER, content=USER_MESSAGE_CONTENT)],
        candidates=candidates,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        render_hash=RENDER_HASH,
        shuffle_seed=None,
        context=ContextWindow(
            text="x",
            target_start_char=0,
            target_end_char=1,
            sentences_before=0,
            sentences_after=0,
        ),
    )


def test_shuffle_per_vote_aggregates_by_sense_key() -> None:
    # Every vote answers index 1, but the per-vote shuffle maps index 1 to a
    # different sense each time. Index-based aggregation would see a (wrong)
    # unanimous "index 1"; key-based aggregation correctly tallies the senses.
    per_vote: dict[int, RenderedTask] = {
        1: _rendered_ordered(ordered_keys=[FIRST_SENSE_KEY, SECOND_SENSE_KEY]),
        2: _rendered_ordered(ordered_keys=[SECOND_SENSE_KEY, FIRST_SENSE_KEY]),
        3: _rendered_ordered(ordered_keys=[FIRST_SENSE_KEY, SECOND_SENSE_KEY]),
    }
    evaluation = run_async(
        evaluate_item(
            rendered=_rendered_ordered(ordered_keys=[FIRST_SENSE_KEY, SECOND_SENSE_KEY]),
            gold_sense_keys=[FIRST_SENSE_KEY],
            client=FakeClient(outputs=[raw_output_for_sense_index(sense_index=1)] * 3),
            config=EvaluationConfig(model=FAKE_MODEL, votes_per_item=3),
            render_for_vote=lambda vote_index: per_vote[vote_index],
        )
    )

    # votes chose sense-1, sense-2, sense-1 -> majority sense-1
    chosen_keys = [vote.chosen_sense_key for vote in evaluation.prediction.votes]
    assert chosen_keys == [FIRST_SENSE_KEY, SECOND_SENSE_KEY, FIRST_SENSE_KEY]
    assert evaluation.prediction.predicted_sense_key == FIRST_SENSE_KEY
    # predicted index reported in the canonical (base) order
    assert evaluation.prediction.predicted_sense_index == 1
    assert evaluation.prediction.is_correct is True

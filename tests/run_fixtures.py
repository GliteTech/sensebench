"""Shared builders for run-artifact tests."""

from __future__ import annotations

from sensebench.datasets.models import DatasetBundle, Document, SenseKey, Sentence, Token, WsdItem
from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR
from sensebench.prompts.models import MessageRole, PromptDefinition
from sensebench.prompts.registry import load_prompt_definition
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    CandidateRecord,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    DatasetReference,
    MessageRecord,
    ModelSourceKind,
    MonosemousPolicyKind,
    PredictionRecord,
    PredictionStatus,
    PromptReference,
    RunMetadata,
    RunnerIdentity,
    RunPolicy,
    RunTotals,
    SamplingParameters,
    TieBreakKind,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from sensebench.verify.runs import RunValidationReport, RunValidationRule

ITEM_ID: str = "i1"
CALL_ID: str = "i1__v1__a1"
PROMPT_ID: str = "p001"
FIRST_SENSE_KEY: SenseKey = "sense-1"
SECOND_SENSE_KEY: SenseKey = "sense-2"


def registered_prompt() -> PromptDefinition:
    return load_prompt_definition(path=PROMPT_REGISTRY_DIR / f"{PROMPT_ID}{PROMPT_JSON_SUFFIX}")


def make_metadata(
    *,
    item_count: int,
    correct_count: int,
    accuracy: float | None,
    call_count: int,
    prompt_id: str = PROMPT_ID,
    content_hash: str | None = None,
    dataset_version: str = "1",
    run_id: str = "run-1",
) -> RunMetadata:
    return RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        created_at="2026-06-12T00:00:00+00:00",
        git_commit="abc",
        runner=RunnerIdentity(github_handle="tester"),
        dataset=DatasetReference(
            dataset_id="fixture",
            dataset_version=dataset_version,
            content_hash=content_hash,
            item_count=item_count,
        ),
        prompt=PromptReference(id=prompt_id, sensebench_version="0.1.0"),
        model=CloudLlmReference(
            kind=CLOUD_LLM_KIND,
            display_name="fake",
            requested_model="fake",
            source_kind=ModelSourceKind.UNKNOWN,
        ),
        sampling=SamplingParameters(),
        policy=RunPolicy(
            votes_per_item=1,
            semantic_reasks_per_invalid_vote=1,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        totals=RunTotals(
            item_count=item_count,
            correct_count=correct_count,
            accuracy=accuracy,
            call_count=call_count,
            usage=TokenUsage(),
            cost=CostBreakdown(total_usd=0.0, source=CostSourceKind.NO_CALLS),
        ),
    )


def two_candidates() -> list[CandidateRecord]:
    return [
        CandidateRecord(index=1, sense_key=FIRST_SENSE_KEY, synset_id="syn-1"),
        CandidateRecord(index=2, sense_key=SECOND_SENSE_KEY, synset_id="syn-2"),
    ]


def voted_prediction(
    *,
    chosen_index: int,
    gold_sense_keys: list[SenseKey],
    is_correct: bool,
) -> PredictionRecord:
    chosen_key = f"sense-{chosen_index}"
    return PredictionRecord(
        item_id=ITEM_ID,
        gold_sense_keys=gold_sense_keys,
        candidates=two_candidates(),
        votes=[
            VoteRecord(
                vote_index=1,
                status=VoteStatus.SUCCESS,
                chosen_sense_index=chosen_index,
                chosen_sense_key=chosen_key,
                call_ids=[CALL_ID],
            )
        ],
        predicted_sense_index=chosen_index,
        predicted_sense_key=chosen_key,
        is_correct=is_correct,
        status=PredictionStatus.SUCCESS,
        was_monosemous=False,
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=0.01, source=CostSourceKind.LITELLM_ESTIMATE),
    )


def monosemous_prediction() -> PredictionRecord:
    return PredictionRecord(
        item_id=ITEM_ID,
        gold_sense_keys=[FIRST_SENSE_KEY],
        candidates=[CandidateRecord(index=1, sense_key=FIRST_SENSE_KEY, synset_id="syn-1")],
        votes=[],
        predicted_sense_index=1,
        predicted_sense_key=FIRST_SENSE_KEY,
        is_correct=True,
        status=PredictionStatus.MONOSEMOUS,
        was_monosemous=True,
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=0.0, source=CostSourceKind.NO_CALLS),
    )


def success_call(*, raw_output: str, call_id: CallID = CALL_ID) -> CallRecord:
    return CallRecord(
        call_id=call_id,
        item_id=ITEM_ID,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=CallStatus.SUCCESS,
        model="fake",
        messages=[MessageRecord(role=MessageRole.USER, content="x")],
        raw_output=raw_output,
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=0.01, source=CostSourceKind.LITELLM_ESTIMATE),
    )


def fixture_dataset(
    *,
    gold_sense_keys: list[SenseKey],
    content_hash: str | None = None,
    item_id: str = ITEM_ID,
) -> DatasetBundle:
    return DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=content_hash,
        documents=[],
        items=[
            WsdItem(
                item_id=item_id,
                document_id="d1",
                sentence_id="s1",
                target_token_index=0,
                target_text="bank",
                lemma="bank",
                pos="NOUN",
                gold_sense_keys=gold_sense_keys,
            )
        ],
    )


def renderable_dataset(*, gold_sense_keys: list[SenseKey]) -> DatasetBundle:
    return DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id="d1",
                sentences=[
                    Sentence(
                        sentence_id="s1",
                        tokens=[
                            Token(text="The"),
                            Token(text="bank", item_id=ITEM_ID),
                            Token(text="was"),
                            Token(text="steep"),
                        ],
                    )
                ],
            )
        ],
        items=[
            WsdItem(
                item_id=ITEM_ID,
                document_id="d1",
                sentence_id="s1",
                target_token_index=1,
                target_text="bank",
                lemma="bank",
                pos="NOUN",
                gold_sense_keys=gold_sense_keys,
            )
        ],
    )


def issue_rules(*, report: RunValidationReport) -> set[RunValidationRule]:
    return {issue.rule for issue in report.issues}

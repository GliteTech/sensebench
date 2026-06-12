"""Shared builders for run-artifact tests."""

from __future__ import annotations

from datetime import UTC, datetime

from sensebench.datasets.models import (
    DatasetBundle,
    DatasetID,
    Document,
    DocumentID,
    ItemID,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)
from sensebench.paths import P001_PROMPT_PATH
from sensebench.prompts.models import MessageRole, PromptDefinition, PromptID
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
    ModelID,
    ModelSourceKind,
    MonosemousPolicyKind,
    PredictionRecord,
    PredictionStatus,
    PromptReference,
    RunID,
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
from sensebench.wordnet import SynsetID

ITEM_ID: ItemID = "i1"
CALL_ID: CallID = "i1__v1__a1"
PROMPT_ID: PromptID = "p001"
DEFAULT_RUN_ID: RunID = "run-1"
DATASET_ID: DatasetID = "fixture"
DATASET_VERSION: str = "1"
MODEL_NAME: ModelID = "fake"
DOCUMENT_ID: DocumentID = "d1"
SENTENCE_ID: SentenceID = "s1"
TARGET_TEXT: str = "bank"
TARGET_LEMMA: str = "bank"
TARGET_POS: str = "NOUN"
RUN_CREATED_AT: datetime = datetime(2026, 6, 12, tzinfo=UTC)
GIT_COMMIT: str = "abc"
RUNNER_GITHUB_HANDLE: str = "tester"
SENSEBENCH_VERSION: str = "0.1.0"
FIRST_SENSE_KEY: SenseKey = "sense-1"
SECOND_SENSE_KEY: SenseKey = "sense-2"
FIRST_SYNSET_ID: SynsetID = "syn-1"
SECOND_SYNSET_ID: SynsetID = "syn-2"
USER_MESSAGE_CONTENT: str = "x"
VOTES_PER_ITEM: int = 1
SEMANTIC_REASKS_PER_INVALID_VOTE: int = 1
SUCCESS_CALL_COST_USD: float = 0.01
NO_CALL_COST_USD: float = 0.0
SENSE_KEYS_BY_INDEX: dict[int, SenseKey] = {
    1: FIRST_SENSE_KEY,
    2: SECOND_SENSE_KEY,
}


def registered_prompt() -> PromptDefinition:
    return load_prompt_definition(path=P001_PROMPT_PATH)


def make_metadata(
    *,
    item_count: int,
    correct_count: int,
    accuracy: float | None,
    call_count: int,
    prompt_id: PromptID = PROMPT_ID,
    content_hash: str | None = None,
    dataset_version: str = DATASET_VERSION,
    run_id: RunID = DEFAULT_RUN_ID,
    github_handle: str | None = RUNNER_GITHUB_HANDLE,
) -> RunMetadata:
    return RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        created_at=RUN_CREATED_AT,
        git_commit=GIT_COMMIT,
        runner=RunnerIdentity(github_handle=github_handle),
        dataset=DatasetReference(
            dataset_id=DATASET_ID,
            dataset_version=dataset_version,
            content_hash=content_hash,
            item_count=item_count,
        ),
        prompt=PromptReference(id=prompt_id, sensebench_version=SENSEBENCH_VERSION),
        model=CloudLlmReference(
            kind=CLOUD_LLM_KIND,
            display_name=MODEL_NAME,
            requested_model=MODEL_NAME,
            source_kind=ModelSourceKind.UNKNOWN,
        ),
        sampling=SamplingParameters(),
        policy=RunPolicy(
            votes_per_item=VOTES_PER_ITEM,
            semantic_reasks_per_invalid_vote=SEMANTIC_REASKS_PER_INVALID_VOTE,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        totals=RunTotals(
            item_count=item_count,
            correct_count=correct_count,
            accuracy=accuracy,
            call_count=call_count,
            usage=TokenUsage(),
            cost=CostBreakdown(total_usd=NO_CALL_COST_USD, source=CostSourceKind.NO_CALLS),
        ),
    )


def two_candidates() -> list[CandidateRecord]:
    return [
        CandidateRecord(index=1, sense_key=FIRST_SENSE_KEY, synset_id=FIRST_SYNSET_ID),
        CandidateRecord(index=2, sense_key=SECOND_SENSE_KEY, synset_id=SECOND_SYNSET_ID),
    ]


def voted_prediction(
    *,
    chosen_index: int,
    gold_sense_keys: list[SenseKey],
    is_correct: bool,
) -> PredictionRecord:
    assert chosen_index in SENSE_KEYS_BY_INDEX, "chosen index has a fixture sense key"
    chosen_key: SenseKey = SENSE_KEYS_BY_INDEX[chosen_index]
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
        cost=CostBreakdown(total_usd=SUCCESS_CALL_COST_USD, source=CostSourceKind.LITELLM_ESTIMATE),
    )


def monosemous_prediction() -> PredictionRecord:
    return PredictionRecord(
        item_id=ITEM_ID,
        gold_sense_keys=[FIRST_SENSE_KEY],
        candidates=[CandidateRecord(index=1, sense_key=FIRST_SENSE_KEY, synset_id=FIRST_SYNSET_ID)],
        votes=[],
        predicted_sense_index=1,
        predicted_sense_key=FIRST_SENSE_KEY,
        is_correct=True,
        status=PredictionStatus.MONOSEMOUS,
        was_monosemous=True,
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=NO_CALL_COST_USD, source=CostSourceKind.NO_CALLS),
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
        model=MODEL_NAME,
        messages=[MessageRecord(role=MessageRole.USER, content=USER_MESSAGE_CONTENT)],
        raw_output=raw_output,
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=SUCCESS_CALL_COST_USD, source=CostSourceKind.LITELLM_ESTIMATE),
    )


def fixture_dataset(
    *,
    gold_sense_keys: list[SenseKey],
    content_hash: str | None = None,
    item_id: ItemID = ITEM_ID,
) -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=content_hash,
        documents=[],
        items=[
            WsdItem(
                item_id=item_id,
                document_id=DOCUMENT_ID,
                sentence_id=SENTENCE_ID,
                target_token_index=0,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=gold_sense_keys,
            )
        ],
    )


def renderable_dataset(*, gold_sense_keys: list[SenseKey]) -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id=DOCUMENT_ID,
                sentences=[
                    Sentence(
                        sentence_id=SENTENCE_ID,
                        tokens=[
                            Token(text="The"),
                            Token(text=TARGET_TEXT, item_id=ITEM_ID),
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
                document_id=DOCUMENT_ID,
                sentence_id=SENTENCE_ID,
                target_token_index=1,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=gold_sense_keys,
            )
        ],
    )


def issue_rules(*, report: RunValidationReport) -> set[RunValidationRule]:
    return {issue.rule for issue in report.issues}

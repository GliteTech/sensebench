from __future__ import annotations

from asyncio import run as run_async
from json import dumps
from pathlib import Path

from sensebench.datasets.models import (
    DatasetBundle,
    Document,
    DocumentID,
    ItemID,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)
from sensebench.prompts.models import SENSE_INDEX_FIELD
from sensebench.runner.client import CompletionClient
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runner.repair import repair_run
from sensebench.runner.run import RunConfig, run_benchmark
from sensebench.runs.loaders import load_run_directory
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    CallID,
    CallRecord,
    CallStatus,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    MessageRecord,
    ModelSourceKind,
    PredictionStatus,
    RunnerIdentity,
    SamplingParameters,
    TokenUsage,
)
from sensebench.verify.runs import verify_run_directory
from tests.run_fixtures import (
    DATASET_ID,
    DATASET_VERSION,
    RUNNER_GITHUB_HANDLE,
    TARGET_LEMMA,
    TARGET_POS,
    TARGET_TEXT,
    registered_prompt,
)

GOOD_ITEM_ID: ItemID = "good-item"
BAD_ITEM_ID: ItemID = "bad-item"
PRIMARY_MODEL: str = "primary-model"
FALLBACK_MODEL: str = "fallback-model"
ORIGINAL_RUN_ID: str = "run-original"
REPAIRED_RUN_ID: str = "run-repaired"
NOOP_RUN_ID: str = "run-noop"


class _ScriptedClient(CompletionClient):
    """Returns one scripted output per attempt for each item; None means an
    empty (content-filter-style) response that fails extraction."""

    def __init__(self, *, outputs_by_item: dict[ItemID, list[str | None]]) -> None:
        self._outputs_by_item = {
            item_id: list(outputs) for item_id, outputs in outputs_by_item.items()
        }

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        outputs = self._outputs_by_item[request.item_id]
        raw_output = outputs.pop(0) if len(outputs) > 0 else None
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
                raw_output=raw_output,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                cost=CostBreakdown(source=CostSourceKind.UNAVAILABLE),
            )
        )


def _two_item_dataset() -> DatasetBundle:
    def _document(*, document_id: DocumentID, sentence_id: SentenceID, item_id: ItemID) -> Document:
        return Document(
            document_id=document_id,
            sentences=[
                Sentence(
                    sentence_id=sentence_id,
                    tokens=[
                        Token(text="The"),
                        Token(text=TARGET_TEXT, item_id=item_id),
                        Token(text="was"),
                        Token(text="steep"),
                    ],
                )
            ],
        )

    def _item(*, item_id: ItemID, document_id: DocumentID, sentence_id: SentenceID) -> WsdItem:
        return WsdItem(
            item_id=item_id,
            document_id=document_id,
            sentence_id=sentence_id,
            target_token_index=1,
            target_text=TARGET_TEXT,
            lemma=TARGET_LEMMA,
            pos=TARGET_POS,
            gold_sense_keys=["not-a-real-sense-key"],
        )

    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            _document(document_id="d-good", sentence_id="s-good", item_id=GOOD_ITEM_ID),
            _document(document_id="d-bad", sentence_id="s-bad", item_id=BAD_ITEM_ID),
        ],
        items=[
            _item(item_id=GOOD_ITEM_ID, document_id="d-good", sentence_id="s-good"),
            _item(item_id=BAD_ITEM_ID, document_id="d-bad", sentence_id="s-bad"),
        ],
    )


def _model(*, name: str) -> CloudLlmReference:
    return CloudLlmReference(
        kind=CLOUD_LLM_KIND,
        display_name=name,
        requested_model=name,
        source_kind=ModelSourceKind.UNKNOWN,
    )


def _original_config(*, tmp_path: Path, dataset: DatasetBundle) -> RunConfig:
    return RunConfig(
        run_id=ORIGINAL_RUN_ID,
        output_root=tmp_path,
        dataset=dataset,
        prompt=registered_prompt(),
        model=_model(name=PRIMARY_MODEL),
        runner=RunnerIdentity(github_handle=RUNNER_GITHUB_HANDLE),
        sampling=SamplingParameters(),
        votes_per_item=1,
        semantic_reasks_per_invalid_vote=1,
        concurrency=2,
        show_progress=False,
    )


def test_repair_run_replaces_only_failed_items_and_records_fallback(tmp_path: Path) -> None:
    dataset = _two_item_dataset()
    original_client = _ScriptedClient(
        outputs_by_item={
            GOOD_ITEM_ID: [dumps({SENSE_INDEX_FIELD: 1})],
            BAD_ITEM_ID: [None, None],  # both attempts empty -> no_valid_vote
        }
    )
    completed = run_async(
        run_benchmark(
            config=_original_config(tmp_path=tmp_path, dataset=dataset), client=original_client
        )
    )
    good_prediction = next(p for p in completed.predictions if p.item_id == GOOD_ITEM_ID)
    bad_prediction = next(p for p in completed.predictions if p.item_id == BAD_ITEM_ID)
    assert good_prediction.status == PredictionStatus.SUCCESS
    assert bad_prediction.status == PredictionStatus.NO_VALID_VOTE

    loaded = load_run_directory(run_dir=completed.run_dir)
    fallback_config = RunConfig(
        run_id=REPAIRED_RUN_ID,
        output_root=tmp_path,
        dataset=dataset,
        prompt=registered_prompt(),
        model=_model(name=FALLBACK_MODEL),
        runner=RunnerIdentity(github_handle=RUNNER_GITHUB_HANDLE),
        sampling=SamplingParameters(),
        votes_per_item=loaded.metadata.policy.votes_per_item,
        semantic_reasks_per_invalid_vote=loaded.metadata.policy.semantic_reasks_per_invalid_vote,
        concurrency=2,
        show_progress=False,
    )
    fallback_client = _ScriptedClient(
        outputs_by_item={BAD_ITEM_ID: [dumps({SENSE_INDEX_FIELD: 2})]}
    )

    new_run_dir = run_async(
        repair_run(
            loaded=loaded,
            fallback_config=fallback_config,
            client=fallback_client,
            new_run_id=REPAIRED_RUN_ID,
            output_root=tmp_path,
        )
    )

    repaired = load_run_directory(run_dir=new_run_dir)
    assert repaired.metadata.totals.fallback_used_count == 1
    assert repaired.metadata.fallback_model is not None
    assert repaired.metadata.fallback_model.requested_model == FALLBACK_MODEL
    assert repaired.metadata.model.resolved_model_counts == {PRIMARY_MODEL: 1}
    assert repaired.metadata.fallback_model.resolved_model_counts == {FALLBACK_MODEL: 1}
    assert repaired.metadata.model.display_name == f"{PRIMARY_MODEL}+fallback:{FALLBACK_MODEL}"

    repaired_good = next(p for p in repaired.predictions if p.item_id == GOOD_ITEM_ID)
    repaired_bad = next(p for p in repaired.predictions if p.item_id == BAD_ITEM_ID)
    original_good_call_ids: set[CallID] = {
        call_id for vote in good_prediction.votes for call_id in vote.call_ids
    }
    repaired_good_call_ids: set[CallID] = {
        call_id for vote in repaired_good.votes for call_id in vote.call_ids
    }
    assert repaired_good_call_ids == original_good_call_ids, (
        "untouched item keeps its original calls"
    )
    assert repaired_bad.status == PredictionStatus.SUCCESS
    repaired_bad_call_ids = {call_id for vote in repaired_bad.votes for call_id in vote.call_ids}
    repaired_calls_by_id = {call.call_id: call for call in repaired.calls}
    assert all(
        repaired_calls_by_id[call_id].model == FALLBACK_MODEL for call_id in repaired_bad_call_ids
    ), "repaired item's calls are attributed to the fallback model"

    # No orphaned or colliding calls: every call is referenced by exactly the
    # votes that use it, and the discarded original failed attempts are gone.
    assert repaired.metadata.totals.call_count == len(repaired.calls)
    referenced_call_ids = {
        call_id
        for prediction in repaired.predictions
        for vote in prediction.votes
        for call_id in vote.call_ids
    }
    assert referenced_call_ids == {call.call_id for call in repaired.calls}

    report = verify_run_directory(
        run_dir=new_run_dir, dataset=dataset, prompt=fallback_config.prompt
    )
    assert not report.has_errors(), [issue.message for issue in report.issues]


def test_repair_run_is_a_noop_when_nothing_failed(tmp_path: Path) -> None:
    dataset = _two_item_dataset()
    original_client = _ScriptedClient(
        outputs_by_item={
            GOOD_ITEM_ID: [dumps({SENSE_INDEX_FIELD: 1})],
            BAD_ITEM_ID: [dumps({SENSE_INDEX_FIELD: 1})],
        }
    )
    completed = run_async(
        run_benchmark(
            config=_original_config(tmp_path=tmp_path, dataset=dataset), client=original_client
        )
    )
    loaded = load_run_directory(run_dir=completed.run_dir)
    fallback_config = RunConfig(
        run_id=NOOP_RUN_ID,
        output_root=tmp_path,
        dataset=dataset,
        prompt=registered_prompt(),
        model=_model(name=FALLBACK_MODEL),
        runner=RunnerIdentity(github_handle=RUNNER_GITHUB_HANDLE),
        sampling=SamplingParameters(),
        votes_per_item=loaded.metadata.policy.votes_per_item,
        semantic_reasks_per_invalid_vote=loaded.metadata.policy.semantic_reasks_per_invalid_vote,
        concurrency=2,
        show_progress=False,
    )
    fallback_client = _ScriptedClient(outputs_by_item={})

    new_run_dir = run_async(
        repair_run(
            loaded=loaded,
            fallback_config=fallback_config,
            client=fallback_client,
            new_run_id=NOOP_RUN_ID,
            output_root=tmp_path,
        )
    )

    repaired = load_run_directory(run_dir=new_run_dir)
    assert repaired.metadata.totals.fallback_used_count == 0
    assert repaired.metadata.model.display_name == PRIMARY_MODEL, (
        "no fallback used, no display-name change"
    )
    assert repaired.metadata.model.resolved_model_counts == {PRIMARY_MODEL: 2}
    assert repaired.metadata.fallback_model is not None
    assert repaired.metadata.fallback_model.resolved_model_counts == {}
    assert repaired.metadata.totals.item_count == loaded.metadata.totals.item_count
    assert repaired.metadata.totals.correct_count == loaded.metadata.totals.correct_count

    report = verify_run_directory(
        run_dir=new_run_dir, dataset=dataset, prompt=fallback_config.prompt
    )
    assert not report.has_errors(), [issue.message for issue in report.issues]

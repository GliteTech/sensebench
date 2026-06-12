from __future__ import annotations

from pathlib import Path

from sensebench.prompts.models import MessageRole
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallRecord,
    CallStatus,
    DatasetReference,
    MessageRecord,
    ModelExecutionKind,
    ModelHostingKind,
    ModelReference,
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
)
from sensebench.verify.runs import verify_run_directory


def test_verify_valid_tiny_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    prediction = PredictionRecord(
        item_id="i1",
        gold_sense_keys=["sense-1"],
        candidates=[],
        votes=[],
        predicted_sense_key="sense-1",
        is_correct=True,
        status=PredictionStatus.MONOSEMOUS,
        was_monosemous=True,
        usage=TokenUsage(),
    )
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-06-12T00:00:00+00:00",
        git_commit="abc",
        runner=RunnerIdentity(github_handle="tester"),
        dataset=DatasetReference(
            dataset_id="fixture",
            dataset_version="1",
            item_count=1,
        ),
        prompt=PromptReference(id="p001", sensebench_version="0.1.0"),
        model=ModelReference(
            execution_kind=ModelExecutionKind.LLM,
            display_name="fake",
            requested_model="fake",
            hosting_kind=ModelHostingKind.CLOUD_API,
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
            item_count=1,
            correct_count=1,
            accuracy=1.0,
            call_count=0,
            usage=TokenUsage(),
        ),
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[prediction], calls=[])

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is False


def test_verify_rejects_bad_call_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    call = CallRecord(
        call_id="c1",
        item_id="i1",
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=CallStatus.SUCCESS,
        model="fake",
        messages=[MessageRecord(role=MessageRole.USER, content="x")],
        raw_output='{"sense_index": 1}',
        usage=TokenUsage(),
    )
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id="run-1",
        created_at="2026-06-12T00:00:00+00:00",
        git_commit="abc",
        runner=RunnerIdentity(github_handle="tester"),
        dataset=DatasetReference(dataset_id="fixture", dataset_version="1", item_count=0),
        prompt=PromptReference(id="p001"),
        model=ModelReference(
            execution_kind=ModelExecutionKind.LLM,
            display_name="fake",
            hosting_kind=ModelHostingKind.CLOUD_API,
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
            item_count=0,
            correct_count=0,
            call_count=0,
            usage=TokenUsage(),
        ),
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[call])

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is True

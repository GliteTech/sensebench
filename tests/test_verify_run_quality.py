from __future__ import annotations

from json import dumps
from pathlib import Path

from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    AttemptKind,
    CostBreakdown,
    CostSourceKind,
    InvalidOutputReason,
    PredictionRecord,
    PredictionStatus,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from tests.run_fixtures import (
    CALL_ID,
    DEFAULT_RUN_ID,
    FIRST_SENSE_KEY,
    SECOND_SENSE_KEY,
    SUCCESS_CALL_COST_USD,
    make_metadata,
    success_call,
    two_candidates,
    voted_prediction,
)
from tools.verify_run_quality import (
    QualityThresholds,
    collect_run_quality,
    quality_gate_failures,
)

STRICT_THRESHOLDS: QualityThresholds = QualityThresholds(
    final_no_valid=0,
    transport_errors=0,
    length_finishes=0,
    max_token_hits=0,
    invalid_attempts=0,
    invalid_votes=0,
)


def _raw_json_sense_index(*, sense_index: int) -> str:
    return dumps({"sense_index": sense_index})


def test_quality_gate_accepts_clean_verified_run(tmp_path: Path) -> None:
    run_dir = tmp_path / DEFAULT_RUN_ID
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=_raw_json_sense_index(sense_index=2))],
    )

    summary = collect_run_quality(run_dir=run_dir, dataset=None)
    failures = quality_gate_failures(summary=summary, thresholds=STRICT_THRESHOLDS)

    assert summary.official_verify_ok is True
    assert failures == []


def test_quality_gate_flags_verified_run_with_final_invalid_vote(tmp_path: Path) -> None:
    run_dir = tmp_path / DEFAULT_RUN_ID
    reask_call_id = "i1__v1__a2"
    metadata = make_metadata(item_count=1, correct_count=0, accuracy=0.0, call_count=2)
    prediction = PredictionRecord(
        item_id="i1",
        gold_sense_keys=[FIRST_SENSE_KEY],
        candidates=two_candidates(),
        votes=[
            VoteRecord(
                vote_index=1,
                status=VoteStatus.INVALID_OUTPUT,
                call_ids=[CALL_ID, reask_call_id],
                invalid_reason=InvalidOutputReason.INVALID_JSON,
            )
        ],
        predicted_sense_index=None,
        predicted_sense_key=None,
        is_correct=None,
        status=PredictionStatus.NO_VALID_VOTE,
        was_monosemous=False,
        usage=TokenUsage(),
        cost=CostBreakdown(
            total_usd=SUCCESS_CALL_COST_USD,
            source=CostSourceKind.LITELLM_ESTIMATE,
        ),
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[
            success_call(raw_output="not json"),
            success_call(raw_output="not json", call_id=reask_call_id).model_copy(
                update={
                    "attempt_index": 2,
                    "attempt_kind": AttemptKind.SEMANTIC_REASK,
                },
            ),
        ],
    )

    summary = collect_run_quality(run_dir=run_dir, dataset=None)
    failures = quality_gate_failures(summary=summary, thresholds=STRICT_THRESHOLDS)

    assert summary.official_verify_ok is True
    assert summary.final_no_valid == 1
    assert summary.invalid_attempts == {InvalidOutputReason.INVALID_JSON.value: 2}
    assert summary.invalid_votes == {InvalidOutputReason.INVALID_JSON.value: 1}
    assert any("final no_valid_vote=1" in failure for failure in failures)
    assert any("invalid_attempts=2" in failure for failure in failures)
    assert any("invalid_votes=1" in failure for failure in failures)

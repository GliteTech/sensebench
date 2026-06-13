from __future__ import annotations

from json import dumps
from pathlib import Path

from sensebench.datasets.models import ItemID, SenseKey
from sensebench.prompts.models import SENSE_INDEX_FIELD, PromptID
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    RUN_SCHEMA_VERSION_V1,
    AttemptKind,
    CallID,
    CostBreakdown,
    CostSourceKind,
    RunID,
    RunMetadata,
    SamplingParameters,
    TokenUsage,
)
from sensebench.verify.runs import (
    CALLS_AFTER_SUCCESS_MESSAGE,
    RunValidationRule,
    verify_run_directory,
)
from tests.run_fixtures import (
    CALL_ID,
    DEFAULT_RUN_ID,
    FIRST_SENSE_KEY,
    FIXTURE_BENCHMARK_SECONDS,
    FIXTURE_HOURLY_RATE_USD,
    SECOND_SENSE_KEY,
    fixture_dataset,
    fixture_machine,
    issue_rules,
    make_metadata,
    monosemous_prediction,
    registered_prompt,
    renderable_dataset,
    self_hosted_model,
    success_call,
    voted_prediction,
)

RUN_DIR_NAME: RunID = DEFAULT_RUN_ID
UNKNOWN_PROMPT_ID: PromptID = "p999"
ORPHAN_CALL_ID: CallID = "i1__v9__a9"
SEMANTIC_REASK_CALL_ID: CallID = "i1__v1__a2"
UNKNOWN_ITEM_ID: ItemID = "i2"
FORGED_GOLD_SENSE_KEY: SenseKey = "bank%1:17:00::"
METADATA_CONTENT_HASH: str = f"sha256:{'a' * 64}"
DATASET_CONTENT_HASH: str = f"sha256:{'b' * 64}"


def raw_output_for_sense_index(*, sense_index: int) -> str:
    return dumps({SENSE_INDEX_FIELD: sense_index})


def test_verify_valid_tiny_run(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=0)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[monosemous_prediction()],
        calls=[],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is False


def test_verify_rejects_bad_call_count(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(item_count=0, correct_count=0, accuracy=None, call_count=0)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=1))],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is True


def test_verify_valid_voted_run_passes_with_prompt(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert report.has_errors() is False


def test_verify_detects_flipped_is_correct(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=1,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=1))],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.CORRECTNESS in issue_rules(report=report)


def test_verify_detects_forged_gold_keys_against_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(gold_sense_keys=[FIRST_SENSE_KEY]),
    )

    assert RunValidationRule.DATASET_GOLD_KEYS in issue_rules(report=report)


def test_verify_detects_vote_mismatching_raw_output(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=1,
        gold_sense_keys=[FIRST_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert RunValidationRule.VOTE_EXTRACTION in issue_rules(report=report)


def test_verify_allows_redundant_same_vote_after_success(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    call_ids: list[CallID] = [CALL_ID, SEMANTIC_REASK_CALL_ID]
    votes = [
        prediction.votes[0].model_copy(
            update={"call_ids": call_ids},
        )
    ]
    prediction = prediction.model_copy(update={"votes": votes})
    reask_call = success_call(
        raw_output=raw_output_for_sense_index(sense_index=2),
        call_id=SEMANTIC_REASK_CALL_ID,
    ).model_copy(
        update={
            "attempt_index": 2,
            "attempt_kind": AttemptKind.SEMANTIC_REASK,
        },
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=2)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[
            success_call(raw_output=raw_output_for_sense_index(sense_index=2)),
            reask_call,
        ],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert report.has_errors() is False


def test_verify_rejects_changed_vote_after_success(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=1,
        gold_sense_keys=[FIRST_SENSE_KEY],
        is_correct=True,
    )
    call_ids: list[CallID] = [CALL_ID, SEMANTIC_REASK_CALL_ID]
    votes = [
        prediction.votes[0].model_copy(
            update={"call_ids": call_ids},
        )
    ]
    prediction = prediction.model_copy(update={"votes": votes})
    reask_call = success_call(
        raw_output=raw_output_for_sense_index(sense_index=2),
        call_id=SEMANTIC_REASK_CALL_ID,
    ).model_copy(
        update={
            "attempt_index": 2,
            "attempt_kind": AttemptKind.SEMANTIC_REASK,
        },
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=2)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[
            success_call(raw_output=raw_output_for_sense_index(sense_index=1)),
            reask_call,
        ],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert RunValidationRule.VOTE_EXTRACTION in issue_rules(report=report)
    assert any(issue.message == CALLS_AFTER_SUCCESS_MESSAGE for issue in report.issues)


def test_verify_detects_content_hash_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        content_hash=METADATA_CONTENT_HASH,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(
            gold_sense_keys=[SECOND_SENSE_KEY],
            content_hash=DATASET_CONTENT_HASH,
        ),
    )

    assert RunValidationRule.DATASET_CONTENT_HASH in issue_rules(report=report)


def test_verify_detects_unregistered_prompt_id(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=0,
        correct_count=0,
        accuracy=None,
        call_count=0,
        prompt_id=UNKNOWN_PROMPT_ID,
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.PROMPT_REFERENCE in issue_rules(report=report)


def test_verify_detects_duplicate_call_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=2)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[
            success_call(raw_output=raw_output_for_sense_index(sense_index=2)),
            success_call(raw_output=raw_output_for_sense_index(sense_index=2)),
        ],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.DUPLICATE_CALL in issue_rules(report=report)


def test_verify_detects_orphan_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=2)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[
            success_call(raw_output=raw_output_for_sense_index(sense_index=2)),
            success_call(
                raw_output=raw_output_for_sense_index(sense_index=2),
                call_id=ORPHAN_CALL_ID,
            ),
        ],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.ORPHAN_CALL in issue_rules(report=report)


def test_verify_handles_call_for_unknown_dataset_item(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(gold_sense_keys=[SECOND_SENSE_KEY], item_id=UNKNOWN_ITEM_ID),
        prompt=registered_prompt(),
    )

    rules: set[RunValidationRule] = issue_rules(report=report)
    assert RunValidationRule.DATASET_ITEMS in rules
    assert RunValidationRule.PROMPT_RENDERING in rules


def test_verify_detects_forged_candidate_set(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    gold_sense_keys: list[SenseKey] = [FORGED_GOLD_SENSE_KEY]
    prediction = voted_prediction(
        chosen_index=1,
        gold_sense_keys=gold_sense_keys,
        is_correct=False,
    )
    metadata = make_metadata(item_count=1, correct_count=0, accuracy=0.0, call_count=1)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=1))],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=renderable_dataset(gold_sense_keys=gold_sense_keys),
        prompt=registered_prompt(),
    )

    assert RunValidationRule.CANDIDATE_SET in issue_rules(report=report)


def _self_hosted_metadata(*, cost: CostBreakdown | None = None) -> RunMetadata:
    return make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        model=self_hosted_model(),
        machine=fixture_machine(),
        cost=cost,
    )


def _machine_time_cost_breakdown(*, total_usd: float) -> CostBreakdown:
    return CostBreakdown(total_usd=total_usd, source=CostSourceKind.MACHINE_TIME_ESTIMATE)


def _write_voted_run(*, run_dir: Path, metadata: RunMetadata) -> None:
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )


def test_verify_v1_run_without_new_sections_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        schema_version=RUN_SCHEMA_VERSION_V1,
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is False


def test_verify_v1_run_with_machine_section_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        schema_version=RUN_SCHEMA_VERSION_V1,
        machine=fixture_machine(),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.SCHEMA_SECTIONS in issue_rules(report=report)


def test_verify_v2_run_requires_execution_section(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
    ).model_copy(update={"execution": None})
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.SCHEMA_SECTIONS in issue_rules(report=report)


def test_verify_self_hosted_run_with_machine_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    expected_cost = FIXTURE_BENCHMARK_SECONDS * FIXTURE_HOURLY_RATE_USD / 3600.0
    metadata = _self_hosted_metadata(
        cost=_machine_time_cost_breakdown(total_usd=expected_cost),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is False


def test_verify_self_hosted_run_without_machine_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        model=self_hosted_model(),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.MACHINE_INFO in issue_rules(report=report)


def test_verify_detects_tampered_machine_time_cost(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    expected_cost = FIXTURE_BENCHMARK_SECONDS * FIXTURE_HOURLY_RATE_USD / 3600.0
    metadata = _self_hosted_metadata(
        cost=_machine_time_cost_breakdown(total_usd=expected_cost * 2),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.MACHINE_COST in issue_rules(report=report)


def test_verify_detects_elapsed_benchmark_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
    )
    tampered_totals = metadata.totals.model_copy(
        update={"elapsed_seconds": FIXTURE_BENCHMARK_SECONDS + 1.0},
    )
    metadata = metadata.model_copy(update={"totals": tampered_totals})
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.EXECUTION_TIMING in issue_rules(report=report)


def test_verify_detects_call_latency_exceeding_benchmark(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(item_count=1, correct_count=1, accuracy=1.0, call_count=1)
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    slow_call = success_call(
        raw_output=raw_output_for_sense_index(sense_index=2),
    ).model_copy(update={"latency_seconds": FIXTURE_BENCHMARK_SECONDS * 10})
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[slow_call],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.EXECUTION_TIMING in issue_rules(report=report)


def test_verify_self_hosted_run_without_revision_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    model = self_hosted_model().model_copy(update={"hf_revision": None})
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        model=model,
        machine=fixture_machine(),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.MODEL_PROVENANCE in issue_rules(report=report)


def test_verify_self_hosted_run_with_revision_passes_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        model=self_hosted_model(),
        machine=fixture_machine(),
    )
    _write_voted_run(run_dir=run_dir, metadata=metadata)

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.MODEL_PROVENANCE not in issue_rules(report=report)


def test_verify_flags_excessive_output_truncation(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        sampling=SamplingParameters(max_tokens=16),
    )
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    clipped_call = success_call(
        raw_output=raw_output_for_sense_index(sense_index=2),
    ).model_copy(update={"usage": TokenUsage(output_tokens=16)})
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[clipped_call],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.OUTPUT_TRUNCATION in issue_rules(report=report)


def test_verify_allows_outputs_below_cap(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN_DIR_NAME
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        sampling=SamplingParameters(max_tokens=256),
    )
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    call = success_call(
        raw_output=raw_output_for_sense_index(sense_index=2),
    ).model_copy(update={"usage": TokenUsage(output_tokens=12)})
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[call],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.OUTPUT_TRUNCATION not in issue_rules(report=report)

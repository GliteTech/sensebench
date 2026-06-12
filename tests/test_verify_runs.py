from __future__ import annotations

from pathlib import Path

from run_fixtures import (
    FIRST_SENSE_KEY,
    SECOND_SENSE_KEY,
    fixture_dataset,
    issue_rules,
    make_metadata,
    monosemous_prediction,
    registered_prompt,
    renderable_dataset,
    success_call,
    voted_prediction,
)

from sensebench.runner.writer import write_run_artifacts
from sensebench.verify.runs import RunValidationRule, verify_run_directory


def test_verify_valid_tiny_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
    run_dir = tmp_path / "run-1"
    metadata = make_metadata(item_count=0, correct_count=0, accuracy=None, call_count=0)
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[],
        calls=[success_call(raw_output='{"sense_index": 1}')],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert report.has_errors() is True


def test_verify_valid_voted_run_passes_with_prompt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        calls=[success_call(raw_output='{"sense_index": 2}')],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert report.has_errors() is False


def test_verify_detects_flipped_is_correct(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        calls=[success_call(raw_output='{"sense_index": 1}')],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.CORRECTNESS in issue_rules(report=report)


def test_verify_detects_forged_gold_keys_against_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        calls=[success_call(raw_output='{"sense_index": 2}')],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(gold_sense_keys=[FIRST_SENSE_KEY]),
    )

    assert RunValidationRule.DATASET_GOLD_KEYS in issue_rules(report=report)


def test_verify_detects_vote_mismatching_raw_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        calls=[success_call(raw_output='{"sense_index": 2}')],
    )

    report = verify_run_directory(run_dir=run_dir, prompt=registered_prompt())

    assert RunValidationRule.VOTE_EXTRACTION in issue_rules(report=report)


def test_verify_detects_content_hash_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        content_hash="sha256:" + "a" * 64,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output='{"sense_index": 2}')],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(
            gold_sense_keys=[SECOND_SENSE_KEY],
            content_hash="sha256:" + "b" * 64,
        ),
    )

    assert RunValidationRule.DATASET_CONTENT_HASH in issue_rules(report=report)


def test_verify_detects_unregistered_prompt_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    metadata = make_metadata(
        item_count=0,
        correct_count=0,
        accuracy=None,
        call_count=0,
        prompt_id="p999",
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.PROMPT_REFERENCE in issue_rules(report=report)


def test_verify_detects_duplicate_call_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
            success_call(raw_output='{"sense_index": 2}'),
            success_call(raw_output='{"sense_index": 2}'),
        ],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.DUPLICATE_CALL in issue_rules(report=report)


def test_verify_detects_orphan_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
            success_call(raw_output='{"sense_index": 2}'),
            success_call(raw_output='{"sense_index": 2}', call_id="i1__v9__a9"),
        ],
    )

    report = verify_run_directory(run_dir=run_dir)

    assert RunValidationRule.ORPHAN_CALL in issue_rules(report=report)


def test_verify_handles_call_for_unknown_dataset_item(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
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
        calls=[success_call(raw_output='{"sense_index": 2}')],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=fixture_dataset(gold_sense_keys=[SECOND_SENSE_KEY], item_id="i2"),
        prompt=registered_prompt(),
    )

    rules = issue_rules(report=report)
    assert RunValidationRule.DATASET_ITEMS in rules
    assert RunValidationRule.PROMPT_RENDERING in rules


def test_verify_detects_forged_candidate_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    gold_sense_keys = ["bank%1:17:00::"]
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
        calls=[success_call(raw_output='{"sense_index": 1}')],
    )

    report = verify_run_directory(
        run_dir=run_dir,
        dataset=renderable_dataset(gold_sense_keys=gold_sense_keys),
        prompt=registered_prompt(),
    )

    assert RunValidationRule.CANDIDATE_SET in issue_rules(report=report)

from __future__ import annotations

from json import dumps
from pathlib import Path

from sensebench.datasets.models import ItemID, SenseKey
from sensebench.prompts.models import SENSE_INDEX_FIELD, PromptID
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import CallID, RunID
from sensebench.verify.runs import RunValidationRule, verify_run_directory
from tests.run_fixtures import (
    DEFAULT_RUN_ID,
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

RUN_DIR_NAME: RunID = DEFAULT_RUN_ID
UNKNOWN_PROMPT_ID: PromptID = "p999"
ORPHAN_CALL_ID: CallID = "i1__v9__a9"
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

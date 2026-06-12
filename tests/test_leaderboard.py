from __future__ import annotations

from json import dumps
from pathlib import Path

from sensebench.leaderboard.aggregate import LeaderboardFile, emit_leaderboard
from sensebench.paths import LEADERBOARD_JSON_PATH, RUN_METADATA_FILENAME, SUBMITTED_RESULTS_DIR
from sensebench.prompts.models import SENSE_INDEX_FIELD
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import RunID
from tests.run_fixtures import (
    DATASET_VERSION,
    SECOND_SENSE_KEY,
    make_metadata,
    success_call,
    voted_prediction,
)

GOOD_RUN_ID: RunID = "run-good"
FORGED_RUN_ID: RunID = "run-forged"
CORRUPT_RUN_ID: RunID = "run-corrupt"


def raw_output_for_sense_index(*, sense_index: int) -> str:
    return dumps({SENSE_INDEX_FIELD: sense_index})


def _write_run(
    *,
    run_dir: Path,
    run_id: RunID,
    chosen_index: int,
    raw_output: str,
) -> None:
    prediction = voted_prediction(
        chosen_index=chosen_index,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        run_id=run_id,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output)],
    )


def test_emit_leaderboard_includes_only_verified_runs(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    output_path = tmp_path / LEADERBOARD_JSON_PATH

    _write_run(
        run_dir=results_dir / GOOD_RUN_ID,
        run_id=GOOD_RUN_ID,
        chosen_index=2,
        raw_output=raw_output_for_sense_index(sense_index=2),
    )
    # Internally consistent forgery: wrong answer marked correct.
    _write_run(
        run_dir=results_dir / FORGED_RUN_ID,
        run_id=FORGED_RUN_ID,
        chosen_index=1,
        raw_output=raw_output_for_sense_index(sense_index=1),
    )
    corrupt_dir = results_dir / CORRUPT_RUN_ID
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / RUN_METADATA_FILENAME).write_text(data="not json", encoding="utf-8")

    emit_leaderboard(results_dir=results_dir, output_path=output_path)

    leaderboard = LeaderboardFile.model_validate_json(output_path.read_text(encoding="utf-8"))
    run_ids: list[RunID] = [entry.run_id for entry in leaderboard.entries]
    assert run_ids == [GOOD_RUN_ID]
    entry = leaderboard.entries[0]
    assert entry.accuracy == 1.0
    assert entry.correct_count == 1
    assert entry.dataset_version == DATASET_VERSION

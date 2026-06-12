from __future__ import annotations

import json
from pathlib import Path

from run_fixtures import SECOND_SENSE_KEY, make_metadata, success_call, voted_prediction

from sensebench.leaderboard.aggregate import emit_leaderboard
from sensebench.runner.writer import write_run_artifacts


def _write_run(*, run_dir: Path, run_id: str, chosen_index: int, raw_output: str) -> None:
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
    results_dir = tmp_path / "results"
    output_path = tmp_path / "leaderboard.json"

    _write_run(
        run_dir=results_dir / "run-good",
        run_id="run-good",
        chosen_index=2,
        raw_output='{"sense_index": 2}',
    )
    # Internally consistent forgery: wrong answer marked correct.
    _write_run(
        run_dir=results_dir / "run-forged",
        run_id="run-forged",
        chosen_index=1,
        raw_output='{"sense_index": 1}',
    )
    corrupt_dir = results_dir / "run-corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "run.json").write_text("not json", encoding="utf-8")

    emit_leaderboard(results_dir=results_dir, output_path=output_path)

    leaderboard = json.loads(output_path.read_text(encoding="utf-8"))
    run_ids = [entry["run_id"] for entry in leaderboard["entries"]]
    assert run_ids == ["run-good"]
    entry = leaderboard["entries"][0]
    assert entry["accuracy"] == 1.0
    assert entry["correct_count"] == 1
    assert entry["dataset_version"] == "1"

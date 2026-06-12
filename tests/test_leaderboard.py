from __future__ import annotations

from json import dumps
from pathlib import Path

from sensebench.leaderboard.aggregate import (
    LeaderboardFile,
    collect_leaderboard_entries,
    emit_leaderboard,
)
from sensebench.paths import LEADERBOARD_JSON_PATH, RUN_METADATA_FILENAME, SUBMITTED_RESULTS_DIR
from sensebench.prompts.models import SENSE_INDEX_FIELD, PromptID
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import RunID
from tests.run_fixtures import (
    DATASET_VERSION,
    PROMPT_ID,
    RUNNER_GITHUB_HANDLE,
    SECOND_SENSE_KEY,
    make_metadata,
    success_call,
    voted_prediction,
)

GOOD_RUN_ID: RunID = "run-good"
FORGED_RUN_ID: RunID = "run-forged"
CORRUPT_RUN_ID: RunID = "run-corrupt"
ANONYMOUS_RUN_ID: RunID = "run-anonymous"
SECOND_PROMPT_RUN_ID: RunID = "run-second-prompt"
SECOND_PROMPT_ID: PromptID = "p002"
PLAIN_SENSE_OUTPUT: str = "2"
GITHUB_HANDLE_ISSUE_TEXT: str = "runner.github_handle"


def raw_output_for_sense_index(*, sense_index: int) -> str:
    return dumps({SENSE_INDEX_FIELD: sense_index})


def _write_run(
    *,
    run_dir: Path,
    run_id: RunID,
    chosen_index: int,
    raw_output: str,
    prompt_id: PromptID = PROMPT_ID,
    github_handle: str | None = RUNNER_GITHUB_HANDLE,
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
        prompt_id=prompt_id,
        github_handle=github_handle,
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


def test_official_leaderboard_rejects_missing_runner_identity(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    _write_run(
        run_dir=results_dir / ANONYMOUS_RUN_ID,
        run_id=ANONYMOUS_RUN_ID,
        chosen_index=2,
        raw_output=raw_output_for_sense_index(sense_index=2),
        github_handle=None,
    )

    official = collect_leaderboard_entries(results_dir=results_dir, official=True)
    assert official.entries == []
    assert any(GITHUB_HANDLE_ISSUE_TEXT in issue.message for issue in official.issues)

    local = collect_leaderboard_entries(results_dir=results_dir, official=False)
    assert [entry.run_id for entry in local.entries] == [ANONYMOUS_RUN_ID]


def test_best_group_key_collapses_prompts_and_runs(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    _write_run(
        run_dir=results_dir / GOOD_RUN_ID,
        run_id=GOOD_RUN_ID,
        chosen_index=2,
        raw_output=raw_output_for_sense_index(sense_index=2),
    )
    _write_run(
        run_dir=results_dir / SECOND_PROMPT_RUN_ID,
        run_id=SECOND_PROMPT_RUN_ID,
        chosen_index=2,
        raw_output=PLAIN_SENSE_OUTPUT,
        prompt_id=SECOND_PROMPT_ID,
    )

    collection = collect_leaderboard_entries(results_dir=results_dir, official=False)

    assert len(collection.entries) == 2
    group_keys = {entry.best_group_key for entry in collection.entries}
    assert len(group_keys) == 1, "same model and dataset share one best-view group"

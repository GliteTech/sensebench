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
from sensebench.runs.models import ModelHostingKind, RunID
from tests.run_fixtures import (
    DATASET_VERSION,
    FIXTURE_BENCHMARK_SECONDS,
    FIXTURE_CONCURRENCY,
    FIXTURE_GPU_COUNT,
    FIXTURE_HF_REVISION,
    FIXTURE_HOURLY_RATE_USD,
    FIXTURE_INFERENCE_ENGINE,
    FIXTURE_INFERENCE_ENGINE_VERSION,
    FIXTURE_QUANTIZATION,
    PROMPT_ID,
    RUNNER_GITHUB_HANDLE,
    SECOND_SENSE_KEY,
    fixture_machine,
    make_metadata,
    self_hosted_model,
    success_call,
    voted_prediction,
)

GOOD_RUN_ID: RunID = "run-good"
FORGED_RUN_ID: RunID = "run-forged"
CORRUPT_RUN_ID: RunID = "run-corrupt"
ANONYMOUS_RUN_ID: RunID = "run-anonymous"
SECOND_PROMPT_RUN_ID: RunID = "run-second-prompt"
SELF_HOSTED_RUN_ID: RunID = "run-self-hosted"
SECOND_PROMPT_ID: PromptID = "p002"
PLAIN_SENSE_OUTPUT: str = "2"
GITHUB_HANDLE_ISSUE_TEXT: str = "runner.github_handle"
EXPECTED_GPU_LABEL: str = "H100 80GB"


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
    assert entry.hosting_kind == ModelHostingKind.CLOUD_API
    assert entry.gpu is None
    assert entry.gpu_count is None
    assert entry.hourly_rate_usd is None
    assert entry.quantization is None
    assert entry.inference_engine is None
    assert entry.benchmark_seconds == FIXTURE_BENCHMARK_SECONDS
    assert entry.seconds_per_item is None
    assert entry.concurrency == FIXTURE_CONCURRENCY


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


def test_self_hosted_run_populates_machine_fields(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
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
        run_id=SELF_HOSTED_RUN_ID,
        model=self_hosted_model(),
        machine=fixture_machine(),
    )
    write_run_artifacts(
        run_dir=results_dir / SELF_HOSTED_RUN_ID,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )

    collection = collect_leaderboard_entries(results_dir=results_dir, official=False)

    assert [entry.run_id for entry in collection.entries] == [SELF_HOSTED_RUN_ID]
    entry = collection.entries[0]
    assert entry.hosting_kind == ModelHostingKind.SELF_HOSTED
    assert entry.gpu == EXPECTED_GPU_LABEL
    assert entry.gpu_count == FIXTURE_GPU_COUNT
    assert entry.quantization == FIXTURE_QUANTIZATION
    assert entry.inference_engine == FIXTURE_INFERENCE_ENGINE
    assert entry.inference_engine_version == FIXTURE_INFERENCE_ENGINE_VERSION
    assert entry.hf_revision == FIXTURE_HF_REVISION
    assert entry.hourly_rate_usd == FIXTURE_HOURLY_RATE_USD
    assert entry.benchmark_seconds == FIXTURE_BENCHMARK_SECONDS
    assert entry.seconds_per_item == FIXTURE_BENCHMARK_SECONDS
    assert entry.concurrency == FIXTURE_CONCURRENCY


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

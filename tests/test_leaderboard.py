from __future__ import annotations

from json import dumps
from pathlib import Path

from pytest import approx

from sensebench.leaderboard.aggregate import (
    OFFICIAL_SEMANTIC_REASKS,
    OFFICIAL_VOTES_PER_ITEM,
    LeaderboardFile,
    _protocol_issues,
    collect_leaderboard_entries,
    emit_leaderboard,
)
from sensebench.paths import LEADERBOARD_JSON_PATH, RUN_METADATA_FILENAME, SUBMITTED_RESULTS_DIR
from sensebench.prompts.models import SENSE_INDEX_FIELD, PromptID
from sensebench.runner.costs import SECONDS_PER_HOUR
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    CostBreakdown,
    CostSourceKind,
    MachineInfo,
    ModelHostingKind,
    MonosemousPolicyKind,
    RunID,
    RunPolicy,
    TieBreakKind,
)
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
MACHINE_TIME_RUN_ID: RunID = "run-machine-time"
UNPRICED_GPU_RUN_ID: RunID = "run-unpriced-gpu"
SECOND_PROMPT_ID: PromptID = "p002"
PLAIN_SENSE_OUTPUT: str = "2"
GITHUB_HANDLE_ISSUE_TEXT: str = "runner.github_handle"
VOTES_PER_ITEM_ISSUE_TEXT: str = "votes_per_item"
SEMANTIC_REASKS_ISSUE_TEXT: str = "semantic_reasks"
EXPECTED_GPU_LABEL: str = "H100 80GB"
EXPECTED_REFERENCE_HOURLY_RATE_USD: float = 2.26
UNPRICED_GPU_NAME: str = "NVIDIA L40S"
UNPRICED_GPU_LABEL: str = "L40S"
GPU_NAME_FIELD: str = "name"
MACHINE_GPU_FIELD: str = "gpu"


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


def _write_machine_time_run(
    *,
    results_dir: Path,
    run_id: RunID,
    machine: MachineInfo,
) -> None:
    """Write a self-hosted run whose cost is machine time priced at the rate actually paid."""
    actual_cost_usd = FIXTURE_BENCHMARK_SECONDS * FIXTURE_HOURLY_RATE_USD / SECONDS_PER_HOUR
    write_run_artifacts(
        run_dir=results_dir / run_id,
        metadata=make_metadata(
            item_count=1,
            correct_count=1,
            accuracy=1.0,
            call_count=1,
            run_id=run_id,
            model=self_hosted_model(),
            machine=machine,
            cost=CostBreakdown(
                total_usd=actual_cost_usd,
                source=CostSourceKind.MACHINE_TIME_ESTIMATE,
            ),
        ),
        predictions=[
            voted_prediction(chosen_index=2, gold_sense_keys=[SECOND_SENSE_KEY], is_correct=True)
        ],
        calls=[success_call(raw_output=raw_output_for_sense_index(sense_index=2))],
    )


def test_machine_time_run_is_compared_at_the_reference_rate(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    _write_machine_time_run(
        results_dir=results_dir,
        run_id=MACHINE_TIME_RUN_ID,
        machine=fixture_machine(),
    )

    collection = collect_leaderboard_entries(results_dir=results_dir, official=False)

    entry = collection.entries[0]
    actual_cost_usd = FIXTURE_BENCHMARK_SECONDS * FIXTURE_HOURLY_RATE_USD / SECONDS_PER_HOUR
    reference_cost_usd = (
        FIXTURE_BENCHMARK_SECONDS * EXPECTED_REFERENCE_HOURLY_RATE_USD / SECONDS_PER_HOUR
    )
    assert entry.hourly_rate_usd == FIXTURE_HOURLY_RATE_USD, "the rate actually paid is preserved"
    assert entry.reference_hourly_rate_usd == EXPECTED_REFERENCE_HOURLY_RATE_USD
    assert entry.cost_usd == approx(actual_cost_usd), "actual run cost stays at the rate paid"
    assert entry.reference_cost_usd == approx(reference_cost_usd)
    assert entry.reference_cost_usd != approx(entry.cost_usd), (
        "this fixture rented above the reference rate, so the two costs must differ"
    )
    assert entry.cost_per_million_items == approx(reference_cost_usd * 1_000_000), (
        "the cross-model comparison metric is priced at the reference rate, not the rate paid"
    )


def test_machine_time_run_on_unpriced_gpu_keeps_actual_cost(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    machine = fixture_machine()
    assert machine.gpu is not None
    unpriced_gpu = machine.gpu.model_copy(update={GPU_NAME_FIELD: UNPRICED_GPU_NAME})
    _write_machine_time_run(
        results_dir=results_dir,
        run_id=UNPRICED_GPU_RUN_ID,
        machine=machine.model_copy(update={MACHINE_GPU_FIELD: unpriced_gpu}),
    )

    collection = collect_leaderboard_entries(results_dir=results_dir, official=False)

    entry = collection.entries[0]
    actual_cost_usd = FIXTURE_BENCHMARK_SECONDS * FIXTURE_HOURLY_RATE_USD / SECONDS_PER_HOUR
    assert entry.gpu == UNPRICED_GPU_LABEL
    assert entry.reference_hourly_rate_usd is None
    assert entry.reference_cost_usd is None, "a GPU class with no reference rate is not re-priced"
    assert entry.cost_per_million_items == approx(actual_cost_usd * 1_000_000), (
        "cost falls back to the rate actually paid rather than dropping off the board"
    )


def test_cloud_run_cost_is_not_re_priced(tmp_path: Path) -> None:
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    _write_run(
        run_dir=results_dir / GOOD_RUN_ID,
        run_id=GOOD_RUN_ID,
        chosen_index=2,
        raw_output=raw_output_for_sense_index(sense_index=2),
    )

    collection = collect_leaderboard_entries(results_dir=results_dir, official=False)

    entry = collection.entries[0]
    assert entry.reference_hourly_rate_usd is None
    assert entry.reference_cost_usd is None, "cloud runs bill at published list prices already"
    assert entry.cost_per_million_items == approx((entry.cost_usd or 0.0) * 1_000_000)


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
    group_keys: set[str] = {entry.best_group_key for entry in collection.entries}
    assert len(group_keys) == 1, "same model and dataset share one best-view group"


def _policy(*, votes: int, reasks: int) -> RunPolicy:
    return RunPolicy(
        votes_per_item=votes,
        semantic_reasks_per_invalid_vote=reasks,
        tie_break=TieBreakKind.EARLIEST_VOTE,
        monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
    )


def test_protocol_issues_accepts_canonical_single_vote() -> None:
    issues: list[str] = _protocol_issues(
        policy=_policy(votes=OFFICIAL_VOTES_PER_ITEM, reasks=OFFICIAL_SEMANTIC_REASKS),
    )

    assert issues == []


def test_protocol_issues_rejects_self_consistency_voting() -> None:
    issues: list[str] = _protocol_issues(policy=_policy(votes=3, reasks=OFFICIAL_SEMANTIC_REASKS))

    assert len(issues) == 1
    assert VOTES_PER_ITEM_ISSUE_TEXT in issues[0]


def test_protocol_issues_rejects_extra_reasks() -> None:
    issues: list[str] = _protocol_issues(policy=_policy(votes=OFFICIAL_VOTES_PER_ITEM, reasks=2))

    assert len(issues) == 1
    assert SEMANTIC_REASKS_ISSUE_TEXT in issues[0]

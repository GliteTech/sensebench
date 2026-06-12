"""Aggregate submitted results into leaderboard.json."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from sensebench.runs.loaders import RUN_METADATA_FILENAME, load_run_directory

DEFAULT_BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_BOOTSTRAP_SEED: int = 12345
CONFIDENCE_LOW_PERCENTILE: float = 2.5
CONFIDENCE_HIGH_PERCENTILE: float = 97.5
LEADERBOARD_SCHEMA_VERSION: str = "sensebench-leaderboard-v1"


class LeaderboardModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AccuracyInterval(LeaderboardModel):
    low: float | None
    high: float | None


class LeaderboardEntry(LeaderboardModel):
    run_id: str
    model: str
    prompt_id: str
    dataset_id: str
    accuracy: float | None
    accuracy_ci: AccuracyInterval
    correct_count: int
    item_count: int
    call_count: int
    cost_usd: float | None


class LeaderboardFile(LeaderboardModel):
    schema_version: str
    entries: list[LeaderboardEntry]


def _bootstrap_accuracy_ci(*, values: list[bool]) -> AccuracyInterval:
    if len(values) == 0:
        return AccuracyInterval(low=None, high=None)
    numeric = np.array([1.0 if value else 0.0 for value in values], dtype=float)
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    estimates = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        sample = rng.choice(numeric, size=len(numeric), replace=True)
        estimates[index] = float(np.mean(sample))
    low, high = np.percentile(
        a=estimates,
        q=[CONFIDENCE_LOW_PERCENTILE, CONFIDENCE_HIGH_PERCENTILE],
    )
    return AccuracyInterval(low=float(low), high=float(high))


def _entry_for_run(*, run_dir: Path) -> LeaderboardEntry:
    loaded = load_run_directory(run_dir=run_dir)
    correctness: list[bool] = [prediction.is_correct is True for prediction in loaded.predictions]
    return LeaderboardEntry(
        run_id=loaded.metadata.run_id,
        model=loaded.metadata.model.display_name,
        prompt_id=loaded.metadata.prompt.id,
        dataset_id=loaded.metadata.dataset.dataset_id,
        accuracy=loaded.metadata.totals.accuracy,
        accuracy_ci=_bootstrap_accuracy_ci(values=correctness),
        correct_count=loaded.metadata.totals.correct_count,
        item_count=loaded.metadata.totals.item_count,
        call_count=loaded.metadata.totals.call_count,
        cost_usd=loaded.metadata.totals.cost_usd,
    )


def emit_leaderboard(*, results_dir: Path, output_path: Path) -> None:
    entries: list[LeaderboardEntry] = []
    if results_dir.exists():
        for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            if (run_dir / RUN_METADATA_FILENAME).exists():
                entries.append(_entry_for_run(run_dir=run_dir))
    entries.sort(
        key=lambda entry: entry.accuracy if entry.accuracy is not None else -1.0, reverse=True
    )
    leaderboard = LeaderboardFile(
        schema_version=LEADERBOARD_SCHEMA_VERSION,
        entries=entries,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        leaderboard.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

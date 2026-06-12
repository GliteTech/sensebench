"""Aggregate submitted results into leaderboard.json."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from sensebench.datasets.models import DatasetID
from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR, RUN_METADATA_FILENAME
from sensebench.prompts.models import PromptDefinition, PromptID
from sensebench.prompts.registry import load_prompt_definition
from sensebench.runs.loaders import LoadedRun, load_run_directory
from sensebench.runs.models import RunID
from sensebench.verify.runs import verify_run_directory

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
    run_id: RunID
    model: str
    prompt_id: PromptID
    dataset_id: DatasetID
    dataset_version: str | None
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
    numeric: NDArray[np.float64] = np.array(
        [1.0 if value else 0.0 for value in values],
        dtype=np.float64,
    )
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    estimates: NDArray[np.float64] = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        sample: NDArray[np.float64] = rng.choice(numeric, size=len(numeric), replace=True)
        estimates[index] = float(np.mean(sample))
    percentiles: NDArray[np.float64] = np.percentile(
        a=estimates,
        q=[CONFIDENCE_LOW_PERCENTILE, CONFIDENCE_HIGH_PERCENTILE],
    )
    return AccuracyInterval(low=float(percentiles[0]), high=float(percentiles[1]))


def _registered_prompt(*, prompt_id: PromptID) -> PromptDefinition | None:
    path = PROMPT_REGISTRY_DIR / f"{prompt_id}{PROMPT_JSON_SUFFIX}"
    try:
        return load_prompt_definition(path=path)
    except (OSError, ValidationError):
        return None


def _entry_for_run(*, loaded: LoadedRun) -> LeaderboardEntry:
    correctness: list[bool] = [prediction.is_correct is True for prediction in loaded.predictions]
    correct_count = sum(1 for value in correctness if value)
    item_count = len(loaded.predictions)
    accuracy = correct_count / item_count if item_count > 0 else None
    return LeaderboardEntry(
        run_id=loaded.metadata.run_id,
        model=loaded.metadata.model.display_name,
        prompt_id=loaded.metadata.prompt.id,
        dataset_id=loaded.metadata.dataset.dataset_id,
        dataset_version=loaded.metadata.dataset.dataset_version,
        accuracy=accuracy,
        accuracy_ci=_bootstrap_accuracy_ci(values=correctness),
        correct_count=correct_count,
        item_count=item_count,
        call_count=len(loaded.calls),
        cost_usd=loaded.metadata.totals.cost.total_usd,
    )


def _verified_entry_for_run(*, run_dir: Path) -> LeaderboardEntry | None:
    try:
        loaded = load_run_directory(run_dir=run_dir)
        prompt = _registered_prompt(prompt_id=loaded.metadata.prompt.id)
        report = verify_run_directory(run_dir=run_dir, prompt=prompt)
    except Exception as exc:
        print(f"skipping {run_dir}: cannot verify ({exc})", file=sys.stderr)
        return None
    if report.has_errors():
        failed_rules = sorted({issue.rule.value for issue in report.issues})
        print(
            f"skipping {run_dir}: failed verification ({', '.join(failed_rules)})",
            file=sys.stderr,
        )
        return None
    return _entry_for_run(loaded=loaded)


def emit_leaderboard(*, results_dir: Path, output_path: Path) -> None:
    entries: list[LeaderboardEntry] = []
    if results_dir.exists():
        for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            if (run_dir / RUN_METADATA_FILENAME).exists():
                entry = _verified_entry_for_run(run_dir=run_dir)
                if entry is not None:
                    entries.append(entry)
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

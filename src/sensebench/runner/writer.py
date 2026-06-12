"""Write run artifacts."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from sensebench.paths import CALLS_FILENAME, PREDICTIONS_FILENAME, RUN_METADATA_FILENAME
from sensebench.runs.models import CallRecord, PredictionRecord, RunMetadata

STAGING_DIR_SUFFIX: str = ".partial"


def write_run_artifacts(
    *,
    run_dir: Path,
    metadata: RunMetadata,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = run_dir.parent / f"{run_dir.name}{STAGING_DIR_SUFFIX}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    metadata_path = staging_dir / RUN_METADATA_FILENAME
    predictions_path = staging_dir / PREDICTIONS_FILENAME
    calls_path = staging_dir / CALLS_FILENAME

    metadata_path.write_text(
        metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json() + "\n")
    with gzip.open(calls_path, mode="wt", encoding="utf-8") as handle:
        for call in calls:
            handle.write(call.model_dump_json() + "\n")
    staging_dir.rename(run_dir)

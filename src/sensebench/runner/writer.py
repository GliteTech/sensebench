"""Write run artifacts."""

from __future__ import annotations

import gzip
from pathlib import Path

from sensebench.runs.loaders import CALLS_FILENAME, PREDICTIONS_FILENAME, RUN_METADATA_FILENAME
from sensebench.runs.models import CallRecord, PredictionRecord, RunMetadata


def write_run_artifacts(
    *,
    run_dir: Path,
    metadata: RunMetadata,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / RUN_METADATA_FILENAME
    predictions_path = run_dir / PREDICTIONS_FILENAME
    calls_path = run_dir / CALLS_FILENAME

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

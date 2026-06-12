"""Load run directories."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from sensebench.paths import CALLS_FILENAME, PREDICTIONS_FILENAME, RUN_METADATA_FILENAME
from sensebench.runs.models import CallRecord, PredictionRecord, RunMetadata


@dataclass(frozen=True, slots=True)
class LoadedRun:
    run_dir: Path
    metadata: RunMetadata
    predictions: list[PredictionRecord]
    calls: list[CallRecord]


def _load_jsonl_models[T_RunModel: BaseModel](
    *,
    path: Path,
    model_cls: type[T_RunModel],
) -> list[T_RunModel]:
    records: list[T_RunModel] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            records.append(model_cls.model_validate_json(line))
    return records


def _load_gzip_jsonl_models[T_RunModel: BaseModel](
    *,
    path: Path,
    model_cls: type[T_RunModel],
) -> list[T_RunModel]:
    records: list[T_RunModel] = []
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            records.append(model_cls.model_validate_json(line))
    return records


def load_run_directory(*, run_dir: Path) -> LoadedRun:
    metadata_path = run_dir / RUN_METADATA_FILENAME
    predictions_path = run_dir / PREDICTIONS_FILENAME
    calls_path = run_dir / CALLS_FILENAME
    metadata = RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    predictions: list[PredictionRecord] = _load_jsonl_models(
        path=predictions_path,
        model_cls=PredictionRecord,
    )
    calls: list[CallRecord] = _load_gzip_jsonl_models(
        path=calls_path,
        model_cls=CallRecord,
    )
    return LoadedRun(
        run_dir=run_dir,
        metadata=metadata,
        predictions=predictions,
        calls=calls,
    )

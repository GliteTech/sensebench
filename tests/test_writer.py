from __future__ import annotations

from pathlib import Path

import pytest
from run_fixtures import make_metadata

from sensebench.paths import RUN_METADATA_FILENAME
from sensebench.runner.writer import STAGING_DIR_SUFFIX, write_run_artifacts


def test_writer_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    metadata = make_metadata(item_count=0, correct_count=0, accuracy=None, call_count=0)
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    with pytest.raises(FileExistsError):
        write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    assert (run_dir / RUN_METADATA_FILENAME).exists()


def test_writer_leaves_no_staging_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    metadata = make_metadata(item_count=0, correct_count=0, accuracy=None, call_count=0)
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    assert (run_dir / RUN_METADATA_FILENAME).exists()
    assert not (tmp_path / f"run-1{STAGING_DIR_SUFFIX}").exists()

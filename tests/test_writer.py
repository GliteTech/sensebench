from __future__ import annotations

from pathlib import Path

from pytest import raises

from sensebench.paths import RUN_METADATA_FILENAME
from sensebench.runner.writer import STAGING_DIR_SUFFIX, write_run_artifacts
from sensebench.runs.models import RunID
from tests.run_fixtures import DEFAULT_RUN_ID, make_metadata

TEST_RUN_ID: RunID = DEFAULT_RUN_ID


def test_writer_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / TEST_RUN_ID
    metadata = make_metadata(
        item_count=0,
        correct_count=0,
        accuracy=None,
        call_count=0,
        run_id=TEST_RUN_ID,
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    with raises(FileExistsError):
        write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    assert (run_dir / RUN_METADATA_FILENAME).exists()


def test_writer_leaves_no_staging_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / TEST_RUN_ID
    metadata = make_metadata(
        item_count=0,
        correct_count=0,
        accuracy=None,
        call_count=0,
        run_id=TEST_RUN_ID,
    )
    write_run_artifacts(run_dir=run_dir, metadata=metadata, predictions=[], calls=[])

    assert (run_dir / RUN_METADATA_FILENAME).exists()
    staging_dir = run_dir.parent / f"{run_dir.name}{STAGING_DIR_SUFFIX}"
    assert not staging_dir.exists()

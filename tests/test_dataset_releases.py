from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch, raises

from sensebench.datasets import releases
from sensebench.datasets.loaders import file_content_hash
from sensebench.datasets.models import DatasetID
from sensebench.datasets.releases import (
    DATASET_CACHE_ENV_VAR,
    DatasetRelease,
    fetch_dataset_release,
    get_dataset_release,
    load_registered_dataset,
)
from sensebench.paths import DATASET_FILENAME, SMOKE_ITEMS_PATH

RELEASE_ID: str = "fixture-v1"
UNKNOWN_RELEASE_ID: str = "no-such-release"
DATASET_ID_FIXTURE: DatasetID = "fixture"
CACHE_DIRNAME: str = "cache"
SAMPLE_JSONL_FILENAME: str = "sample.jsonl"
DOWNLOAD_RELEASE_ATTR: str = "_download_release"
VALID_PAYLOAD: bytes = b'{"item": 1}\n'
CORRUPT_PAYLOAD: bytes = b"corrupted bytes"
UNEXPECTED_PAYLOAD: bytes = b"unexpected content"
RELEASE_URL: str = "https://invalid.example/items.jsonl"


def _release_for_payload(
    *,
    payload: bytes,
    tmp_path: Path,
    item_count: int = 1,
) -> DatasetRelease:
    sample_path = tmp_path / SAMPLE_JSONL_FILENAME
    sample_path.write_bytes(payload)
    return DatasetRelease(
        release_id=RELEASE_ID,
        dataset_id=DATASET_ID_FIXTURE,
        url=RELEASE_URL,
        content_hash=file_content_hash(path=sample_path),
        item_count=item_count,
    )


def _seed_cache(*, cache_dir: Path, payload: bytes) -> Path:
    cached_path = cache_dir / RELEASE_ID / DATASET_FILENAME
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(payload)
    return cached_path


def _fail_download(*, release: DatasetRelease, target: Path) -> None:
    raise AssertionError("download must not be attempted")


def test_get_dataset_release_rejects_unknown_id() -> None:
    with raises(KeyError):
        get_dataset_release(release_id=UNKNOWN_RELEASE_ID)


def test_fetch_uses_hash_verified_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / CACHE_DIRNAME
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    cached_path = _seed_cache(cache_dir=cache_dir, payload=VALID_PAYLOAD)
    release = _release_for_payload(payload=VALID_PAYLOAD, tmp_path=tmp_path)
    monkeypatch.setattr(
        target=releases,
        name=DOWNLOAD_RELEASE_ATTR,
        value=_fail_download,
    )

    assert fetch_dataset_release(release=release) == cached_path


def test_fetch_downloads_when_cache_is_corrupt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / CACHE_DIRNAME
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    _seed_cache(cache_dir=cache_dir, payload=CORRUPT_PAYLOAD)
    release = _release_for_payload(payload=VALID_PAYLOAD, tmp_path=tmp_path)

    def _fake_download(*, release: DatasetRelease, target: Path) -> None:
        target.write_bytes(VALID_PAYLOAD)

    monkeypatch.setattr(
        target=releases,
        name=DOWNLOAD_RELEASE_ATTR,
        value=_fake_download,
    )

    fetched_path = fetch_dataset_release(release=release)

    assert fetched_path.read_bytes() == VALID_PAYLOAD


def test_fetch_rejects_download_with_wrong_hash(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / CACHE_DIRNAME
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    release = _release_for_payload(payload=VALID_PAYLOAD, tmp_path=tmp_path)

    def _fake_download(*, release: DatasetRelease, target: Path) -> None:
        target.write_bytes(UNEXPECTED_PAYLOAD)

    monkeypatch.setattr(
        target=releases,
        name=DOWNLOAD_RELEASE_ATTR,
        value=_fake_download,
    )

    with raises(RuntimeError):
        fetch_dataset_release(release=release)


def test_load_registered_dataset_checks_item_count(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / CACHE_DIRNAME
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = SMOKE_ITEMS_PATH.read_bytes()
    _seed_cache(cache_dir=cache_dir, payload=payload)
    release = _release_for_payload(payload=payload, tmp_path=tmp_path, item_count=2)
    monkeypatch.setattr(
        target=releases,
        name=DOWNLOAD_RELEASE_ATTR,
        value=_fail_download,
    )

    with raises(RuntimeError):
        load_registered_dataset(release=release)


def test_load_registered_dataset_records_release_identity(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / CACHE_DIRNAME
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = SMOKE_ITEMS_PATH.read_bytes()
    _seed_cache(cache_dir=cache_dir, payload=payload)
    release = _release_for_payload(payload=payload, tmp_path=tmp_path, item_count=1)
    monkeypatch.setattr(
        target=releases,
        name=DOWNLOAD_RELEASE_ATTR,
        value=_fail_download,
    )

    bundle = load_registered_dataset(release=release)

    assert bundle.dataset_id == DATASET_ID_FIXTURE
    assert bundle.dataset_version == RELEASE_ID
    assert bundle.content_hash == release.content_hash
    assert len(bundle.items) == 1, "registered dataset loads all fixture items"

from __future__ import annotations

from pathlib import Path

import pytest

import sensebench.datasets.releases as releases_module
from sensebench.datasets.loaders import file_content_hash
from sensebench.datasets.releases import (
    DATASET_CACHE_ENV_VAR,
    DATASET_FILENAME,
    DatasetRelease,
    fetch_dataset_release,
    get_dataset_release,
    load_registered_dataset,
)

RELEASE_ID: str = "fixture-v1"
SMOKE_ITEMS_PATH: Path = Path(__file__).parent / "data" / "smoke_items.jsonl"


def _release_for_payload(*, payload: bytes, tmp_path: Path, item_count: int = 1) -> DatasetRelease:
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_bytes(payload)
    return DatasetRelease(
        release_id=RELEASE_ID,
        dataset_id="fixture",
        url="https://invalid.example/items.jsonl",
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
    with pytest.raises(KeyError):
        get_dataset_release(release_id="no-such-release")


def test_fetch_uses_hash_verified_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = b'{"item": 1}\n'
    cached_path = _seed_cache(cache_dir=cache_dir, payload=payload)
    release = _release_for_payload(payload=payload, tmp_path=tmp_path)
    monkeypatch.setattr(releases_module, "_download_release", _fail_download)

    assert fetch_dataset_release(release=release) == cached_path


def test_fetch_downloads_when_cache_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = b'{"item": 1}\n'
    _seed_cache(cache_dir=cache_dir, payload=b"corrupted bytes")
    release = _release_for_payload(payload=payload, tmp_path=tmp_path)

    def _fake_download(*, release: DatasetRelease, target: Path) -> None:
        target.write_bytes(payload)

    monkeypatch.setattr(releases_module, "_download_release", _fake_download)

    fetched_path = fetch_dataset_release(release=release)

    assert fetched_path.read_bytes() == payload


def test_fetch_rejects_download_with_wrong_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    release = _release_for_payload(payload=b'{"item": 1}\n', tmp_path=tmp_path)

    def _fake_download(*, release: DatasetRelease, target: Path) -> None:
        target.write_bytes(b"unexpected content")

    monkeypatch.setattr(releases_module, "_download_release", _fake_download)

    with pytest.raises(RuntimeError):
        fetch_dataset_release(release=release)


def test_load_registered_dataset_checks_item_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = SMOKE_ITEMS_PATH.read_bytes()
    _seed_cache(cache_dir=cache_dir, payload=payload)
    release = _release_for_payload(payload=payload, tmp_path=tmp_path, item_count=2)
    monkeypatch.setattr(releases_module, "_download_release", _fail_download)

    with pytest.raises(RuntimeError):
        load_registered_dataset(release=release)


def test_load_registered_dataset_records_release_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(DATASET_CACHE_ENV_VAR, str(cache_dir))
    payload = SMOKE_ITEMS_PATH.read_bytes()
    _seed_cache(cache_dir=cache_dir, payload=payload)
    release = _release_for_payload(payload=payload, tmp_path=tmp_path, item_count=1)
    monkeypatch.setattr(releases_module, "_download_release", _fail_download)

    bundle = load_registered_dataset(release=release)

    assert bundle.dataset_id == "fixture"
    assert bundle.dataset_version == RELEASE_ID
    assert bundle.content_hash == release.content_hash
    assert len(bundle.items) == 1

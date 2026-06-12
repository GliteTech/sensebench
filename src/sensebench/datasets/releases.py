"""Registered dataset releases and hash-verified downloads."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sensebench import __version__
from sensebench.datasets.loaders import file_content_hash, load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle, DatasetID

DATASET_CACHE_ENV_VAR: str = "SENSEBENCH_CACHE_DIR"
DATASET_FILENAME: str = "items.jsonl"
DOWNLOAD_SUFFIX: str = ".download"
DOWNLOAD_TIMEOUT_SECONDS: float = 60.0


@dataclass(frozen=True, slots=True)
class DatasetRelease:
    release_id: str
    dataset_id: DatasetID
    url: str
    content_hash: str
    item_count: int


DATASET_RELEASES: dict[str, DatasetRelease] = {
    "lexen-v0.1.0": DatasetRelease(
        release_id="lexen-v0.1.0",
        dataset_id="lexen",
        url="https://github.com/GliteTech/lexen/releases/download/lexen-v0.1.0/items.jsonl",
        content_hash="sha256:0222f3be1b54975692f2be67f271db0a351eb627e327e346d6b8155f9d1ba856",
        item_count=4917,
    ),
}


def get_dataset_release(*, release_id: str) -> DatasetRelease:
    release = DATASET_RELEASES.get(release_id)
    if release is None:
        known_releases = ", ".join(sorted(DATASET_RELEASES))
        raise KeyError(
            f"unknown dataset release {release_id}; registered releases: {known_releases}"
        )
    return release


def dataset_cache_dir() -> Path:
    override = os.environ.get(DATASET_CACHE_ENV_VAR)
    if override is not None and len(override) > 0:
        return Path(override)
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return cache_root / "sensebench" / "datasets"


def _download_release(*, release: DatasetRelease, target: Path) -> None:
    print(f"Downloading dataset {release.release_id} from {release.url}", file=sys.stderr)
    request = urllib.request.Request(
        release.url,
        headers={"User-Agent": f"sensebench/{__version__}"},
    )
    with (
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        target.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)


def fetch_dataset_release(*, release: DatasetRelease) -> Path:
    target = dataset_cache_dir() / release.release_id / DATASET_FILENAME
    if target.exists():
        if file_content_hash(path=target) == release.content_hash:
            return target
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    download_path = target.with_name(f"{target.name}{DOWNLOAD_SUFFIX}")
    _download_release(release=release, target=download_path)
    downloaded_hash = file_content_hash(path=download_path)
    if downloaded_hash != release.content_hash:
        download_path.unlink()
        raise RuntimeError(
            f"downloaded dataset {release.release_id} has content hash {downloaded_hash}, "
            f"expected {release.content_hash}"
        )
    download_path.replace(target)
    return target


def load_registered_dataset(*, release: DatasetRelease) -> DatasetBundle:
    path = fetch_dataset_release(release=release)
    bundle = load_jsonl_dataset(
        path=path,
        dataset_id=release.dataset_id,
        dataset_version=release.release_id,
    )
    if len(bundle.items) != release.item_count:
        raise RuntimeError(
            f"dataset {release.release_id} has {len(bundle.items)} items, "
            f"registry expects {release.item_count}"
        )
    return bundle

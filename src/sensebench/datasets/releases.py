"""Registered dataset releases and hash-verified downloads."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from shutil import copyfileobj
from sys import stderr
from urllib.request import Request, urlopen

from sensebench import __version__
from sensebench.datasets.loaders import file_content_hash, load_jsonl_dataset
from sensebench.datasets.models import ContentHash, DatasetBundle, DatasetID, DatasetVersion
from sensebench.paths import (
    DATASET_FILENAME,
    DATASETS_CACHE_DIRNAME,
    DEFAULT_CACHE_DIRNAME,
    DEFAULT_LEXEN_RELEASE_ID,
    DOWNLOAD_SUFFIX,
    LEXEN_DATASET_ID,
    SENSEBENCH_CACHE_DIRNAME,
)

DATASET_CACHE_ENV_VAR: str = "SENSEBENCH_CACHE_DIR"
XDG_CACHE_HOME_ENV_VAR: str = "XDG_CACHE_HOME"
DOWNLOAD_TIMEOUT_SECONDS: float = 60.0
DATASET_USER_AGENT_HEADER: str = "User-Agent"
DATASET_USER_AGENT: str = f"sensebench/{__version__}"


@dataclass(frozen=True, slots=True)
class DatasetRelease:
    release_id: DatasetVersion
    dataset_id: DatasetID
    url: str
    content_hash: ContentHash
    item_count: int


DATASET_RELEASES: dict[DatasetVersion, DatasetRelease] = {
    DEFAULT_LEXEN_RELEASE_ID: DatasetRelease(
        release_id=DEFAULT_LEXEN_RELEASE_ID,
        dataset_id=LEXEN_DATASET_ID,
        url="https://github.com/GliteTech/lexen/releases/download/lexen-v1/items.jsonl",
        content_hash="sha256:5fd4382b93f19087a1e31f6dd7d1db17c1eb17ff80fcbe1d3fdd55c0c3ecefe8",
        item_count=4861,
    ),
}


def get_dataset_release(*, release_id: DatasetVersion) -> DatasetRelease:
    release = DATASET_RELEASES.get(release_id)
    if release is None:
        known_releases = ", ".join(sorted(DATASET_RELEASES))
        raise KeyError(
            f"unknown dataset release {release_id}; registered releases: {known_releases}"
        )
    return release


def dataset_cache_dir() -> Path:
    override = environ.get(DATASET_CACHE_ENV_VAR)
    if override is not None and len(override) > 0:
        return Path(override)
    xdg_cache_home = environ.get(XDG_CACHE_HOME_ENV_VAR)
    cache_root = (
        Path(xdg_cache_home)
        if xdg_cache_home is not None and len(xdg_cache_home) > 0
        else Path.home() / DEFAULT_CACHE_DIRNAME
    )
    return cache_root / SENSEBENCH_CACHE_DIRNAME / DATASETS_CACHE_DIRNAME


def _download_release(*, release: DatasetRelease, target: Path) -> None:
    print(f"Downloading dataset {release.release_id} from {release.url}", file=stderr)
    request = Request(
        url=release.url,
        headers={DATASET_USER_AGENT_HEADER: DATASET_USER_AGENT},
    )
    with (
        urlopen(url=request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        target.open(mode="wb") as handle,
    ):
        copyfileobj(fsrc=response, fdst=handle)


def fetch_dataset_release(*, release: DatasetRelease) -> Path:
    target = dataset_cache_dir() / release.release_id / DATASET_FILENAME
    if target.exists():
        if file_content_hash(path=target) == release.content_hash:
            return target
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    download_path = target.with_name(f"{target.name}{DOWNLOAD_SUFFIX}")
    try:
        _download_release(release=release, target=download_path)
        downloaded_hash = file_content_hash(path=download_path)
        if downloaded_hash != release.content_hash:
            raise RuntimeError(
                f"downloaded dataset {release.release_id} has content hash {downloaded_hash}, "
                f"expected {release.content_hash}"
            )
        download_path.replace(target)
    finally:
        if download_path.exists():
            download_path.unlink()
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

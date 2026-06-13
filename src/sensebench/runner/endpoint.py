"""Probe OpenAI-compatible self-hosted endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from sensebench.runs.models import ModelID

HOSTED_VLLM_PREFIX: str = "hosted_vllm/"
OPENAI_PREFIX: str = "openai/"
LOCAL_HOSTNAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
VLLM_ENGINE_NAME: str = "vllm"
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 10.0
OPENAI_API_PATH_SUFFIX: str = "/v1"
MODELS_ENDPOINT_PATH: str = "/models"
VERSION_ENDPOINT_PATH: str = "/version"
MODELS_DATA_FIELD: str = "data"
MODEL_ID_FIELD: str = "id"
VERSION_FIELD: str = "version"


@dataclass(frozen=True, slots=True)
class EndpointProbe:
    served_model_ids: list[ModelID]
    engine_version: str | None


def is_local_endpoint(*, base_url: str) -> bool:
    hostname = urlsplit(base_url).hostname
    if hostname is None:
        return False
    return hostname in LOCAL_HOSTNAMES


def litellm_model_id(*, model: ModelID) -> ModelID:
    if model.startswith(HOSTED_VLLM_PREFIX) or model.startswith(OPENAI_PREFIX):
        return model
    return f"{HOSTED_VLLM_PREFIX}{model}"


def served_model_id(*, requested_model: ModelID) -> ModelID:
    for prefix in (HOSTED_VLLM_PREFIX, OPENAI_PREFIX):
        if requested_model.startswith(prefix):
            return requested_model[len(prefix) :]
    return requested_model


def _fetch_json(*, url: str, timeout_seconds: float) -> object:
    with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _served_model_ids(*, base_url: str, timeout_seconds: float) -> list[ModelID]:
    url = f"{base_url.rstrip('/')}{MODELS_ENDPOINT_PATH}"
    try:
        payload = _fetch_json(url=url, timeout_seconds=timeout_seconds)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot reach OpenAI-compatible endpoint at {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected response from {url}: not a JSON object")
    data = payload.get(MODELS_DATA_FIELD)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response from {url}: missing model list")
    model_ids: list[ModelID] = []
    for entry in data:
        if isinstance(entry, dict):
            model_id = entry.get(MODEL_ID_FIELD)
            if isinstance(model_id, str) and len(model_id) > 0:
                model_ids.append(model_id)
    return model_ids


def _server_root_url(*, base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith(OPENAI_API_PATH_SUFFIX):
        return trimmed[: -len(OPENAI_API_PATH_SUFFIX)]
    return trimmed


def _engine_version(*, base_url: str, timeout_seconds: float) -> str | None:
    url = f"{_server_root_url(base_url=base_url)}{VERSION_ENDPOINT_PATH}"
    try:
        payload = _fetch_json(url=url, timeout_seconds=timeout_seconds)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get(VERSION_FIELD)
    if isinstance(version, str) and len(version) > 0:
        return version
    return None


def probe_openai_endpoint(
    *,
    base_url: str,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> EndpointProbe:
    return EndpointProbe(
        served_model_ids=_served_model_ids(base_url=base_url, timeout_seconds=timeout_seconds),
        engine_version=_engine_version(base_url=base_url, timeout_seconds=timeout_seconds),
    )

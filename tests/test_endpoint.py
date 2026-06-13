from __future__ import annotations

import pytest
from pytest import MonkeyPatch

import sensebench.runner.endpoint as endpoint_module
from sensebench.runner.endpoint import (
    MODEL_ID_FIELD,
    MODELS_DATA_FIELD,
    VERSION_FIELD,
    is_local_endpoint,
    litellm_model_id,
    probe_openai_endpoint,
    served_model_id,
)

LOCAL_BASE_URL: str = "http://localhost:8000/v1"
LOOPBACK_BASE_URL: str = "http://127.0.0.1:8000/v1"
REMOTE_BASE_URL: str = "https://api.example.com/v1"
BARE_MODEL_ID: str = "Qwen/Qwen3.6-27B-FP8"
OTHER_MODEL_ID: str = "other-model"
HOSTED_VLLM_MODEL_ID: str = f"hosted_vllm/{BARE_MODEL_ID}"
OPENAI_PREFIXED_MODEL_ID: str = f"openai/{BARE_MODEL_ID}"
VLLM_VERSION: str = "0.22.1"
MODELS_URL: str = f"{LOCAL_BASE_URL}/models"
VERSION_URL: str = "http://localhost:8000/version"
FETCH_JSON_ATTR: str = "_fetch_json"


def test_is_local_endpoint() -> None:
    assert is_local_endpoint(base_url=LOCAL_BASE_URL) is True
    assert is_local_endpoint(base_url=LOOPBACK_BASE_URL) is True
    assert is_local_endpoint(base_url=REMOTE_BASE_URL) is False
    assert is_local_endpoint(base_url="not a url") is False


def test_litellm_model_id_prefixes_bare_names() -> None:
    assert litellm_model_id(model=BARE_MODEL_ID) == HOSTED_VLLM_MODEL_ID
    assert litellm_model_id(model=HOSTED_VLLM_MODEL_ID) == HOSTED_VLLM_MODEL_ID
    assert litellm_model_id(model=OPENAI_PREFIXED_MODEL_ID) == OPENAI_PREFIXED_MODEL_ID


def test_served_model_id_strips_provider_prefixes() -> None:
    assert served_model_id(requested_model=HOSTED_VLLM_MODEL_ID) == BARE_MODEL_ID
    assert served_model_id(requested_model=OPENAI_PREFIXED_MODEL_ID) == BARE_MODEL_ID
    assert served_model_id(requested_model=BARE_MODEL_ID) == BARE_MODEL_ID


def test_probe_openai_endpoint_parses_models_and_version(monkeypatch: MonkeyPatch) -> None:
    payloads: dict[str, object] = {
        MODELS_URL: {
            MODELS_DATA_FIELD: [
                {MODEL_ID_FIELD: BARE_MODEL_ID},
                {MODEL_ID_FIELD: OTHER_MODEL_ID},
            ]
        },
        VERSION_URL: {VERSION_FIELD: VLLM_VERSION},
    }

    def fake_fetch_json(*, url: str, timeout_seconds: float) -> object:
        return payloads[url]

    monkeypatch.setattr(
        target=endpoint_module,
        name=FETCH_JSON_ATTR,
        value=fake_fetch_json,
    )

    probe = probe_openai_endpoint(base_url=LOCAL_BASE_URL)

    assert probe.served_model_ids == [BARE_MODEL_ID, OTHER_MODEL_ID]
    assert probe.engine_version == VLLM_VERSION


def test_probe_openai_endpoint_version_failure_is_not_fatal(monkeypatch: MonkeyPatch) -> None:
    def fake_fetch_json(*, url: str, timeout_seconds: float) -> object:
        if url == MODELS_URL:
            return {MODELS_DATA_FIELD: [{MODEL_ID_FIELD: BARE_MODEL_ID}]}
        raise OSError("no version endpoint")

    monkeypatch.setattr(
        target=endpoint_module,
        name=FETCH_JSON_ATTR,
        value=fake_fetch_json,
    )

    probe = probe_openai_endpoint(base_url=LOCAL_BASE_URL)

    assert probe.served_model_ids == [BARE_MODEL_ID]
    assert probe.engine_version is None


def test_probe_openai_endpoint_unreachable_models_is_fatal(monkeypatch: MonkeyPatch) -> None:
    def fake_fetch_json(*, url: str, timeout_seconds: float) -> object:
        raise OSError("connection refused")

    monkeypatch.setattr(
        target=endpoint_module,
        name=FETCH_JSON_ATTR,
        value=fake_fetch_json,
    )

    with pytest.raises(RuntimeError, match="cannot reach"):
        probe_openai_endpoint(base_url=LOCAL_BASE_URL)

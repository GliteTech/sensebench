from __future__ import annotations

from pathlib import Path

import pytest

from sensebench.cli import DEFAULT_RUN_CONCURRENCY, _build_parser, _uses_native_gemini_api, main
from sensebench.paths import (
    DEFAULT_LEXEN_RELEASE_ID,
    RUN_METADATA_FILENAME,
    SITE_OUTPUT_DIR,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.runs.models import RunMetadata
from tests.run_fixtures import make_metadata

RUN_ARGS: list[str] = ["run", "--prompt", "p001", "--model", "fake", "--run-id", "run-1"]
SET_RUNNER_HANDLE: str = "octocat"
SET_RUNNER_NAME: str = "Octo Cat"


def test_cli_uses_registered_dataset_default() -> None:
    parser = _build_parser()

    args = parser.parse_args(["render", "--prompt", "p001", "--limit", "1"])

    assert args.dataset == DEFAULT_LEXEN_RELEASE_ID
    assert args.dataset_jsonl is None


def test_run_cli_uses_512_default_concurrency() -> None:
    parser = _build_parser()

    args = parser.parse_args(["run", "--prompt", "p001", "--model", "fake", "--run-id", "run-1"])

    assert args.concurrency == DEFAULT_RUN_CONCURRENCY
    assert DEFAULT_RUN_CONCURRENCY == 512
    assert args.no_progress is False


def test_run_cli_run_id_is_optional() -> None:
    parser = _build_parser()

    args = parser.parse_args(["run", "--prompt", "p001", "--model", "fake"])

    assert args.run_id is None
    assert args.skip_preflight is False


def test_run_cli_can_disable_progress() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        ["run", "--prompt", "p001", "--model", "fake", "--run-id", "run-1", "--no-progress"]
    )

    assert args.no_progress is True


def test_run_cli_can_disable_thinking() -> None:
    parser = _build_parser()

    args = parser.parse_args([*RUN_ARGS, "--disable-thinking"])

    assert args.disable_thinking is True


def test_run_cli_uses_native_gemini_api_for_direct_gemini_provider() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "run",
            "--prompt",
            "p001",
            "--model",
            "gemini/gemma-4-26b-a4b-it",
            "--api-provider",
            "Gemini API",
        ]
    )

    assert _uses_native_gemini_api(args=args) is True


def test_run_cli_rejects_non_positive_votes() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--votes", "0"])


def test_run_cli_rejects_non_positive_concurrency() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--concurrency", "0"])


def test_run_cli_accepts_requests_per_minute() -> None:
    parser = _build_parser()

    args = parser.parse_args([*RUN_ARGS, "--requests-per-minute", "20.5"])

    assert args.requests_per_minute == 20.5


def test_run_cli_rejects_non_positive_requests_per_minute() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--requests-per-minute", "0"])


def test_run_cli_rejects_unknown_source_kind() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--source-kind", "bogus"])


def test_site_build_cli_defaults() -> None:
    parser = _build_parser()

    args = parser.parse_args(["site", "build"])

    assert args.results_dir == str(SUBMITTED_RESULTS_DIR)
    assert args.output_dir == str(SITE_OUTPUT_DIR)
    assert args.strict is False


def test_set_runner_requires_github_handle() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["set-runner", "runs/run-1"])


def test_set_runner_rewrites_runner_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        github_handle=None,
    )
    metadata_path = run_dir / RUN_METADATA_FILENAME
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "set-runner",
            str(run_dir),
            "--github-handle",
            SET_RUNNER_HANDLE,
            "--runner-name",
            SET_RUNNER_NAME,
        ]
    )

    assert exit_code == 0
    updated = RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    assert updated.runner.github_handle == SET_RUNNER_HANDLE
    assert updated.runner.name == SET_RUNNER_NAME
    assert updated.totals == metadata.totals
    assert updated.run_id == metadata.run_id

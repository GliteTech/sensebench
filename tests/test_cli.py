from __future__ import annotations

from pathlib import Path

import pytest

from sensebench.cli import (
    DEFAULT_RUN_CONCURRENCY,
    DEFAULT_SELF_HOSTED_CONCURRENCY,
    _build_parser,
    _machine_info,
    _resolved_concurrency,
    _self_hosted_model_reference,
    main,
)
from sensebench.paths import (
    DEFAULT_LEXEN_RELEASE_ID,
    RUN_METADATA_FILENAME,
    SITE_OUTPUT_DIR,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.runs.models import MachineInfo, ModelHostingKind, RunMetadata
from tests.run_fixtures import fixture_machine, make_metadata

RUN_COMMAND: str = "run"
RENDER_COMMAND: str = "render"
SITE_COMMAND: str = "site"
BUILD_COMMAND: str = "build"
SET_RUNNER_COMMAND: str = "set-runner"
MACHINE_INFO_COMMAND: str = "machine-info"
PROMPT_ARG: str = "--prompt"
MODEL_ARG: str = "--model"
RUN_ID_ARG: str = "--run-id"
PROMPT_ID: str = "p001"
FAKE_MODEL_ID: str = "fake"
RUN_ID: str = "run-1"
RUN_ARGS: list[str] = [
    RUN_COMMAND,
    PROMPT_ARG,
    PROMPT_ID,
    MODEL_ARG,
    FAKE_MODEL_ID,
    RUN_ID_ARG,
    RUN_ID,
]
SET_RUNNER_HANDLE: str = "octocat"
SET_RUNNER_NAME: str = "Octo Cat"
SELF_HOSTED_KIND_VALUE: str = "self_hosted"
LOCAL_ENDPOINT_BASE_URL: str = "http://localhost:8000/v1"
REMOTE_ENDPOINT_BASE_URL: str = "https://gpu-box.example.com/v1"


def _run_args(*, extra: list[str] | None = None, include_run_id: bool = True) -> list[str]:
    args = list(RUN_ARGS if include_run_id else RUN_ARGS[:5])
    if extra is not None:
        args.extend(extra)
    return args


def test_cli_uses_registered_dataset_default() -> None:
    parser = _build_parser()

    args = parser.parse_args([RENDER_COMMAND, PROMPT_ARG, PROMPT_ID, "--limit", "1"])

    assert args.dataset == DEFAULT_LEXEN_RELEASE_ID
    assert args.dataset_jsonl is None


def test_run_cli_resolves_concurrency_by_hosting_kind() -> None:
    parser = _build_parser()

    args = parser.parse_args(_run_args())

    assert args.concurrency is None
    assert args.no_progress is False
    assert DEFAULT_RUN_CONCURRENCY == 512
    assert DEFAULT_SELF_HOSTED_CONCURRENCY == 256
    assert (
        _resolved_concurrency(args=args, hosting_kind=ModelHostingKind.CLOUD_API)
        == DEFAULT_RUN_CONCURRENCY
    )
    assert (
        _resolved_concurrency(args=args, hosting_kind=ModelHostingKind.SELF_HOSTED)
        == DEFAULT_SELF_HOSTED_CONCURRENCY
    )
    explicit = parser.parse_args(_run_args(extra=["--concurrency", "7"]))
    assert _resolved_concurrency(args=explicit, hosting_kind=ModelHostingKind.SELF_HOSTED) == 7


def test_run_cli_run_id_is_optional() -> None:
    parser = _build_parser()

    args = parser.parse_args(_run_args(include_run_id=False))

    assert args.run_id is None
    assert args.skip_preflight is False


def test_run_cli_can_disable_progress() -> None:
    parser = _build_parser()

    args = parser.parse_args(_run_args(extra=["--no-progress"]))

    assert args.no_progress is True


def test_run_cli_rejects_non_positive_votes() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_run_args(extra=["--votes", "0"]))


def test_run_cli_rejects_non_positive_concurrency() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_run_args(extra=["--concurrency", "0"]))


def test_run_cli_rejects_unknown_source_kind() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_run_args(extra=["--source-kind", "bogus"]))


def test_site_build_cli_defaults() -> None:
    parser = _build_parser()

    args = parser.parse_args([SITE_COMMAND, BUILD_COMMAND])

    assert args.results_dir == str(SUBMITTED_RESULTS_DIR)
    assert args.output_dir == str(SITE_OUTPUT_DIR)
    assert args.strict is False


def test_set_runner_requires_github_handle() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([SET_RUNNER_COMMAND, "runs/run-1"])


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
            SET_RUNNER_COMMAND,
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


def test_run_cli_parses_self_hosted_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            *RUN_ARGS,
            "--hosting-kind",
            SELF_HOSTED_KIND_VALUE,
            "--endpoint-base-url",
            LOCAL_ENDPOINT_BASE_URL,
            "--hourly-rate-usd",
            "2.49",
            "--provider",
            "vast.ai",
            "--instance-id",
            "40430336",
            "--warmup-calls",
            "8",
            "--machine-info-json",
            "machine.json",
        ]
    )

    assert args.hourly_rate_usd == 2.49
    assert args.provider == "vast.ai"
    assert args.instance_id == "40430336"
    assert args.warmup_calls == 8
    assert args.machine_info_json == "machine.json"


def test_run_cli_rejects_negative_warmup_calls() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_run_args(extra=["--warmup-calls", "-1"]))


def test_run_cli_rejects_negative_hourly_rate() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_run_args(extra=["--hourly-rate-usd", "-1"]))


def test_self_hosted_model_reference_prefixes_requested_model() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            RUN_COMMAND,
            PROMPT_ARG,
            PROMPT_ID,
            MODEL_ARG,
            "Qwen/Qwen3.6-27B-FP8",
            "--hosting-kind",
            SELF_HOSTED_KIND_VALUE,
            "--endpoint-base-url",
            LOCAL_ENDPOINT_BASE_URL,
        ]
    )

    model = _self_hosted_model_reference(args=args)

    assert model.requested_model == "hosted_vllm/Qwen/Qwen3.6-27B-FP8"
    assert model.display_name == "Qwen/Qwen3.6-27B-FP8"


def test_machine_info_command_emits_machine_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([MACHINE_INFO_COMMAND, "--provider", "vast.ai", "--hourly-rate-usd", "2.5"])

    assert exit_code == 0
    machine = MachineInfo.model_validate_json(capsys.readouterr().out)
    assert machine.provider == "vast.ai"
    assert machine.hourly_rate_usd == 2.5


def test_machine_info_helper_merges_overrides(tmp_path: Path) -> None:
    machine_path = tmp_path / "machine.json"
    machine_path.write_text(fixture_machine().model_dump_json(), encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            *RUN_ARGS,
            "--hosting-kind",
            SELF_HOSTED_KIND_VALUE,
            "--endpoint-base-url",
            REMOTE_ENDPOINT_BASE_URL,
            "--machine-info-json",
            str(machine_path),
            "--hourly-rate-usd",
            "9.99",
        ]
    )

    machine = _machine_info(args=args, endpoint_base_url=REMOTE_ENDPOINT_BASE_URL)

    assert machine is not None
    assert machine.gpu == fixture_machine().gpu
    assert machine.hourly_rate_usd == 9.99
    assert machine.provider == fixture_machine().provider


def test_machine_info_helper_remote_endpoint_without_json_is_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            *RUN_ARGS,
            "--hosting-kind",
            SELF_HOSTED_KIND_VALUE,
            "--endpoint-base-url",
            REMOTE_ENDPOINT_BASE_URL,
        ]
    )

    machine = _machine_info(args=args, endpoint_base_url=REMOTE_ENDPOINT_BASE_URL)

    assert machine is None

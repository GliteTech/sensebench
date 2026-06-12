from __future__ import annotations

import pytest

from sensebench.cli import DEFAULT_RUN_CONCURRENCY, _build_parser
from sensebench.paths import DEFAULT_LEXEN_RELEASE_ID

RUN_ARGS: list[str] = ["run", "--prompt", "p001", "--model", "fake", "--run-id", "run-1"]


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


def test_run_cli_rejects_non_positive_votes() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--votes", "0"])


def test_run_cli_rejects_non_positive_concurrency() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--concurrency", "0"])


def test_run_cli_rejects_unknown_source_kind() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([*RUN_ARGS, "--source-kind", "bogus"])

"""Strict quality gate for SenseBench reruns."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensebench.datasets.models import DatasetBundle
from sensebench.datasets.releases import get_dataset_release, load_registered_dataset
from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR
from sensebench.prompts.models import PromptDefinition
from sensebench.prompts.registry import load_prompt_definition
from sensebench.runner.extract import InvalidSenseIndexExtraction, extract_sense_index
from sensebench.verify.runs import RunValidationIssue, verify_run_directory

CALLS_STEM: str = "calls"
PREDICTIONS_STEM: str = "predictions"
RUN_METADATA_FILENAME: str = "run.json"
JSONL_SUFFIX: str = ".jsonl"
GZIP_JSONL_SUFFIX: str = ".jsonl.gz"
SUCCESS_STATUS: str = "success"
NO_VALID_VOTE_STATUS: str = "no_valid_vote"
LENGTH_FINISH_NEEDLES: tuple[str, ...] = ("length", "max_token")
RAW_RESPONSE_FIELD: str = "raw_response"
CHOICES_FIELD: str = "choices"
FINISH_REASON_FIELD: str = "finish_reason"
RESPONSE_METADATA_FIELD: str = "response_metadata"
USAGE_FIELD: str = "usage"
OUTPUT_TOKENS_FIELD: str = "output_tokens"


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Problem-count thresholds allowed before a rerun is rejected."""

    final_no_valid: int
    transport_errors: int
    length_finishes: int
    max_token_hits: int
    invalid_attempts: int
    invalid_votes: int


@dataclass(frozen=True, slots=True)
class RunQualitySummary:
    """Computed quality stats for one run directory."""

    run_dir: Path
    run_id: str
    prompt_id: str
    official_verify_ok: bool
    official_verify_issues: list[str]
    item_count: int
    accuracy: float | None
    final_no_valid: int
    transport_errors: int
    length_finishes: int
    max_token_hits: int
    invalid_attempts: Counter[str]
    invalid_votes: Counter[str]
    transport_error_kinds: Counter[str]

    @property
    def invalid_attempt_count(self) -> int:
        return sum(self.invalid_attempts.values())

    @property
    def invalid_vote_count(self) -> int:
        return sum(self.invalid_votes.values())


def _artifact_path(*, run_dir: Path, stem: str) -> Path:
    for suffix in (JSONL_SUFFIX, GZIP_JSONL_SUFFIX):
        path = run_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{run_dir}: missing {stem}.jsonl or {stem}.jsonl.gz")


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(GZIP_JSONL_SUFFIX):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _finish_reasons(call: Mapping[str, Any]) -> list[str]:
    values: list[object] = [
        call.get(FINISH_REASON_FIELD),
        _mapping_get(call.get(RESPONSE_METADATA_FIELD), FINISH_REASON_FIELD),
    ]
    raw_response = call.get(RAW_RESPONSE_FIELD)
    choices = _mapping_get(raw_response, CHOICES_FIELD)
    if isinstance(choices, list):
        for choice in choices:
            values.append(_mapping_get(choice, FINISH_REASON_FIELD))
    return [str(value) for value in values if value is not None]


def _mapping_get(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _is_length_finish(call: Mapping[str, Any]) -> bool:
    for finish_reason in _finish_reasons(call=call):
        normalized = finish_reason.lower()
        if any(needle in normalized for needle in LENGTH_FINISH_NEEDLES):
            return True
    return False


def _is_max_token_hit(*, call: Mapping[str, Any], max_tokens: int | None) -> bool:
    if max_tokens is None:
        return False
    output_tokens = _mapping_get(call.get(USAGE_FIELD), OUTPUT_TOKENS_FIELD)
    return isinstance(output_tokens, int) and output_tokens >= max_tokens


def _official_issue_text(issue: RunValidationIssue) -> str:
    return f"[{issue.rule.value}] {issue.location}: {issue.message}"


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    with (run_dir / RUN_METADATA_FILENAME).open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_prompt(prompt_id: str) -> PromptDefinition:
    prompt_path = PROMPT_REGISTRY_DIR / f"{prompt_id}{PROMPT_JSON_SUFFIX}"
    return load_prompt_definition(path=prompt_path)


def collect_run_quality(
    *,
    run_dir: Path,
    dataset: DatasetBundle | None,
) -> RunQualitySummary:
    metadata = _load_run_metadata(run_dir=run_dir)
    prompt_id = str(metadata["prompt"]["id"])
    prompt = _load_prompt(prompt_id=prompt_id)
    report = verify_run_directory(run_dir=run_dir, dataset=dataset, prompt=prompt)

    predictions_path = _artifact_path(
        run_dir=run_dir,
        stem=PREDICTIONS_STEM,
    )
    predictions = list(_iter_jsonl_records(predictions_path))
    calls_path = _artifact_path(run_dir=run_dir, stem=CALLS_STEM)
    candidate_counts = {
        str(prediction["item_id"]): len(prediction.get("candidates", []))
        for prediction in predictions
    }
    prediction_statuses = Counter(str(prediction.get("status")) for prediction in predictions)
    invalid_votes: Counter[str] = Counter()
    for prediction in predictions:
        votes = prediction.get("votes")
        if not isinstance(votes, list):
            continue
        for vote in votes:
            reason = _mapping_get(vote, "invalid_reason")
            if reason is not None:
                invalid_votes[str(reason)] += 1

    transport_errors = 0
    length_finishes = 0
    max_token_hits = 0
    invalid_attempts: Counter[str] = Counter()
    transport_error_kinds: Counter[str] = Counter()
    max_tokens = _mapping_get(metadata.get("sampling"), "max_tokens")
    max_tokens = max_tokens if isinstance(max_tokens, int) else None

    for call in _iter_jsonl_records(calls_path):
        status = str(call.get("status"))
        if status != SUCCESS_STATUS:
            transport_errors += 1
            error_kind = call.get("error_kind") or status
            transport_error_kinds[str(error_kind)] += 1
            continue
        if _is_length_finish(call=call):
            length_finishes += 1
        if _is_max_token_hit(call=call, max_tokens=max_tokens):
            max_token_hits += 1
        item_id = str(call.get("item_id"))
        candidate_count = candidate_counts.get(item_id)
        if candidate_count is None:
            continue
        extraction = extract_sense_index(
            text=call.get("raw_output"),
            output_mode=prompt.output.mode,
            candidate_count=candidate_count,
        )
        if isinstance(extraction, InvalidSenseIndexExtraction):
            invalid_attempts[extraction.invalid_reason.value] += 1

    totals = metadata.get("totals", {})
    return RunQualitySummary(
        run_dir=run_dir,
        run_id=str(metadata["run_id"]),
        prompt_id=prompt_id,
        official_verify_ok=not report.has_errors(),
        official_verify_issues=[_official_issue_text(issue=issue) for issue in report.issues],
        item_count=int(totals.get("item_count", len(predictions))),
        accuracy=totals.get("accuracy") if isinstance(totals.get("accuracy"), float) else None,
        final_no_valid=prediction_statuses[NO_VALID_VOTE_STATUS],
        transport_errors=transport_errors,
        length_finishes=length_finishes,
        max_token_hits=max_token_hits,
        invalid_attempts=invalid_attempts,
        invalid_votes=invalid_votes,
        transport_error_kinds=transport_error_kinds,
    )


def quality_gate_failures(
    *,
    summary: RunQualitySummary,
    thresholds: QualityThresholds,
) -> list[str]:
    failures: list[str] = []
    if not summary.official_verify_ok:
        failures.append(
            f"official verification failed with {len(summary.official_verify_issues)} issue(s)",
        )
    if summary.final_no_valid > thresholds.final_no_valid:
        failures.append(
            f"final no_valid_vote={summary.final_no_valid} "
            f"> allowed {thresholds.final_no_valid}",
        )
    if summary.transport_errors > thresholds.transport_errors:
        failures.append(
            f"transport_errors={summary.transport_errors} "
            f"> allowed {thresholds.transport_errors}",
        )
    if summary.length_finishes > thresholds.length_finishes:
        failures.append(
            f"length_finishes={summary.length_finishes} "
            f"> allowed {thresholds.length_finishes}",
        )
    if summary.max_token_hits > thresholds.max_token_hits:
        failures.append(
            f"max_token_hits={summary.max_token_hits} > allowed {thresholds.max_token_hits}",
        )
    if summary.invalid_attempt_count > thresholds.invalid_attempts:
        failures.append(
            f"invalid_attempts={summary.invalid_attempt_count} "
            f"> allowed {thresholds.invalid_attempts}",
        )
    if summary.invalid_vote_count > thresholds.invalid_votes:
        failures.append(
            f"invalid_votes={summary.invalid_vote_count} > allowed {thresholds.invalid_votes}",
        )
    return failures


def _summary_dict(*, summary: RunQualitySummary, failures: list[str]) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "run_dir": str(summary.run_dir),
        "prompt": summary.prompt_id,
        "official_verify_ok": summary.official_verify_ok,
        "official_verify_issues": summary.official_verify_issues,
        "item_count": summary.item_count,
        "accuracy": summary.accuracy,
        "final_no_valid": summary.final_no_valid,
        "transport_errors": summary.transport_errors,
        "transport_error_kinds": dict(summary.transport_error_kinds),
        "length_finishes": summary.length_finishes,
        "max_token_hits": summary.max_token_hits,
        "invalid_attempts": dict(summary.invalid_attempts),
        "invalid_votes": dict(summary.invalid_votes),
        "failures": failures,
    }


def _print_text_summary(*, summary: RunQualitySummary, failures: list[str]) -> None:
    status = "FAILED" if len(failures) > 0 else "OK"
    accuracy = "n/a" if summary.accuracy is None else f"{summary.accuracy:.6f}"
    print(f"{summary.run_id}: {status}")
    print(f"  prompt={summary.prompt_id} items={summary.item_count} accuracy={accuracy}")
    print(
        "  "
        f"no_valid={summary.final_no_valid} "
        f"transport={summary.transport_errors} "
        f"length={summary.length_finishes} "
        f"max_token_hits={summary.max_token_hits}",
    )
    print(
        "  "
        f"invalid_attempts={dict(summary.invalid_attempts)} "
        f"invalid_votes={dict(summary.invalid_votes)}",
    )
    for failure in failures:
        print(f"  failure: {failure}")
    for issue in summary.official_verify_issues[:5]:
        print(f"  verify: {issue}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--dataset", default="lexen-v1")
    parser.add_argument("--allow-final-no-valid", type=int, default=0)
    parser.add_argument("--allow-transport-errors", type=int, default=0)
    parser.add_argument("--allow-length-finishes", type=int, default=0)
    parser.add_argument("--allow-max-token-hits", type=int, default=0)
    parser.add_argument("--allow-invalid-attempts", type=int, default=0)
    parser.add_argument("--allow-invalid-votes", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    dataset = load_registered_dataset(release=get_dataset_release(release_id=str(args.dataset)))
    thresholds = QualityThresholds(
        final_no_valid=args.allow_final_no_valid,
        transport_errors=args.allow_transport_errors,
        length_finishes=args.allow_length_finishes,
        max_token_hits=args.allow_max_token_hits,
        invalid_attempts=args.allow_invalid_attempts,
        invalid_votes=args.allow_invalid_votes,
    )
    payloads: list[dict[str, Any]] = []
    has_failures = False
    for run_dir in args.run_dirs:
        summary = collect_run_quality(run_dir=run_dir, dataset=dataset)
        failures = quality_gate_failures(summary=summary, thresholds=thresholds)
        has_failures = has_failures or len(failures) > 0
        payload = _summary_dict(summary=summary, failures=failures)
        payloads.append(payload)
        if not args.json:
            _print_text_summary(summary=summary, failures=failures)
    if args.json:
        print(json.dumps(payloads, indent=2, sort_keys=True))
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

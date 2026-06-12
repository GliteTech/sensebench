"""Aggregate submitted results into leaderboard data."""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from sensebench.datasets.models import DatasetBundle, DatasetID
from sensebench.datasets.releases import (
    DatasetRelease,
    get_dataset_release,
    load_registered_dataset,
)
from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR, RUN_METADATA_FILENAME
from sensebench.prompts.models import PromptDefinition, PromptID
from sensebench.prompts.registry import load_prompt_definition
from sensebench.runs.loaders import LoadedRun, load_run_directory
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    SELF_HOSTED_LLM_KIND,
    CloudLlmReference,
    ModelHostingKind,
    ModelReference,
    PredictionRecord,
    PredictionStatus,
    RunID,
    TokenUsage,
    VoteStatus,
)
from sensebench.verify.runs import RunValidationIssue, verify_run_directory

DEFAULT_BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_BOOTSTRAP_SEED: int = 12345
CONFIDENCE_LOW_PERCENTILE: float = 2.5
CONFIDENCE_HIGH_PERCENTILE: float = 97.5
LEADERBOARD_SCHEMA_VERSION: str = "sensebench-leaderboard-v4"
RUN_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9._-]+$")
UNKNOWN_MODEL_HOSTING_KIND: str = "unknown"
MISSING_DATASET_VERSION_GROUP_VALUE: str = "dataset_version:none"
RANK_FIELD: str = "rank"


class LeaderboardModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AccuracyInterval(LeaderboardModel):
    low: float | None
    high: float | None


class LeaderboardEntry(LeaderboardModel):
    rank: int
    run_id: RunID
    run_url: str
    created_at: str
    git_commit: str | None
    runner_github_handle: str | None
    runner_name: str | None
    model: str
    requested_model: str
    resolved_model: str | None
    model_kind: str
    hosting_kind: ModelHostingKind | str
    source_kind: str
    llm_vendor: str | None
    api_provider: str | None
    license: str | None
    model_url: str | None
    reasoning_effort: str | None
    prompt_id: PromptID
    prompt_name: str | None
    dataset_id: DatasetID
    dataset_version: str | None
    dataset_content_hash: str | None
    accuracy: float | None
    accuracy_ci: AccuracyInterval
    correct_count: int
    item_count: int
    call_count: int
    success_count: int
    monosemous_count: int
    no_candidates_count: int
    no_valid_vote_count: int
    invalid_output_vote_count: int
    transport_error_vote_count: int
    input_tokens: int | None
    input_uncached_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None
    total_tokens: int | None
    tokens_per_item: float | None
    cost_source: str
    cost_usd: float | None
    input_uncached_usd: float | None
    input_cached_usd: float | None
    output_usd: float | None
    input_uncached_unit_price_usd: float | None
    input_cached_unit_price_usd: float | None
    output_unit_price_usd: float | None
    cost_per_million_items: float | None
    elapsed_seconds: float | None
    best_group_key: str


class LeaderboardFile(LeaderboardModel):
    schema_version: str
    generated_at: str
    entries: list[LeaderboardEntry]


@dataclass(frozen=True, slots=True)
class LeaderboardCollectionIssue:
    run_dir: Path
    message: str


@dataclass(frozen=True, slots=True)
class LeaderboardCollection:
    entries: list[LeaderboardEntry]
    issues: list[LeaderboardCollectionIssue]


@dataclass(frozen=True, slots=True)
class VerifiedEntryResult:
    entry: LeaderboardEntry | None
    issues: list[LeaderboardCollectionIssue]


@dataclass(frozen=True, slots=True, order=True)
class LeaderboardSortKey:
    negative_accuracy: float
    cost_per_million_items: float
    negative_created_at_timestamp: float
    run_id: RunID


class LeaderboardBuildError(RuntimeError):
    def __init__(self, *, issues: list[LeaderboardCollectionIssue]) -> None:
        self.issues = issues
        joined = "\n".join(f"{issue.run_dir}: {issue.message}" for issue in issues)
        super().__init__(f"leaderboard build failed:\n{joined}")


def bootstrap_accuracy_ci(*, values: list[bool]) -> AccuracyInterval:
    if len(values) == 0:
        return AccuracyInterval(low=None, high=None)
    numeric: NDArray[np.float64] = np.array(
        object=[1.0 if value else 0.0 for value in values],
        dtype=np.float64,
    )
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    estimates: NDArray[np.float64] = np.empty(
        shape=DEFAULT_BOOTSTRAP_RESAMPLES,
        dtype=np.float64,
    )
    for index in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        sample: NDArray[np.float64] = rng.choice(
            a=numeric,
            size=len(numeric),
            replace=True,
        )
        estimates[index] = float(np.mean(sample))
    percentiles: NDArray[np.float64] = np.percentile(
        a=estimates,
        q=[CONFIDENCE_LOW_PERCENTILE, CONFIDENCE_HIGH_PERCENTILE],
    )
    return AccuracyInterval(low=float(percentiles[0]), high=float(percentiles[1]))


def _registered_prompt(*, prompt_id: PromptID) -> PromptDefinition | None:
    path = PROMPT_REGISTRY_DIR / f"{prompt_id}{PROMPT_JSON_SUFFIX}"
    try:
        return load_prompt_definition(path=path)
    except (OSError, ValidationError):
        return None


def _divide(*, numerator: float | int | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return float(numerator) / denominator


def _token_total(*, usage: TokenUsage) -> int | None:
    values: list[int | None] = [
        usage.input_tokens,
        usage.output_tokens,
        usage.reasoning_output_tokens,
    ]
    present: list[int] = [value for value in values if value is not None]
    if len(present) == 0:
        return None
    return sum(present)


def _input_uncached_tokens(*, usage: TokenUsage) -> int | None:
    if usage.input_tokens is None:
        return None
    if usage.cached_input_tokens is None:
        return usage.input_tokens
    return max(usage.input_tokens - usage.cached_input_tokens, 0)


def _status_counts(*, predictions: list[PredictionRecord]) -> Counter[PredictionStatus]:
    return Counter(prediction.status for prediction in predictions)


def _invalid_vote_count(*, predictions: list[PredictionRecord]) -> int:
    return sum(
        1
        for prediction in predictions
        for vote in prediction.votes
        if vote.status == VoteStatus.INVALID_OUTPUT
    )


def _transport_error_vote_count(*, predictions: list[PredictionRecord]) -> int:
    return sum(
        1
        for prediction in predictions
        for vote in prediction.votes
        if vote.status == VoteStatus.TRANSPORT_ERROR
    )


def _hosting_kind(*, model_kind: str) -> ModelHostingKind | str:
    if model_kind == CLOUD_LLM_KIND:
        return ModelHostingKind.CLOUD_API
    if model_kind == SELF_HOSTED_LLM_KIND:
        return ModelHostingKind.SELF_HOSTED
    return UNKNOWN_MODEL_HOSTING_KIND


def _api_provider(*, model: ModelReference) -> str | None:
    if isinstance(model, CloudLlmReference):
        return model.api_provider
    return None


def _reasoning_effort(*, model: ModelReference) -> str | None:
    if isinstance(model, CloudLlmReference):
        return model.reasoning_effort
    return None


def _best_group_key(*, loaded: LoadedRun) -> str:
    metadata = loaded.metadata
    return "|".join(
        [
            metadata.model.display_name,
            metadata.dataset.dataset_version
            if metadata.dataset.dataset_version is not None
            else MISSING_DATASET_VERSION_GROUP_VALUE,
        ]
    )


def _entry_for_run(
    *,
    loaded: LoadedRun,
    prompt: PromptDefinition | None,
    rank: int,
) -> LeaderboardEntry:
    metadata = loaded.metadata
    correctness: list[bool] = [prediction.is_correct is True for prediction in loaded.predictions]
    correct_count = sum(1 for value in correctness if value)
    item_count = len(loaded.predictions)
    accuracy = correct_count / item_count if item_count > 0 else None
    usage = metadata.totals.usage
    total_tokens = _token_total(usage=usage)
    status_counts = _status_counts(predictions=loaded.predictions)
    cost = metadata.totals.cost
    cost_usd = cost.total_usd
    model = metadata.model
    model_kind = model.kind
    return LeaderboardEntry(
        rank=rank,
        run_id=metadata.run_id,
        run_url=f"runs/{metadata.run_id}/",
        created_at=metadata.created_at.isoformat(),
        git_commit=metadata.git_commit,
        runner_github_handle=metadata.runner.github_handle,
        runner_name=metadata.runner.name,
        model=model.display_name,
        requested_model=model.requested_model,
        resolved_model=model.resolved_model,
        model_kind=model_kind,
        hosting_kind=_hosting_kind(model_kind=model_kind),
        source_kind=model.source_kind.value,
        llm_vendor=model.llm_vendor,
        api_provider=_api_provider(model=model),
        license=model.license,
        model_url=model.model_url,
        reasoning_effort=_reasoning_effort(model=model),
        prompt_id=metadata.prompt.id,
        prompt_name=prompt.name if prompt is not None else None,
        dataset_id=metadata.dataset.dataset_id,
        dataset_version=metadata.dataset.dataset_version,
        dataset_content_hash=metadata.dataset.content_hash,
        accuracy=accuracy,
        accuracy_ci=bootstrap_accuracy_ci(values=correctness),
        correct_count=correct_count,
        item_count=item_count,
        call_count=len(loaded.calls),
        success_count=status_counts.get(PredictionStatus.SUCCESS, 0),
        monosemous_count=status_counts.get(PredictionStatus.MONOSEMOUS, 0),
        no_candidates_count=status_counts.get(PredictionStatus.NO_CANDIDATES, 0),
        no_valid_vote_count=status_counts.get(PredictionStatus.NO_VALID_VOTE, 0),
        invalid_output_vote_count=_invalid_vote_count(predictions=loaded.predictions),
        transport_error_vote_count=_transport_error_vote_count(predictions=loaded.predictions),
        input_tokens=usage.input_tokens,
        input_uncached_tokens=_input_uncached_tokens(usage=usage),
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_output_tokens=usage.reasoning_output_tokens,
        total_tokens=total_tokens,
        tokens_per_item=_divide(numerator=total_tokens, denominator=item_count),
        cost_source=cost.source.value,
        cost_usd=cost_usd,
        input_uncached_usd=cost.input_uncached_usd,
        input_cached_usd=cost.input_cached_usd,
        output_usd=cost.output_usd,
        input_uncached_unit_price_usd=cost.input_uncached_unit_price_usd,
        input_cached_unit_price_usd=cost.input_cached_unit_price_usd,
        output_unit_price_usd=cost.output_unit_price_usd,
        cost_per_million_items=None
        if cost_usd is None or item_count <= 0
        else (cost_usd / item_count) * 1_000_000,
        elapsed_seconds=metadata.totals.elapsed_seconds,
        best_group_key=_best_group_key(loaded=loaded),
    )


def _release_for_metadata(*, loaded: LoadedRun) -> DatasetRelease:
    metadata = loaded.metadata
    release_id = metadata.dataset.dataset_version
    if release_id is None:
        raise ValueError("dataset_version must name a registered release")
    release = get_dataset_release(release_id=release_id)
    if metadata.dataset.dataset_id != release.dataset_id:
        raise ValueError(
            f"dataset_id {metadata.dataset.dataset_id} does not match release "
            f"{release.dataset_id}"
        )
    if metadata.dataset.item_count != release.item_count:
        raise ValueError(
            f"item_count {metadata.dataset.item_count} does not match release "
            f"{release.item_count}"
        )
    if metadata.dataset.content_hash != release.content_hash:
        raise ValueError(
            f"content_hash {metadata.dataset.content_hash} does not match release "
            f"{release.content_hash}"
        )
    return release


def _official_dataset(
    *,
    loaded: LoadedRun,
    cache: dict[str, DatasetBundle],
) -> DatasetBundle:
    release = _release_for_metadata(loaded=loaded)
    dataset = cache.get(release.release_id)
    if dataset is None:
        dataset = load_registered_dataset(release=release)
        cache[release.release_id] = dataset
    return dataset


def _format_verification_issues(*, issues: list[RunValidationIssue]) -> str:
    failed_rules: list[str] = sorted({issue.rule.value for issue in issues})
    return f"failed verification ({', '.join(failed_rules)})"


def _eligibility_issues(*, loaded: LoadedRun, official: bool) -> list[str]:
    issues: list[str] = []
    run_id = loaded.metadata.run_id
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        issues.append("run_id must contain only lowercase letters, numbers, '.', '_', and '-'")
    github_handle = loaded.metadata.runner.github_handle
    if official and (github_handle is None or len(github_handle.strip()) == 0):
        issues.append(
            "official submissions require runner.github_handle in run.json "
            "(set it with: sensebench set-runner <run-dir> --github-handle <handle>)"
        )
    return issues


def _verified_entry_for_run(
    *,
    run_dir: Path,
    official: bool,
    dataset_cache: dict[str, DatasetBundle],
) -> VerifiedEntryResult:
    try:
        loaded = load_run_directory(run_dir=run_dir)
    except Exception as exc:
        return VerifiedEntryResult(
            entry=None,
            issues=[
                LeaderboardCollectionIssue(run_dir=run_dir, message=f"cannot load ({exc})"),
            ],
        )

    local_issues: list[LeaderboardCollectionIssue] = [
        LeaderboardCollectionIssue(run_dir=run_dir, message=message)
        for message in _eligibility_issues(loaded=loaded, official=official)
    ]
    prompt = _registered_prompt(prompt_id=loaded.metadata.prompt.id)
    if prompt is None:
        local_issues.append(
            LeaderboardCollectionIssue(
                run_dir=run_dir,
                message=f"prompt {loaded.metadata.prompt.id} is not registered",
            )
        )

    dataset: DatasetBundle | None = None
    if official:
        try:
            dataset = _official_dataset(loaded=loaded, cache=dataset_cache)
        except Exception as exc:
            local_issues.append(
                LeaderboardCollectionIssue(
                    run_dir=run_dir,
                    message=f"not eligible for official leaderboard ({exc})",
                )
            )

    if len(local_issues) > 0:
        return VerifiedEntryResult(entry=None, issues=local_issues)

    try:
        report = verify_run_directory(run_dir=run_dir, dataset=dataset, prompt=prompt)
    except Exception as exc:
        return VerifiedEntryResult(
            entry=None,
            issues=[
                LeaderboardCollectionIssue(run_dir=run_dir, message=f"cannot verify ({exc})"),
            ],
        )
    if report.has_errors():
        return VerifiedEntryResult(
            entry=None,
            issues=[
                LeaderboardCollectionIssue(
                    run_dir=run_dir,
                    message=_format_verification_issues(issues=report.issues),
                ),
            ],
        )
    return VerifiedEntryResult(
        entry=_entry_for_run(loaded=loaded, prompt=prompt, rank=0),
        issues=[],
    )


def _created_at_timestamp(*, value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _sort_key(entry: LeaderboardEntry) -> LeaderboardSortKey:
    accuracy = entry.accuracy if entry.accuracy is not None else -1.0
    cost = (
        entry.cost_per_million_items
        if entry.cost_per_million_items is not None
        else float("inf")
    )
    created_at = _created_at_timestamp(value=entry.created_at)
    created_at_sort_value = created_at if created_at is not None else float("-inf")
    return LeaderboardSortKey(
        negative_accuracy=-accuracy,
        cost_per_million_items=cost,
        negative_created_at_timestamp=-created_at_sort_value,
        run_id=entry.run_id,
    )


def _ranked(*, entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    sorted_entries: list[LeaderboardEntry] = sorted(entries, key=_sort_key)
    return [
        entry.model_copy(update={RANK_FIELD: rank})
        for rank, entry in enumerate(sorted_entries, 1)
    ]


def collect_leaderboard_entries(
    *,
    results_dir: Path,
    official: bool = False,
    fail_on_invalid: bool = False,
) -> LeaderboardCollection:
    entries: list[LeaderboardEntry] = []
    issues: list[LeaderboardCollectionIssue] = []
    seen_run_ids: set[RunID] = set()
    dataset_cache: dict[str, DatasetBundle] = {}
    if results_dir.exists():
        for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            if not (run_dir / RUN_METADATA_FILENAME).exists():
                continue
            result = _verified_entry_for_run(
                run_dir=run_dir,
                official=official,
                dataset_cache=dataset_cache,
            )
            issues.extend(result.issues)
            if result.entry is None:
                continue
            if result.entry.run_id in seen_run_ids:
                issues.append(
                    LeaderboardCollectionIssue(
                        run_dir=run_dir,
                        message=f"duplicate run_id {result.entry.run_id}",
                    )
                )
                continue
            seen_run_ids.add(result.entry.run_id)
            entries.append(result.entry)
    if fail_on_invalid and len(issues) > 0:
        raise LeaderboardBuildError(issues=issues)
    return LeaderboardCollection(entries=_ranked(entries=entries), issues=issues)


def _generated_at() -> str:
    return datetime.now(tz=UTC).isoformat()


def leaderboard_file(*, entries: list[LeaderboardEntry]) -> LeaderboardFile:
    return LeaderboardFile(
        schema_version=LEADERBOARD_SCHEMA_VERSION,
        generated_at=_generated_at(),
        entries=entries,
    )


def emit_leaderboard(
    *,
    results_dir: Path,
    output_path: Path,
    official: bool = False,
    strict: bool = False,
) -> None:
    collection = collect_leaderboard_entries(
        results_dir=results_dir,
        official=official,
        fail_on_invalid=strict,
    )
    for issue in collection.issues:
        print(f"skipping {issue.run_dir}: {issue.message}", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        leaderboard_file(entries=collection.entries).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

"""Validate SenseBench run directories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetBundle, DatasetIndex
from sensebench.prompts.models import PromptDefinition
from sensebench.prompts.render import render_task
from sensebench.runs.loaders import (
    CALLS_FILENAME,
    PREDICTIONS_FILENAME,
    RUN_METADATA_FILENAME,
    load_run_directory,
)
from sensebench.runs.models import AttemptKind, CallRecord, PredictionRecord
from sensebench.wordnet import SenseCandidate, get_candidate_senses

ACCURACY_TOLERANCE: float = 1e-12


class RunValidationRule(StrEnum):
    LOADABLE = "loadable"
    RUN_ID_MATCHES_DIRECTORY = "run_id_matches_directory"
    ITEM_COUNT = "item_count"
    DUPLICATE_ITEM = "duplicate_item"
    CALL_COUNT = "call_count"
    CORRECT_COUNT = "correct_count"
    ACCURACY = "accuracy"
    CANDIDATE_INDEX = "candidate_index"
    VOTE_CALL_REFERENCE = "vote_call_reference"
    DATASET_ITEMS = "dataset_items"
    PROMPT_RENDERING = "prompt_rendering"


@dataclass(frozen=True, slots=True)
class RunValidationIssue:
    rule: RunValidationRule
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class RunValidationReport:
    run_dir: Path
    issues: list[RunValidationIssue]

    def has_errors(self) -> bool:
        return len(self.issues) > 0


def _correct_count(*, predictions: list[PredictionRecord]) -> int:
    return sum(1 for prediction in predictions if prediction.is_correct is True)


def _candidate_issues(*, prediction: PredictionRecord) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    indexes: list[int] = [candidate.index for candidate in prediction.candidates]
    if len(indexes) != len(set(indexes)):
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.CANDIDATE_INDEX,
                location=prediction.item_id,
                message="candidate indexes must be unique",
            )
        )
    if indexes != sorted(indexes):
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.CANDIDATE_INDEX,
                location=prediction.item_id,
                message="candidate indexes must be sorted",
            )
        )
    return issues


def _vote_reference_issues(
    *,
    prediction: PredictionRecord,
    calls_by_id: dict[str, CallRecord],
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    for vote in prediction.votes:
        for call_id in vote.call_ids:
            if call_id not in calls_by_id:
                issues.append(
                    RunValidationIssue(
                        rule=RunValidationRule.VOTE_CALL_REFERENCE,
                        location=f"{prediction.item_id}:{vote.vote_index}",
                        message=f"unknown call_id {call_id}",
                    )
                )
    return issues


def _basic_issues(
    *,
    run_dir: Path,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
    loaded_run_id: str,
    expected_item_count: int,
    expected_call_count: int,
    expected_correct_count: int,
    expected_accuracy: float | None,
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    if run_dir.name != loaded_run_id:
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.RUN_ID_MATCHES_DIRECTORY,
                location=str(run_dir),
                message=f"directory name must match run_id {loaded_run_id}",
            )
        )
    if len(predictions) != expected_item_count:
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.ITEM_COUNT,
                location=PREDICTIONS_FILENAME,
                message=f"expected {expected_item_count}, found {len(predictions)}",
            )
        )
    if len(calls) != expected_call_count:
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.CALL_COUNT,
                location=CALLS_FILENAME,
                message=f"expected {expected_call_count}, found {len(calls)}",
            )
        )
    if _correct_count(predictions=predictions) != expected_correct_count:
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.CORRECT_COUNT,
                location=RUN_METADATA_FILENAME,
                message="correct_count does not match predictions",
            )
        )
    if expected_accuracy is not None and len(predictions) > 0:
        recomputed = _correct_count(predictions=predictions) / len(predictions)
        if abs(recomputed - expected_accuracy) > ACCURACY_TOLERANCE:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.ACCURACY,
                    location=RUN_METADATA_FILENAME,
                    message="accuracy does not match predictions",
                )
            )
    return issues


def _duplicate_item_issues(*, predictions: list[PredictionRecord]) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.item_id in seen:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.DUPLICATE_ITEM,
                    location=prediction.item_id,
                    message="duplicate prediction item_id",
                )
            )
        seen.add(prediction.item_id)
    return issues


def _dataset_issues(
    *,
    predictions: list[PredictionRecord],
    dataset: DatasetBundle,
) -> list[RunValidationIssue]:
    expected: set[str] = {item.item_id for item in dataset.items}
    observed: set[str] = {prediction.item_id for prediction in predictions}
    if expected == observed:
        return []
    missing: list[str] = sorted(expected - observed)
    extra: list[str] = sorted(observed - expected)
    return [
        RunValidationIssue(
            rule=RunValidationRule.DATASET_ITEMS,
            location=PREDICTIONS_FILENAME,
            message=f"missing={missing[:10]} extra={extra[:10]}",
        )
    ]


def _message_payload(*, call: CallRecord) -> list[dict[str, str]]:
    return [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in call.messages
    ]


def _render_payload(
    *,
    dataset_index: DatasetIndex,
    prompt: PromptDefinition,
    item_id: str,
) -> list[dict[str, str]]:
    item = dataset_index.items_by_id[item_id]
    candidates: list[SenseCandidate] = get_candidate_senses(lemma=item.lemma, pos=item.pos)
    rendered = render_task(
        prompt=prompt,
        item=item,
        dataset_index=dataset_index,
        candidates=candidates,
    )
    return [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in rendered.messages
    ]


def _prompt_rendering_issues(
    *,
    calls: list[CallRecord],
    dataset: DatasetBundle,
    prompt: PromptDefinition,
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    rendered_by_item: dict[str, list[dict[str, str]]] = {}
    dataset_index = build_dataset_index(bundle=dataset)
    for call in calls:
        if call.attempt_kind != AttemptKind.INITIAL:
            continue
        expected = rendered_by_item.get(call.item_id)
        if expected is None:
            expected = _render_payload(
                dataset_index=dataset_index,
                prompt=prompt,
                item_id=call.item_id,
            )
            rendered_by_item[call.item_id] = expected
        if _message_payload(call=call) != expected:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.PROMPT_RENDERING,
                    location=call.call_id,
                    message="stored call messages differ from deterministic render",
                )
            )
    return issues


def verify_run_directory(
    *,
    run_dir: Path,
    dataset: DatasetBundle | None = None,
    prompt: PromptDefinition | None = None,
) -> RunValidationReport:
    try:
        loaded = load_run_directory(run_dir=run_dir)
    except (OSError, ValidationError) as exc:
        return RunValidationReport(
            run_dir=run_dir,
            issues=[
                RunValidationIssue(
                    rule=RunValidationRule.LOADABLE,
                    location=str(run_dir),
                    message=str(exc),
                )
            ],
        )

    issues = _basic_issues(
        run_dir=run_dir,
        predictions=loaded.predictions,
        calls=loaded.calls,
        loaded_run_id=loaded.metadata.run_id,
        expected_item_count=loaded.metadata.totals.item_count,
        expected_call_count=loaded.metadata.totals.call_count,
        expected_correct_count=loaded.metadata.totals.correct_count,
        expected_accuracy=loaded.metadata.totals.accuracy,
    )
    issues.extend(_duplicate_item_issues(predictions=loaded.predictions))
    calls_by_id: dict[str, CallRecord] = {call.call_id: call for call in loaded.calls}
    for prediction in loaded.predictions:
        issues.extend(_candidate_issues(prediction=prediction))
        issues.extend(_vote_reference_issues(prediction=prediction, calls_by_id=calls_by_id))
    if dataset is not None:
        issues.extend(_dataset_issues(predictions=loaded.predictions, dataset=dataset))
    if dataset is not None and prompt is not None:
        issues.extend(_prompt_rendering_issues(calls=loaded.calls, dataset=dataset, prompt=prompt))
    return RunValidationReport(run_dir=run_dir, issues=issues)

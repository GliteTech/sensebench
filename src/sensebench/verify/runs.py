"""Validate SenseBench run directories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetBundle, DatasetIndex, ItemID, SenseKey
from sensebench.paths import CALLS_FILENAME, PREDICTIONS_FILENAME, RUN_METADATA_FILENAME
from sensebench.prompts.models import OutputMode, PromptDefinition, PromptID
from sensebench.prompts.registry import registered_prompt_paths
from sensebench.prompts.render import ChatMessage, RenderedTask, render_task
from sensebench.runner.evaluate import choose_prediction, prediction_is_correct
from sensebench.runner.extract import ValidSenseIndexExtraction, extract_sense_index
from sensebench.runs.loaders import load_run_directory
from sensebench.runs.models import (
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    InvalidOutputReason,
    MessageRecord,
    PredictionRecord,
    PredictionStatus,
    RunID,
    RunMetadata,
    RunPolicy,
    VoteRecord,
    VoteStatus,
)
from sensebench.wordnet import SenseCandidate, SynsetID, get_candidate_senses, wordnet_version

ACCURACY_TOLERANCE: float = 1e-12
MESSAGE_ROLE_FIELD: str = "role"
MESSAGE_CONTENT_FIELD: str = "content"
DATASET_ITEM_DIFF_SAMPLE_LIMIT: int = 10
CALLS_AFTER_SUCCESS_MESSAGE: str = "calls recorded after a successful extraction"


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
    DUPLICATE_CALL = "duplicate_call"
    ORPHAN_CALL = "orphan_call"
    PREDICTION_CONSISTENCY = "prediction_consistency"
    PREDICTION_DECISION = "prediction_decision"
    VOTE_EXTRACTION = "vote_extraction"
    CORRECTNESS = "correctness"
    PROMPT_REFERENCE = "prompt_reference"
    DATASET_ITEMS = "dataset_items"
    DATASET_GOLD_KEYS = "dataset_gold_keys"
    DATASET_CONTENT_HASH = "dataset_content_hash"
    CANDIDATE_SET = "candidate_set"
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
    calls_by_id: dict[CallID, CallRecord],
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


def _call_set_issues(
    *,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    seen_call_ids: set[CallID] = set()
    for call in calls:
        if call.call_id in seen_call_ids:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.DUPLICATE_CALL,
                    location=call.call_id,
                    message="duplicate call_id",
                )
            )
        seen_call_ids.add(call.call_id)
    referenced_call_ids: set[CallID] = {
        call_id
        for prediction in predictions
        for vote in prediction.votes
        for call_id in vote.call_ids
    }
    for call in calls:
        if call.call_id not in referenced_call_ids:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.ORPHAN_CALL,
                    location=call.call_id,
                    message="call is not referenced by any vote",
                )
            )
    return issues


def _basic_issues(
    *,
    run_dir: Path,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
    loaded_run_id: RunID,
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
    seen: set[ItemID] = set()
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
    expected: set[ItemID] = {item.item_id for item in dataset.items}
    observed: set[ItemID] = {prediction.item_id for prediction in predictions}
    if expected == observed:
        return []
    missing: list[ItemID] = sorted(expected - observed)
    extra: list[ItemID] = sorted(observed - expected)
    return [
        RunValidationIssue(
            rule=RunValidationRule.DATASET_ITEMS,
            location=PREDICTIONS_FILENAME,
            message=(
                f"missing={missing[:DATASET_ITEM_DIFF_SAMPLE_LIMIT]} "
                f"extra={extra[:DATASET_ITEM_DIFF_SAMPLE_LIMIT]}"
            ),
        )
    ]


def _sense_keys_by_index(*, prediction: PredictionRecord) -> dict[int, SenseKey]:
    return {candidate.index: candidate.sense_key for candidate in prediction.candidates}


def _prediction_consistency_issues(*, prediction: PredictionRecord) -> list[RunValidationIssue]:
    messages: list[str] = []
    sense_keys_by_index = _sense_keys_by_index(prediction=prediction)

    if prediction.predicted_sense_index is None:
        if prediction.predicted_sense_key is not None:
            messages.append("predicted_sense_key must be null when predicted_sense_index is null")
    else:
        expected_key = sense_keys_by_index.get(prediction.predicted_sense_index)
        if prediction.predicted_sense_key != expected_key:
            messages.append(
                f"predicted_sense_key {prediction.predicted_sense_key} does not match "
                f"candidate {prediction.predicted_sense_index}"
            )

    if prediction.status == PredictionStatus.MONOSEMOUS:
        if not prediction.was_monosemous:
            messages.append("monosemous prediction must set was_monosemous")
        if len(prediction.candidates) != 1:
            messages.append("monosemous prediction must record exactly one candidate")
        elif prediction.predicted_sense_index != prediction.candidates[0].index:
            messages.append("monosemous prediction must predict the only candidate")
        if len(prediction.votes) != 0:
            messages.append("monosemous prediction must not record votes")
    elif prediction.status == PredictionStatus.NO_CANDIDATES:
        if prediction.was_monosemous:
            messages.append("no_candidates prediction must not set was_monosemous")
        if len(prediction.candidates) != 0:
            messages.append("no_candidates prediction must not record candidates")
        if len(prediction.votes) != 0:
            messages.append("no_candidates prediction must not record votes")
        if prediction.predicted_sense_index is not None:
            messages.append("no_candidates prediction must not record a predicted sense")
        if prediction.is_correct is not None:
            messages.append("no_candidates prediction must record is_correct as null")
    else:
        if prediction.was_monosemous:
            messages.append("voted prediction must not set was_monosemous")
        if len(prediction.candidates) < 2:
            messages.append("voted prediction must record at least two candidates")
        is_success = prediction.status == PredictionStatus.SUCCESS
        if is_success and prediction.predicted_sense_key is None:
            messages.append("success prediction must record a predicted sense")
        if not is_success and prediction.predicted_sense_key is not None:
            messages.append("no_valid_vote prediction must not record a predicted sense")

    for vote in prediction.votes:
        if vote.status == VoteStatus.SUCCESS:
            if vote.chosen_sense_index is None:
                messages.append(
                    f"vote {vote.vote_index}: successful vote must record chosen_sense_index"
                )
            elif vote.chosen_sense_key != sense_keys_by_index.get(vote.chosen_sense_index):
                messages.append(
                    f"vote {vote.vote_index}: chosen_sense_key does not match "
                    f"candidate {vote.chosen_sense_index}"
                )
        elif vote.chosen_sense_index is not None or vote.chosen_sense_key is not None:
            messages.append(f"vote {vote.vote_index}: failed vote must not record a chosen sense")

    return [
        RunValidationIssue(
            rule=RunValidationRule.PREDICTION_CONSISTENCY,
            location=prediction.item_id,
            message=message,
        )
        for message in messages
    ]


def _decision_issues(
    *,
    prediction: PredictionRecord,
    policy: RunPolicy,
) -> list[RunValidationIssue]:
    if prediction.status in (PredictionStatus.MONOSEMOUS, PredictionStatus.NO_CANDIDATES):
        return []
    messages: list[str] = []
    expected_vote_indexes: list[int] = list(range(1, policy.votes_per_item + 1))
    observed_vote_indexes: list[int] = [vote.vote_index for vote in prediction.votes]
    if observed_vote_indexes != expected_vote_indexes:
        messages.append(
            f"vote indexes {observed_vote_indexes} do not match "
            f"policy votes_per_item {policy.votes_per_item}"
        )
    replayed_index = choose_prediction(votes=prediction.votes)
    if prediction.predicted_sense_index != replayed_index:
        messages.append(
            f"predicted_sense_index {prediction.predicted_sense_index} does not match "
            f"replayed vote decision {replayed_index}"
        )
    return [
        RunValidationIssue(
            rule=RunValidationRule.PREDICTION_DECISION,
            location=prediction.item_id,
            message=message,
        )
        for message in messages
    ]


@dataclass(frozen=True, slots=True)
class _ReplayedVote:
    status: VoteStatus
    chosen_sense_index: int | None
    chosen_sense_key: SenseKey | None
    invalid_reason: InvalidOutputReason | str | None


@dataclass(frozen=True, slots=True)
class _VoteReplayResult:
    replayed_vote: _ReplayedVote | None
    issues: list[RunValidationIssue]


@dataclass(frozen=True, slots=True)
class _CandidateFingerprint:
    index: int
    sense_key: SenseKey
    synset_id: SynsetID


def _stored_vote_fields(*, vote: VoteRecord) -> _ReplayedVote:
    return _ReplayedVote(
        status=vote.status,
        chosen_sense_index=vote.chosen_sense_index,
        chosen_sense_key=vote.chosen_sense_key,
        invalid_reason=vote.invalid_reason,
    )


def _replay_vote(
    *,
    calls: list[CallRecord],
    output_mode: OutputMode,
    candidate_count: int,
    sense_keys_by_index: dict[int, SenseKey],
    max_attempts: int,
    location: str,
) -> _VoteReplayResult:
    issues: list[RunValidationIssue] = []

    def _issue(message: str) -> RunValidationIssue:
        return RunValidationIssue(
            rule=RunValidationRule.VOTE_EXTRACTION,
            location=location,
            message=message,
        )

    successful_vote: _ReplayedVote | None = None
    invalid_reason: InvalidOutputReason | str | None = None
    for call_position, call in enumerate(calls):
        is_last_call = call_position == len(calls) - 1
        if call.status == CallStatus.TRANSPORT_ERROR:
            if successful_vote is not None:
                issues.append(_issue(CALLS_AFTER_SUCCESS_MESSAGE))
                return _VoteReplayResult(replayed_vote=successful_vote, issues=issues)
            if not is_last_call:
                issues.append(_issue("calls recorded after a transport error"))
            return _VoteReplayResult(
                replayed_vote=_ReplayedVote(
                    status=VoteStatus.TRANSPORT_ERROR,
                    chosen_sense_index=None,
                    chosen_sense_key=None,
                    invalid_reason=call.error_kind,
                ),
                issues=issues,
            )
        try:
            extracted = extract_sense_index(
                text=call.raw_output,
                output_mode=output_mode,
                candidate_count=candidate_count,
            )
        except Exception as exc:
            issues.append(_issue(f"extraction replay raised {type(exc).__name__}: {exc}"))
            return _VoteReplayResult(replayed_vote=None, issues=issues)
        if isinstance(extracted, ValidSenseIndexExtraction):
            replayed_success = _ReplayedVote(
                status=VoteStatus.SUCCESS,
                chosen_sense_index=extracted.sense_index,
                chosen_sense_key=sense_keys_by_index.get(extracted.sense_index),
                invalid_reason=None,
            )
            if successful_vote is None:
                successful_vote = replayed_success
                continue
            if replayed_success != successful_vote:
                issues.append(_issue(CALLS_AFTER_SUCCESS_MESSAGE))
                return _VoteReplayResult(replayed_vote=successful_vote, issues=issues)
            continue
        if successful_vote is not None:
            issues.append(_issue(CALLS_AFTER_SUCCESS_MESSAGE))
            return _VoteReplayResult(replayed_vote=successful_vote, issues=issues)
        invalid_reason = extracted.invalid_reason
    if successful_vote is not None:
        return _VoteReplayResult(replayed_vote=successful_vote, issues=issues)
    if len(calls) != max_attempts:
        issues.append(
            _issue(f"invalid-output vote must record {max_attempts} attempt(s), found {len(calls)}")
        )
    return _VoteReplayResult(
        replayed_vote=_ReplayedVote(
            status=VoteStatus.INVALID_OUTPUT,
            chosen_sense_index=None,
            chosen_sense_key=None,
            invalid_reason=invalid_reason,
        ),
        issues=issues,
    )


def _vote_calls(
    *,
    vote: VoteRecord,
    calls_by_id: dict[CallID, CallRecord],
) -> list[CallRecord] | None:
    calls: list[CallRecord] = []
    for call_id in vote.call_ids:
        call = calls_by_id.get(call_id)
        if call is None:
            return None
        calls.append(call)
    return calls


def _call_structure_issues(
    *,
    prediction: PredictionRecord,
    vote: VoteRecord,
    calls: list[CallRecord],
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    for call_position, call in enumerate(calls):
        messages: list[str] = []
        expected_kind = AttemptKind.INITIAL if call_position == 0 else AttemptKind.SEMANTIC_REASK
        if call.attempt_kind != expected_kind:
            messages.append(f"attempt_kind must be {expected_kind.value}")
        if call.attempt_index != call_position + 1:
            messages.append(f"attempt_index must be {call_position + 1}")
        if call.item_id != prediction.item_id:
            messages.append("call item_id does not match prediction")
        if call.vote_index != vote.vote_index:
            messages.append("call vote_index does not match vote")
        issues.extend(
            RunValidationIssue(
                rule=RunValidationRule.VOTE_EXTRACTION,
                location=call.call_id,
                message=message,
            )
            for message in messages
        )
    return issues


def _vote_extraction_issues(
    *,
    prediction: PredictionRecord,
    calls_by_id: dict[CallID, CallRecord],
    output_mode: OutputMode,
    policy: RunPolicy,
) -> list[RunValidationIssue]:
    if prediction.status in (PredictionStatus.MONOSEMOUS, PredictionStatus.NO_CANDIDATES):
        return []
    issues: list[RunValidationIssue] = []
    sense_keys_by_index = _sense_keys_by_index(prediction=prediction)
    max_attempts = policy.semantic_reasks_per_invalid_vote + 1
    for vote in prediction.votes:
        location = f"{prediction.item_id}:{vote.vote_index}"
        calls = _vote_calls(vote=vote, calls_by_id=calls_by_id)
        if calls is None:
            continue  # unknown call ids are reported by the vote_call_reference rule
        if len(calls) == 0:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.VOTE_EXTRACTION,
                    location=location,
                    message="vote records no calls",
                )
            )
            continue
        if len(calls) > max_attempts:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.VOTE_EXTRACTION,
                    location=location,
                    message=f"vote records {len(calls)} calls, "
                    f"policy allows at most {max_attempts}",
                )
            )
        issues.extend(_call_structure_issues(prediction=prediction, vote=vote, calls=calls))
        replay_result = _replay_vote(
            calls=calls,
            output_mode=output_mode,
            candidate_count=len(prediction.candidates),
            sense_keys_by_index=sense_keys_by_index,
            max_attempts=max_attempts,
            location=location,
        )
        issues.extend(replay_result.issues)
        if replay_result.replayed_vote is None:
            continue
        replayed = replay_result.replayed_vote
        stored = _stored_vote_fields(vote=vote)
        if stored != replayed:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.VOTE_EXTRACTION,
                    location=location,
                    message=(
                        f"stored vote (status={stored.status.value}, "
                        f"index={stored.chosen_sense_index}, key={stored.chosen_sense_key}, "
                        f"reason={stored.invalid_reason}) does not match extraction replay "
                        f"(status={replayed.status.value}, index={replayed.chosen_sense_index}, "
                        f"key={replayed.chosen_sense_key}, reason={replayed.invalid_reason})"
                    ),
                )
            )
    return issues


def _correctness_issues(*, predictions: list[PredictionRecord]) -> list[RunValidationIssue]:
    needs_wordnet = any(
        prediction.predicted_sense_key is not None
        and prediction.predicted_sense_key not in prediction.gold_sense_keys
        for prediction in predictions
    )
    if needs_wordnet:
        try:
            wordnet_version()
        except Exception as exc:
            return [
                RunValidationIssue(
                    rule=RunValidationRule.CORRECTNESS,
                    location=PREDICTIONS_FILENAME,
                    message=f"cannot recompute is_correct: {exc}",
                )
            ]
    issues: list[RunValidationIssue] = []
    for prediction in predictions:
        recomputed = prediction_is_correct(
            predicted_sense_key=prediction.predicted_sense_key,
            gold_sense_keys=prediction.gold_sense_keys,
        )
        if prediction.is_correct != recomputed:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.CORRECTNESS,
                    location=prediction.item_id,
                    message=f"is_correct {prediction.is_correct} does not match "
                    f"recomputed {recomputed}",
                )
            )
    return issues


def _gold_key_issues(
    *,
    predictions: list[PredictionRecord],
    dataset: DatasetBundle,
) -> list[RunValidationIssue]:
    gold_by_item: dict[ItemID, list[SenseKey]] = {
        item.item_id: item.gold_sense_keys for item in dataset.items
    }
    issues: list[RunValidationIssue] = []
    for prediction in predictions:
        expected = gold_by_item.get(prediction.item_id)
        if expected is None:
            continue  # unknown items are reported by the dataset_items rule
        if prediction.gold_sense_keys != expected:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.DATASET_GOLD_KEYS,
                    location=prediction.item_id,
                    message=f"gold_sense_keys {prediction.gold_sense_keys} "
                    f"do not match dataset {expected}",
                )
            )
    return issues


def _content_hash_issues(
    *,
    metadata: RunMetadata,
    dataset: DatasetBundle,
) -> list[RunValidationIssue]:
    recorded = metadata.dataset.content_hash
    expected = dataset.content_hash
    if recorded is None or expected is None or recorded == expected:
        return []
    return [
        RunValidationIssue(
            rule=RunValidationRule.DATASET_CONTENT_HASH,
            location=RUN_METADATA_FILENAME,
            message=f"recorded dataset content_hash {recorded} "
            f"does not match verification dataset {expected}",
        )
    ]


def _registered_prompt_ids() -> set[PromptID]:
    return {path.stem for path in registered_prompt_paths()}


def _prompt_reference_issues(
    *,
    metadata: RunMetadata,
    prompt: PromptDefinition | None,
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    if metadata.prompt.id not in _registered_prompt_ids():
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.PROMPT_REFERENCE,
                location=RUN_METADATA_FILENAME,
                message=f"prompt {metadata.prompt.id} is not in the registered prompt registry",
            )
        )
    if prompt is not None and prompt.id != metadata.prompt.id:
        issues.append(
            RunValidationIssue(
                rule=RunValidationRule.PROMPT_REFERENCE,
                location=RUN_METADATA_FILENAME,
                message=f"verification prompt {prompt.id} "
                f"does not match run prompt {metadata.prompt.id}",
            )
        )
    return issues


def _messages_payload(*, messages: Sequence[ChatMessage | MessageRecord]) -> list[dict[str, str]]:
    return [
        {
            MESSAGE_ROLE_FIELD: message.role.value,
            MESSAGE_CONTENT_FIELD: message.content,
        }
        for message in messages
    ]


class _RenderCache:
    def __init__(self, *, dataset_index: DatasetIndex, prompt: PromptDefinition) -> None:
        self._dataset_index = dataset_index
        self._prompt = prompt
        self._rendered_by_item: dict[ItemID, RenderedTask | None] = {}

    def rendered(self, *, item_id: ItemID) -> RenderedTask | None:
        if item_id not in self._rendered_by_item:
            item = self._dataset_index.items_by_id.get(item_id)
            if item is None:
                self._rendered_by_item[item_id] = None
            else:
                candidates: list[SenseCandidate] = get_candidate_senses(
                    lemma=item.lemma,
                    pos=item.pos,
                )
                self._rendered_by_item[item_id] = render_task(
                    prompt=self._prompt,
                    item=item,
                    dataset_index=self._dataset_index,
                    candidates=candidates,
                )
        return self._rendered_by_item[item_id]


def _candidate_set_issues(
    *,
    predictions: list[PredictionRecord],
    render_cache: _RenderCache,
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    for prediction in predictions:
        rendered = render_cache.rendered(item_id=prediction.item_id)
        if rendered is None:
            continue  # items missing from the dataset are reported by the dataset_items rule
        expected: list[_CandidateFingerprint] = [
            _CandidateFingerprint(
                index=candidate.index,
                sense_key=candidate.sense_key,
                synset_id=candidate.synset_id,
            )
            for candidate in rendered.candidates
        ]
        observed: list[_CandidateFingerprint] = [
            _CandidateFingerprint(
                index=candidate.index,
                sense_key=candidate.sense_key,
                synset_id=candidate.synset_id,
            )
            for candidate in prediction.candidates
        ]
        if observed != expected:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.CANDIDATE_SET,
                    location=prediction.item_id,
                    message="stored candidates differ from deterministic re-derivation",
                )
            )
    return issues


def _prompt_rendering_issues(
    *,
    calls: list[CallRecord],
    render_cache: _RenderCache,
) -> list[RunValidationIssue]:
    issues: list[RunValidationIssue] = []
    for call in calls:
        if call.attempt_kind != AttemptKind.INITIAL:
            continue
        rendered = render_cache.rendered(item_id=call.item_id)
        if rendered is None:
            issues.append(
                RunValidationIssue(
                    rule=RunValidationRule.PROMPT_RENDERING,
                    location=call.call_id,
                    message=f"call references item {call.item_id} missing from the dataset",
                )
            )
            continue
        expected = _messages_payload(messages=rendered.messages)
        if _messages_payload(messages=call.messages) != expected:
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

    issues: list[RunValidationIssue] = _basic_issues(
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
    issues.extend(_call_set_issues(predictions=loaded.predictions, calls=loaded.calls))
    issues.extend(_prompt_reference_issues(metadata=loaded.metadata, prompt=prompt))
    verification_prompt: PromptDefinition | None = None
    if prompt is not None and prompt.id == loaded.metadata.prompt.id:
        verification_prompt = prompt
    calls_by_id: dict[CallID, CallRecord] = {call.call_id: call for call in loaded.calls}
    for prediction in loaded.predictions:
        issues.extend(_candidate_issues(prediction=prediction))
        issues.extend(_vote_reference_issues(prediction=prediction, calls_by_id=calls_by_id))
        issues.extend(_prediction_consistency_issues(prediction=prediction))
        issues.extend(_decision_issues(prediction=prediction, policy=loaded.metadata.policy))
        if verification_prompt is not None:
            issues.extend(
                _vote_extraction_issues(
                    prediction=prediction,
                    calls_by_id=calls_by_id,
                    output_mode=verification_prompt.output.mode,
                    policy=loaded.metadata.policy,
                )
            )
    issues.extend(_correctness_issues(predictions=loaded.predictions))
    if dataset is not None:
        issues.extend(_dataset_issues(predictions=loaded.predictions, dataset=dataset))
        issues.extend(_gold_key_issues(predictions=loaded.predictions, dataset=dataset))
        issues.extend(_content_hash_issues(metadata=loaded.metadata, dataset=dataset))
    if dataset is not None and verification_prompt is not None:
        render_cache = _RenderCache(
            dataset_index=build_dataset_index(bundle=dataset),
            prompt=verification_prompt,
        )
        issues.extend(
            _candidate_set_issues(predictions=loaded.predictions, render_cache=render_cache)
        )
        issues.extend(_prompt_rendering_issues(calls=loaded.calls, render_cache=render_cache))
    return RunValidationReport(run_dir=run_dir, issues=issues)

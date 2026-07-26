"""Reference WSD baselines scored on registered dataset items at build time."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from sensebench.datasets.models import (
    DatasetBundle,
    DatasetPos,
    ItemID,
    LemmaText,
    SenseKey,
    WsdItem,
)
from sensebench.leaderboard.aggregate import AccuracyInterval, SchemeScore, bootstrap_accuracy_ci
from sensebench.leaderboard.schemes import (
    SCHEMES,
    ConceptMap,
    is_scoreable,
    load_concept_map,
    scheme_correct,
)
from sensebench.paths import (
    BEM_BASELINE_PATH,
    CONSEC_BASELINE_PATH,
    ESCHER_BASELINE_PATH,
    GLITE_LENS_BASELINE_PATH,
)
from sensebench.runner.evaluate import prediction_is_correct
from sensebench.wordnet import get_candidate_senses, wordnet_version

KEY_FILE_FIELD_COUNT: int = 2
MFS_BASELINE_LABEL: str = "MFS (WordNet first sense)"
MFS_SOURCE_NOTE: str = (
    "Most frequent sense baseline: WordNet 3.0's first (frequency-ranked) sense for the "
    "target lemma and part of speech, computed directly on the dataset items."
)
MARU_PREDICTIONS_URL: str = "https://github.com/SapienzaNLP/wsd-hard-benchmark"


class BaselineKind(StrEnum):
    COMPUTED_WORDNET_MFS = "computed_wordnet_mfs"
    PUBLISHED_PREDICTIONS = "published_predictions"
    REPRODUCED_PREDICTIONS = "reproduced_predictions"


class BaselineModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Baseline(BaselineModel):
    label: str
    kind: BaselineKind
    accuracy: float
    accuracy_ci: AccuracyInterval
    correct_count: int
    item_count: int
    scheme_scores: dict[str, SchemeScore]
    dataset_version: str | None
    source_note: str
    source_url: str | None


@dataclass(frozen=True, slots=True)
class BaselinePredictionSpec:
    label: str
    path: Path
    kind: BaselineKind
    source_note: str
    source_url: str | None


class FirstSenseCacheKey(NamedTuple):
    lemma: LemmaText
    pos: DatasetPos


BASELINE_PREDICTION_SPECS: tuple[BaselinePredictionSpec, ...] = (
    BaselinePredictionSpec(
        label="BEM",
        path=BEM_BASELINE_PATH,
        kind=BaselineKind.PUBLISHED_PREDICTIONS,
        source_note=(
            "Bi-Encoder Model (Blevins & Zettlemoyer 2020); per-item predictions released by "
            "Maru et al. 2022, scored on this dataset's items."
        ),
        source_url=MARU_PREDICTIONS_URL,
    ),
    BaselinePredictionSpec(
        label="ESCHER",
        path=ESCHER_BASELINE_PATH,
        kind=BaselineKind.REPRODUCED_PREDICTIONS,
        source_note=(
            "ESCHER (Barba et al. 2021; SemCor training); predictions reproduced by Glite, "
            "79.6 F1 on Raganato ALL (-1.1 of the published 80.7 F1), scored on this "
            "dataset's items."
        ),
        source_url="https://github.com/SapienzaNLP/esc",
    ),
    BaselinePredictionSpec(
        label="ConSeC",
        path=CONSEC_BASELINE_PATH,
        kind=BaselineKind.REPRODUCED_PREDICTIONS,
        source_note=(
            "ConSeC (Barba et al. 2021); predictions reproduced by Glite (SemCor + WordNet "
            "Gloss+Examples training, 82.9 F1 on Raganato ALL, -0.3 of the published "
            "83.2 F1), scored on this dataset's items."
        ),
        source_url="https://github.com/SapienzaNLP/consec",
    ),
    BaselinePredictionSpec(
        label="Glite LENS",
        path=GLITE_LENS_BASELINE_PATH,
        kind=BaselineKind.PUBLISHED_PREDICTIONS,
        source_note=(
            "Glite LENS (ModernBERT bi-encoder); shipped seed-42 predictions trained on "
            "SemCor-GPT5.5, the GPT-5.5-relabeled corpus rather than original SemCor "
            "(83.7 F1 on Raganato ALL; 3-seed mean 83.6). This row demonstrates the "
            "relabel-and-retrain result, so the LENS-ESCHER margin is not an architecture-only "
            "comparison; because its training labels share a model family with the lexEN "
            "triage, its lexEN score is confirmatory under the paper's Section 6.4 rule."
        ),
        source_url="https://github.com/GliteTech/research-semcor-relabeling#glite-lens",
    ),
)


@dataclass(frozen=True, slots=True)
class BaselineScore:
    correctness: list[bool]


def _load_key_file(*, path: Path) -> dict[ItemID, SenseKey]:
    predictions: dict[ItemID, SenseKey] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if len(line) == 0:
            continue
        fields: list[str] = line.split()
        if len(fields) != KEY_FILE_FIELD_COUNT:
            raise ValueError(f"{path}:{line_number}: expected 'instance_id sense_key'")
        instance_id, sense_key = fields
        if instance_id in predictions:
            raise ValueError(f"{path}:{line_number}: duplicate instance_id {instance_id}")
        predictions[instance_id] = sense_key
    return predictions


def _item_correct(*, item: WsdItem, predicted_sense_key: SenseKey | None) -> bool:
    return (
        prediction_is_correct(
            predicted_sense_key=predicted_sense_key,
            gold_sense_keys=item.gold_sense_keys,
        )
        is True
    )


def _score(
    *,
    dataset: DatasetBundle,
    predictions: dict[ItemID, SenseKey],
    label: str,
) -> BaselineScore | None:
    missing: list[ItemID] = [
        item.item_id for item in dataset.items if item.item_id not in predictions
    ]
    if len(missing) == len(dataset.items):
        return None
    if len(missing) > 0:
        raise ValueError(
            f"baseline {label} is missing predictions for {len(missing)} dataset items "
            f"(first: {missing[0]})"
        )
    correctness: list[bool] = [
        _item_correct(item=item, predicted_sense_key=predictions[item.item_id])
        for item in dataset.items
    ]
    return BaselineScore(correctness=correctness)


def _mfs_predictions(*, dataset: DatasetBundle) -> dict[ItemID, SenseKey]:
    first_sense_cache: dict[FirstSenseCacheKey, SenseKey | None] = {}
    predictions: dict[ItemID, SenseKey] = {}
    for item in dataset.items:
        cache_key = FirstSenseCacheKey(lemma=item.lemma, pos=item.pos)
        if cache_key not in first_sense_cache:
            candidates = get_candidate_senses(lemma=item.lemma, pos=item.pos)
            first_sense_cache[cache_key] = candidates[0].sense_key if len(candidates) > 0 else None
        first_sense = first_sense_cache[cache_key]
        if first_sense is not None:
            predictions[item.item_id] = first_sense
    return predictions


def _scheme_score(*, correctness: list[bool]) -> SchemeScore:
    correct_count = sum(1 for value in correctness if value)
    item_count = len(correctness)
    return SchemeScore(
        accuracy=correct_count / item_count if item_count > 0 else None,
        accuracy_ci=bootstrap_accuracy_ci(values=correctness),
        correct_count=correct_count,
        item_count=item_count,
    )


def _scheme_scores_for_predictions(
    *,
    dataset: DatasetBundle,
    predictions: dict[ItemID, SenseKey],
    concept_map: ConceptMap,
) -> dict[str, SchemeScore]:
    per_scheme: dict[str, list[bool]] = {scheme.scheme_id: [] for scheme in SCHEMES}
    for item in dataset.items:
        predicted = predictions.get(item.item_id)
        for scheme in SCHEMES:
            if not is_scoreable(item=item, gold_source=scheme.gold_source):
                continue
            per_scheme[scheme.scheme_id].append(
                scheme_correct(
                    scheme=scheme,
                    predicted_sense_key=predicted,
                    item=item,
                    concept_map=concept_map,
                )
            )
    return {
        scheme_id: _scheme_score(correctness=correctness)
        for scheme_id, correctness in per_scheme.items()
    }


def _baseline(
    *,
    label: str,
    kind: BaselineKind,
    correctness: list[bool],
    scheme_scores: dict[str, SchemeScore],
    dataset: DatasetBundle,
    source_note: str,
    source_url: str | None,
) -> Baseline:
    correct_count = sum(1 for value in correctness if value)
    item_count = len(correctness)
    assert item_count > 0, "baseline accuracy requires at least one item"
    return Baseline(
        label=label,
        kind=kind,
        accuracy=correct_count / item_count,
        accuracy_ci=bootstrap_accuracy_ci(values=correctness),
        correct_count=correct_count,
        item_count=item_count,
        scheme_scores=scheme_scores,
        dataset_version=dataset.dataset_version,
        source_note=source_note,
        source_url=source_url,
    )


def _mfs_baseline(*, dataset: DatasetBundle, concept_map: ConceptMap) -> Baseline:
    predictions = _mfs_predictions(dataset=dataset)
    correctness: list[bool] = [
        _item_correct(item=item, predicted_sense_key=predictions.get(item.item_id))
        for item in dataset.items
    ]
    return _baseline(
        label=MFS_BASELINE_LABEL,
        kind=BaselineKind.COMPUTED_WORDNET_MFS,
        correctness=correctness,
        scheme_scores=_scheme_scores_for_predictions(
            dataset=dataset, predictions=predictions, concept_map=concept_map
        ),
        dataset=dataset,
        source_note=MFS_SOURCE_NOTE,
        source_url=None,
    )


def score_baselines(*, dataset: DatasetBundle) -> list[Baseline]:
    """Score reference baselines on the dataset.

    Prediction-file baselines that cover none of the dataset's items are skipped;
    partial coverage is an error.
    """
    if len(dataset.items) == 0:
        return []
    wordnet_version()
    concept_map = load_concept_map()
    baselines: list[Baseline] = [_mfs_baseline(dataset=dataset, concept_map=concept_map)]
    for spec in BASELINE_PREDICTION_SPECS:
        predictions = _load_key_file(path=spec.path)
        score = _score(dataset=dataset, predictions=predictions, label=spec.label)
        if score is None:
            continue
        baselines.append(
            _baseline(
                label=spec.label,
                kind=spec.kind,
                correctness=score.correctness,
                scheme_scores=_scheme_scores_for_predictions(
                    dataset=dataset, predictions=predictions, concept_map=concept_map
                ),
                dataset=dataset,
                source_note=spec.source_note,
                source_url=spec.source_url,
            )
        )
    return baselines

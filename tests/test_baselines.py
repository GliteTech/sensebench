from __future__ import annotations

from pathlib import Path

from pytest import raises

from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle, WsdItem
from sensebench.leaderboard.baselines import (
    BASELINE_PREDICTION_SPECS,
    MFS_BASELINE_LABEL,
    BaselineKind,
    score_baselines,
)
from sensebench.paths import DEFAULT_LEXEN_RELEASE_ID, LEXEN_DATASET_ID

SMOKE_ITEMS_PATH: Path = Path("tests/data/smoke_items.jsonl")
UNKNOWN_ITEM_ID: str = "i1"
UNKNOWN_GOLD_SENSE_KEY: str = "sense-1"
EXPECTED_BASELINE_COUNT: int = 1 + len(BASELINE_PREDICTION_SPECS)


def _smoke_dataset() -> DatasetBundle:
    return load_jsonl_dataset(
        path=SMOKE_ITEMS_PATH,
        dataset_id=LEXEN_DATASET_ID,
        dataset_version=DEFAULT_LEXEN_RELEASE_ID,
    )


def _unknown_item() -> WsdItem:
    return WsdItem(
        item_id=UNKNOWN_ITEM_ID,
        document_id="d1",
        sentence_id="s1",
        target_token_index=0,
        target_text="bank",
        lemma="bank",
        pos="NOUN",
        gold_sense_keys=[UNKNOWN_GOLD_SENSE_KEY],
    )


def test_score_baselines_covers_raganato_items() -> None:
    dataset = _smoke_dataset()

    baselines = score_baselines(dataset=dataset)

    assert len(baselines) == EXPECTED_BASELINE_COUNT
    labels = [baseline.label for baseline in baselines]
    assert labels[0] == MFS_BASELINE_LABEL
    assert {spec.label for spec in BASELINE_PREDICTION_SPECS} <= set(labels)
    mfs = baselines[0]
    assert mfs.kind == BaselineKind.COMPUTED_WORDNET_MFS
    # The smoke item's gold sense is not WordNet's first sense for art/NOUN.
    assert mfs.accuracy == 0.0
    for baseline in baselines:
        assert baseline.dataset_version == DEFAULT_LEXEN_RELEASE_ID
        assert baseline.item_count == len(dataset.items)
        assert baseline.accuracy_ci.low is not None
        assert baseline.accuracy_ci.high is not None


def test_score_baselines_skips_files_without_coverage() -> None:
    dataset = DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=None,
        documents=[],
        items=[_unknown_item()],
    )

    baselines = score_baselines(dataset=dataset)

    assert [baseline.label for baseline in baselines] == [MFS_BASELINE_LABEL]
    assert baselines[0].accuracy == 0.0


def test_score_baselines_rejects_partial_coverage() -> None:
    smoke = _smoke_dataset()
    dataset = DatasetBundle(
        dataset_id=smoke.dataset_id,
        dataset_version=smoke.dataset_version,
        dataset_revision=None,
        content_hash=None,
        documents=[],
        items=[*smoke.items, _unknown_item()],
    )

    with raises(ValueError, match="missing predictions"):
        score_baselines(dataset=dataset)


def test_score_baselines_empty_dataset_returns_nothing() -> None:
    dataset = DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=None,
        documents=[],
        items=[],
    )

    assert score_baselines(dataset=dataset) == []

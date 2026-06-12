from __future__ import annotations

import json
from pathlib import Path

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.loaders import (
    ORIGINAL_DOCUMENT_ID_METADATA_KEY,
    ORIGINAL_SENTENCE_ID_METADATA_KEY,
    load_jsonl_dataset,
)
from sensebench.datasets.models import DatasetID, DocumentID, ItemID, SenseKey, SentenceID

DATASET_ID_FIXTURE: DatasetID = "fixture"
ITEM_ID_FIELD: str = "item_id"
DOCUMENT_ID_FIELD: str = "document_id"
SENTENCE_ID_FIELD: str = "sentence_id"
SENTENCE_INDEX_FIELD: str = "sentence_index"
TARGET_TOKEN_INDEX_FIELD: str = "target_token_index"
TARGET_TEXT_FIELD: str = "target_text"
LEMMA_FIELD: str = "lemma"
POS_FIELD: str = "pos"
GOLD_SENSE_KEYS_FIELD: str = "gold_sense_keys"
SENTENCES_FIELD: str = "sentences"
METADATA_FIELD: str = "metadata"
DOCUMENT_ID: DocumentID = "d1"
FIRST_ITEM_ID: ItemID = "i1"
SECOND_ITEM_ID: ItemID = "i2"
FIRST_SENTENCE_ID: SentenceID = "d1.s1"
SECOND_SENTENCE_ID: SentenceID = "d1.s2"
FIRST_WORD: str = "alpha"
SECOND_WORD: str = "beta"
NOUN_POS: str = "NOUN"
FIRST_SENSE_KEY: SenseKey = "alpha%1:00:00::"
SECOND_SENSE_KEY: SenseKey = "beta%1:00:00::"


def _write_jsonl(*, path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _dataset_row(
    *,
    item_id: ItemID,
    sentence_id: SentenceID,
    word: str,
    sense_key: SenseKey,
) -> dict[str, object]:
    return {
        ITEM_ID_FIELD: item_id,
        DOCUMENT_ID_FIELD: DOCUMENT_ID,
        SENTENCE_ID_FIELD: sentence_id,
        SENTENCE_INDEX_FIELD: 0,
        TARGET_TOKEN_INDEX_FIELD: 0,
        TARGET_TEXT_FIELD: word,
        LEMMA_FIELD: word,
        POS_FIELD: NOUN_POS,
        GOLD_SENSE_KEYS_FIELD: [sense_key],
        SENTENCES_FIELD: [[word]],
        METADATA_FIELD: {},
    }


def test_jsonl_loader_keeps_repeated_original_document_ids_separate(tmp_path: Path) -> None:
    dataset_path = tmp_path / "items.jsonl"
    _write_jsonl(
        path=dataset_path,
        rows=[
            _dataset_row(
                item_id=FIRST_ITEM_ID,
                sentence_id=FIRST_SENTENCE_ID,
                word=FIRST_WORD,
                sense_key=FIRST_SENSE_KEY,
            ),
            _dataset_row(
                item_id=SECOND_ITEM_ID,
                sentence_id=SECOND_SENTENCE_ID,
                word=SECOND_WORD,
                sense_key=SECOND_SENSE_KEY,
            ),
        ],
    )

    bundle = load_jsonl_dataset(path=dataset_path, dataset_id=DATASET_ID_FIXTURE)
    index = build_dataset_index(bundle=bundle)

    assert len(index.documents_by_id) == 2
    assert bundle.items[0].document_id != bundle.items[1].document_id
    assert bundle.items[0].metadata[ORIGINAL_DOCUMENT_ID_METADATA_KEY] == DOCUMENT_ID
    assert bundle.items[0].metadata[ORIGINAL_SENTENCE_ID_METADATA_KEY] == FIRST_SENTENCE_ID
    first_window = build_context_window(
        index=index,
        item=bundle.items[0],
        previous_sentences=0,
        next_sentences=0,
    )
    second_window = build_context_window(
        index=index,
        item=bundle.items[1],
        previous_sentences=0,
        next_sentences=0,
    )

    assert first_window.text == FIRST_WORD
    assert second_window.text == SECOND_WORD

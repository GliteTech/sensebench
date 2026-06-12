from __future__ import annotations

from pathlib import Path

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.loaders import (
    ORIGINAL_DOCUMENT_ID_METADATA_KEY,
    ORIGINAL_SENTENCE_ID_METADATA_KEY,
    JsonDatasetRecord,
    load_jsonl_dataset,
)
from sensebench.datasets.models import DatasetID, DocumentID, ItemID, SenseKey, SentenceID

DATASET_ID_FIXTURE: DatasetID = "fixture"
ITEMS_JSONL_FILENAME: str = "items.jsonl"
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


def _write_jsonl(*, path: Path, rows: list[JsonDatasetRecord]) -> None:
    with path.open(mode="w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _dataset_row(
    *,
    item_id: ItemID,
    sentence_id: SentenceID,
    word: str,
    sense_key: SenseKey,
) -> JsonDatasetRecord:
    return JsonDatasetRecord(
        item_id=item_id,
        document_id=DOCUMENT_ID,
        sentence_id=sentence_id,
        sentence_index=0,
        target_token_index=0,
        target_text=word,
        lemma=word,
        pos=NOUN_POS,
        gold_sense_keys=[sense_key],
        sentences=[[word]],
    )


def test_jsonl_loader_keeps_repeated_original_document_ids_separate(tmp_path: Path) -> None:
    dataset_path = tmp_path / ITEMS_JSONL_FILENAME
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

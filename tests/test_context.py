from __future__ import annotations

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.models import (
    DatasetBundle,
    DatasetID,
    DatasetPos,
    Document,
    DocumentID,
    ItemID,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)

DATASET_ID: DatasetID = "fixture"
DATASET_VERSION: str = "1"
DOCUMENT_ID: DocumentID = "d1"
PREVIOUS_SENTENCE_ID: SentenceID = "s1"
TARGET_SENTENCE_ID: SentenceID = "s2"
NEXT_SENTENCE_ID: SentenceID = "s3"
ITEM_ID: ItemID = "i1"
TARGET_TEXT: str = "bank"
TARGET_LEMMA: str = "bank"
TARGET_POS: DatasetPos = DatasetPos.NOUN
GOLD_SENSE_KEY: SenseKey = "bank%1:14:00::"
PREVIOUS_SENTENCES: int = 1
NEXT_SENTENCES: int = 1
EXPECTED_CONTEXT_TEXT: str = "Before The bank closed After"
DETOKENIZED_TARGET_SENTENCE_ID: SentenceID = "s2d"
DETOKENIZED_ITEM_ID: ItemID = "i2"
DETOKENIZED_EXPECTED_TEXT: str = 'Before The "bank", closed After'


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id=DOCUMENT_ID,
                sentences=[
                    Sentence(sentence_id=PREVIOUS_SENTENCE_ID, tokens=[Token(text="Before")]),
                    Sentence(
                        sentence_id=TARGET_SENTENCE_ID,
                        tokens=[
                            Token(text="The"),
                            Token(text=TARGET_TEXT, item_id=ITEM_ID),
                            Token(text="closed"),
                        ],
                    ),
                    Sentence(sentence_id=NEXT_SENTENCE_ID, tokens=[Token(text="After")]),
                ],
            )
        ],
        items=[
            WsdItem(
                item_id=ITEM_ID,
                document_id=DOCUMENT_ID,
                sentence_id=TARGET_SENTENCE_ID,
                target_token_index=1,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=[GOLD_SENSE_KEY],
            )
        ],
    )


def test_context_window_uses_token_offset() -> None:
    bundle = _bundle()
    index = build_dataset_index(bundle=bundle)
    window = build_context_window(
        index=index,
        item=bundle.items[0],
        previous_sentences=PREVIOUS_SENTENCES,
        next_sentences=NEXT_SENTENCES,
    )

    assert window.text == EXPECTED_CONTEXT_TEXT
    assert window.text[window.target_start_char : window.target_end_char] == TARGET_TEXT
    assert window.sentences_before == PREVIOUS_SENTENCES
    assert window.sentences_after == NEXT_SENTENCES


def _detokenized_bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id=DOCUMENT_ID,
                sentences=[
                    Sentence(sentence_id=PREVIOUS_SENTENCE_ID, tokens=[Token(text="Before")]),
                    Sentence(
                        sentence_id=DETOKENIZED_TARGET_SENTENCE_ID,
                        tokens=[
                            Token(text="The"),
                            Token(text="``"),
                            Token(text=TARGET_TEXT, item_id=DETOKENIZED_ITEM_ID),
                            Token(text="''"),
                            Token(text=","),
                            Token(text="closed"),
                        ],
                    ),
                    Sentence(sentence_id=NEXT_SENTENCE_ID, tokens=[Token(text="After")]),
                ],
            )
        ],
        items=[
            WsdItem(
                item_id=DETOKENIZED_ITEM_ID,
                document_id=DOCUMENT_ID,
                sentence_id=DETOKENIZED_TARGET_SENTENCE_ID,
                target_token_index=2,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=[GOLD_SENSE_KEY],
            )
        ],
    )


def test_context_window_detokenizes_and_preserves_target_offset() -> None:
    bundle = _detokenized_bundle()
    index = build_dataset_index(bundle=bundle)
    window = build_context_window(
        index=index,
        item=bundle.items[0],
        previous_sentences=PREVIOUS_SENTENCES,
        next_sentences=NEXT_SENTENCES,
        detokenize=True,
    )

    assert window.text == DETOKENIZED_EXPECTED_TEXT
    assert window.text[window.target_start_char : window.target_end_char] == TARGET_TEXT

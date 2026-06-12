from __future__ import annotations

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.models import DatasetBundle, Document, Sentence, Token, WsdItem


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id="d1",
                sentences=[
                    Sentence(sentence_id="s1", tokens=[Token(text="Before")]),
                    Sentence(
                        sentence_id="s2",
                        tokens=[
                            Token(text="The"),
                            Token(text="bank", item_id="i1"),
                            Token(text="closed"),
                        ],
                    ),
                    Sentence(sentence_id="s3", tokens=[Token(text="After")]),
                ],
            )
        ],
        items=[
            WsdItem(
                item_id="i1",
                document_id="d1",
                sentence_id="s2",
                target_token_index=1,
                target_text="bank",
                lemma="bank",
                pos="NOUN",
                gold_sense_keys=["bank%1:14:00::"],
            )
        ],
    )


def test_context_window_uses_token_offset() -> None:
    bundle = _bundle()
    index = build_dataset_index(bundle=bundle)
    window = build_context_window(
        index=index,
        item=bundle.items[0],
        previous_sentences=1,
        next_sentences=1,
    )

    assert window.text == "Before The bank closed After"
    assert window.text[window.target_start_char : window.target_end_char] == "bank"
    assert window.sentences_before == 1
    assert window.sentences_after == 1

"""Deterministic document-relative context windows."""

from __future__ import annotations

from dataclasses import dataclass

from sensebench.datasets.detokenize import detokenize_pieces
from sensebench.datasets.models import (
    DatasetBundle,
    DatasetIndex,
    Document,
    DocumentID,
    Sentence,
    SentenceID,
    WsdItem,
)

SENTENCE_SEPARATOR: str = " "


@dataclass(frozen=True, slots=True)
class ContextWindow:
    text: str
    target_start_char: int
    target_end_char: int
    sentences_before: int
    sentences_after: int


@dataclass(frozen=True, slots=True)
class CharSpan:
    start: int
    end: int


def build_dataset_index(*, bundle: DatasetBundle) -> DatasetIndex:
    documents_by_id: dict[DocumentID, Document] = {
        document.document_id: document for document in bundle.documents
    }
    document_sentence_indexes: dict[DocumentID, dict[SentenceID, int]] = {}
    for document in bundle.documents:
        document_sentence_indexes[document.document_id] = {
            sentence.sentence_id: index for index, sentence in enumerate(document.sentences)
        }
    return DatasetIndex(
        documents_by_id=documents_by_id,
        document_sentence_indexes=document_sentence_indexes,
        items_by_id={item.item_id: item for item in bundle.items},
    )


def sentence_text(*, sentence: Sentence) -> str:
    return SENTENCE_SEPARATOR.join(token.text for token in sentence.tokens)


def _target_offsets(
    *,
    sentence: Sentence,
    target_token_index: int,
    target_text: str,
) -> CharSpan:
    if target_token_index < 0 or target_token_index >= len(sentence.tokens):
        raise ValueError(f"target_token_index out of range: {target_token_index}")
    current_start = 0
    for token_index, token in enumerate(sentence.tokens):
        token_start = current_start
        token_end = token_start + len(token.text)
        if token_index == target_token_index:
            if token.text != target_text:
                raise ValueError(
                    f"target token text mismatch: expected {target_text}, got {token.text}"
                )
            return CharSpan(start=token_start, end=token_end)
        current_start = token_end
        if token_index < len(sentence.tokens) - 1:
            current_start += len(SENTENCE_SEPARATOR)
    raise ValueError(f"target_token_index out of range: {target_token_index}")


def _joined_offsets(
    *,
    sentence_texts: list[str],
    local_target_sentence_index: int,
    target_start_in_sentence: int,
    target_end_in_sentence: int,
) -> CharSpan:
    offset = 0
    for sentence_index in range(local_target_sentence_index):
        offset += len(sentence_texts[sentence_index]) + len(SENTENCE_SEPARATOR)
    return CharSpan(start=offset + target_start_in_sentence, end=offset + target_end_in_sentence)


def _detokenized_window(
    *,
    selected: list[Sentence],
    local_target_sentence_index: int,
    target_token_index: int,
    target_text: str,
    item_id: str,
) -> ContextWindow:
    surfaces: list[str] = []
    target_flat_index = -1
    for sentence_index, sentence in enumerate(selected):
        for token_index, token in enumerate(sentence.tokens):
            if (
                sentence_index == local_target_sentence_index
                and token_index == target_token_index
            ):
                target_flat_index = len(surfaces)
            surfaces.append(token.text)
    if target_flat_index < 0:
        raise ValueError(f"target_token_index out of range: {target_token_index}")
    pieces = detokenize_pieces(surfaces=surfaces)
    text = ""
    target_start = -1
    target_end = -1
    for piece_index, piece in enumerate(pieces):
        if piece.leading_space:
            text += SENTENCE_SEPARATOR
        if piece_index == target_flat_index:
            target_start = len(text)
            target_end = target_start + len(piece.text)
        text += piece.text
    if text[target_start:target_end] != target_text:
        raise ValueError(f"context target offset mismatch for {item_id}")
    return ContextWindow(
        text=text,
        target_start_char=target_start,
        target_end_char=target_end,
        sentences_before=local_target_sentence_index,
        sentences_after=len(selected) - local_target_sentence_index - 1,
    )


def build_context_window(
    *,
    index: DatasetIndex,
    item: WsdItem,
    previous_sentences: int,
    next_sentences: int,
    detokenize: bool = False,
) -> ContextWindow:
    if previous_sentences < 0:
        raise ValueError("previous_sentences must be non-negative")
    if next_sentences < 0:
        raise ValueError("next_sentences must be non-negative")

    document = index.documents_by_id[item.document_id]
    sentence_index = index.document_sentence_indexes[item.document_id][item.sentence_id]
    target_sentence = document.sentences[sentence_index]
    target_span = _target_offsets(
        sentence=target_sentence,
        target_token_index=item.target_token_index,
        target_text=item.target_text,
    )

    first_sentence_index = max(0, sentence_index - previous_sentences)
    last_sentence_exclusive = min(len(document.sentences), sentence_index + next_sentences + 1)
    selected: list[Sentence] = document.sentences[first_sentence_index:last_sentence_exclusive]
    local_target_sentence_index = sentence_index - first_sentence_index
    if detokenize:
        return _detokenized_window(
            selected=selected,
            local_target_sentence_index=local_target_sentence_index,
            target_token_index=item.target_token_index,
            target_text=item.target_text,
            item_id=item.item_id,
        )
    selected_texts: list[str] = [sentence_text(sentence=sentence) for sentence in selected]
    context_text = SENTENCE_SEPARATOR.join(selected_texts)
    absolute_span = _joined_offsets(
        sentence_texts=selected_texts,
        local_target_sentence_index=local_target_sentence_index,
        target_start_in_sentence=target_span.start,
        target_end_in_sentence=target_span.end,
    )
    if context_text[absolute_span.start : absolute_span.end] != item.target_text:
        raise ValueError(f"context target offset mismatch for {item.item_id}")
    return ContextWindow(
        text=context_text,
        target_start_char=absolute_span.start,
        target_end_char=absolute_span.end,
        sentences_before=local_target_sentence_index,
        sentences_after=len(selected) - local_target_sentence_index - 1,
    )

"""Internal dataset dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

type DatasetID = str
type DocumentID = str
type SentenceID = str
type ItemID = str
type SenseKey = str


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    lemma: str | None = None
    pos: str | None = None
    item_id: ItemID | None = None


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: SentenceID
    tokens: list[Token]


@dataclass(frozen=True, slots=True)
class Document:
    document_id: DocumentID
    sentences: list[Sentence]


@dataclass(frozen=True, slots=True)
class WsdItem:
    item_id: ItemID
    document_id: DocumentID
    sentence_id: SentenceID
    target_token_index: int
    target_text: str
    lemma: str
    pos: str
    gold_sense_keys: list[SenseKey]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    dataset_id: DatasetID
    dataset_version: str | None
    dataset_revision: str | None
    content_hash: str | None
    documents: list[Document]
    items: list[WsdItem]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    documents_by_id: dict[DocumentID, Document]
    document_sentence_indexes: dict[DocumentID, dict[SentenceID, int]]
    items_by_id: dict[ItemID, WsdItem]

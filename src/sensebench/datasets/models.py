"""Internal dataset dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

type DatasetID = str
type DatasetVersion = str
type DatasetRevision = str
type ContentHash = str
type DocumentID = str
type SentenceID = str
type ItemID = str
type LemmaText = str
type SenseKey = str


class DatasetPos(StrEnum):
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    lemma: LemmaText | None = None
    pos: DatasetPos | None = None
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
    lemma: LemmaText
    pos: DatasetPos
    gold_sense_keys: list[SenseKey]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    dataset_id: DatasetID
    dataset_version: DatasetVersion | None
    dataset_revision: DatasetRevision | None
    content_hash: ContentHash | None
    documents: list[Document]
    items: list[WsdItem]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    documents_by_id: dict[DocumentID, Document]
    document_sentence_indexes: dict[DocumentID, dict[SentenceID, int]]
    items_by_id: dict[ItemID, WsdItem]

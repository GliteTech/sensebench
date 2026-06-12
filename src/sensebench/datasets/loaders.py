"""Dataset loaders for local JSONL files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sensebench.datasets.models import (
    DatasetBundle,
    DatasetID,
    Document,
    DocumentID,
    ItemID,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)

DEFAULT_DATASET_VERSION: str | None = None
HASH_CHUNK_SIZE_BYTES: int = 1024 * 1024
CONTEXT_DOCUMENT_ID_SUFFIX: str = "::context"
ORIGINAL_DOCUMENT_ID_METADATA_KEY: str = "original_document_id"
ORIGINAL_SENTENCE_ID_METADATA_KEY: str = "original_sentence_id"


class JsonDatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: ItemID
    document_id: DocumentID
    sentence_id: SentenceID | None = None
    sentence_index: int = Field(ge=0)
    target_token_index: int = Field(ge=0)
    target_text: str = Field(min_length=1)
    lemma: str = Field(min_length=1)
    pos: str = Field(min_length=1)
    gold_sense_keys: list[SenseKey] = Field(min_length=1)
    sentences: list[list[str]] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


def file_content_hash(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE_BYTES)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tokens_from_sentence(
    *,
    words: list[str],
    item_id: ItemID,
    target_index: int,
) -> list[Token]:
    tokens: list[Token] = []
    for token_index, word in enumerate(words):
        tokens.append(
            Token(
                text=word,
                item_id=item_id if token_index == target_index else None,
            )
        )
    return tokens


def _context_document_id(*, record: JsonDatasetRecord) -> DocumentID:
    return f"{record.item_id}{CONTEXT_DOCUMENT_ID_SUFFIX}"


def _item_metadata(*, record: JsonDatasetRecord) -> dict[str, str]:
    metadata: dict[str, str] = dict(record.metadata)
    metadata[ORIGINAL_DOCUMENT_ID_METADATA_KEY] = record.document_id
    if record.sentence_id is not None:
        metadata[ORIGINAL_SENTENCE_ID_METADATA_KEY] = record.sentence_id
    return metadata


def _bundle_from_records(
    *,
    records: list[JsonDatasetRecord],
    dataset_id: DatasetID,
    dataset_version: str | None,
    dataset_revision: str | None,
    content_hash: str | None,
) -> DatasetBundle:
    documents: list[Document] = []
    items: list[WsdItem] = []
    for record in records:
        context_document_id = _context_document_id(record=record)
        sentences: list[Sentence] = []
        for sentence_index, words in enumerate(record.sentences):
            sentence_id = (
                record.sentence_id
                if sentence_index == record.sentence_index and record.sentence_id is not None
                else f"{record.item_id}.s{sentence_index}"
            )
            tokens = _tokens_from_sentence(
                words=words,
                item_id=record.item_id,
                target_index=record.target_token_index
                if sentence_index == record.sentence_index
                else -1,
            )
            sentences.append(Sentence(sentence_id=sentence_id, tokens=tokens))
        item_sentence_id = (
            record.sentence_id
            if record.sentence_id is not None
            else f"{record.item_id}.s{record.sentence_index}"
        )
        documents.append(Document(document_id=context_document_id, sentences=sentences))
        items.append(
            WsdItem(
                item_id=record.item_id,
                document_id=context_document_id,
                sentence_id=item_sentence_id,
                target_token_index=record.target_token_index,
                target_text=record.target_text,
                lemma=record.lemma,
                pos=record.pos,
                gold_sense_keys=record.gold_sense_keys,
                metadata=_item_metadata(record=record),
            )
        )
    return DatasetBundle(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_revision=dataset_revision,
        content_hash=content_hash,
        documents=documents,
        items=items,
    )


def load_jsonl_dataset(
    *,
    path: Path,
    dataset_id: DatasetID,
    dataset_version: str | None = DEFAULT_DATASET_VERSION,
    dataset_revision: str | None = None,
) -> DatasetBundle:
    records: list[JsonDatasetRecord] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            records.append(JsonDatasetRecord.model_validate_json(line))
    return _bundle_from_records(
        records=records,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_revision=dataset_revision,
        content_hash=file_content_hash(path=path),
    )

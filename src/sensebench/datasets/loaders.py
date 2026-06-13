"""Dataset loaders for local JSONL files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sensebench.datasets.models import (
    ContentHash,
    DatasetBundle,
    DatasetID,
    DatasetPos,
    DatasetRevision,
    DatasetVersion,
    Document,
    DocumentID,
    ItemID,
    LemmaText,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)

DEFAULT_DATASET_VERSION: str | None = None
HASH_CHUNK_SIZE_BYTES: int = 1024 * 1024
CONTENT_HASH_PREFIX: str = "sha256:"
CONTEXT_DOCUMENT_ID_SUFFIX: str = "::context"
ORIGINAL_DOCUMENT_ID_METADATA_KEY: str = "original_document_id"
ORIGINAL_SENTENCE_ID_METADATA_KEY: str = "original_sentence_id"
GENERATED_SENTENCE_ID_SEPARATOR: str = ".s"


class JsonDatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: ItemID
    document_id: DocumentID
    sentence_id: SentenceID | None = None
    sentence_index: int = Field(ge=0)
    target_token_index: int = Field(ge=0)
    target_text: str = Field(min_length=1)
    lemma: LemmaText = Field(min_length=1)
    pos: DatasetPos
    gold_sense_keys: list[SenseKey] = Field(min_length=1)
    sentences: list[list[str]] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_indexes(self) -> JsonDatasetRecord:
        if self.sentence_index >= len(self.sentences):
            raise ValueError("sentence_index is within sentences")
        target_sentence = self.sentences[self.sentence_index]
        if self.target_token_index >= len(target_sentence):
            raise ValueError("target_token_index is within target sentence")
        return self


def file_content_hash(*, path: Path) -> ContentHash:
    digest = sha256()
    with path.open(mode="rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE_BYTES)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    return f"{CONTENT_HASH_PREFIX}{digest.hexdigest()}"


def _tokens_from_sentence(
    *,
    words: list[str],
    item_id: ItemID,
    target_index: int | None,
) -> list[Token]:
    tokens: list[Token] = []
    for token_index, word in enumerate(words):
        token_item_id = (
            item_id if target_index is not None and token_index == target_index else None
        )
        tokens.append(
            Token(
                text=word,
                item_id=token_item_id,
            )
        )
    return tokens


def _context_document_id(*, record: JsonDatasetRecord) -> DocumentID:
    return f"{record.item_id}{CONTEXT_DOCUMENT_ID_SUFFIX}"


def _generated_sentence_id(*, item_id: ItemID, sentence_index: int) -> SentenceID:
    return f"{item_id}{GENERATED_SENTENCE_ID_SEPARATOR}{sentence_index}"


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
    dataset_version: DatasetVersion | None,
    dataset_revision: DatasetRevision | None,
    content_hash: ContentHash | None,
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
                else _generated_sentence_id(
                    item_id=record.item_id,
                    sentence_index=sentence_index,
                )
            )
            target_index = (
                record.target_token_index if sentence_index == record.sentence_index else None
            )
            tokens = _tokens_from_sentence(
                words=words,
                item_id=record.item_id,
                target_index=target_index,
            )
            sentences.append(Sentence(sentence_id=sentence_id, tokens=tokens))
        item_sentence_id = (
            record.sentence_id
            if record.sentence_id is not None
            else _generated_sentence_id(
                item_id=record.item_id,
                sentence_index=record.sentence_index,
            )
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
    dataset_version: DatasetVersion | None = DEFAULT_DATASET_VERSION,
    dataset_revision: DatasetRevision | None = None,
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

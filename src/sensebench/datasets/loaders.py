"""Dataset loaders for local JSONL and Hugging Face datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sensebench.datasets.models import (
    DatasetBundle,
    Document,
    DocumentID,
    ItemID,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)

DEFAULT_HF_DATASET_NAME: str = "GliteTech/lexen"
DEFAULT_HF_SPLIT: str = "test"
DEFAULT_DATASET_VERSION: str | None = None
EMPTY_CONTENT_HASH: str = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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


def _content_hash(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
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


def _bundle_from_records(
    *,
    records: list[JsonDatasetRecord],
    dataset_id: str,
    dataset_version: str | None,
    dataset_revision: str | None,
    content_hash: str | None,
) -> DatasetBundle:
    documents: list[Document] = []
    items: list[WsdItem] = []
    for record in records:
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
        documents.append(Document(document_id=record.document_id, sentences=sentences))
        items.append(
            WsdItem(
                item_id=record.item_id,
                document_id=record.document_id,
                sentence_id=item_sentence_id,
                target_token_index=record.target_token_index,
                target_text=record.target_text,
                lemma=record.lemma,
                pos=record.pos,
                gold_sense_keys=record.gold_sense_keys,
                metadata=record.metadata,
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
    dataset_id: str,
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
        content_hash=_content_hash(path=path) if path.exists() else EMPTY_CONTENT_HASH,
    )


def _record_from_hf_row(*, row: dict[str, Any]) -> JsonDatasetRecord:
    return JsonDatasetRecord.model_validate(row)


def load_hf_dataset(
    *,
    dataset_name: str = DEFAULT_HF_DATASET_NAME,
    split: str = DEFAULT_HF_SPLIT,
    revision: str | None = None,
    dataset_version: str | None = DEFAULT_DATASET_VERSION,
) -> DatasetBundle:
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(dataset_name, split=split, revision=revision)
    records: list[JsonDatasetRecord] = [_record_from_hf_row(row=dict(row)) for row in loaded]
    return _bundle_from_records(
        records=records,
        dataset_id=dataset_name,
        dataset_version=dataset_version,
        dataset_revision=revision,
        content_hash=None,
    )

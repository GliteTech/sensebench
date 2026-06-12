"""Small NLTK WordNet wrapper used by SenseBench."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sensebench.datasets.models import SenseKey

EXPECTED_WORDNET_VERSION: str = "3.0"
WORDNET_CORPUS_ID: str = "wordnet"

type SynsetID = str
type LemmaText = str
type NormalizedLemma = str
type SynonymText = str


class WordNetPos(StrEnum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "a"
    ADVERB = "r"


PROJECT_POS_TO_WORDNET: dict[str, WordNetPos] = {
    "NOUN": WordNetPos.NOUN,
    "VERB": WordNetPos.VERB,
    "ADJ": WordNetPos.ADJECTIVE,
    "ADV": WordNetPos.ADVERB,
    "n": WordNetPos.NOUN,
    "v": WordNetPos.VERB,
    "a": WordNetPos.ADJECTIVE,
    "s": WordNetPos.ADJECTIVE,
    "r": WordNetPos.ADVERB,
}


@dataclass(frozen=True, slots=True)
class SenseCandidate:
    sense_key: SenseKey
    synset_id: SynsetID
    pos: WordNetPos
    definition: str
    synonyms: list[SynonymText]
    examples: list[str]


def _wordnet() -> Any:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install nltk to use SenseBench WordNet support.") from exc
    return wn


def _normalize_lemma(*, value: LemmaText | SynonymText) -> NormalizedLemma:
    return "".join(
        character for character in value.lower().replace("_", " ") if character.isalnum()
    )


def wordnet_pos(*, pos: str) -> WordNetPos | None:
    return PROJECT_POS_TO_WORDNET.get(pos)


def _synset_pos(*, synset: Any) -> WordNetPos:
    raw_pos = str(synset.pos())
    pos = wordnet_pos(pos=raw_pos)
    if pos is None:
        raise ValueError(f"unsupported WordNet POS: {raw_pos}")
    return pos


def _download_wordnet_corpus() -> None:
    import nltk  # type: ignore[import-untyped]

    print("Downloading the NLTK WordNet corpus (one-time setup)...", file=sys.stderr)
    nltk.download(info_or_id=WORDNET_CORPUS_ID, quiet=True)


def wordnet_version() -> str:
    wn = _wordnet()
    try:
        version = str(wn.get_version())
    except LookupError:
        _download_wordnet_corpus()
        version = str(wn.get_version())
    if version != EXPECTED_WORDNET_VERSION:
        raise RuntimeError(f"Expected WordNet {EXPECTED_WORDNET_VERSION}, got {version}")
    return version


def _target_sense_key(*, synset: Any, target_lemma: LemmaText) -> SenseKey:
    normalized_target = _normalize_lemma(value=target_lemma)
    lemmas: list[Any] = list(synset.lemmas())
    for lemma in lemmas:
        if _normalize_lemma(value=str(lemma.name())) == normalized_target:
            return str(lemma.key())
    if len(lemmas) == 0:
        raise ValueError(f"synset has no lemmas: {synset.name()}")
    return str(lemmas[0].key())


def _synonyms(*, synset: Any, target_lemma: LemmaText) -> list[SynonymText]:
    normalized_target = _normalize_lemma(value=target_lemma)
    values: list[SynonymText] = []
    seen: set[NormalizedLemma] = set()
    lemma_names: list[Any] = list(synset.lemma_names())
    for lemma_name in lemma_names:
        rendered: SynonymText = str(lemma_name).replace("_", " ")
        normalized = _normalize_lemma(value=rendered)
        if normalized == normalized_target or normalized in seen:
            continue
        seen.add(normalized)
        values.append(rendered)
    return values


def get_candidate_senses(*, lemma: LemmaText, pos: str) -> list[SenseCandidate]:
    wn = _wordnet()
    wn_pos = wordnet_pos(pos=pos)
    synsets: list[Any] = list(
        wn.synsets(
            lemma=lemma,
            pos=wn_pos.value if wn_pos is not None else None,
        )
    )
    candidates: list[SenseCandidate] = []
    for synset in synsets:
        examples: list[str] = [str(example).strip() for example in synset.examples()]
        candidates.append(
            SenseCandidate(
                sense_key=_target_sense_key(synset=synset, target_lemma=lemma),
                synset_id=str(synset.name()),
                pos=_synset_pos(synset=synset),
                definition=str(synset.definition()),
                synonyms=_synonyms(synset=synset, target_lemma=lemma),
                examples=examples,
            )
        )
    return candidates


def sense_keys_match(*, predicted_sense_key: SenseKey, gold_sense_keys: list[SenseKey]) -> bool:
    if predicted_sense_key in gold_sense_keys:
        return True
    wn = _wordnet()
    try:
        predicted_synset = wn.lemma_from_key(predicted_sense_key).synset().name()
    except Exception:
        return False
    for gold_sense_key in gold_sense_keys:
        try:
            gold_synset = wn.lemma_from_key(gold_sense_key).synset().name()
        except Exception:
            continue
        if predicted_synset == gold_synset:
            return True
    return False

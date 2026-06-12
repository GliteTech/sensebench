"""Small NLTK WordNet wrapper used by SenseBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sensebench.datasets.models import SenseKey

EXPECTED_WORDNET_VERSION: str = "3.0"

type SynsetID = str
type WordNetPos = str

PROJECT_POS_TO_WORDNET: dict[str, WordNetPos] = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "n": "n",
    "v": "v",
    "a": "a",
    "s": "a",
    "r": "r",
}


@dataclass(frozen=True, slots=True)
class SenseCandidate:
    sense_key: SenseKey
    synset_id: SynsetID
    pos: WordNetPos
    definition: str
    synonyms: list[str]
    examples: list[str]


def _wordnet() -> Any:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install nltk to use SenseBench WordNet support.") from exc
    return wn


def _normalize_lemma(*, value: str) -> str:
    return "".join(
        character for character in value.lower().replace("_", " ") if character.isalnum()
    )


def wordnet_pos(*, pos: str) -> WordNetPos | None:
    return PROJECT_POS_TO_WORDNET.get(pos)


def wordnet_version() -> str:
    wn = _wordnet()
    version = str(wn.get_version())
    if version != EXPECTED_WORDNET_VERSION:
        raise RuntimeError(f"Expected WordNet {EXPECTED_WORDNET_VERSION}, got {version}")
    return version


def _target_sense_key(*, synset: Any, target_lemma: str) -> SenseKey:
    normalized_target = _normalize_lemma(value=target_lemma)
    lemmas: list[Any] = list(synset.lemmas())
    for lemma in lemmas:
        if _normalize_lemma(value=str(lemma.name())) == normalized_target:
            return str(lemma.key())
    if len(lemmas) == 0:
        raise ValueError(f"synset has no lemmas: {synset.name()}")
    return str(lemmas[0].key())


def _synonyms(*, synset: Any, target_lemma: str) -> list[str]:
    normalized_target = _normalize_lemma(value=target_lemma)
    values: list[str] = []
    seen: set[str] = set()
    lemma_names: list[Any] = list(synset.lemma_names())
    for lemma_name in lemma_names:
        rendered = str(lemma_name).replace("_", " ")
        normalized = _normalize_lemma(value=rendered)
        if normalized == normalized_target or normalized in seen:
            continue
        seen.add(normalized)
        values.append(rendered)
    return values


def get_candidate_senses(*, lemma: str, pos: str) -> list[SenseCandidate]:
    wn = _wordnet()
    wn_pos = wordnet_pos(pos=pos)
    synsets: list[Any] = list(wn.synsets(lemma, pos=wn_pos))
    candidates: list[SenseCandidate] = []
    for synset in synsets:
        examples: list[str] = [str(example).strip() for example in synset.examples()]
        candidates.append(
            SenseCandidate(
                sense_key=_target_sense_key(synset=synset, target_lemma=lemma),
                synset_id=str(synset.name()),
                pos=str(synset.pos()),
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

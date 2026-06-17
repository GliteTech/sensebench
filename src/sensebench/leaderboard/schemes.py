"""Scoring schemes: re-score a prediction against any of nine label sets.

A scheme is one of three gold label sources -- lexEN v1, Maru 2022 (ALLamended), and the
original Raganato 2017 ALL labels -- crossed with three sense granularities -- WordNet
fine-grained sense keys, Glite coarse-grained concepts, and CSI coarse-grained concepts
(Lacerra et al. 2020, a public third-party inventory). The lexEN WordNet fine-grained scheme
is the default and the dataset's native gold.

Fine-grained correctness reuses the WordNet matcher (`prediction_is_correct`). Coarse
correctness maps the predicted sense key and the gold keys through a vendored concept map
(Glite or CSI) and tests concept membership; it also inherits a fine-grained hit, so coarse
accuracy is always at least the fine-grained accuracy for the same gold source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import assert_never

from sensebench.datasets.models import SenseKey, WsdItem
from sensebench.paths import (
    CSI_ALIASES_PATH,
    CSI_CONCEPT_MAP_PATH,
    GLITE_ALIASES_PATH,
    GLITE_CONCEPT_MAP_PATH,
)
from sensebench.runner.evaluate import prediction_is_correct

MARU2022_SENSE_KEYS_METADATA_KEY: str = "maru2022_sense_keys"
RAGANATO_SENSE_KEYS_METADATA_KEY: str = "raganato_original_sense_keys"
UNMAPPED_CONCEPT_PREFIX: str = "unmapped:"


class GoldSource(StrEnum):
    LEXEN = "lexen"
    MARU2022 = "maru2022"
    RAGANATO = "raganato"


class Granularity(StrEnum):
    FINE = "fine"
    COARSE = "coarse"
    CSI = "csi"


@dataclass(frozen=True, slots=True)
class Scheme:
    scheme_id: str
    label: str
    gold_label: str
    granularity_label: str
    gold_source: GoldSource
    granularity: Granularity


GOLD_SOURCE_LABELS: dict[GoldSource, str] = {
    GoldSource.LEXEN: "lexEN v1",
    GoldSource.MARU2022: "Maru 2022 (ALLamended)",
    GoldSource.RAGANATO: "Raganato 2017 (original)",
}
GRANULARITY_LABELS: dict[Granularity, str] = {
    Granularity.FINE: "WordNet fine-grained",
    Granularity.COARSE: "Glite coarse-grained",
    Granularity.CSI: "CSI coarse-grained (Lacerra 2020)",
}


def _build_schemes() -> tuple[Scheme, ...]:
    schemes: list[Scheme] = []
    for source in (GoldSource.LEXEN, GoldSource.MARU2022, GoldSource.RAGANATO):
        for granularity in (Granularity.FINE, Granularity.COARSE, Granularity.CSI):
            gold_label = GOLD_SOURCE_LABELS[source]
            granularity_label = GRANULARITY_LABELS[granularity]
            schemes.append(
                Scheme(
                    scheme_id=f"{source.value}_{granularity.value}",
                    label=f"{gold_label} · {granularity_label}",
                    gold_label=gold_label,
                    granularity_label=granularity_label,
                    gold_source=source,
                    granularity=granularity,
                )
            )
    return tuple(schemes)


SCHEMES: tuple[Scheme, ...] = _build_schemes()
SCHEME_BY_ID: dict[str, Scheme] = {scheme.scheme_id: scheme for scheme in SCHEMES}
SCHEME_IDS: tuple[str, ...] = tuple(scheme.scheme_id for scheme in SCHEMES)
DEFAULT_SCHEME_ID: str = "lexen_fine"


@dataclass(frozen=True, slots=True)
class ConceptMap:
    direct: dict[SenseKey, str]
    aliases: dict[SenseKey, str]

    def concept_for(self, sense_key: SenseKey) -> str:
        concept = self.direct.get(sense_key)
        if concept is not None:
            return concept
        concept = self.aliases.get(sense_key)
        if concept is not None:
            return concept
        return f"{UNMAPPED_CONCEPT_PREFIX}{sense_key}"


@lru_cache(maxsize=1)
def load_concept_map() -> ConceptMap:
    direct: dict[SenseKey, str] = {}
    with GLITE_CONCEPT_MAP_PATH.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            row = json.loads(line)
            direct[row["sense_key"]] = row["concept_id"]
    aliases: dict[SenseKey, str] = {}
    payload = json.loads(GLITE_ALIASES_PATH.read_text(encoding="utf-8"))
    for alias in payload["aliases"]:
        aliases[alias["source_sense_key"]] = alias["concept_id"]
    return ConceptMap(direct=direct, aliases=aliases)


@lru_cache(maxsize=1)
def load_concept_map_csi() -> ConceptMap:
    """Vendored CSI (Lacerra 2020) sense-key -> composite-concept map; same schema as Glite."""
    direct: dict[SenseKey, str] = {}
    with CSI_CONCEPT_MAP_PATH.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if len(line) == 0:
                continue
            row = json.loads(line)
            direct[row["sense_key"]] = row["concept_id"]
    aliases: dict[SenseKey, str] = {}
    payload = json.loads(CSI_ALIASES_PATH.read_text(encoding="utf-8"))
    for alias in payload["aliases"]:
        aliases[alias["source_sense_key"]] = alias["concept_id"]
    return ConceptMap(direct=direct, aliases=aliases)


def coarse_concept_map(granularity: Granularity) -> ConceptMap:
    """The vendored concept map backing a coarse granularity (Glite or CSI)."""
    if granularity == Granularity.CSI:
        return load_concept_map_csi()
    return load_concept_map()


def gold_fine_keys(*, item: WsdItem, gold_source: GoldSource) -> list[SenseKey]:
    """The gold WordNet sense keys for `gold_source` on this item (possibly empty)."""
    match gold_source:
        case GoldSource.LEXEN:
            return list(item.gold_sense_keys)
        case GoldSource.MARU2022:
            return item.metadata.get(MARU2022_SENSE_KEYS_METADATA_KEY, "").split()
        case GoldSource.RAGANATO:
            return item.metadata.get(RAGANATO_SENSE_KEYS_METADATA_KEY, "").split()
        case _:
            assert_never(gold_source)


def is_scoreable(*, item: WsdItem, gold_source: GoldSource) -> bool:
    """Whether the item carries a non-empty gold set for this source (else it is skipped)."""
    return len(gold_fine_keys(item=item, gold_source=gold_source)) > 0


def _fine_correct(*, predicted_sense_key: SenseKey | None, gold_keys: list[SenseKey]) -> bool:
    return (
        prediction_is_correct(predicted_sense_key=predicted_sense_key, gold_sense_keys=gold_keys)
        is True
    )


def scheme_correct(
    *,
    scheme: Scheme,
    predicted_sense_key: SenseKey | None,
    item: WsdItem,
    concept_map: ConceptMap,
) -> bool:
    """Correctness of `predicted_sense_key` under `scheme`. Callers must check `is_scoreable`.

    A `None` prediction (no valid vote) is a miss. Coarse correctness inherits a fine-grained
    hit, so coarse is always >= fine for the same gold source.
    """
    gold_keys = gold_fine_keys(item=item, gold_source=scheme.gold_source)
    fine = _fine_correct(predicted_sense_key=predicted_sense_key, gold_keys=gold_keys)
    if scheme.granularity == Granularity.FINE:
        return fine
    if fine:
        return True
    if predicted_sense_key is None:
        return False
    # COARSE uses the passed (Glite) map; CSI self-resolves its own vendored map, so the
    # CSI schemes score correctly even though callers pass the Glite map for all schemes.
    cmap = load_concept_map_csi() if scheme.granularity == Granularity.CSI else concept_map
    predicted_concept = cmap.concept_for(predicted_sense_key)
    gold_concepts = {cmap.concept_for(key) for key in gold_keys}
    return predicted_concept in gold_concepts


def scheme_correctness(
    *,
    predicted_sense_key: SenseKey | None,
    item: WsdItem,
    concept_map: ConceptMap,
) -> dict[str, bool | None]:
    """Per-scheme correctness for one prediction. `None` means the item is not scoreable
    under that scheme (its gold source is empty); a bool counts toward that scheme."""
    results: dict[str, bool | None] = {}
    for scheme in SCHEMES:
        if not is_scoreable(item=item, gold_source=scheme.gold_source):
            results[scheme.scheme_id] = None
            continue
        results[scheme.scheme_id] = scheme_correct(
            scheme=scheme,
            predicted_sense_key=predicted_sense_key,
            item=item,
            concept_map=concept_map,
        )
    return results

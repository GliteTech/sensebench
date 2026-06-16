from __future__ import annotations

from sensebench.datasets.models import DatasetPos, SenseKey, WsdItem
from sensebench.leaderboard.schemes import (
    DEFAULT_SCHEME_ID,
    SCHEME_BY_ID,
    SCHEME_IDS,
    SCHEMES,
    GoldSource,
    Granularity,
    gold_fine_keys,
    is_scoreable,
    load_concept_map,
    scheme_correct,
    scheme_correctness,
)

# accept%2:31:00:: and accept%2:31:01:: are different WordNet sense keys that map to one Glite
# concept, so a prediction of one against the other is a fine-grained miss but a coarse hit.
ACCEPT_GOLD: SenseKey = "accept%2:31:00::"
ACCEPT_SAME_CONCEPT: SenseKey = "accept%2:31:01::"
UNMAPPED_KEY: SenseKey = "analyse%2:31:00::"


def _item(
    *,
    lexen: list[SenseKey],
    maru2022: str = "",
    raganato: str = "",
) -> WsdItem:
    return WsdItem(
        item_id="i1",
        document_id="d1",
        sentence_id="s1",
        target_token_index=0,
        target_text="accept",
        lemma="accept",
        pos=DatasetPos.VERB,
        gold_sense_keys=lexen,
        metadata={
            "maru2022_sense_keys": maru2022,
            "raganato_original_sense_keys": raganato,
        },
    )


def test_scheme_registry_has_nine_schemes() -> None:
    assert len(SCHEMES) == 9
    assert set(SCHEME_IDS) == {
        "lexen_fine",
        "lexen_coarse",
        "lexen_csi",
        "maru2022_fine",
        "maru2022_coarse",
        "maru2022_csi",
        "raganato_fine",
        "raganato_coarse",
        "raganato_csi",
    }
    assert DEFAULT_SCHEME_ID == "lexen_fine"
    assert set(SCHEME_BY_ID) == set(SCHEME_IDS)


def test_concept_map_loads_and_marks_unmapped() -> None:
    concept_map = load_concept_map()
    assert len(concept_map.direct) > 0
    assert len(concept_map.aliases) > 0
    assert concept_map.concept_for(ACCEPT_GOLD).startswith("ct:")
    assert concept_map.concept_for(UNMAPPED_KEY) == f"unmapped:{UNMAPPED_KEY}"


def test_gold_fine_keys_parses_whitespace_strings() -> None:
    item = _item(lexen=[ACCEPT_GOLD], maru2022="a%1:00:00:: b%1:00:00::", raganato="c%1:00:00::")
    assert gold_fine_keys(item=item, gold_source=GoldSource.LEXEN) == [ACCEPT_GOLD]
    assert gold_fine_keys(item=item, gold_source=GoldSource.MARU2022) == [
        "a%1:00:00::",
        "b%1:00:00::",
    ]
    assert gold_fine_keys(item=item, gold_source=GoldSource.RAGANATO) == ["c%1:00:00::"]


def test_is_scoreable_requires_nonempty_gold() -> None:
    item = _item(lexen=[ACCEPT_GOLD], maru2022="", raganato="x%1:00:00::")
    assert is_scoreable(item=item, gold_source=GoldSource.LEXEN) is True
    assert is_scoreable(item=item, gold_source=GoldSource.MARU2022) is False
    assert is_scoreable(item=item, gold_source=GoldSource.RAGANATO) is True


def test_fine_scheme_is_exact_sense_match() -> None:
    item = _item(lexen=[ACCEPT_GOLD])
    concept_map = load_concept_map()
    fine = SCHEME_BY_ID["lexen_fine"]
    assert scheme_correct(
        scheme=fine, predicted_sense_key=ACCEPT_GOLD, item=item, concept_map=concept_map
    )
    assert not scheme_correct(
        scheme=fine, predicted_sense_key=ACCEPT_SAME_CONCEPT, item=item, concept_map=concept_map
    )


def test_coarse_scheme_credits_same_concept() -> None:
    item = _item(lexen=[ACCEPT_GOLD])
    concept_map = load_concept_map()
    coarse = SCHEME_BY_ID["lexen_coarse"]
    # Fine miss, but the predicted sense maps to the same Glite concept as the gold sense.
    assert scheme_correct(
        scheme=coarse, predicted_sense_key=ACCEPT_SAME_CONCEPT, item=item, concept_map=concept_map
    )


def test_coarse_inherits_fine_hit() -> None:
    item = _item(lexen=[ACCEPT_GOLD])
    concept_map = load_concept_map()
    coarse = SCHEME_BY_ID["lexen_coarse"]
    assert scheme_correct(
        scheme=coarse, predicted_sense_key=ACCEPT_GOLD, item=item, concept_map=concept_map
    )


def test_unmapped_prediction_is_coarse_miss() -> None:
    item = _item(lexen=[ACCEPT_GOLD])
    concept_map = load_concept_map()
    coarse = SCHEME_BY_ID["lexen_coarse"]
    assert not scheme_correct(
        scheme=coarse, predicted_sense_key=UNMAPPED_KEY, item=item, concept_map=concept_map
    )


def test_none_prediction_is_a_miss_in_every_scheme() -> None:
    item = _item(lexen=[ACCEPT_GOLD], maru2022=ACCEPT_GOLD, raganato=ACCEPT_GOLD)
    concept_map = load_concept_map()
    for scheme in SCHEMES:
        assert not scheme_correct(
            scheme=scheme, predicted_sense_key=None, item=item, concept_map=concept_map
        )


def test_coarse_is_never_worse_than_fine() -> None:
    item = _item(lexen=[ACCEPT_GOLD], maru2022=ACCEPT_GOLD, raganato=ACCEPT_GOLD)
    concept_map = load_concept_map()
    for predicted in (ACCEPT_GOLD, ACCEPT_SAME_CONCEPT, UNMAPPED_KEY, None):
        for source in (GoldSource.LEXEN, GoldSource.MARU2022, GoldSource.RAGANATO):
            fine = scheme_correct(
                scheme=SCHEME_BY_ID[f"{source.value}_fine"],
                predicted_sense_key=predicted,
                item=item,
                concept_map=concept_map,
            )
            coarse = scheme_correct(
                scheme=SCHEME_BY_ID[f"{source.value}_coarse"],
                predicted_sense_key=predicted,
                item=item,
                concept_map=concept_map,
            )
            assert coarse or not fine


def test_scheme_correctness_skips_unscoreable_sources() -> None:
    item = _item(lexen=[ACCEPT_GOLD], maru2022="", raganato=ACCEPT_GOLD)
    concept_map = load_concept_map()
    results = scheme_correctness(
        predicted_sense_key=ACCEPT_GOLD, item=item, concept_map=concept_map
    )
    assert results["lexen_fine"] is True
    assert results["raganato_fine"] is True
    assert results["maru2022_fine"] is None
    assert results["maru2022_coarse"] is None
    assert set(results) == set(SCHEME_IDS)


def test_granularity_enum_values() -> None:
    assert Granularity.FINE.value == "fine"
    assert Granularity.COARSE.value == "coarse"

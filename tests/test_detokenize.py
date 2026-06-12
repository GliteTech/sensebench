from __future__ import annotations

from sensebench.datasets.detokenize import detokenize_pieces, detokenize_text

PERIOD_TOKENS: list[str] = ["The", "end", "."]
PERIOD_EXPECTED: str = "The end."

COMMA_TOKENS: list[str] = ["here", "in", "Earth", ",", "they", "say"]
COMMA_EXPECTED: str = "here in Earth, they say"

CONTRACTION_TOKENS: list[str] = ["Thursday", "'s", "online", "edition"]
CONTRACTION_EXPECTED: str = "Thursday's online edition"

NEGATION_TOKENS: list[str] = ["it", "does", "n't", "matter"]
NEGATION_EXPECTED: str = "it doesn't matter"

DOUBLE_QUOTE_TOKENS: list[str] = ["``", "dual", "capability", ".", "''"]
DOUBLE_QUOTE_EXPECTED: str = '"dual capability."'

SINGLE_QUOTE_TOKENS: list[str] = ["truly", "`", "alien", "'", "life"]
SINGLE_QUOTE_EXPECTED: str = "truly 'alien' life"

BRACKET_TOKENS: list[str] = ["see", "(", "Mono", "Lake", ")", "now"]
BRACKET_EXPECTED: str = "see (Mono Lake) now"

TOKENIZED_TOKENS: list[str] = ["essential", "for", "life", "-", "carbon", ",", "hydrogen", "."]
TOKENIZED_EXPECTED: str = "essential for life - carbon, hydrogen."


def test_detokenize_attaches_sentence_punctuation() -> None:
    assert detokenize_text(surfaces=PERIOD_TOKENS) == PERIOD_EXPECTED
    assert detokenize_text(surfaces=COMMA_TOKENS) == COMMA_EXPECTED
    assert detokenize_text(surfaces=TOKENIZED_TOKENS) == TOKENIZED_EXPECTED


def test_detokenize_joins_contractions() -> None:
    assert detokenize_text(surfaces=CONTRACTION_TOKENS) == CONTRACTION_EXPECTED
    assert detokenize_text(surfaces=NEGATION_TOKENS) == NEGATION_EXPECTED


def test_detokenize_rewrites_quotes() -> None:
    assert detokenize_text(surfaces=DOUBLE_QUOTE_TOKENS) == DOUBLE_QUOTE_EXPECTED
    assert detokenize_text(surfaces=SINGLE_QUOTE_TOKENS) == SINGLE_QUOTE_EXPECTED


def test_detokenize_handles_brackets() -> None:
    assert detokenize_text(surfaces=BRACKET_TOKENS) == BRACKET_EXPECTED


def test_detokenize_pieces_align_with_input() -> None:
    pieces = detokenize_pieces(surfaces=DOUBLE_QUOTE_TOKENS)

    assert len(pieces) == len(DOUBLE_QUOTE_TOKENS)
    assert pieces[0].leading_space is False
    assert pieces[0].text == '"'
    # "dual" follows the opening quote and must not get a leading space.
    assert pieces[1].leading_space is False
    assert pieces[1].text == "dual"

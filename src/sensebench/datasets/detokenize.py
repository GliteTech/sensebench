"""Deterministic detokenization of Penn-Treebank-style tokens into natural text.

The Senseval/SemEval source data in the Raganato unified evaluation framework is
Penn-Treebank tokenized: punctuation is space-separated and quotes use the ``/''
convention. This module joins those tokens back into natural English with a small,
auditable rule set (no external dependency, fully deterministic), so a prompt can
present the model with ordinary prose instead of tokenized text.

Reconstruction is best-effort: PTB tokenization is lossy (original whitespace is not
recorded), so this targets the common, unambiguous cases — punctuation attachment,
contractions, brackets, and quotation marks.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tokens that attach to the previous token with no leading space.
GLUE_LEFT_TOKENS: frozenset[str] = frozenset(
    {
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        ")",
        "]",
        "}",
        "%",
        "''",
        "'",
        "'s",
        "'S",
        "n't",
        "N'T",
        "'re",
        "'RE",
        "'ve",
        "'VE",
        "'m",
        "'M",
        "'ll",
        "'LL",
        "'d",
        "'D",
    }
)

# Opener tokens after which the next token attaches with no leading space.
GLUE_RIGHT_TOKENS: frozenset[str] = frozenset({"(", "[", "{", "``", "`", "$"})

# Surface rewrites toward natural typography.
SURFACE_REWRITES: dict[str, str] = {
    "``": '"',
    "''": '"',
    "`": "'",
}


@dataclass(frozen=True, slots=True)
class DetokenizedPiece:
    leading_space: bool
    text: str


def detokenize_pieces(*, surfaces: list[str]) -> list[DetokenizedPiece]:
    """Return one piece per input token: whether it needs a leading space and its rendered text.

    Pieces align 1:1 with ``surfaces`` (order and count preserved), so callers can map a
    piece back to its original token, e.g. to wrap a target token.
    """
    pieces: list[DetokenizedPiece] = []
    previous_glues_right = False
    for index, surface in enumerate(surfaces):
        leading_space = not (
            index == 0 or surface in GLUE_LEFT_TOKENS or previous_glues_right
        )
        pieces.append(
            DetokenizedPiece(
                leading_space=leading_space,
                text=SURFACE_REWRITES.get(surface, surface),
            )
        )
        previous_glues_right = surface in GLUE_RIGHT_TOKENS
    return pieces


def detokenize_text(*, surfaces: list[str]) -> str:
    return "".join(
        f"{' ' if piece.leading_space else ''}{piece.text}"
        for piece in detokenize_pieces(surfaces=surfaces)
    )

"""Compose the social sharing card from the published Pareto figure.

The card is the paper's cost/accuracy figure with a header band above it. The
figure itself is the hand-curated publication artifact (per-run label offsets and
frontier names are tuned by hand in the paper repository), so it is committed
here as an asset rather than regenerated: see `tools/assets/pareto-figure.png`.

To refresh it, re-render the figure in the paper repository, trim its margins,
overwrite the asset, and rerun this script:

    uv run python tools/make_og_card.py

The header deliberately carries no run or model counts, so the card does not go
stale as runs are merged.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.image import imread

FIGURE_PATH: Path = Path("tools/assets/pareto-figure.png")
CARD_PATH: Path = Path("src/sensebench/site/static/og-card.png")

CARD_WIDTH_PX: int = 1200
CARD_HEIGHT_PX: int = 630
CARD_DPI: int = 100

HEADER_HEIGHT_FRACTION: float = 0.205
FIGURE_SIDE_PADDING_FRACTION: float = 0.012

INK: str = "#171717"
MUTED: str = "#6F747C"
RULE: str = "#D7DCD8"
PAPER: str = "#FFFFFF"

TITLE_SIZE: int = 33
SUBTITLE_SIZE: int = 17
URL_SIZE: int = 17

TITLE_TEXT: str = "SenseBench"
SUBTITLE_TEXT: str = "LLM word sense disambiguation · cost vs accuracy on lexEN"
URL_TEXT: str = "sense-bench.com"

SERIF_FAMILY: list[str] = ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"]


def _render_card(*, figure_path: Path, card_path: Path) -> None:
    mpl.rcParams.update({"font.family": "serif", "font.serif": SERIF_FAMILY})

    card = plt.figure(
        figsize=(CARD_WIDTH_PX / CARD_DPI, CARD_HEIGHT_PX / CARD_DPI),
        dpi=CARD_DPI,
        facecolor=PAPER,
    )

    header_top = 1.0 - HEADER_HEIGHT_FRACTION
    card.text(0.038, 0.975, TITLE_TEXT, size=TITLE_SIZE, weight="bold", color=INK, va="top")
    card.text(0.038, 0.888, SUBTITLE_TEXT, size=SUBTITLE_SIZE, color=MUTED, va="top")
    card.text(
        0.962,
        0.975,
        URL_TEXT,
        size=URL_SIZE,
        color=MUTED,
        ha="right",
        va="top",
    )
    card.add_artist(
        plt.Line2D(
            [0.038, 0.962],
            [header_top, header_top],
            color=RULE,
            linewidth=1.0,
            transform=card.transFigure,
        )
    )

    plot = card.add_axes(
        (
            FIGURE_SIDE_PADDING_FRACTION,
            0.0,
            1.0 - 2 * FIGURE_SIDE_PADDING_FRACTION,
            header_top,
        )
    )
    plot.imshow(imread(figure_path), interpolation="antialiased")
    plot.set_axis_off()

    card_path.parent.mkdir(parents=True, exist_ok=True)
    card.savefig(card_path, dpi=CARD_DPI, facecolor=PAPER)
    plt.close(card)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figure-path", type=Path, default=FIGURE_PATH)
    parser.add_argument("--card-path", type=Path, default=CARD_PATH)
    args = parser.parse_args()

    if not args.figure_path.exists():
        parser.error(f"{args.figure_path}: not found")

    _render_card(figure_path=args.figure_path, card_path=args.card_path)
    print(f"wrote {args.card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

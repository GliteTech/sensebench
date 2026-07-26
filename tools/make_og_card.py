"""Render the social sharing card from the verified leaderboard.

The card is the cost/accuracy Pareto chart, redrawn for small-format legibility:
share previews render around 500px wide, so the publication figure's per-point
callouts and family legend become unreadable. Here only the frontier endpoints
and the best open-weight run are labelled, and the family legend collapses to
open weights versus proprietary.

Costs come from the aggregated leaderboard rather than from run artifacts, so
self-hosted runs are priced at the same reference GPU rates the site uses and the
card cannot disagree with the chart it advertises.

Regenerate after merging runs that change the frontier:

    uv run sensebench leaderboard
    uv run python tools/make_og_card.py

Verify the committed card still matches the leaderboard:

    uv run python tools/make_og_card.py --check
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from math import isfinite, log10
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from sensebench.leaderboard.aggregate import LeaderboardEntry
from sensebench.paths import DEFAULT_LEXEN_RELEASE_ID

CARD_PATH: Path = Path("src/sensebench/site/static/og-card.png")
STATE_PATH: Path = Path("tools/og_card_state.json")
LEADERBOARD_JSON_CANDIDATES: tuple[Path, ...] = (
    Path("_site/data/leaderboard.json"),
    Path("leaderboard.json"),
)
ENTRIES_KEY: str = "entries"
STATE_ACCURACY_DECIMALS: int = 4

CARD_WIDTH_PX: int = 1200
CARD_HEIGHT_PX: int = 630
CARD_DPI: int = 100

HEADLINE_PROMPT_ID: str = "p001"
OPEN_SOURCE_KIND: str = "open_source"

INK: str = "#171717"
MUTED: str = "#6F747C"
GRID: str = "#E5E7EB"
OPEN_WEIGHTS_COLOR: str = "#14966B"
PROPRIETARY_COLOR: str = "#2374B7"
FRONTIER_COLOR: str = "#2D333B"
PAPER: str = "#FFFFFF"

TITLE_SIZE: int = 34
SUBTITLE_SIZE: int = 17
AXIS_LABEL_SIZE: int = 17
TICK_SIZE: int = 15
ANNOTATION_SIZE: int = 15
LEGEND_SIZE: int = 15

TITLE_TEXT: str = "SenseBench"
SUBTITLE_TEXT: str = "LLM word sense disambiguation · cost vs accuracy"
FOOTER_TEXT: str = "sense-bench.com"
X_AXIS_LABEL: str = "cost (USD per 1M items, log scale)"
Y_AXIS_LABEL: str = "accuracy (%)"
OPEN_WEIGHTS_LABEL: str = "open weights"
PROPRIETARY_LABEL: str = "proprietary"
FRONTIER_LABEL: str = "Pareto frontier"

ACCURACY_HEADROOM_PCT: float = 0.6
COST_AXIS_LOW_MARGIN: float = 1.7
COST_AXIS_HIGH_MARGIN: float = 3.4
KNEE_ACCURACY_TOLERANCE_PCT: float = 7.0
FALLBACK_MODEL_MARKER: str = "+fallback"
LABEL_RIGHT_ALIGN_FRACTION: float = 0.72
LABEL_LEFT_ALIGN_FRACTION: float = 0.12


@dataclass(frozen=True, slots=True)
class CardPoint:
    label: str
    accuracy_pct: float
    cost_per_million: float
    is_open_weights: bool
    is_fallback: bool


@dataclass(frozen=True, slots=True)
class Annotation:
    point: CardPoint
    place_below: bool


@dataclass(frozen=True, slots=True)
class CardState:
    run_count: int
    model_count: int
    top_accuracy_pct: float
    plotted_point_count: int


def _default_leaderboard_json() -> Path | None:
    return next((path for path in LEADERBOARD_JSON_CANDIDATES if path.exists()), None)


def _load_entries(path: Path) -> list[LeaderboardEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [LeaderboardEntry.model_validate(row) for row in payload[ENTRIES_KEY]]


def _comparable_points(entries: list[LeaderboardEntry]) -> list[CardPoint]:
    points: list[CardPoint] = []
    for entry in entries:
        cost = entry.cost_per_million_items
        accuracy = entry.accuracy
        if cost is None or accuracy is None:
            continue
        if entry.prompt_id != HEADLINE_PROMPT_ID:
            continue
        if entry.dataset_version != DEFAULT_LEXEN_RELEASE_ID:
            continue
        if not isfinite(cost) or not isfinite(accuracy) or cost <= 0:
            continue
        points.append(
            CardPoint(
                label=entry.display_label or entry.model,
                accuracy_pct=accuracy * 100.0,
                cost_per_million=cost,
                is_open_weights=entry.source_kind == OPEN_SOURCE_KIND,
                is_fallback=FALLBACK_MODEL_MARKER in entry.model,
            )
        )
    return points


def _pareto_frontier(points: list[CardPoint]) -> list[CardPoint]:
    frontier = [
        point
        for point in points
        if not any(
            other.accuracy_pct >= point.accuracy_pct
            and other.cost_per_million <= point.cost_per_million
            and (
                other.accuracy_pct > point.accuracy_pct
                or other.cost_per_million < point.cost_per_million
            )
            for other in points
        )
    ]
    return sorted(frontier, key=lambda point: (point.cost_per_million, point.accuracy_pct))


def _annotated_points(frontier: list[CardPoint], points: list[CardPoint]) -> list[Annotation]:
    """Best overall, the frontier knee, and the best plain open-weights run.

    The cheapest frontier point is deliberately not labelled: it is whatever
    scored worst while still being cheapest, which advertises nothing. Fallback
    hybrids are skipped because their composite labels do not read as a model.
    The knee label goes below its point, since the frontier rises steeply there
    and a label above it would sit on the line.
    """
    if len(frontier) == 0:
        return []
    best = max(frontier, key=lambda point: point.accuracy_pct)
    knee = min(
        (
            point
            for point in frontier
            if point.accuracy_pct >= best.accuracy_pct - KNEE_ACCURACY_TOLERANCE_PCT
        ),
        key=lambda point: point.cost_per_million,
        default=best,
    )
    chosen = [Annotation(point=best, place_below=False)]
    if knee != best:
        chosen.insert(0, Annotation(point=knee, place_below=True))
    plain_open = [point for point in points if point.is_open_weights and not point.is_fallback]
    if len(plain_open) > 0:
        best_open = max(plain_open, key=lambda point: point.accuracy_pct)
        if all(best_open != annotation.point for annotation in chosen):
            chosen.append(Annotation(point=best_open, place_below=False))
    return chosen


def _format_cost_tick(value: float, _position: int) -> str:
    if value >= 1000:
        return f"${value / 1000:g}k"
    return f"${value:g}"


def _render_card(*, points: list[CardPoint], state: CardState, path: Path) -> None:
    figure = plt.figure(
        figsize=(CARD_WIDTH_PX / CARD_DPI, CARD_HEIGHT_PX / CARD_DPI),
        dpi=CARD_DPI,
        facecolor=PAPER,
    )
    figure.text(0.055, 0.905, TITLE_TEXT, size=TITLE_SIZE, weight="bold", color=INK)
    figure.text(0.055, 0.845, SUBTITLE_TEXT, size=SUBTITLE_SIZE, color=MUTED)
    figure.text(
        0.945,
        0.905,
        f"{state.run_count} verified runs · {state.model_count} models",
        size=SUBTITLE_SIZE,
        color=INK,
        ha="right",
    )
    figure.text(
        0.945,
        0.845,
        f"best {state.top_accuracy_pct:.2f}% on lexEN v1",
        size=SUBTITLE_SIZE,
        color=MUTED,
        ha="right",
    )
    figure.text(0.945, 0.045, FOOTER_TEXT, size=SUBTITLE_SIZE, color=MUTED, ha="right")

    axes = figure.add_axes((0.075, 0.165, 0.87, 0.62))
    axes.set_facecolor(PAPER)
    axes.set_xscale("log")
    axes.grid(True, which="major", color=GRID, linewidth=0.9)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRID)

    open_points = [point for point in points if point.is_open_weights]
    closed_points = [point for point in points if not point.is_open_weights]
    for bucket, color in ((closed_points, PROPRIETARY_COLOR), (open_points, OPEN_WEIGHTS_COLOR)):
        axes.scatter(
            [point.cost_per_million for point in bucket],
            [point.accuracy_pct for point in bucket],
            s=46,
            color=color,
            alpha=0.75,
            linewidths=0,
            zorder=2,
        )

    frontier = _pareto_frontier(points)
    axes.plot(
        [point.cost_per_million for point in frontier],
        [point.accuracy_pct for point in frontier],
        color=FRONTIER_COLOR,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        zorder=3,
    )

    accuracies = [point.accuracy_pct for point in points]
    costs = [point.cost_per_million for point in points]
    axes.set_ylim(min(accuracies) - ACCURACY_HEADROOM_PCT, max(accuracies) + 2.6)
    axes.set_xlim(
        min(costs) / COST_AXIS_LOW_MARGIN,
        max(costs) * COST_AXIS_HIGH_MARGIN,
    )

    low_cost, high_cost = axes.get_xlim()
    log_span = log10(high_cost) - log10(low_cost)
    for annotation in _annotated_points(frontier, points):
        point = annotation.point
        position = (log10(point.cost_per_million) - log10(low_cost)) / log_span
        if position > LABEL_RIGHT_ALIGN_FRACTION:
            alignment, x_offset = "right", -8
        elif position < LABEL_LEFT_ALIGN_FRACTION:
            alignment, x_offset = "left", 8
        else:
            alignment, x_offset = "center", 0
        axes.annotate(
            point.label,
            xy=(point.cost_per_million, point.accuracy_pct),
            xytext=(x_offset, -24 if annotation.place_below else 14),
            textcoords="offset points",
            size=ANNOTATION_SIZE,
            color=INK,
            ha=alignment,
            zorder=4,
        )
    axes.set_xlabel(X_AXIS_LABEL, size=AXIS_LABEL_SIZE, color=MUTED)
    axes.set_ylabel(Y_AXIS_LABEL, size=AXIS_LABEL_SIZE, color=MUTED)
    axes.tick_params(labelsize=TICK_SIZE, colors=MUTED, length=0)
    axes.xaxis.set_major_formatter(FuncFormatter(_format_cost_tick))

    axes.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=OPEN_WEIGHTS_COLOR, markersize=9,
                   label=OPEN_WEIGHTS_LABEL),
            Line2D([], [], marker="o", linestyle="", color=PROPRIETARY_COLOR, markersize=9,
                   label=PROPRIETARY_LABEL),
            Line2D([], [], linestyle=(0, (5, 3)), color=FRONTIER_COLOR, label=FRONTIER_LABEL),
        ],
        loc="lower right",
        fontsize=LEGEND_SIZE,
        frameon=False,
        labelcolor=MUTED,
        handletextpad=0.5,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=CARD_DPI, facecolor=PAPER)
    plt.close(figure)


def _card_state(*, entries: list[LeaderboardEntry], points: list[CardPoint]) -> CardState:
    accuracies = [entry.accuracy for entry in entries if entry.accuracy is not None]
    top_accuracy_pct = max(accuracies) * 100.0 if len(accuracies) > 0 else 0.0
    return CardState(
        run_count=len(entries),
        model_count=len({entry.model for entry in entries}),
        # Rounded here so the recomputed state compares equal to the stored one.
        top_accuracy_pct=round(top_accuracy_pct, STATE_ACCURACY_DECIMALS),
        plotted_point_count=len(points),
    )


def _write_state(*, state: CardState, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_count": state.run_count,
                "model_count": state.model_count,
                "top_accuracy_pct": state.top_accuracy_pct,
                "plotted_point_count": state.plotted_point_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_state(path: Path) -> CardState | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CardState(
        run_count=int(payload["run_count"]),
        model_count=int(payload["model_count"]),
        top_accuracy_pct=float(payload["top_accuracy_pct"]),
        plotted_point_count=int(payload["plotted_point_count"]),
    )


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-json", type=Path, default=None)
    parser.add_argument("--card-path", type=Path, default=CARD_PATH)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed card is stale instead of rewriting it.",
    )
    args = parser.parse_args()

    leaderboard_json = args.leaderboard_json or _default_leaderboard_json()
    if leaderboard_json is None:
        parser.error(
            "no leaderboard JSON found; run `sensebench leaderboard` or pass --leaderboard-json"
        )
    if not leaderboard_json.exists():
        parser.error(f"{leaderboard_json}: not found")

    entries = _load_entries(leaderboard_json)
    points = _comparable_points(entries)
    if len(points) == 0:
        parser.error(f"no comparable {HEADLINE_PROMPT_ID}/{DEFAULT_LEXEN_RELEASE_ID} runs found")
    state = _card_state(entries=entries, points=points)

    if args.check:
        committed = _read_state(args.state_path)
        if committed is None:
            print(f"{args.state_path}: missing; run tools/make_og_card.py")
            return 1
        if not args.card_path.exists():
            print(f"{args.card_path}: missing; run tools/make_og_card.py")
            return 1
        if committed != state:
            print(f"{args.card_path}: stale ({committed} != {state}); run tools/make_og_card.py")
            return 1
        print(f"{args.card_path}: up to date ({state.run_count} runs, {state.model_count} models)")
        return 0

    _render_card(points=points, state=state, path=args.card_path)
    _write_state(state=state, path=args.state_path)
    print(
        f"wrote {args.card_path} "
        f"({state.run_count} runs, {state.model_count} models, "
        f"{state.plotted_point_count} plotted, best {state.top_accuracy_pct:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

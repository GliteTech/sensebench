"""Verify the homepage answer block still matches the leaderboard.

The block in `index.html.j2` is written by an agent rather than generated, so its figures
can drift as runs are merged. This checks every claim in it against the built site data and
fails when one is stale. Rewrite the block with `.agents/skills/write-homepage-answer`.

Run against a built site:

    uv run sensebench site build --results-dir results --output-dir _site --strict
    uv run python tools/check_homepage_answer.py
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

INDEX_HTML: Path = Path("_site/index.html")
LEADERBOARD_JSON: Path = Path("_site/data/leaderboard.json")

BLOCK_PATTERN: re.Pattern[str] = re.compile(
    r'<section class="answer-block".*?</section>', re.DOTALL
)
PERCENT_PATTERN: re.Pattern[str] = re.compile(r"(\d+\.\d{2})%")
COUNT_PATTERN: re.Pattern[str] = re.compile(r"(\d{1,3}(?:,\d{3})+|\b\d{2,4}\b)")
TIME_PATTERN: re.Pattern[str] = re.compile(r'<time datetime="(\d{4}-\d{2}-\d{2})"')

REQUIRED_BASELINE_LABELS: tuple[str, ...] = ("MFS (WordNet first sense)", "ConSeC", "Glite LENS")


@dataclass(frozen=True, slots=True)
class Failure:
    claim: str
    detail: str


def _pct(value: float) -> str:
    return f"{value * 100:.2f}"


def _check(*, index_html: Path, leaderboard_json: Path) -> list[Failure]:
    html_text = index_html.read_text(encoding="utf-8")
    match = BLOCK_PATTERN.search(html_text)
    if match is None:
        return [Failure("answer block", f'no <section class="answer-block"> in {index_html}')]
    block = match.group(0)

    payload = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    entries = payload["entries"]
    baselines = {row["label"]: row for row in payload.get("baselines", [])}
    summary = payload["summary"]
    top = entries[0]

    failures: list[Failure] = []

    # Every figure the prose is allowed to state, derived from live data.
    allowed = {_pct(entry["accuracy"]) for entry in entries if entry.get("accuracy") is not None}
    allowed |= {_pct(row["accuracy"]) for row in baselines.values() if row.get("accuracy")}
    allowed |= {_pct(top["accuracy_ci"]["low"]), _pct(top["accuracy_ci"]["high"])}

    for stated in PERCENT_PATTERN.findall(block):
        if stated not in allowed:
            failures.append(
                Failure(f"{stated}%", "matches no current run, baseline or confidence bound")
            )

    def require(label: str, value: str) -> None:
        if value not in block:
            failures.append(Failure(label, f"expected {value!r}, not present in the block"))

    require("top accuracy", f"{_pct(top['accuracy'])}%")
    require("CI low", f"{_pct(top['accuracy_ci']['low'])}")
    require("CI high", f"{_pct(top['accuracy_ci']['high'])}")
    require("top model", top["display_label"] or top["model"])
    require("correct count", f"{top['correct_count']:,}")
    require("item count", f"{top['item_count']:,}")
    require("run count", str(summary["verified_run_count"]))
    require("model count", str(summary["model_count"]))

    for label in REQUIRED_BASELINE_LABELS:
        row = baselines.get(label)
        if row is None:
            failures.append(Failure(f"baseline {label}", "absent from the built site data"))
            continue
        require(f"baseline {label}", f"{_pct(row['accuracy'])}%")

    latest_run_date = max(entry["created_at"] for entry in entries)[:10]
    stated_dates = TIME_PATTERN.findall(block)
    if len(stated_dates) == 0:
        failures.append(Failure("<time>", "the block states no machine-readable date"))
    elif stated_dates[0] != latest_run_date:
        failures.append(
            Failure("<time>", f"states {stated_dates[0]}, latest run is {latest_run_date}")
        )

    return failures


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--index-html", type=Path, default=INDEX_HTML)
    parser.add_argument("--leaderboard-json", type=Path, default=LEADERBOARD_JSON)
    args = parser.parse_args()

    for path in (args.index_html, args.leaderboard_json):
        if not path.exists():
            parser.error(f"{path}: not found; run `sensebench site build` first")

    failures = _check(index_html=args.index_html, leaderboard_json=args.leaderboard_json)
    if len(failures) > 0:
        print(f"{args.index_html}: homepage answer block is stale")
        for failure in failures:
            print(f"  {failure.claim}: {failure.detail}")
        print("\nRewrite it with .agents/skills/write-homepage-answer, then rebuild the site.")
        return 1

    print(f"{args.index_html}: homepage answer block matches the leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

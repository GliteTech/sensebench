"""Derive the leaderboard's reference GPU hourly rates from submitted runs.

The rates live as frozen constants in `sensebench.leaderboard.gpu`; this script is how they are
computed and audited. Each rate is the mean price paid for a GPU class across distinct rentals
(a rental is one provider instance), so an instance that served many runs does not pull the mean
toward its own price.

Run it after adding self-hosted results to see whether the frozen table still reflects what we
pay, then update `GPU_REFERENCE_HOURLY_RATE_USD` deliberately:

    uv run python tools/compute_gpu_rates.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sensebench.leaderboard.gpu import GPU_REFERENCE_HOURLY_RATE_USD, gpu_label
from sensebench.paths import RUN_METADATA_FILENAME, SUBMITTED_RESULTS_DIR

RATE_DECIMALS: int = 2


@dataclass(frozen=True, slots=True)
class Rental:
    provider: str | None
    instance_id: str | None
    hourly_rate_usd: float


@dataclass(frozen=True, slots=True)
class GpuClassRates:
    rentals: list[Rental]
    run_count: int

    @property
    def reference_hourly_rate_usd(self) -> float:
        rates = [rental.hourly_rate_usd for rental in self.rentals]
        return round(sum(rates) / len(rates), RATE_DECIMALS)


def _rental_for_run(*, run_json: Path) -> tuple[str, Rental] | None:
    try:
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    machine = metadata.get("machine")
    if machine is None:
        return None
    gpu = machine.get("gpu")
    rate = machine.get("hourly_rate_usd")
    if gpu is None or rate is None:
        return None
    return (
        gpu_label(name=gpu["name"]),
        Rental(
            provider=machine.get("provider"),
            instance_id=machine.get("instance_id"),
            hourly_rate_usd=float(rate),
        ),
    )


def collect_gpu_class_rates(*, results_dir: Path) -> dict[str, GpuClassRates]:
    rentals_by_class: dict[str, set[Rental]] = defaultdict(set)
    runs_by_class: dict[str, int] = defaultdict(int)
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        found = _rental_for_run(run_json=run_dir / RUN_METADATA_FILENAME)
        if found is None:
            continue
        label, rental = found
        rentals_by_class[label].add(rental)
        runs_by_class[label] += 1
    return {
        label: GpuClassRates(
            rentals=sorted(rentals, key=lambda rental: rental.hourly_rate_usd),
            run_count=runs_by_class[label],
        )
        for label, rentals in rentals_by_class.items()
    }


def _report(*, rates_by_class: dict[str, GpuClassRates]) -> bool:
    matches_frozen = True
    for label, rates in sorted(rates_by_class.items()):
        observed = [rental.hourly_rate_usd for rental in rates.rentals]
        reference = rates.reference_hourly_rate_usd
        frozen = GPU_REFERENCE_HOURLY_RATE_USD.get(label)
        if frozen is None:
            status = "MISSING from GPU_REFERENCE_HOURLY_RATE_USD"
            matches_frozen = False
        elif frozen != reference:
            status = f"STALE: frozen ${frozen:.2f}/h"
            matches_frozen = False
        else:
            status = "matches frozen table"
        print(f"{label}: ${reference:.2f}/h  ({status})")
        print(f"  {rates.run_count} runs across {len(rates.rentals)} rentals, ", end="")
        print(f"paid ${min(observed):.2f}/h-${max(observed):.2f}/h")
        for rental in rates.rentals:
            print(
                f"    ${rental.hourly_rate_usd:.4f}/h  "
                f"{rental.provider or 'unknown'} instance {rental.instance_id or 'unknown'}"
            )
    return matches_frozen


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SUBMITTED_RESULTS_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when the frozen table does not match the computed rates",
    )
    args = parser.parse_args(argv)
    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    rates_by_class = collect_gpu_class_rates(results_dir=args.results_dir)
    if len(rates_by_class) == 0:
        print(f"no runs with a recorded hourly rate under {args.results_dir}", file=sys.stderr)
        return 1
    matches_frozen = _report(rates_by_class=rates_by_class)
    if args.check and not matches_frozen:
        print(
            "\nfrozen rates are out of date; update GPU_REFERENCE_HOURLY_RATE_USD",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

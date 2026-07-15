"""GPU classes and the reference hourly rates the leaderboard prices them at.

`gpu_label` folds the raw driver-reported GPU name recorded in a run (`NVIDIA H100 80GB HBM3`)
into a canonical class label (`H100 80GB`). Runs are compared by class, not by the exact board
string the driver happened to report.

Each class has a fixed reference rate. A self-hosted run records the real rate its machine was
rented at (`machine.hourly_rate_usd` in run.json) and its actual cost is computed from that
rate, but spot prices move between rentals: the same H100 class cost between $1.60/h and
$2.54/h across the rentals behind this leaderboard. Pricing each model against whatever the
market charged the hour it happened to run makes actual cost useless for comparing models, so
the leaderboard prices every machine-time run at its class's reference rate instead.

A reference rate is the mean of what we actually paid for that class, averaged over distinct
rentals so one long-lived instance serving many runs does not dominate the mean. The rates are
frozen constants rather than a build-time average, which would let each new submission silently
re-price every run already on the board. Re-derive them with
`uv run python tools/compute_gpu_rates.py` and update this table deliberately when the rentals
behind `results/` change enough to matter, moving `GPU_REFERENCE_RATES_AS_OF` in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass

GPU_NAME_PREFIXES_TO_STRIP: tuple[str, ...] = ("NVIDIA ", "GeForce ")


@dataclass(frozen=True, slots=True)
class GpuLabelPattern:
    substring: str
    label: str


GPU_LABEL_PATTERNS: tuple[GpuLabelPattern, ...] = (
    GpuLabelPattern(substring="B300", label="B300 288GB"),
    GpuLabelPattern(substring="H200", label="H200 141GB"),
    GpuLabelPattern(substring="H100", label="H100 80GB"),
    GpuLabelPattern(substring="A100", label="A100 80GB"),
    GpuLabelPattern(substring="RTX 4090", label="RTX 4090"),
)

# The date the rates below were last computed from the rentals behind `results/`. Update it in
# the same commit as any rate change: it is published on the leaderboard so readers can tell how
# old the prices they are being ranked by are.
GPU_REFERENCE_RATES_AS_OF: str = "2026-07-14"

GPU_REFERENCE_HOURLY_RATE_USD: dict[str, float] = {
    "A100 80GB": 1.09,
    "H100 80GB": 2.26,
    "H200 141GB": 3.66,
    "B300 288GB": 6.72,
}


def gpu_label(*, name: str) -> str:
    for pattern in GPU_LABEL_PATTERNS:
        if pattern.substring in name:
            return pattern.label
    label = name
    for prefix in GPU_NAME_PREFIXES_TO_STRIP:
        label = label.removeprefix(prefix)
    return label


def reference_hourly_rate_usd(*, gpu_label: str | None) -> float | None:
    if gpu_label is None:
        return None
    return GPU_REFERENCE_HOURLY_RATE_USD.get(gpu_label)

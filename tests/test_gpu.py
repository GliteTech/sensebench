from __future__ import annotations

from sensebench.leaderboard.gpu import (
    GPU_REFERENCE_HOURLY_RATE_USD,
    gpu_label,
    reference_hourly_rate_usd,
)


def test_gpu_label_folds_driver_names_into_classes() -> None:
    assert gpu_label(name="NVIDIA H100 80GB HBM3") == "H100 80GB"
    assert gpu_label(name="NVIDIA H200") == "H200 141GB"
    assert gpu_label(name="NVIDIA A100-SXM4-80GB") == "A100 80GB"
    assert gpu_label(name="NVIDIA B300 SXM6 AC") == "B300 288GB"


def test_gpu_label_strips_vendor_prefix_from_unmatched_names() -> None:
    assert gpu_label(name="NVIDIA L40S") == "L40S", "unknown boards keep their driver name"


def test_every_reference_rate_is_keyed_by_a_gpu_label() -> None:
    for label in GPU_REFERENCE_HOURLY_RATE_USD:
        assert gpu_label(name=label) == label, (
            f"reference rate key {label!r} is not a label gpu_label() can produce, "
            "so no run will ever match it"
        )


def test_reference_hourly_rate_is_known_for_benchmarked_classes() -> None:
    assert reference_hourly_rate_usd(gpu_label="H100 80GB") == 2.26


def test_reference_hourly_rate_is_absent_for_unpriced_classes() -> None:
    assert reference_hourly_rate_usd(gpu_label="L40S") is None
    assert reference_hourly_rate_usd(gpu_label=None) is None

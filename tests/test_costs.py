from __future__ import annotations

from pytest import approx

from sensebench.runner.costs import sum_costs
from sensebench.runs.models import CostBreakdown, CostSourceKind


def test_sum_costs_preserves_shared_unit_prices() -> None:
    cost = sum_costs(
        costs=[
            CostBreakdown(
                total_usd=0.10,
                input_uncached_usd=0.03,
                output_usd=0.07,
                input_uncached_unit_price_usd=0.001,
                output_unit_price_usd=0.002,
                source=CostSourceKind.LITELLM_ESTIMATE,
            ),
            CostBreakdown(
                total_usd=0.20,
                input_uncached_usd=0.06,
                output_usd=0.14,
                input_uncached_unit_price_usd=0.001,
                output_unit_price_usd=0.002,
                source=CostSourceKind.LITELLM_ESTIMATE,
            ),
        ],
    )

    assert cost.total_usd == approx(0.30), "total cost matches summed input totals"
    assert cost.input_uncached_usd == approx(0.09), "uncached input cost matches summed inputs"
    assert cost.output_usd == approx(0.21), "output cost matches summed outputs"
    assert cost.input_uncached_unit_price_usd == 0.001, "shared uncached input price is preserved"
    assert cost.output_unit_price_usd == 0.002, "shared output price is preserved"
    assert cost.source == CostSourceKind.LITELLM_ESTIMATE, "shared cost source is preserved"


def test_sum_costs_preserves_provider_reported_source() -> None:
    cost = sum_costs(
        costs=[
            CostBreakdown(total_usd=0.10, source=CostSourceKind.PROVIDER_REPORTED),
            CostBreakdown(total_usd=0.00, source=CostSourceKind.NO_CALLS),
        ],
    )

    assert cost.total_usd == approx(0.10), "total cost includes provider-reported calls"
    assert cost.source == CostSourceKind.PROVIDER_REPORTED, "provider source is preserved"

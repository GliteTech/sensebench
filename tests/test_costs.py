from __future__ import annotations

import pytest

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

    assert cost.total_usd == pytest.approx(0.30)
    assert cost.input_uncached_usd == pytest.approx(0.09)
    assert cost.output_usd == pytest.approx(0.21)
    assert cost.input_uncached_unit_price_usd == 0.001
    assert cost.output_unit_price_usd == 0.002
    assert cost.source == CostSourceKind.LITELLM_ESTIMATE

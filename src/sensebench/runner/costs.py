"""Cost helpers for run artifacts."""

from __future__ import annotations

from sensebench.runs.models import CostBreakdown, CostSourceKind

SECONDS_PER_HOUR: float = 3600.0


def unavailable_cost() -> CostBreakdown:
    return CostBreakdown(source=CostSourceKind.UNAVAILABLE)


def machine_time_cost(*, benchmark_seconds: float, hourly_rate_usd: float) -> CostBreakdown:
    return CostBreakdown(
        total_usd=benchmark_seconds * hourly_rate_usd / SECONDS_PER_HOUR,
        source=CostSourceKind.MACHINE_TIME_ESTIMATE,
    )


def no_call_cost() -> CostBreakdown:
    return CostBreakdown(
        total_usd=0.0,
        input_uncached_usd=0.0,
        input_cached_usd=0.0,
        output_usd=0.0,
        source=CostSourceKind.NO_CALLS,
    )


def _sum_optional_floats(*, values: list[float | None]) -> float | None:
    observed_values: list[float] = [value for value in values if value is not None]
    if len(observed_values) == 0:
        return None
    return sum(observed_values)


def _shared_unit_price(*, values: list[float | None]) -> float | None:
    observed_values: list[float] = [value for value in values if value is not None]
    if len(observed_values) == 0:
        return None
    first_value = observed_values[0]
    if all(value == first_value for value in observed_values):
        return first_value
    return None


def _combined_cost_source(*, costs: list[CostBreakdown]) -> CostSourceKind:
    if len(costs) == 0:
        return CostSourceKind.NO_CALLS
    sources: set[CostSourceKind] = {cost.source for cost in costs}
    if sources == {CostSourceKind.NO_CALLS}:
        return CostSourceKind.NO_CALLS
    if CostSourceKind.PROVIDER_REPORTED in sources:
        return CostSourceKind.PROVIDER_REPORTED
    if CostSourceKind.LITELLM_ESTIMATE in sources:
        return CostSourceKind.LITELLM_ESTIMATE
    return CostSourceKind.UNAVAILABLE


def sum_costs(*, costs: list[CostBreakdown]) -> CostBreakdown:
    total_usd = _sum_optional_floats(values=[cost.total_usd for cost in costs])
    input_uncached_usd = _sum_optional_floats(
        values=[cost.input_uncached_usd for cost in costs],
    )
    input_cached_usd = _sum_optional_floats(values=[cost.input_cached_usd for cost in costs])
    output_usd = _sum_optional_floats(values=[cost.output_usd for cost in costs])
    return CostBreakdown(
        total_usd=total_usd,
        input_uncached_usd=input_uncached_usd,
        input_cached_usd=input_cached_usd,
        output_usd=output_usd,
        input_uncached_unit_price_usd=_shared_unit_price(
            values=[cost.input_uncached_unit_price_usd for cost in costs],
        ),
        input_cached_unit_price_usd=_shared_unit_price(
            values=[cost.input_cached_unit_price_usd for cost in costs],
        ),
        output_unit_price_usd=_shared_unit_price(
            values=[cost.output_unit_price_usd for cost in costs],
        ),
        source=_combined_cost_source(costs=costs),
    )

"""Continuous linear optimizer for the 24-hour GridWise schedule.

This module deliberately accepts either dictionaries or Pydantic-style objects so
it can be integrated with the shared API schemas without coupling the numerical
engine to a particular Pydantic version.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Mapping, Sequence

import pulp


HOURS = tuple(range(24))
OUTPUT_EPSILON = 1e-7


class OptimizationError(RuntimeError):
    """Raised when an optimization request cannot produce an optimal schedule."""

    def __init__(self, message: str, *, solver_status: str | None = None) -> None:
        super().__init__(message)
        self.solver_status = solver_status


@dataclass(frozen=True)
class _CompiledDirectives:
    solar_factor: tuple[float, ...]
    minimum_energy: tuple[float, ...]
    no_charge: frozenset[int]
    no_discharge: frozenset[int]
    max_grid: tuple[float | None, ...]


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> str:
    """Return a stable string for either a plain string or a string-valued Enum."""

    return str(getattr(value, "value", value))


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OptimizationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OptimizationError(f"{label} must be a finite number") from exc
    if not isfinite(result):
        raise OptimizationError(f"{label} must be a finite number")
    return result


def _directive_hours(adjustment: Any, directive_type: str) -> tuple[int, ...]:
    raw_hours = _field(adjustment, "hours")
    if not isinstance(raw_hours, Sequence) or isinstance(raw_hours, (str, bytes)):
        raise OptimizationError(f"{directive_type}.hours must be a sequence")
    hours: list[int] = []
    for raw_hour in raw_hours:
        if isinstance(raw_hour, bool) or not isinstance(raw_hour, int):
            raise OptimizationError(f"{directive_type}.hours must contain integers")
        if raw_hour not in HOURS:
            raise OptimizationError(f"{directive_type}.hours contains {raw_hour}, outside 0..23")
        hours.append(raw_hour)
    if len(hours) != len(set(hours)):
        raise OptimizationError(f"{directive_type}.hours must be unique")
    return tuple(hours)


def _compile_directives(
    directives: Sequence[Any], base_minimum: float, capacity: float
) -> _CompiledDirectives:
    solar_factor = [1.0] * 24
    minimum_energy = [base_minimum] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    max_grid: list[float | None] = [None] * 24

    for directive in directives:
        directive_type = _enum_value(_field(directive, "directive_type"))
        applies = bool(_optional_field(directive, "applies", directive_type != "no_op"))
        if directive_type == "no_op":
            if applies:
                raise OptimizationError("no_op must use applies=false")
            continue
        if not applies:
            raise OptimizationError(f"{directive_type} must use applies=true")

        adjustment = _field(directive, "structured_adjustment")
        if adjustment is None:
            raise OptimizationError(f"{directive_type} requires structured_adjustment")
        hours = _directive_hours(adjustment, directive_type)

        if directive_type == "solar_reduction":
            factor = _number(_field(adjustment, "factor"), "solar_reduction.factor")
            if not 0.0 <= factor <= 1.0:
                raise OptimizationError("solar_reduction.factor must be between 0 and 1")
            for hour in hours:
                solar_factor[hour] = min(solar_factor[hour], factor)
        elif directive_type == "minimum_battery_reserve":
            reserve = _number(
                _field(adjustment, "minimum_energy_kwh"),
                "minimum_battery_reserve.minimum_energy_kwh",
            )
            if not 0.0 <= reserve <= capacity:
                raise OptimizationError(
                    "minimum_battery_reserve.minimum_energy_kwh must be within capacity"
                )
            for hour in hours:
                minimum_energy[hour] = max(minimum_energy[hour], reserve)
        elif directive_type == "no_charge_window":
            no_charge.update(hours)
        elif directive_type == "no_discharge_window":
            no_discharge.update(hours)
        elif directive_type == "max_grid_window":
            cap = _number(_field(adjustment, "max_grid_kwh"), "max_grid_window.max_grid_kwh")
            if cap < 0.0:
                raise OptimizationError("max_grid_window.max_grid_kwh must be non-negative")
            for hour in hours:
                max_grid[hour] = cap if max_grid[hour] is None else min(max_grid[hour], cap)
        else:
            raise OptimizationError(f"Unsupported directive type: {directive_type}")

    return _CompiledDirectives(
        solar_factor=tuple(solar_factor),
        minimum_energy=tuple(minimum_energy),
        no_charge=frozenset(no_charge),
        no_discharge=frozenset(no_discharge),
        max_grid=tuple(max_grid),
    )


def _normalize_request(request: Any) -> tuple[list[Any], Any]:
    raw_hours = list(_field(request, "hours"))
    if len(raw_hours) != 24:
        raise OptimizationError("Exactly 24 hourly entries are required")

    by_hour: dict[int, Any] = {}
    for row in raw_hours:
        hour = _field(row, "hour")
        if isinstance(hour, bool) or not isinstance(hour, int) or hour not in HOURS:
            raise OptimizationError("Hour indices must be integers from 0 through 23")
        if hour in by_hour:
            raise OptimizationError(f"Duplicate hour: {hour}")
        by_hour[hour] = row
    if set(by_hour) != set(HOURS):
        raise OptimizationError("Hourly entries must cover every hour from 0 through 23")
    return [by_hour[hour] for hour in HOURS], _field(request, "battery")


def _clean(value: float) -> float:
    return 0.0 if abs(value) <= OUTPUT_EPSILON else float(value)


def solve(request: Any, directives: Sequence[Any]) -> dict[str, Any]:
    """Solve a GridWise request and return rows plus internal diagnostics.

    Args:
        request: Mapping or object with ``hours`` and ``battery`` fields.
        directives: Validated directive-interpretation mappings or objects.

    Returns:
        A dictionary containing ``hourly_plan``, aggregate totals, solver status,
        and measured solver time.

    Raises:
        OptimizationError: If inputs are invalid or CBC does not find an optimum.
    """

    hours, battery = _normalize_request(request)
    capacity = _number(_field(battery, "capacity_kwh"), "battery.capacity_kwh")
    initial = _number(_field(battery, "initial_energy_kwh"), "battery.initial_energy_kwh")
    base_minimum = _number(_field(battery, "minimum_energy_kwh"), "battery.minimum_energy_kwh")
    max_charge = _number(
        _field(battery, "max_charge_kwh_per_hour"), "battery.max_charge_kwh_per_hour"
    )
    max_discharge = _number(
        _field(battery, "max_discharge_kwh_per_hour"),
        "battery.max_discharge_kwh_per_hour",
    )
    if min(capacity, initial, base_minimum, max_charge, max_discharge) < 0.0:
        raise OptimizationError("Battery values must be non-negative")
    if not base_minimum <= initial <= capacity:
        raise OptimizationError("Battery must satisfy minimum <= initial <= capacity")

    demand: list[float] = []
    solar: list[float] = []
    tariff: list[float] = []
    for hour, row in enumerate(hours):
        demand.append(_number(_field(row, "demand_kwh"), f"hours[{hour}].demand_kwh"))
        solar.append(_number(_field(row, "solar_kwh"), f"hours[{hour}].solar_kwh"))
        tariff.append(
            _number(_field(row, "tariff_bdt_per_kwh"), f"hours[{hour}].tariff_bdt_per_kwh")
        )
    if min(demand + solar + tariff) < 0.0:
        raise OptimizationError("Demand, solar, and tariff values must be non-negative")

    compiled = _compile_directives(directives, base_minimum, capacity)
    model = pulp.LpProblem("gridwise_24_hour_schedule", pulp.LpMinimize)
    hour_indices = list(HOURS)
    grid = pulp.LpVariable.dicts("grid", hour_indices, lowBound=0.0, cat=pulp.LpContinuous)
    solar_used = pulp.LpVariable.dicts(
        "solar_used", hour_indices, lowBound=0.0, cat=pulp.LpContinuous
    )
    delta = pulp.LpVariable.dicts(
        "battery_delta",
        hour_indices,
        lowBound=-max_discharge,
        upBound=max_charge,
        cat=pulp.LpContinuous,
    )
    energy = pulp.LpVariable.dicts("battery_energy", hour_indices, cat=pulp.LpContinuous)

    model += pulp.lpSum(grid[hour] * tariff[hour] for hour in HOURS), "total_grid_cost"
    for hour in HOURS:
        model += solar_used[hour] <= solar[hour] * compiled.solar_factor[hour]
        model += grid[hour] + solar_used[hour] == demand[hour] + delta[hour]
        previous_energy: Any = initial if hour == 0 else energy[hour - 1]
        model += energy[hour] == previous_energy + delta[hour]
        model += energy[hour] >= compiled.minimum_energy[hour]
        model += energy[hour] <= capacity
        if hour in compiled.no_charge:
            model += delta[hour] <= 0.0
        if hour in compiled.no_discharge:
            model += delta[hour] >= 0.0
        if compiled.max_grid[hour] is not None:
            model += grid[hour] <= compiled.max_grid[hour]
    model += energy[23] == initial, "end_of_day_battery_neutrality"

    started = perf_counter()
    solver_error: Exception | None = None
    for _attempt in range(2):
        try:
            status_code = model.solve(
                pulp.PULP_CBC_CMD(msg=False, mip=False, threads=1)
            )
            break
        except (pulp.PulpSolverError, OSError, IndexError) as exc:
            # PuLP/CBC can very rarely leave a truncated temporary solution file
            # on Windows. Retry the identical model once; never alter constraints.
            solver_error = exc
    else:
        raise OptimizationError("CBC could not execute the optimization model") from solver_error
    solve_time_seconds = perf_counter() - started
    solver_status = pulp.LpStatus.get(status_code, str(status_code))
    if solver_status != "Optimal":
        raise OptimizationError(
            f"CBC did not find an optimal schedule (status: {solver_status})",
            solver_status=solver_status,
        )

    hourly_plan: list[dict[str, Any]] = []
    for hour in HOURS:
        grid_value = _clean(float(pulp.value(grid[hour])))
        solar_value = _clean(float(pulp.value(solar_used[hour])))
        delta_value = _clean(float(pulp.value(delta[hour])))
        energy_value = _clean(float(pulp.value(energy[hour])))
        if delta_value > OUTPUT_EPSILON:
            action = "charge"
            battery_kwh = delta_value
        elif delta_value < -OUTPUT_EPSILON:
            action = "discharge"
            battery_kwh = -delta_value
        else:
            action = "idle"
            battery_kwh = 0.0
        hourly_plan.append(
            {
                "hour": hour,
                "grid_kwh": grid_value,
                "solar_used_kwh": solar_value,
                "battery_action": action,
                "battery_kwh": battery_kwh,
                "battery_energy_after_kwh": energy_value,
            }
        )

    total_grid = sum(row["grid_kwh"] for row in hourly_plan)
    total_cost = sum(hourly_plan[hour]["grid_kwh"] * tariff[hour] for hour in HOURS)
    peak_grid = max(row["grid_kwh"] for row in hourly_plan)
    return {
        "solver_status": solver_status,
        "hourly_plan": hourly_plan,
        "total_grid_kwh": _clean(total_grid),
        "total_cost_bdt": _clean(total_cost),
        "peak_grid_kwh": _clean(peak_grid),
        "solve_time_seconds": solve_time_seconds,
    }


def optimize(request: Any, directives: Sequence[Any]) -> list[Any]:
    """Return the hourly rows expected by the shared API integration contract.

    In the integrated application, rows are validated into Member 1's frozen
    ``HourlyPlan`` model. The lightweight dictionary fallback keeps this module
    independently runnable on the Member 3 branch before the shared schemas are
    merged.
    """

    rows = solve(request, directives)["hourly_plan"]
    try:
        from app.schemas import HourlyPlan
    except ModuleNotFoundError as exc:
        if exc.name != "app.schemas":
            raise
        return rows
    return [HourlyPlan.model_validate(row) for row in rows]


__all__ = ["OptimizationError", "optimize", "solve"]

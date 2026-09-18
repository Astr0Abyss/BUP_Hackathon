"""Independent replay of the public schedule against the original inputs.

No optimizer/compiler state is imported. Revalidation deliberately passes through
JSON, so mutated or model_construct-created Pydantic instances are not trusted.
"""

from collections.abc import Sequence
from math import fsum, isfinite

from app.errors import ServiceError
from app.schemas import DirectiveInterpretation, OptimizeRequest, OptimizeResponse

TOLERANCE = 0.01


class ReplayValidationError(ServiceError):
    code = "replay_validation_failed"
    public_message = "The generated schedule failed validation."


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReplayValidationError(reason)


def _close(actual: float, expected: float, reason: str) -> None:
    _require(
        isfinite(actual) and isfinite(expected)
        and abs(actual - expected) <= TOLERANCE,
        reason,
    )


def validate_response(
    request: OptimizeRequest,
    response: OptimizeResponse,
    directives: Sequence[DirectiveInterpretation],
) -> None:
    """Raise ReplayValidationError on any structural or numerical violation."""
    try:
        _replay(request, response, directives)
    except ReplayValidationError:
        raise
    except Exception as exc:
        # All malformed structures, serialization errors and overflow are failures.
        raise ReplayValidationError("Invalid replay inputs.") from exc


def _replay(request, response, directives) -> None:
    response = OptimizeResponse.model_validate_json(response.model_dump_json())
    directives = [
        DirectiveInterpretation.model_validate_json(item.model_dump_json())
        for item in directives
    ]
    _require(response.scenario_id == request.scenario_id, "Scenario mismatch.")
    expected_indices = list(range(len(request.operator_notes)))
    _require([d.note_index for d in directives] == expected_indices, "Invalid directive mapping.")
    _require(
        [d.note_index for d in response.directive_interpretation] == expected_indices,
        "Invalid returned note mapping.",
    )
    _require(
        response.directive_interpretation == directives,
        "Returned directives differ from validated directives.",
    )
    hours = {row.hour: row for row in request.hours}
    battery = request.battery
    solar = [hours[h].solar_kwh for h in range(24)]
    reserve = [battery.minimum_energy_kwh] * 24
    caps = [float("inf")] * 24
    no_charge, no_discharge = set(), set()
    for directive in directives:
        kind = directive.directive_type
        adjustment = directive.structured_adjustment
        if kind == "no_op":
            continue
        for h in adjustment.hours:
            if kind == "solar_reduction":
                # Each reduction is relative to the original forecast. Overlaps
                # must satisfy every applicable upper bound, not compound factors.
                solar[h] = min(solar[h], hours[h].solar_kwh * adjustment.factor)
            elif kind == "minimum_battery_reserve":
                _require(adjustment.minimum_energy_kwh <= battery.capacity_kwh, "Reserve exceeds capacity.")
                reserve[h] = max(reserve[h], adjustment.minimum_energy_kwh)
            elif kind == "no_charge_window":
                no_charge.add(h)
            elif kind == "no_discharge_window":
                no_discharge.add(h)
            elif kind == "max_grid_window":
                caps[h] = min(caps[h], adjustment.max_grid_kwh)

    energy = battery.initial_energy_kwh
    for row in response.hourly_plan:
        h = row.hour
        charge = row.battery_kwh if row.battery_action == "charge" else 0.0
        discharge = row.battery_kwh if row.battery_action == "discharge" else 0.0
        _require(charge <= battery.max_charge_kwh_per_hour + TOLERANCE, "Charge rate exceeded.")
        _require(discharge <= battery.max_discharge_kwh_per_hour + TOLERANCE, "Discharge rate exceeded.")
        _require(h not in no_charge or charge <= TOLERANCE, "Charging forbidden.")
        _require(h not in no_discharge or discharge <= TOLERANCE, "Discharging forbidden.")
        energy += charge - discharge
        _close(row.battery_energy_after_kwh, energy, "Battery transition mismatch.")
        # Check both independently replayed and published states at the bounds.
        for state in (energy, row.battery_energy_after_kwh):
            _require(reserve[h] - TOLERANCE <= state <= battery.capacity_kwh + TOLERANCE, "Battery bounds violated.")
        _require(row.solar_used_kwh <= solar[h] + TOLERANCE, "Solar exceeded.")
        _require(row.grid_kwh <= caps[h] + TOLERANCE, "Grid cap exceeded.")
        _close(row.grid_kwh + row.solar_used_kwh + discharge,
               hours[h].demand_kwh + charge, "Energy balance violated.")

    _close(energy, battery.initial_energy_kwh, "Final battery neutrality violated.")
    _close(response.hourly_plan[-1].battery_energy_after_kwh,
           battery.initial_energy_kwh, "Published final battery neutrality violated.")
    _close(response.total_grid_kwh, fsum(r.grid_kwh for r in response.hourly_plan), "Grid total mismatch.")
    _close(response.total_cost_bdt,
           fsum(r.grid_kwh * hours[r.hour].tariff_bdt_per_kwh for r in response.hourly_plan),
           "Cost total mismatch.")
    _close(response.peak_grid_kwh, max(r.grid_kwh for r in response.hourly_plan), "Peak mismatch.")

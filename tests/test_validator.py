import json
from pathlib import Path

import pytest

from app.response import build_response
from app.schemas import DirectiveInterpretation, HourlyPlan, OptimizeRequest, OptimizeResponse
from app.validator import ReplayValidationError, validate_response


def fixture_plan(actions=None, directive_type="no_op", adjustment=None):
    request = OptimizeRequest.model_validate({
        "scenario_id": "REPLAY-TEST", "operator_notes": ["Synthetic test note."],
        "hours": [{"hour": h, "demand_kwh": 100, "solar_kwh": 20,
                   "tariff_bdt_per_kwh": h + 1} for h in range(24)],
        "battery": {"capacity_kwh": 100, "initial_energy_kwh": 50,
                    "minimum_energy_kwh": 20, "max_charge_kwh_per_hour": 20,
                    "max_discharge_kwh_per_hour": 20},
    })
    directives = [DirectiveInterpretation.model_validate({
        "note_index": 0, "applies": directive_type != "no_op",
        "directive_type": directive_type, "structured_adjustment": adjustment,
        "explanation": "Synthetic test directive.",
    })]
    energy, rows = 50, []
    for h in range(24):
        delta = (actions or {}).get(h, 0)
        energy += delta
        rows.append(HourlyPlan(
            hour=h, grid_kwh=90 + delta, solar_used_kwh=10,
            battery_action="charge" if delta > 0 else "discharge" if delta < 0 else "idle",
            battery_kwh=abs(delta), battery_energy_after_kwh=energy,
        ))
    return request, build_response(request, directives, rows), directives


def test_valid_schedule_and_unordered_request():
    request, response, directives = fixture_plan({0: 10, 1: -10})
    request.hours.reverse()
    validate_response(request, response, directives)


@pytest.mark.parametrize("case", [
    "scenario", "23_rows", "duplicate_hour", "negative_grid", "solar",
    "transition", "charge_rate", "discharge_rate", "no_charge", "no_discharge",
    "reserve", "capacity", "grid_cap", "balance", "neutrality",
    "grid_total", "cost_total", "peak", "idle_magnitude",
])
def test_all_required_invalid_plans_are_rejected(case):
    actions = {
        "charge_rate": {0: 21, 1: -21},
        "discharge_rate": {0: -21, 1: 21},
        "no_charge": {0: 10, 1: -10},
        "no_discharge": {0: 10, 1: -10},
        "capacity": {0: 20, 1: 20, 2: 20, 3: -20, 4: -20, 5: -20},
        "neutrality": {23: 10},
    }.get(case)
    kind, adjustment = {
        "solar": ("solar_reduction", {"hours": [0], "factor": 0.25}),
        "reserve": ("minimum_battery_reserve", {"hours": [0], "minimum_energy_kwh": 60}),
        "grid_cap": ("max_grid_window", {"hours": [0], "max_grid_kwh": 80}),
        "no_charge": ("no_charge_window", {"hours": [0]}),
        "no_discharge": ("no_discharge_window", {"hours": [1]}),
    }.get(case, ("no_op", None))
    request, response, directives = fixture_plan(actions, kind, adjustment)
    # Mutating already-created instances exercises replay, not constructor checks.
    if case == "scenario": response.scenario_id = "WRONG"
    if case == "23_rows": response.hourly_plan.pop()
    if case == "duplicate_hour": response.hourly_plan[1].hour = 0
    if case == "negative_grid": response.hourly_plan[0].grid_kwh = -1
    if case == "transition": response.hourly_plan[0].battery_energy_after_kwh += 1
    if case == "balance": response.hourly_plan[0].grid_kwh += 1
    if case == "grid_total": response.total_grid_kwh += 1
    if case == "cost_total": response.total_cost_bdt += 1
    if case == "peak": response.peak_grid_kwh += 1
    if case == "idle_magnitude": response.hourly_plan[0].battery_kwh = 1
    with pytest.raises(ReplayValidationError):
        validate_response(request, response, directives)


@pytest.mark.parametrize("field", ["grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_public_numbers(field, value):
    request, response, directives = fixture_plan()
    setattr(response.hourly_plan[0], field, value)
    with pytest.raises(ReplayValidationError):
        validate_response(request, response, directives)


@pytest.mark.parametrize("mutation", ["missing", "index", "changed", "action"])
def test_mapping_and_action_integrity(mutation):
    request, response, directives = fixture_plan()
    response = response.model_copy(deep=True)
    if mutation == "missing": response.directive_interpretation.clear()
    if mutation == "index": response.directive_interpretation[0].note_index = 1
    if mutation == "changed": response.directive_interpretation[0].explanation = "Altered"
    if mutation == "action": response.hourly_plan[0].battery_action = "export"
    with pytest.raises(ReplayValidationError):
        validate_response(request, response, directives)


def test_absolute_tolerance_and_accumulated_state():
    request, response, directives = fixture_plan()
    response.total_cost_bdt += 0.005
    validate_response(request, response, directives)
    # Small per-hour falsifications cannot accumulate by replaying reported state.
    for h, row in enumerate(response.hourly_plan):
        row.battery_energy_after_kwh += (h + 1) * 0.005
    with pytest.raises(ReplayValidationError):
        validate_response(request, response, directives)


PUBLIC_CASES = json.loads((Path(__file__).resolve().parents[1] /
    "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", PUBLIC_CASES, ids=lambda case: case["id"])
def test_official_reference_replay(case):
    request = OptimizeRequest.model_validate(case["input"])
    response = OptimizeResponse.model_validate(case["expected_output"])
    validate_response(request, response, response.directive_interpretation)


def test_builder_orders_rows_and_recalculates_using_hour_keys():
    request, response, directives = fixture_plan({0: 10, 23: -10})
    request.hours.reverse()
    rebuilt = build_response(request, directives, list(reversed(response.hourly_plan)))
    assert rebuilt.total_cost_bdt == sum(row.grid_kwh * (row.hour + 1) for row in rebuilt.hourly_plan)
    assert rebuilt.total_grid_kwh == sum(row.grid_kwh for row in rebuilt.hourly_plan)
    assert rebuilt.peak_grid_kwh == 100
    validate_response(request, rebuilt, directives)


@pytest.mark.parametrize("difference,accepted", [(0.009, True), (0.011, False)])
def test_tolerance_boundary(difference, accepted):
    request, response, directives = fixture_plan()
    response.total_cost_bdt += difference
    if accepted:
        validate_response(request, response, directives)
    else:
        with pytest.raises(ReplayValidationError):
            validate_response(request, response, directives)


@pytest.mark.parametrize("field", ["total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_totals(field, value):
    request, response, directives = fixture_plan()
    setattr(response, field, value)
    with pytest.raises(ReplayValidationError):
        validate_response(request, response, directives)


def test_base_reserve_is_enforced():
    request, response, directives = fixture_plan({0: -20, 1: -20, 2: 20, 3: 20})
    with pytest.raises(ReplayValidationError, match="Battery bounds"):
        validate_response(request, response, directives)

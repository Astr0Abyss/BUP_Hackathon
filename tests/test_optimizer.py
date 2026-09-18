from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pulp
import pytest

from app.optimizer import OptimizationError, optimize, solve


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
TOLERANCE = 0.01


def _samples() -> list[dict]:
    with SAMPLES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def _compiled_expectations(case: dict) -> tuple[list[float], list[float], set[int], set[int], list]:
    request = case["input"]
    battery = request["battery"]
    factors = [1.0] * 24
    reserves = [float(battery["minimum_energy_kwh"])] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    grid_caps: list[float | None] = [None] * 24
    for directive in case["expected_output"]["directive_interpretation"]:
        kind = directive["directive_type"]
        if kind == "no_op":
            continue
        adjustment = directive["structured_adjustment"]
        hours = adjustment["hours"]
        if kind == "solar_reduction":
            for hour in hours:
                factors[hour] *= adjustment["factor"]
        elif kind == "minimum_battery_reserve":
            for hour in hours:
                reserves[hour] = max(reserves[hour], adjustment["minimum_energy_kwh"])
        elif kind == "no_charge_window":
            no_charge.update(hours)
        elif kind == "no_discharge_window":
            no_discharge.update(hours)
        elif kind == "max_grid_window":
            for hour in hours:
                cap = adjustment["max_grid_kwh"]
                grid_caps[hour] = cap if grid_caps[hour] is None else min(grid_caps[hour], cap)
    return factors, reserves, no_charge, no_discharge, grid_caps


def _validity_errors(case: dict, result: dict) -> list[str]:
    request = case["input"]
    by_hour = {row["hour"]: row for row in request["hours"]}
    battery = request["battery"]
    factors, reserves, no_charge, no_discharge, grid_caps = _compiled_expectations(case)
    plan = result["hourly_plan"]
    errors: list[str] = []
    if len(plan) != 24 or [row["hour"] for row in plan] != list(range(24)):
        errors.append("plan does not contain ordered hours 0..23")
        return errors

    previous_energy = float(battery["initial_energy_kwh"])
    for row in plan:
        hour = row["hour"]
        source = by_hour[hour]
        action = row["battery_action"]
        amount = row["battery_kwh"]
        signed_delta = amount if action == "charge" else -amount if action == "discharge" else 0.0
        expected_energy = previous_energy + signed_delta
        effective_solar = source["solar_kwh"] * factors[hour]

        if row["grid_kwh"] < -TOLERANCE:
            errors.append(f"hour {hour}: negative grid")
        if row["solar_used_kwh"] < -TOLERANCE or row["solar_used_kwh"] > effective_solar + TOLERANCE:
            errors.append(f"hour {hour}: solar limit")
        if action not in {"charge", "discharge", "idle"}:
            errors.append(f"hour {hour}: invalid action")
        if action == "idle" and abs(amount) > TOLERANCE:
            errors.append(f"hour {hour}: nonzero idle amount")
        if action == "charge" and amount > battery["max_charge_kwh_per_hour"] + TOLERANCE:
            errors.append(f"hour {hour}: charge rate")
        if action == "discharge" and amount > battery["max_discharge_kwh_per_hour"] + TOLERANCE:
            errors.append(f"hour {hour}: discharge rate")
        if abs(row["battery_energy_after_kwh"] - expected_energy) > TOLERANCE:
            errors.append(f"hour {hour}: battery transition")
        if row["battery_energy_after_kwh"] < reserves[hour] - TOLERANCE:
            errors.append(f"hour {hour}: reserve")
        if row["battery_energy_after_kwh"] > battery["capacity_kwh"] + TOLERANCE:
            errors.append(f"hour {hour}: capacity")
        if hour in no_charge and action == "charge" and amount > TOLERANCE:
            errors.append(f"hour {hour}: no-charge window")
        if hour in no_discharge and action == "discharge" and amount > TOLERANCE:
            errors.append(f"hour {hour}: no-discharge window")
        if grid_caps[hour] is not None and row["grid_kwh"] > grid_caps[hour] + TOLERANCE:
            errors.append(f"hour {hour}: grid cap")
        balance = row["grid_kwh"] + row["solar_used_kwh"] - source["demand_kwh"] - signed_delta
        if abs(balance) > TOLERANCE:
            errors.append(f"hour {hour}: energy balance")
        previous_energy = row["battery_energy_after_kwh"]

    if abs(previous_energy - battery["initial_energy_kwh"]) > TOLERANCE:
        errors.append("end-of-day neutrality")
    recomputed_grid = sum(row["grid_kwh"] for row in plan)
    recomputed_cost = sum(row["grid_kwh"] * by_hour[row["hour"]]["tariff_bdt_per_kwh"] for row in plan)
    recomputed_peak = max(row["grid_kwh"] for row in plan)
    if abs(result["total_grid_kwh"] - recomputed_grid) > TOLERANCE:
        errors.append("total grid")
    if abs(result["total_cost_bdt"] - recomputed_cost) > TOLERANCE:
        errors.append("total cost")
    if abs(result["peak_grid_kwh"] - recomputed_peak) > TOLERANCE:
        errors.append("peak grid")
    return errors


@pytest.mark.parametrize("case", _samples(), ids=lambda case: case["id"])
def test_official_public_sample(case: dict, request: pytest.FixtureRequest) -> None:
    directives = case["expected_output"]["directive_interpretation"]
    result = solve(case["input"], directives)
    reference_cost = case["expected_output"]["total_cost_bdt"]
    difference = abs(result["total_cost_bdt"] - reference_cost)
    errors = _validity_errors(case, result)
    request.node.user_properties.extend(
        [
            ("solver_status", result["solver_status"]),
            ("cost", result["total_cost_bdt"]),
            ("reference_cost", reference_cost),
            ("absolute_difference", difference),
            ("validity", "PASS" if not errors else "FAIL"),
            ("solve_time_seconds", result["solve_time_seconds"]),
        ]
    )
    print(
        f"{case['id']} | {result['solver_status']} | {result['total_cost_bdt']:.6f} | "
        f"{reference_cost:.6f} | {difference:.6f} | "
        f"{'PASS' if not errors else 'FAIL'} | {result['solve_time_seconds']:.6f}s"
    )
    assert result["solver_status"] == "Optimal"
    assert errors == []
    assert difference <= TOLERANCE


def _profile(value: float | list[float]) -> list[float]:
    if isinstance(value, list):
        assert len(value) == 24
        return [float(item) for item in value]
    return [float(value)] * 24


def _directive(kind: str, adjustment: dict, note_index: int = 0) -> dict:
    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": kind,
        "structured_adjustment": adjustment,
        "explanation": "Synthetic optimizer regression directive.",
    }


def _synthetic_case(
    case_id: str,
    *,
    demand: float | list[float] = 10.0,
    solar: float | list[float] = 0.0,
    tariff: float | list[float] = 1.0,
    battery: dict | None = None,
    directives: list[dict] | None = None,
) -> dict:
    demand_profile = _profile(demand)
    solar_profile = _profile(solar)
    tariff_profile = _profile(tariff)
    battery_values = {
        "capacity_kwh": 20.0,
        "initial_energy_kwh": 10.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 5.0,
        "max_discharge_kwh_per_hour": 5.0,
    }
    if battery:
        battery_values.update(battery)
    interpretations = directives or []
    return {
        "id": case_id,
        "input": {
            "scenario_id": case_id,
            "operator_notes": ["Synthetic optimizer-only test."],
            "hours": [
                {
                    "hour": hour,
                    "demand_kwh": demand_profile[hour],
                    "solar_kwh": solar_profile[hour],
                    "tariff_bdt_per_kwh": tariff_profile[hour],
                }
                for hour in range(24)
            ],
            "battery": battery_values,
        },
        "expected_output": {"directive_interpretation": interpretations},
    }


def _part2_cases() -> list[dict]:
    zero_battery = {
        "capacity_kwh": 0.0,
        "initial_energy_kwh": 0.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 0.0,
        "max_discharge_kwh_per_hour": 0.0,
    }
    cases: list[dict] = []

    # Combined hard directives.
    cases.append(
        _synthetic_case(
            "reserve-plus-grid-cap",
            directives=[
                _directive(
                    "minimum_battery_reserve",
                    {"hours": [18], "minimum_energy_kwh": 5.0},
                ),
                _directive("max_grid_window", {"hours": [18], "max_grid_kwh": 5.0}, 1),
            ],
        )
    )
    cases.append(
        _synthetic_case(
            "solar-reduction-plus-grid-cap",
            solar=10.0,
            battery=zero_battery,
            directives=[
                _directive("solar_reduction", {"hours": [12], "factor": 0.5}),
                _directive("max_grid_window", {"hours": [12], "max_grid_kwh": 5.0}, 1),
            ],
        )
    )
    cases.append(
        _synthetic_case(
            "no-charge-plus-reserve",
            battery={"initial_energy_kwh": 5.0},
            directives=[
                _directive("no_charge_window", {"hours": [5]}),
                _directive(
                    "minimum_battery_reserve", {"hours": [5], "minimum_energy_kwh": 5.0}, 1
                ),
            ],
        )
    )
    cases.append(
        _synthetic_case(
            "no-discharge-plus-grid-cap",
            directives=[
                _directive("no_discharge_window", {"hours": [6]}),
                _directive("max_grid_window", {"hours": [6], "max_grid_kwh": 10.0}, 1),
            ],
        )
    )
    cases.append(
        _synthetic_case(
            "simultaneous-no-charge-no-discharge",
            directives=[
                _directive("no_charge_window", {"hours": [7]}),
                _directive("no_discharge_window", {"hours": [7]}, 1),
            ],
        )
    )
    cases.append(
        _synthetic_case(
            "multiple-non-conflicting-directives",
            solar=10.0,
            directives=[
                _directive("solar_reduction", {"hours": [8], "factor": 0.5}),
                _directive(
                    "minimum_battery_reserve", {"hours": [9], "minimum_energy_kwh": 4.0}, 1
                ),
                _directive("no_charge_window", {"hours": [10]}, 2),
                _directive("no_discharge_window", {"hours": [11]}, 3),
                _directive("max_grid_window", {"hours": [12], "max_grid_kwh": 10.0}, 4),
            ],
        )
    )

    # Battery boundaries.
    cases.extend(
        [
            _synthetic_case(
                "initial-equals-minimum",
                battery={"capacity_kwh": 10.0, "initial_energy_kwh": 5.0, "minimum_energy_kwh": 5.0},
            ),
            _synthetic_case(
                "initial-equals-capacity",
                battery={"capacity_kwh": 10.0, "initial_energy_kwh": 10.0},
            ),
            _synthetic_case(
                "zero-max-charge",
                battery={"initial_energy_kwh": 0.0, "max_charge_kwh_per_hour": 0.0},
            ),
            _synthetic_case(
                "zero-max-discharge",
                battery={"max_discharge_kwh_per_hour": 0.0},
            ),
            _synthetic_case("zero-capacity", battery=zero_battery),
            _synthetic_case(
                "reserve-equals-capacity",
                battery={"capacity_kwh": 10.0, "initial_energy_kwh": 10.0},
                directives=[
                    _directive(
                        "minimum_battery_reserve",
                        {"hours": [8], "minimum_energy_kwh": 10.0},
                    )
                ],
            ),
            _synthetic_case(
                "directive-reserve-zero",
                directives=[
                    _directive(
                        "minimum_battery_reserve", {"hours": [8], "minimum_energy_kwh": 0.0}
                    )
                ],
            ),
        ]
    )

    # Solar limits, factors, surplus, and curtailment.
    cases.extend(
        [
            _synthetic_case("zero-solar", battery=zero_battery),
            _synthetic_case("solar-surplus", demand=1.0, solar=5.0, battery=zero_battery),
            _synthetic_case(
                "solar-factor-zero",
                demand=1.0,
                solar=5.0,
                battery=zero_battery,
                directives=[_directive("solar_reduction", {"hours": [5], "factor": 0.0})],
            ),
            _synthetic_case(
                "solar-factor-one",
                demand=1.0,
                solar=5.0,
                battery=zero_battery,
                directives=[_directive("solar_reduction", {"hours": [5], "factor": 1.0})],
            ),
            _synthetic_case("solar-curtailment", demand=0.0, solar=5.0, battery=zero_battery),
        ]
    )

    # Grid caps, including a cap that requires advance charging.
    grid_demand = [0.0] * 24
    grid_demand[10] = 10.0
    grid_tariff = [1.0] * 24
    grid_tariff[10] = 100.0
    cases.extend(
        [
            _synthetic_case(
                "zero-grid-cap-feasible",
                demand=5.0,
                solar=5.0,
                battery=zero_battery,
                directives=[_directive("max_grid_window", {"hours": [6], "max_grid_kwh": 0.0})],
            ),
            _synthetic_case(
                "tight-grid-cap-precharge",
                demand=grid_demand,
                tariff=grid_tariff,
                battery={
                    "capacity_kwh": 5.0,
                    "initial_energy_kwh": 0.0,
                    "minimum_energy_kwh": 0.0,
                    "max_charge_kwh_per_hour": 5.0,
                    "max_discharge_kwh_per_hour": 5.0,
                },
                directives=[
                    _directive("max_grid_window", {"hours": [10], "max_grid_kwh": 5.0})
                ],
            ),
        ]
    )

    # Tariff degeneracy and sharp price signals.
    sharp_tariff = [1.0] * 24
    sharp_tariff[12] = 100.0
    cheap_tariff = [10.0] * 24
    cheap_tariff[1] = 0.0
    cheap_tariff[2] = 100.0
    empty_battery = {
        "capacity_kwh": 5.0,
        "initial_energy_kwh": 0.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 5.0,
        "max_discharge_kwh_per_hour": 5.0,
    }
    cases.extend(
        [
            _synthetic_case("equal-tariffs", tariff=7.0),
            _synthetic_case("zero-tariffs", tariff=0.0),
            _synthetic_case("sharp-expensive-peak", tariff=sharp_tariff, battery=empty_battery),
            _synthetic_case("cheap-precharge-period", tariff=cheap_tariff, battery=empty_battery),
        ]
    )

    # End-of-day restoration, including a restriction at hour 23.
    last_hour_tariff = [50.0] * 24
    last_hour_tariff[0] = 100.0
    last_hour_tariff[23] = 0.0
    restricted_tariff = list(last_hour_tariff)
    restricted_tariff[22] = 0.0
    restore_battery = {
        "capacity_kwh": 5.0,
        "initial_energy_kwh": 5.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 5.0,
        "max_discharge_kwh_per_hour": 5.0,
    }
    cases.extend(
        [
            _synthetic_case(
                "last-hour-restoration", tariff=last_hour_tariff, battery=restore_battery
            ),
            _synthetic_case(
                "restriction-near-hour-23",
                tariff=restricted_tariff,
                battery=restore_battery,
                directives=[_directive("no_charge_window", {"hours": [23]})],
            ),
        ]
    )
    return cases


def test_part2_combined_and_numerical_edge_case_matrix() -> None:
    timings: list[float] = []
    for case in _part2_cases():
        directives = case["expected_output"]["directive_interpretation"]
        result = solve(case["input"], directives)
        errors = _validity_errors(case, result)
        assert errors == [], f"{case['id']}: {errors}"
        assert result["solver_status"] == "Optimal"
        timings.append(result["solve_time_seconds"])

        by_hour = {row["hour"]: row for row in result["hourly_plan"]}
        if case["id"] == "simultaneous-no-charge-no-discharge":
            assert by_hour[7]["battery_action"] == "idle"
            assert by_hour[7]["battery_kwh"] == pytest.approx(0.0, abs=TOLERANCE)
        elif case["id"] == "solar-factor-zero":
            assert by_hour[5]["solar_used_kwh"] == pytest.approx(0.0, abs=TOLERANCE)
        elif case["id"] == "solar-factor-one":
            assert by_hour[5]["grid_kwh"] == pytest.approx(0.0, abs=TOLERANCE)
        elif case["id"] == "solar-curtailment":
            assert sum(row["solar_used_kwh"] for row in result["hourly_plan"]) == pytest.approx(
                0.0, abs=TOLERANCE
            )
        elif case["id"] == "tight-grid-cap-precharge":
            assert by_hour[10]["grid_kwh"] <= 5.0 + TOLERANCE
            assert any(row["battery_action"] == "charge" for row in result["hourly_plan"][:10])
        elif case["id"] == "sharp-expensive-peak":
            assert by_hour[12]["battery_action"] == "discharge"
        elif case["id"] == "cheap-precharge-period":
            assert by_hour[1]["battery_action"] == "charge"
            assert by_hour[2]["battery_action"] == "discharge"
        elif case["id"] == "last-hour-restoration":
            assert by_hour[23]["battery_action"] == "charge"
        elif case["id"] == "restriction-near-hour-23":
            assert by_hour[23]["battery_action"] != "charge"
            assert by_hour[22]["battery_action"] == "charge"

    print(
        f"PART2_EDGE_CASES={len(timings)} "
        f"AVG_SOLVE_SECONDS={mean(timings):.6f} WORST_SOLVE_SECONDS={max(timings):.6f}"
    )


def test_fractional_values_are_not_rounded() -> None:
    case = _synthetic_case(
        "fractional-precision",
        demand=1.23456789,
        solar=0.0,
        battery={
            "capacity_kwh": 0.0,
            "initial_energy_kwh": 0.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 0.0,
            "max_discharge_kwh_per_hour": 0.0,
        },
    )
    result = solve(case["input"], [])
    assert result["hourly_plan"][0]["grid_kwh"] == pytest.approx(1.23456789, abs=1e-7)
    assert _validity_errors(case, result) == []


@pytest.mark.parametrize(
    ("factors", "expected_solar_upper_bound"),
    [
        ([0.5, 0.5], 50.0),
        ([0.8, 0.3], 30.0),
    ],
)
def test_overlapping_solar_reductions_use_strictest_active_factor(
    factors: list[float], expected_solar_upper_bound: float
) -> None:
    hour = 5
    zero_battery = {
        "capacity_kwh": 0.0,
        "initial_energy_kwh": 0.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 0.0,
        "max_discharge_kwh_per_hour": 0.0,
    }
    directives = [
        _directive(
            "solar_reduction",
            {"hours": [hour], "factor": factor},
            note_index=index,
        )
        for index, factor in enumerate(factors)
    ]
    case = _synthetic_case(
        "overlapping-solar-reductions",
        demand=100.0,
        solar=100.0,
        battery=zero_battery,
        directives=directives,
    )
    result = solve(case["input"], directives)
    assert result["hourly_plan"][hour]["solar_used_kwh"] == pytest.approx(
        expected_solar_upper_bound, abs=TOLERANCE
    )


def test_infeasible_model_raises_typed_failure_without_relaxation() -> None:
    demand = [0.0] * 24
    demand[4] = 10.0
    case = _synthetic_case(
        "infeasible-grid-cap",
        demand=demand,
        battery={
            "capacity_kwh": 0.0,
            "initial_energy_kwh": 0.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 0.0,
            "max_discharge_kwh_per_hour": 0.0,
        },
        directives=[_directive("max_grid_window", {"hours": [4], "max_grid_kwh": 0.0})],
    )
    with pytest.raises(OptimizationError) as raised:
        solve(case["input"], case["expected_output"]["directive_interpretation"])
    assert raised.value.solver_status == "Infeasible"
    assert "did not find an optimal schedule" in str(raised.value)


def test_optimize_integration_contract_returns_only_hourly_rows() -> None:
    case = _synthetic_case("integration-contract")
    rows = optimize(case["input"], [])
    assert isinstance(rows, list)
    assert len(rows) == 24
    assert [row["hour"] if isinstance(row, dict) else row.hour for row in rows] == list(range(24))


def test_transient_cbc_execution_failure_is_retried_once(monkeypatch) -> None:
    original_solve = pulp.LpProblem.solve
    attempts = 0

    def flaky_solve(model, solver=None, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IndexError("synthetic truncated CBC solution")
        return original_solve(model, solver, **kwargs)

    monkeypatch.setattr(pulp.LpProblem, "solve", flaky_solve)
    result = solve(_synthetic_case("transient-cbc-failure")["input"], [])
    assert result["solver_status"] == "Optimal"
    assert attempts == 2


def test_persistent_cbc_execution_failure_is_typed(monkeypatch) -> None:
    attempts = 0

    def broken_solve(_model, _solver=None, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("synthetic CBC execution failure")

    monkeypatch.setattr(pulp.LpProblem, "solve", broken_solve)
    with pytest.raises(OptimizationError, match="CBC could not execute") as raised:
        solve(_synthetic_case("persistent-cbc-failure")["input"], [])
    assert attempts == 2
    assert isinstance(raised.value.__cause__, OSError)

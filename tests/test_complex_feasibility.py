from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.optimizer import OptimizationError, solve
from app.response import build_response
from app.schemas import DirectiveInterpretation, OptimizeRequest, OptimizeResponse
from app.validator import validate_response


HOURS = [
    {"hour": 0, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 8},
    {"hour": 1, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 8},
    {"hour": 2, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 8},
    {"hour": 3, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 8},
    {"hour": 4, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 9},
    {"hour": 5, "demand_kwh": 100, "solar_kwh": 20, "tariff_bdt_per_kwh": 9},
    {"hour": 6, "demand_kwh": 110, "solar_kwh": 50, "tariff_bdt_per_kwh": 10},
    {"hour": 7, "demand_kwh": 120, "solar_kwh": 80, "tariff_bdt_per_kwh": 11},
    {"hour": 8, "demand_kwh": 130, "solar_kwh": 120, "tariff_bdt_per_kwh": 12},
    {"hour": 9, "demand_kwh": 140, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
    {"hour": 10, "demand_kwh": 150, "solar_kwh": 150, "tariff_bdt_per_kwh": 14},
    {"hour": 11, "demand_kwh": 150, "solar_kwh": 160, "tariff_bdt_per_kwh": 15},
    {"hour": 12, "demand_kwh": 160, "solar_kwh": 170, "tariff_bdt_per_kwh": 16},
    {"hour": 13, "demand_kwh": 140, "solar_kwh": 150, "tariff_bdt_per_kwh": 30},
    {"hour": 14, "demand_kwh": 130, "solar_kwh": 160, "tariff_bdt_per_kwh": 32},
    {"hour": 15, "demand_kwh": 150, "solar_kwh": 150, "tariff_bdt_per_kwh": 28},
    {"hour": 16, "demand_kwh": 140, "solar_kwh": 120, "tariff_bdt_per_kwh": 25},
    {"hour": 17, "demand_kwh": 150, "solar_kwh": 80, "tariff_bdt_per_kwh": 27},
    {"hour": 18, "demand_kwh": 180, "solar_kwh": 20, "tariff_bdt_per_kwh": 30},
    {"hour": 19, "demand_kwh": 190, "solar_kwh": 0, "tariff_bdt_per_kwh": 31},
    {"hour": 20, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 29},
    {"hour": 21, "demand_kwh": 150, "solar_kwh": 0, "tariff_bdt_per_kwh": 20},
    {"hour": 22, "demand_kwh": 120, "solar_kwh": 0, "tariff_bdt_per_kwh": 15},
    {"hour": 23, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
]

BATTERY = {
    "capacity_kwh": 500,
    "initial_energy_kwh": 300,
    "minimum_energy_kwh": 30,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100,
}

FEASIBLE_NOTES = [
    "Solar output will fall to 20% of normal from 1 PM until 3 PM.",
    "The battery must not discharge from 1 PM until 3 PM.",
]

INFEASIBLE_NOTES = FEASIBLE_NOTES + [
    "Grid import must not exceed 100 kWh per hour from 1 PM until 3 PM.",
]

FEASIBLE_DIRECTIVES = [
    {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "Usable solar is limited to 20% of forecast from 1 PM until 3 PM.",
    },
    {
        "note_index": 1,
        "applies": True,
        "directive_type": "no_discharge_window",
        "structured_adjustment": {"hours": [13, 14]},
        "explanation": "Battery discharge is prohibited from 1 PM until 3 PM.",
    },
]

INFEASIBLE_DIRECTIVES = FEASIBLE_DIRECTIVES + [
    {
        "note_index": 2,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": [13, 14], "max_grid_kwh": 100.0},
        "explanation": "Grid import is capped at 100 kWh from 1 PM until 3 PM.",
    },
]


def _request(scenario_id: str, notes: list[str]) -> dict:
    return {
        "scenario_id": scenario_id,
        "operator_notes": notes,
        "hours": deepcopy(HOURS),
        "battery": deepcopy(BATTERY),
    }


def _models(raw: list[dict]) -> list[DirectiveInterpretation]:
    return [DirectiveInterpretation.model_validate(item) for item in raw]


def test_grid_complex_003_optimizer_and_replay_are_feasible() -> None:
    request = OptimizeRequest.model_validate(_request("GRID-COMPLEX-003", FEASIBLE_NOTES))
    directives = _models(FEASIBLE_DIRECTIVES)

    result = solve(request, directives)
    assert result["solver_status"] == "Optimal"
    assert len(result["hourly_plan"]) == 24

    by_hour = {row["hour"]: row for row in result["hourly_plan"]}
    assert by_hour[13]["solar_used_kwh"] <= 30.0 + 0.01
    assert by_hour[14]["solar_used_kwh"] <= 32.0 + 0.01

    assert not (
        by_hour[13]["battery_action"] == "discharge"
        and by_hour[13]["battery_kwh"] > 0.01
    )
    assert not (
        by_hour[14]["battery_action"] == "discharge"
        and by_hour[14]["battery_kwh"] > 0.01
    )

    assert by_hour[23]["battery_energy_after_kwh"] == pytest.approx(300.0, abs=0.01)

    response = build_response(
        request,
        directives,
        [main.HourlyPlan.model_validate(row) for row in result["hourly_plan"]],
    )
    final = OptimizeResponse.model_validate_json(response.model_dump_json())
    validate_response(request, final, directives)


def test_grid_complex_003_full_pipeline_with_fake_llm_returns_200(monkeypatch) -> None:
    async def fake_interpret(_request):
        return deepcopy(FEASIBLE_DIRECTIVES)

    monkeypatch.setattr(main, "interpret_notes", fake_interpret)

    with TestClient(main.app) as client:
        response = client.post(
            "/optimize-energy",
            json=_request("GRID-COMPLEX-003", FEASIBLE_NOTES),
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scenario_id"] == "GRID-COMPLEX-003"
    assert len(body["hourly_plan"]) == 24
    assert body["directive_interpretation"][0]["structured_adjustment"] == {
        "hours": [13, 14],
        "factor": 0.2,
    }
    assert body["directive_interpretation"][1]["structured_adjustment"] == {
        "hours": [13, 14],
    }


def test_grid_complex_002_is_provably_infeasible() -> None:
    # Hour 13: demand=140, effective solar=150*0.2=30,
    # grid cap=100, and discharge is prohibited.
    # Maximum supply is therefore 130 < 140.
    assert 100.0 + (150.0 * 0.2) < 140.0

    request = OptimizeRequest.model_validate(_request("GRID-COMPLEX-002", INFEASIBLE_NOTES))
    directives = _models(INFEASIBLE_DIRECTIVES)

    with pytest.raises(OptimizationError) as raised:
        solve(request, directives)

    assert raised.value.solver_status == "Infeasible"


def test_grid_complex_002_full_pipeline_fails_safely(monkeypatch) -> None:
    async def fake_interpret(_request):
        return deepcopy(INFEASIBLE_DIRECTIVES)

    monkeypatch.setattr(main, "interpret_notes", fake_interpret)

    with TestClient(main.app) as client:
        response = client.post(
            "/optimize-energy",
            json=_request("GRID-COMPLEX-002", INFEASIBLE_NOTES),
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "The service could not complete the request.",
        }
    }
    assert "secret" not in response.text.lower()
    assert "traceback" not in response.text.lower()

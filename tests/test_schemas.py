from __future__ import annotations

from copy import deepcopy
from math import inf, nan

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import (
    BatteryAction,
    DirectiveInterpretation,
    DirectiveType,
    HourlyPlan,
    OptimizeRequest,
    SolarReductionAdjustment,
)


def valid_request_data() -> dict:
    return {
        "scenario_id": "GRID-101",
        "operator_notes": ["Keep today's schedule unchanged."],
        "hours": [
            {
                "hour": hour,
                "demand_kwh": 100,
                "solar_kwh": 10,
                "tariff_bdt_per_kwh": 7,
            }
            for hour in range(24)
        ],
        "battery": {
            "capacity_kwh": 500,
            "initial_energy_kwh": 200,
            "minimum_energy_kwh": 50,
            "max_charge_kwh_per_hour": 100,
            "max_discharge_kwh_per_hour": 100,
        },
    }


def test_valid_official_request_structure() -> None:
    request = OptimizeRequest.model_validate(valid_request_data())

    assert request.scenario_id == "GRID-101"
    assert [entry.hour for entry in request.hours] == list(range(24))


@pytest.mark.parametrize("scenario_id", ["", "   "])
def test_scenario_id_must_be_non_empty(scenario_id: str) -> None:
    payload = valid_request_data()
    payload["scenario_id"] = scenario_id

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


@pytest.mark.parametrize("operator_notes", [[], [""], ["  "], ["a", "b", "c", "d"]])
def test_operator_notes_require_one_to_three_non_empty_strings(
    operator_notes: list[str],
) -> None:
    payload = valid_request_data()
    payload["operator_notes"] = operator_notes

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


@pytest.mark.parametrize("hour_count", [23, 25])
def test_hours_require_exactly_24_entries(hour_count: int) -> None:
    payload = valid_request_data()
    if hour_count == 23:
        payload["hours"].pop()
    else:
        payload["hours"].append(
            {
                "hour": 23,
                "demand_kwh": 100,
                "solar_kwh": 10,
                "tariff_bdt_per_kwh": 7,
            }
        )

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


def test_hours_require_every_unique_index() -> None:
    payload = valid_request_data()
    payload["hours"][23]["hour"] = 22

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"])
@pytest.mark.parametrize("invalid_value", [-1, inf, -inf, nan])
def test_hour_numbers_must_be_finite_and_non_negative(
    field: str, invalid_value: float
) -> None:
    payload = valid_request_data()
    payload["hours"][0][field] = invalid_value

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


@pytest.mark.parametrize(
    "missing_field",
    [
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    ],
)
def test_all_battery_fields_are_required(missing_field: str) -> None:
    payload = valid_request_data()
    del payload["battery"][missing_field]

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("minimum", "initial", "capacity"),
    [(51, 50, 100), (0, 101, 100)],
)
def test_battery_energy_order(
    minimum: float, initial: float, capacity: float
) -> None:
    payload = valid_request_data()
    payload["battery"].update(
        minimum_energy_kwh=minimum,
        initial_energy_kwh=initial,
        capacity_kwh=capacity,
    )

    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate(payload)


def test_directive_model_freezes_solar_reduction_shape() -> None:
    directive = DirectiveInterpretation.model_validate(
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
            "explanation": "Only 20% of forecast solar remains usable.",
        }
    )

    assert directive.directive_type is DirectiveType.SOLAR_REDUCTION
    assert isinstance(directive.structured_adjustment, SolarReductionAdjustment)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Irrelevant note.",
        },
        {
            "note_index": 0,
            "applies": False,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [2, 3]},
            "explanation": "Charging is unavailable.",
        },
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [14, 13], "factor": 0.2},
            "explanation": "Unsorted hours.",
        },
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [13], "factor": 1.2},
            "explanation": "Invalid factor.",
        },
    ],
)
def test_invalid_directive_semantics_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DirectiveInterpretation.model_validate(payload)


@pytest.mark.parametrize("action", list(BatteryAction))
def test_all_official_battery_actions_are_supported(action: BatteryAction) -> None:
    entry = HourlyPlan(
        hour=0,
        grid_kwh=100,
        solar_used_kwh=0,
        battery_action=action,
        battery_kwh=0,
        battery_energy_after_kwh=50,
    )

    assert entry.battery_action is action


def test_idle_action_requires_zero_battery_kwh() -> None:
    with pytest.raises(ValidationError):
        HourlyPlan(
            hour=0,
            grid_kwh=100,
            solar_used_kwh=0,
            battery_action="idle",
            battery_kwh=1,
            battery_energy_after_kwh=50,
        )


def test_invalid_post_request_returns_controlled_http_400() -> None:
    payload = deepcopy(valid_request_data())
    payload["hours"].pop()
    client = TestClient(app)

    response = client.post("/optimize-energy", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_valid_post_without_runtime_credentials_fails_safely() -> None:
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/optimize-energy", json=valid_request_data())

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "The service could not complete the request.",
        }
    }

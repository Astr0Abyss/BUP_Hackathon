"""Shared, canonical API models for the GridWise service."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model used to keep the public JSON contract explicit."""

    model_config = ConfigDict(extra="forbid", strict=True)


FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
HourIndex = Annotated[int, Field(ge=0, le=23)]
NoteIndex = Annotated[int, Field(ge=0)]
NonEmptyText = Annotated[str, Field(min_length=1)]
HoursList = Annotated[list[HourIndex], Field(min_length=1, max_length=24)]


def _require_trimmed_non_empty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_unique_ascending_hours(hours: list[int]) -> list[int]:
    if len(hours) != len(set(hours)):
        raise ValueError("hours must contain unique values")
    if hours != sorted(hours):
        raise ValueError("hours must be in ascending order")
    return hours


class HourInput(StrictModel):
    hour: HourIndex
    demand_kwh: FiniteNonNegative
    solar_kwh: FiniteNonNegative
    tariff_bdt_per_kwh: FiniteNonNegative


class BatteryInput(StrictModel):
    capacity_kwh: FiniteNonNegative
    initial_energy_kwh: FiniteNonNegative
    minimum_energy_kwh: FiniteNonNegative
    max_charge_kwh_per_hour: FiniteNonNegative
    max_discharge_kwh_per_hour: FiniteNonNegative

    @model_validator(mode="after")
    def validate_energy_order(self) -> BatteryInput:
        if not (
            self.minimum_energy_kwh
            <= self.initial_energy_kwh
            <= self.capacity_kwh
        ):
            raise ValueError(
                "battery energy must satisfy minimum_energy_kwh <= "
                "initial_energy_kwh <= capacity_kwh"
            )
        return self


class OptimizeRequest(StrictModel):
    scenario_id: NonEmptyText
    operator_notes: Annotated[list[str], Field(min_length=1, max_length=3)]
    hours: Annotated[list[HourInput], Field(min_length=24, max_length=24)]
    battery: BatteryInput

    @field_validator("scenario_id")
    @classmethod
    def validate_scenario_id(cls, value: str) -> str:
        return _require_trimmed_non_empty(value, "scenario_id")

    @field_validator("operator_notes")
    @classmethod
    def validate_operator_notes(cls, notes: list[str]) -> list[str]:
        return [
            _require_trimmed_non_empty(note, f"operator_notes[{index}]")
            for index, note in enumerate(notes)
        ]

    @field_validator("hours")
    @classmethod
    def validate_complete_day(cls, hours: list[HourInput]) -> list[HourInput]:
        indices = [entry.hour for entry in hours]
        if len(indices) != len(set(indices)):
            raise ValueError("hours must contain 24 unique hour values")
        if set(indices) != set(range(24)):
            raise ValueError("hours must contain every integer from 0 through 23")
        return hours


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class SolarReductionAdjustment(StrictModel):
    hours: HoursList
    factor: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    _validate_hours = field_validator("hours")(_require_unique_ascending_hours)


class MinimumBatteryReserveAdjustment(StrictModel):
    hours: HoursList
    minimum_energy_kwh: FiniteNonNegative

    _validate_hours = field_validator("hours")(_require_unique_ascending_hours)


class WindowAdjustment(StrictModel):
    hours: HoursList

    _validate_hours = field_validator("hours")(_require_unique_ascending_hours)


class MaxGridWindowAdjustment(StrictModel):
    hours: HoursList
    max_grid_kwh: FiniteNonNegative

    _validate_hours = field_validator("hours")(_require_unique_ascending_hours)


StructuredAdjustment = (
    SolarReductionAdjustment
    | MinimumBatteryReserveAdjustment
    | WindowAdjustment
    | MaxGridWindowAdjustment
    | None
)


class DirectiveInterpretation(StrictModel):
    note_index: NoteIndex
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment
    explanation: NonEmptyText

    @field_validator("directive_type", mode="before")
    @classmethod
    def parse_directive_type(cls, value: object) -> object:
        if isinstance(value, str):
            return DirectiveType(value)
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        return _require_trimmed_non_empty(value, "explanation")

    @model_validator(mode="after")
    def validate_directive_shape(self) -> DirectiveInterpretation:
        expected_type: dict[DirectiveType, type[StrictModel]] = {
            DirectiveType.SOLAR_REDUCTION: SolarReductionAdjustment,
            DirectiveType.MINIMUM_BATTERY_RESERVE: MinimumBatteryReserveAdjustment,
            DirectiveType.NO_CHARGE_WINDOW: WindowAdjustment,
            DirectiveType.NO_DISCHARGE_WINDOW: WindowAdjustment,
            DirectiveType.MAX_GRID_WINDOW: MaxGridWindowAdjustment,
        }

        if self.directive_type is DirectiveType.NO_OP:
            if self.applies or self.structured_adjustment is not None:
                raise ValueError(
                    "no_op requires applies=false and structured_adjustment=null"
                )
            return self

        if not self.applies:
            raise ValueError("non-no_op directives require applies=true")

        required = expected_type[self.directive_type]
        if not isinstance(self.structured_adjustment, required):
            raise ValueError(
                f"{self.directive_type.value} requires a "
                f"{required.__name__} structured_adjustment"
            )
        return self


class BatteryAction(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


class HourlyPlan(StrictModel):
    hour: HourIndex
    grid_kwh: FiniteNonNegative
    solar_used_kwh: FiniteNonNegative
    battery_action: BatteryAction
    battery_kwh: FiniteNonNegative
    battery_energy_after_kwh: FiniteNonNegative

    @field_validator("battery_action", mode="before")
    @classmethod
    def parse_battery_action(cls, value: object) -> object:
        if isinstance(value, str):
            return BatteryAction(value)
        return value

    @model_validator(mode="after")
    def validate_idle_action(self) -> HourlyPlan:
        if self.battery_action is BatteryAction.IDLE and self.battery_kwh != 0:
            raise ValueError("battery_kwh must be 0 when battery_action is idle")
        return self


class OptimizeResponse(StrictModel):
    scenario_id: NonEmptyText
    directive_interpretation: Annotated[
        list[DirectiveInterpretation], Field(min_length=1, max_length=3)
    ]
    hourly_plan: Annotated[list[HourlyPlan], Field(min_length=24, max_length=24)]
    total_grid_kwh: FiniteNonNegative
    total_cost_bdt: FiniteNonNegative
    peak_grid_kwh: FiniteNonNegative
    plan_summary: NonEmptyText

    @field_validator("scenario_id", "plan_summary")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        return _require_trimmed_non_empty(value, "text field")

    @field_validator("directive_interpretation")
    @classmethod
    def validate_note_order(
        cls, directives: list[DirectiveInterpretation]
    ) -> list[DirectiveInterpretation]:
        indices = [directive.note_index for directive in directives]
        if indices != list(range(len(directives))):
            raise ValueError(
                "directive_interpretation must be in note_index order 0..N-1"
            )
        return directives

    @field_validator("hourly_plan")
    @classmethod
    def validate_plan_hours(cls, plan: list[HourlyPlan]) -> list[HourlyPlan]:
        indices = [entry.hour for entry in plan]
        if indices != list(range(24)):
            raise ValueError("hourly_plan must contain hours 0 through 23 in order")
        return plan


class HealthResponse(StrictModel):
    status: Annotated[str, Field(pattern="^ok$")]

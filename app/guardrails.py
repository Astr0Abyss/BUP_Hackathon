"""Deterministic validation for untrusted GridWise LLM interpretations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas import DirectiveInterpretation, OptimizeRequest


ALLOWED_DIRECTIVE_TYPES = frozenset(
    {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }
)

_ENTRY_KEYS = {
    "note_index",
    "applies",
    "directive_type",
    "structured_adjustment",
    "explanation",
}

_ADJUSTMENT_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}


class GuardrailViolation(ValueError):
    """Raised when model output cannot be safely compiled into directives."""


def _plain(value: Any) -> Any:
    """Convert Pydantic values to plain Python without depending on shared models."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailViolation(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise GuardrailViolation(f"{field} must be finite")
    return result


def _normalized_hours(value: Any, note_index: int) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GuardrailViolation(f"note {note_index}: hours must be an array")

    hours = list(value)
    if not hours:
        raise GuardrailViolation(f"note {note_index}: hours must not be empty")
    if any(isinstance(hour, bool) or not isinstance(hour, int) for hour in hours):
        raise GuardrailViolation(f"note {note_index}: hours must contain integers")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise GuardrailViolation(f"note {note_index}: hours must be within 0..23")
    if len(set(hours)) != len(hours):
        raise GuardrailViolation(f"note {note_index}: hours must be unique")

    # The official plan explicitly permits harmless normalization of valid hours.
    return sorted(hours)


def _validate_adjustment(
    directive_type: str,
    adjustment: Any,
    *,
    note_index: int,
    battery_capacity_kwh: float,
) -> dict[str, Any]:
    adjustment = _plain(adjustment)
    if not isinstance(adjustment, Mapping):
        raise GuardrailViolation(
            f"note {note_index}: {directive_type} requires an adjustment object"
        )

    expected_keys = _ADJUSTMENT_KEYS[directive_type]
    actual_keys = set(adjustment)
    if actual_keys != expected_keys:
        raise GuardrailViolation(
            f"note {note_index}: {directive_type} adjustment keys must be "
            f"{sorted(expected_keys)}; got {sorted(actual_keys)}"
        )

    normalized = {"hours": _normalized_hours(adjustment["hours"], note_index)}

    if directive_type == "solar_reduction":
        factor = _finite_number(adjustment["factor"], "factor")
        if not 0 <= factor <= 1:
            raise GuardrailViolation(f"note {note_index}: factor must be within 0..1")
        normalized["factor"] = factor
    elif directive_type == "minimum_battery_reserve":
        reserve = _finite_number(
            adjustment["minimum_energy_kwh"], "minimum_energy_kwh"
        )
        if reserve < 0 or reserve > battery_capacity_kwh:
            raise GuardrailViolation(
                f"note {note_index}: reserve must be within 0..battery capacity"
            )
        normalized["minimum_energy_kwh"] = reserve
    elif directive_type == "max_grid_window":
        cap = _finite_number(adjustment["max_grid_kwh"], "max_grid_kwh")
        if cap < 0:
            raise GuardrailViolation(f"note {note_index}: grid cap must be non-negative")
        normalized["max_grid_kwh"] = cap

    return normalized


def validate_interpretations(
    raw_interpretations: Any,
    *,
    note_count: int,
    battery_capacity_kwh: float,
) -> list[dict[str, Any]]:
    """Validate and normalize one model result for all input notes.

    The function accepts either the interpretations array itself or a wrapper with
    one ``interpretations`` field. It never invents entries or converts invalid
    output to ``no_op``.
    """

    if isinstance(note_count, bool) or not isinstance(note_count, int) or not 1 <= note_count <= 3:
        raise GuardrailViolation("note_count must be an integer from 1 through 3")

    capacity = _finite_number(battery_capacity_kwh, "battery_capacity_kwh")
    if capacity < 0:
        raise GuardrailViolation("battery_capacity_kwh must be non-negative")

    raw_interpretations = _plain(raw_interpretations)
    if isinstance(raw_interpretations, Mapping):
        if set(raw_interpretations) != {"interpretations"}:
            raise GuardrailViolation("model output wrapper may contain only interpretations")
        raw_interpretations = raw_interpretations["interpretations"]

    if isinstance(raw_interpretations, (str, bytes)) or not isinstance(
        raw_interpretations, Sequence
    ):
        raise GuardrailViolation("model output must contain an interpretations array")

    entries = [_plain(item) for item in raw_interpretations]
    if len(entries) != note_count:
        raise GuardrailViolation(
            f"expected {note_count} interpretation entries; got {len(entries)}"
        )

    normalized: list[dict[str, Any]] = []
    indexes: list[int] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise GuardrailViolation(f"entry {position} must be an object")
        if set(entry) != _ENTRY_KEYS:
            raise GuardrailViolation(
                f"entry {position} fields must be exactly {sorted(_ENTRY_KEYS)}"
            )

        note_index = entry["note_index"]
        if isinstance(note_index, bool) or not isinstance(note_index, int):
            raise GuardrailViolation(f"entry {position}: note_index must be an integer")
        indexes.append(note_index)

        directive_type = entry["directive_type"]
        if directive_type not in ALLOWED_DIRECTIVE_TYPES:
            raise GuardrailViolation(
                f"note {note_index}: unsupported directive type {directive_type!r}"
            )

        applies = entry["applies"]
        if not isinstance(applies, bool):
            raise GuardrailViolation(f"note {note_index}: applies must be boolean")

        explanation = entry["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise GuardrailViolation(f"note {note_index}: explanation must be non-empty")

        if directive_type == "no_op":
            if applies is not False or entry["structured_adjustment"] is not None:
                raise GuardrailViolation(
                    f"note {note_index}: no_op requires applies=false and null adjustment"
                )
            adjustment = None
        else:
            if applies is not True:
                raise GuardrailViolation(
                    f"note {note_index}: real directives require applies=true"
                )
            adjustment = _validate_adjustment(
                directive_type,
                entry["structured_adjustment"],
                note_index=note_index,
                battery_capacity_kwh=capacity,
            )

        normalized.append(
            {
                "note_index": note_index,
                "applies": applies,
                "directive_type": directive_type,
                "structured_adjustment": adjustment,
                "explanation": explanation.strip(),
            }
        )

    expected_indexes = list(range(note_count))
    if indexes != expected_indexes:
        raise GuardrailViolation(
            f"note indexes must be exactly {expected_indexes} in order; got {indexes}"
        )

    return deepcopy(normalized)


def validate_and_normalize(
    raw: object,
    request: OptimizeRequest,
) -> list[DirectiveInterpretation]:
    """Validate untrusted output and return Member 1 schema instances."""
    validated = validate_interpretations(
        raw,
        note_count=len(request.operator_notes),
        battery_capacity_kwh=request.battery.capacity_kwh,
    )
    from app.schemas import DirectiveInterpretation

    return [
        DirectiveInterpretation.model_validate(item)
        for item in validated
    ]

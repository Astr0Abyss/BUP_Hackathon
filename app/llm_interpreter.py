"""Bounded OpenAI interpretation and CodeCraft backup; requires Python 3.11+."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import asyncio
from contextlib import AsyncExitStack
import json
import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from app.guardrails import validate_interpretations

if TYPE_CHECKING:
    from app.schemas import OptimizeRequest


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_CODECRAFT_MODEL = "claude-fable-5"
TOTAL_TIMEOUT_SECONDS = 24.0
ATTEMPT_TIMEOUT_SECONDS = 7.0


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _HoursAdjustment(_StrictModel):
    hours: list[int]


class _SolarAdjustment(_StrictModel):
    hours: list[int]
    factor: float


class _ReserveAdjustment(_StrictModel):
    hours: list[int]
    minimum_energy_kwh: float


class _GridCapAdjustment(_StrictModel):
    hours: list[int]
    max_grid_kwh: float


class _InterpretationDraft(_StrictModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    structured_adjustment: (
        _SolarAdjustment
        | _ReserveAdjustment
        | _GridCapAdjustment
        | _HoursAdjustment
        | None
    )
    explanation: str


class _InterpretationBatch(_StrictModel):
    interpretations: list[_InterpretationDraft]


SYSTEM_PROMPT = """You interpret GridWise smart-campus operator notes.

Return exactly one interpretation for every supplied note, in note_index order.
The model performs interpretation only; never optimize or create an energy plan.

Allowed directives and exact structured_adjustment shapes:
- solar_reduction: {"hours": [...], "factor": number}
- minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": number}
- no_charge_window: {"hours": [...]}
- no_discharge_window: {"hours": [...]}
- max_grid_window: {"hours": [...], "max_grid_kwh": number}
- no_op: null

Rules:
- Whole-hour windows are start-inclusive and end-exclusive.
- 1 PM to 3 PM means [13, 14]; 6 PM to 9 PM means [18, 19, 20].
- Treat from/between/until/through time windows with the same end-exclusive rule.
  In "from one until three PM", PM applies to both endpoints.
- Hours must be unique integers 0..23 in ascending order.
- solar_reduction factor is the usable fraction remaining. An 80% reduction means
  factor 0.2; reduced to 20% also means factor 0.2.
- "Reduced by X%" leaves 1-X; "reduced to X%", "X% remains", fractions such as
  one-fifth, and "half remains" directly state the remaining factor.
- Convert an explicit percentage-of-capacity reserve to kWh using the supplied
  battery capacity.
- "Charging unavailable" prevents energy entering the battery (no_charge_window).
  "Battery cannot supply power" prevents energy leaving it (no_discharge_window).
- Feeder limits, transformer limits, and grid intake/import caps are
  max_grid_window. Keep/maintain/retain-at-least language is a battery reserve.
- Notes unrelated to the current 24-hour energy schedule are no_op.
- Future events, meetings, notices, and seminars remain no_op even when they
  mention times, percentages, electricity, energy, or numbers.
- Interpret repeated note text independently at each supplied note_index.
- no_op requires applies=false and structured_adjustment=null.
- Every other directive requires applies=true.
- Do not modify or emit demand, solar forecasts, tariffs, battery configuration,
  rate limits, or any unsupported directive.
"""


class InterpretationFailure(RuntimeError):
    """Safe public error without provider response bodies or chained secrets."""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("battery must be a mapping or model with mapping-compatible fields")


def _battery_payload(battery: Any) -> dict[str, float]:
    source = _as_mapping(battery)
    required = (
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    )
    missing = [field for field in required if field not in source]
    if missing:
        raise ValueError(f"battery is missing required fields: {missing}")
    return {field: source[field] for field in required}


def _default_client(*, backup: bool = False) -> Any:
    from openai import AsyncOpenAI

    key = os.getenv("CODECRAFT_API_KEY" if backup else "OPENAI_API_KEY")
    if not key:
        raise InterpretationFailure("Provider credentials are unavailable")
    base_url = (
        os.getenv("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")
        if backup
        else "https://api.openai.com/v1"
    )
    return AsyncOpenAI(
        api_key=key,
        base_url=base_url,
        max_retries=0,
        timeout=ATTEMPT_TIMEOUT_SECONDS,
    )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


async def _request(
    client: Any,
    model: str,
    messages: list,
    *,
    backup: bool,
    timeout: float,
) -> Any:
    bounded = client.with_options(max_retries=0, timeout=timeout)
    if not backup:
        response = await bounded.responses.parse(
            model=model, reasoning={"effort": "low"}, input=messages,
            text_format=_InterpretationBatch, max_output_tokens=1800, store=False,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("Missing or incomplete interpretation")
        return response.output_parsed
    # Portable OpenAI-compatible interface; no Responses/reasoning assumptions.
    response = await bounded.chat.completions.create(
        model=model, messages=messages, max_tokens=1800,
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
        raise ValueError("Missing or incomplete backup interpretation")
    return json.loads(choice.message.content, object_pairs_hook=_json_object)


def _retry_primary(exc: Exception) -> bool:
    # Throttling/auth/billing/bad parameters go straight to backup. Never
    # immediately retry ahead of a provider's Retry-After header.
    status = getattr(exc, "status_code", None)
    return status is None or status in (408, 409) or status >= 500


async def interpret_operator_notes_async(
    operator_notes: Sequence[str],
    battery: Any,
    *,
    client: Any | None = None,
    model: str | None = None,
    backup_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Use one primary retry and one backup within a 24-second budget."""

    if isinstance(operator_notes, (str, bytes)) or not isinstance(operator_notes, Sequence):
        raise ValueError("operator_notes must be a sequence of 1 to 3 strings")
    notes = list(operator_notes)
    if not 1 <= len(notes) <= 3:
        raise ValueError("operator_notes must contain 1 to 3 entries")
    if any(not isinstance(note, str) or not note.strip() for note in notes):
        raise ValueError("operator_notes entries must be non-empty strings")

    battery_payload = _battery_payload(battery)
    request_payload = {
        "operator_notes": [
            {"note_index": index, "text": note} for index, note in enumerate(notes)
        ],
        "battery": battery_payload,
    }
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(request_payload, separators=(",", ":"))},
    ]
    backup_model = os.getenv("CODECRAFT_MODEL", DEFAULT_CODECRAFT_MODEL)
    backup_enabled = backup_client is not None or bool(os.getenv("CODECRAFT_API_KEY"))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TOTAL_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout_at(deadline), AsyncExitStack() as stack:
            primary = client
            if primary is None:
                try:
                    primary = await stack.enter_async_context(_default_client())
                except Exception:
                    primary = None
            if primary is not None:
                for attempt in range(2):
                    remaining = deadline - loop.time()
                    if attempt and backup_enabled and remaining < 2 * ATTEMPT_TIMEOUT_SECONDS:
                        break
                    if remaining <= 0:
                        break
                    try:
                        timeout = min(ATTEMPT_TIMEOUT_SECONDS, remaining)
                        async with asyncio.timeout(timeout):
                            parsed = await _request(
                                primary,
                                selected_model,
                                messages,
                                backup=False,
                                timeout=timeout,
                            )
                            return validate_interpretations(
                                parsed, note_count=len(notes),
                                battery_capacity_kwh=battery_payload["capacity_kwh"],
                            )
                    except Exception as exc:
                        if not _retry_primary(exc):
                            break
                        messages = [messages[0], {
                            "role": "system",
                            "content": (
                                "The prior attempt failed. Reinterpret every original "
                                "note. Check indexes, exact adjustment fields, hours "
                                "and numeric bounds."
                            ),
                        }, messages[-1]]
            if backup_enabled and loop.time() < deadline:
                backup = backup_client
                if backup is None:
                    backup = await stack.enter_async_context(_default_client(backup=True))
                backup_messages = [messages[0], {
                    "role": "system",
                    "content": (
                        "Return only a JSON object with one field interpretations, "
                        "an array. Each entry has exactly note_index, applies, "
                        "directive_type, structured_adjustment, explanation. "
                        "Do not include markdown."
                    ),
                }, messages[-1]]
                timeout = min(ATTEMPT_TIMEOUT_SECONDS, deadline - loop.time())
                async with asyncio.timeout(timeout):
                    parsed = await _request(backup, backup_model, backup_messages,
                                            backup=True, timeout=timeout)
                    return validate_interpretations(
                        parsed, note_count=len(notes),
                        battery_capacity_kwh=battery_payload["capacity_kwh"],
                    )
    except Exception:
        pass
    raise InterpretationFailure("Operator-note interpretation is unavailable") from None


def interpret_operator_notes(
    operator_notes: Sequence[str], battery: Any, *, client: Any | None = None,
    model: str | None = None, backup_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper; async routes should await the async entry point."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(interpret_operator_notes_async(
            operator_notes, battery, client=client, model=model,
            backup_client=backup_client,
        ))
    raise RuntimeError("Use await interpret_operator_notes_async from an async route")


async def interpret_notes(request: OptimizeRequest) -> object:
    """Member 1 compatibility entry point."""
    return await interpret_operator_notes_async(
        request.operator_notes,
        request.battery,
    )


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "InterpretationFailure",
    "interpret_notes",
    "interpret_operator_notes",
    "interpret_operator_notes_async",
]

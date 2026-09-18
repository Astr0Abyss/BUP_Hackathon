"""Build public responses from actual schedule rows, never solver totals."""

from collections.abc import Sequence
from math import fsum

from app.schemas import DirectiveInterpretation, HourlyPlan, OptimizeRequest, OptimizeResponse


def build_response(
    request: OptimizeRequest,
    directives: Sequence[DirectiveInterpretation],
    hourly_plan: Sequence[HourlyPlan],
) -> OptimizeResponse:
    rows = sorted(hourly_plan, key=lambda row: row.hour)
    tariffs = {row.hour: row.tariff_bdt_per_kwh for row in request.hours}
    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=sorted(directives, key=lambda item: item.note_index),
        hourly_plan=rows,
        total_grid_kwh=fsum(row.grid_kwh for row in rows),
        total_cost_bdt=fsum(row.grid_kwh * tariffs[row.hour] for row in rows),
        peak_grid_kwh=max(row.grid_kwh for row in rows),
        plan_summary="Schedules grid, solar and battery energy under the validated operator directives, restoring the initial battery energy at day end.",
    )

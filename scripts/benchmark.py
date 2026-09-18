"""Lightweight benchmark for the GridWise optimizer and optional deployed API.

The local benchmark uses organizer-provided expected directive interpretations as
oracle inputs. Supplying ``--api-url`` additionally sends the unmodified official
requests through the complete deployed HTTP pipeline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.optimizer import OptimizationError, solve  # noqa: E402


SAMPLES_PATH = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
TOLERANCE = 0.01


def _load_cases() -> list[dict[str, Any]]:
    with SAMPLES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def _local_run(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = solve(
        case["input"], case["expected_output"]["directive_interpretation"]
    )
    wall_seconds = time.perf_counter() - started
    reference_cost = float(case["expected_output"]["total_cost_bdt"])
    difference = abs(float(result["total_cost_bdt"]) - reference_cost)
    valid = (
        result["solver_status"] == "Optimal"
        and len(result["hourly_plan"]) == 24
        and [row["hour"] for row in result["hourly_plan"]] == list(range(24))
        and difference <= TOLERANCE
    )
    return {
        "case_id": case["id"],
        "status": result["solver_status"],
        "cost": result["total_cost_bdt"],
        "reference_cost": reference_cost,
        "difference": difference,
        "valid": valid,
        "solver_seconds": result["solve_time_seconds"],
        "wall_seconds": wall_seconds,
    }


def _api_run(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/optimize-energy"
    payload = json.dumps(case["input"], separators=(",", ":")).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = response.status
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        status_code = exc.code
        body = None
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "case_id": case["id"],
            "status_code": None,
            "latency_seconds": time.perf_counter() - started,
            "valid": False,
            "error": type(exc).__name__,
        }

    latency = time.perf_counter() - started
    reference_cost = float(case["expected_output"]["total_cost_bdt"])
    cost = body.get("total_cost_bdt") if isinstance(body, dict) else None
    difference = abs(float(cost) - reference_cost) if isinstance(cost, (int, float)) else None
    valid = (
        status_code == 200
        and isinstance(body, dict)
        and len(body.get("hourly_plan", [])) == 24
        and difference is not None
        and difference <= TOLERANCE
    )
    return {
        "case_id": case["id"],
        "status_code": status_code,
        "cost": cost,
        "reference_cost": reference_cost,
        "difference": difference,
        "latency_seconds": latency,
        "valid": valid,
    }


def _print_local(rows: list[dict[str, Any]]) -> None:
    print("case_id status cost reference difference valid solver_s wall_s")
    for row in rows:
        print(
            f"{row['case_id']} {row['status']} {row['cost']:.6f} "
            f"{row['reference_cost']:.6f} {row['difference']:.6f} "
            f"{'PASS' if row['valid'] else 'FAIL'} "
            f"{row['solver_seconds']:.6f} {row['wall_seconds']:.6f}"
        )
    solver_times = [row["solver_seconds"] for row in rows]
    wall_times = [row["wall_seconds"] for row in rows]
    print(
        "local_summary "
        f"passed={sum(row['valid'] for row in rows)}/{len(rows)} "
        f"solver_avg_s={statistics.mean(solver_times):.6f} "
        f"solver_worst_s={max(solver_times):.6f} "
        f"wall_avg_s={statistics.mean(wall_times):.6f} "
        f"wall_worst_s={max(wall_times):.6f}"
    )


def _print_api(rows: list[dict[str, Any]]) -> None:
    print("api_case_id http cost reference difference valid latency_s")
    for row in rows:
        cost = "-" if row.get("cost") is None else f"{row['cost']:.6f}"
        difference = "-" if row.get("difference") is None else f"{row['difference']:.6f}"
        print(
            f"{row['case_id']} {row.get('status_code')} {cost} "
            f"{row.get('reference_cost', '-')} {difference} "
            f"{'PASS' if row['valid'] else 'FAIL'} {row['latency_seconds']:.6f}"
        )
    latencies = [row["latency_seconds"] for row in rows]
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    print(
        "api_summary "
        f"passed={sum(row['valid'] for row in rows)}/{len(rows)} "
        f"avg_s={statistics.mean(latencies):.6f} "
        f"p95_s={ordered[p95_index]:.6f} worst_s={max(latencies):.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", help="Optional deployed API base URL")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    cases = _load_cases()
    try:
        local_rows = [_local_run(case) for _ in range(args.repeats) for case in cases]
    except OptimizationError as exc:
        print(f"local optimizer failure: {exc}", file=sys.stderr)
        return 1
    _print_local(local_rows)

    api_rows: list[dict[str, Any]] = []
    if args.api_url:
        api_rows = [
            _api_run(case, args.api_url, args.timeout)
            for _ in range(args.repeats)
            for case in cases
        ]
        _print_api(api_rows)

    return 0 if all(row["valid"] for row in local_rows + api_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

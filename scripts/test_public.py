"""Run the official public cases against a live GridWise API."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
REQUIRED_RESPONSE_FIELDS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}
TOLERANCE = 0.01


def _compare_interpretations(actual: Any, expected: list[dict[str, Any]]) -> list[str]:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return ["interpretation count mismatch"]
    failures: list[str] = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        if not isinstance(got, dict):
            failures.append(f"interpretation {index} is not an object")
            continue
        for field in ("note_index", "applies", "directive_type"):
            if got.get(field) != want.get(field):
                failures.append(f"interpretation {index} {field} mismatch")
        got_adjustment = got.get("structured_adjustment")
        want_adjustment = want.get("structured_adjustment")
        if want_adjustment is None:
            if got_adjustment is not None:
                failures.append(f"interpretation {index} adjustment must be null")
            continue
        if not isinstance(got_adjustment, dict):
            failures.append(f"interpretation {index} adjustment is not an object")
            continue
        if got_adjustment.get("hours") != want_adjustment.get("hours"):
            failures.append(f"interpretation {index} hours mismatch")
        for field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if field in want_adjustment:
                value = got_adjustment.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    failures.append(f"interpretation {index} {field} is not finite")
                elif abs(value - want_adjustment[field]) > TOLERANCE:
                    failures.append(f"interpretation {index} {field} mismatch")
    return failures


def evaluate(case: dict[str, Any], response: httpx.Response) -> list[str]:
    failures: list[str] = []
    if response.status_code != 200:
        return [f"HTTP {response.status_code}"]
    try:
        payload = response.json()
    except ValueError:
        return ["response is not valid JSON"]
    if not isinstance(payload, dict):
        return ["response JSON is not an object"]
    missing = REQUIRED_RESPONSE_FIELDS - payload.keys()
    if missing:
        failures.append(f"missing fields: {', '.join(sorted(missing))}")
    if payload.get("scenario_id") != case["input"]["scenario_id"]:
        failures.append("scenario_id mismatch")
    if len(payload.get("directive_interpretation", [])) != len(case["input"]["operator_notes"]):
        failures.append("interpretation count mismatch")
    if len(payload.get("hourly_plan", [])) != 24:
        failures.append("hourly_plan count mismatch")
    failures.extend(_compare_interpretations(
        payload.get("directive_interpretation"),
        case["expected_output"]["directive_interpretation"],
    ))
    return failures


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    endpoint = f"{args.base_url.rstrip('/')}/optimize-energy"
    latencies: list[float] = []
    failed: list[tuple[str, list[str]]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post(endpoint, json=case["input"])
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                try:
                    response.json()
                    json_ok = True
                except ValueError:
                    json_ok = False
                failures = evaluate(case, response)
                status = response.status_code
            except httpx.HTTPError as exc:
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                failures = [f"request failed: {type(exc).__name__}"]
                status = "ERROR"
                json_ok = False
            label = "PASS" if not failures else "FAIL"
            print(f"{case['id']}: {label} status={status} latency={elapsed:.3f}s json={'yes' if json_ok else 'no'}")
            if failures:
                failed.append((case["id"], failures))

    passed = len(cases) - len(failed)
    print(f"\nSummary: {passed}/{len(cases)} passed")
    if latencies:
        print(f"Latency: p50={statistics.median(latencies):.3f}s p95={percentile(latencies, 0.95):.3f}s")
    for case_id, failures in failed:
        print(f"- {case_id}: {'; '.join(failures)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

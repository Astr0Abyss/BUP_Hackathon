"""Optional smoke tests for a real deployed API."""

import json
import os
import time
from pathlib import Path

import httpx
import pytest


BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="BASE_URL is not configured")
CASES_FILE = (
    Path(__file__).resolve().parents[1]
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def test_external_health() -> None:
    response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_external_optimize_energy_contract_and_timeout() -> None:
    sample = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"][0]["input"]
    started = time.perf_counter()
    response = httpx.post(
        f"{BASE_URL}/optimize-energy",
        json=sample,
        timeout=30.0,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0
    assert response.status_code == 200
    payload = response.json()
    assert {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    } <= payload.keys()

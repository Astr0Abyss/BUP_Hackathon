import asyncio
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schemas import DirectiveInterpretation, HourlyPlan, OptimizeResponse


CASES = json.loads((Path(__file__).resolve().parents[1] /
    "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json").read_text(encoding="utf-8"))["cases"]


def install_test_components(monkeypatch, case):
    calls = []
    async def interpret(request):
        calls.append("interpret")
        return case["expected_output"]["directive_interpretation"]
    def guardrails(raw, request):
        calls.append("guardrails")
        return [DirectiveInterpretation.model_validate(d) for d in raw]
    def optimizer(request, directives):
        calls.append("optimizer")
        return [HourlyPlan.model_validate(r) for r in case["expected_output"]["hourly_plan"]]
    for module_name, function_name, function in [
        ("app.llm_interpreter", "interpret_notes", interpret),
        ("app.guardrails", "validate_and_normalize", guardrails),
        ("app.optimizer", "optimize", optimizer),
    ]:
        module = ModuleType(module_name)
        setattr(module, function_name, function)
        monkeypatch.setitem(sys.modules, module_name, module)
    return calls


@pytest.mark.parametrize("index", range(10))
def test_complete_route_with_public_reference_test_components(monkeypatch, index):
    case = CASES[index]
    calls = install_test_components(monkeypatch, case)
    original_replay = main.validate_response
    seen = []
    def replay(request, response, directives):
        calls.append("replay")
        seen.append(response.model_dump(mode="json"))
        original_replay(request, response, directives)
    monkeypatch.setattr(main, "validate_response", replay)
    with TestClient(main.app) as client:
        result = client.post("/optimize-energy", json=case["input"])
    assert result.status_code == 200, result.text
    assert calls == ["interpret", "guardrails", "optimizer", "replay"]
    assert result.json() == seen[0]
    for key in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        assert result.json()[key] == pytest.approx(case["expected_output"][key], abs=0.01)


@pytest.mark.parametrize("stage", ["interpret_notes", "validate_and_normalize", "optimize", "validate_response"])
def test_each_stage_failure_is_sanitized(monkeypatch, stage):
    install_test_components(monkeypatch, CASES[0])
    def fail(*args):
        raise RuntimeError("Bearer secret-api-key; private traceback")
    async def async_fail(*args):
        fail()
    monkeypatch.setattr(main, stage, async_fail if stage == "interpret_notes" else fail)
    with TestClient(main.app) as client:
        response = client.post("/optimize-energy", json=CASES[0]["input"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "The service could not complete the request."}}


def test_invalid_optimizer_schedule_never_returns_200(monkeypatch):
    install_test_components(monkeypatch, CASES[0])
    original = main.optimize
    def invalid(request, directives):
        rows = original(request, directives)
        rows[0].grid_kwh += 1
        return rows
    monkeypatch.setattr(main, "optimize", invalid)
    with TestClient(main.app) as client:
        response = client.post("/optimize-energy", json=CASES[0]["input"])
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "replay_validation_failed"


def test_no_op_only_pipeline(monkeypatch):
    from test_validator import fixture_plan
    request, response, directives = fixture_plan()
    case = {"input": request.model_dump(mode="json"), "expected_output": response.model_dump(mode="json")}
    install_test_components(monkeypatch, case)
    with TestClient(main.app) as client:
        result = client.post("/optimize-energy", json=case["input"])
    assert result.status_code == 200
    assert result.json()["directive_interpretation"][0]["directive_type"] == "no_op"


@pytest.mark.parametrize("stage", ["async", "sync"])
def test_whole_request_timeout(monkeypatch, stage):
    install_test_components(monkeypatch, CASES[0])
    monkeypatch.setattr(main, "REQUEST_TIMEOUT_SECONDS", 0.02)
    async def slow_async(request):
        await asyncio.sleep(0.2)
    def slow_sync(*args):
        time.sleep(0.2)
        return []
    monkeypatch.setattr(main, "interpret_notes" if stage == "async" else "optimize",
                        slow_async if stage == "async" else slow_sync)
    with TestClient(main.app) as client:
        started = time.monotonic()
        result = client.post("/optimize-energy", json=CASES[0]["input"])
        elapsed = time.monotonic() - started
    assert result.status_code == 500
    assert elapsed < 0.15


def test_original_inputs_survive_component_mutation(monkeypatch):
    install_test_components(monkeypatch, CASES[0])
    original = main.optimize
    def corrupt(request, directives):
        rows = original(request, directives)
        request.hours[0].demand_kwh += 10
        rows[0].grid_kwh += 10
        return rows
    monkeypatch.setattr(main, "optimize", corrupt)
    with TestClient(main.app) as client:
        result = client.post("/optimize-energy", json=CASES[0]["input"])
    assert result.status_code == 500


def test_malformed_json_remains_400():
    with TestClient(main.app) as client:
        result = client.post("/optimize-energy", content="{", headers={"Content-Type": "application/json"})
    assert result.status_code == 400


def test_serialized_schedule_is_replayed_not_only_in_memory_model(monkeypatch):
    install_test_components(monkeypatch, CASES[0])
    original_dump = OptimizeResponse.model_dump_json
    def changed_on_wire(self, *args, **kwargs):
        payload = json.loads(original_dump(self, *args, **kwargs))
        payload["hourly_plan"][0]["grid_kwh"] += 5
        return json.dumps(payload)
    monkeypatch.setattr(OptimizeResponse, "model_dump_json", changed_on_wire)
    with TestClient(main.app) as client:
        result = client.post("/optimize-energy", json=CASES[0]["input"])
    assert result.status_code == 500
    assert result.json()["error"]["code"] == "replay_validation_failed"

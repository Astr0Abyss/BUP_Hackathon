import asyncio
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.llm_interpreter import (
    DEFAULT_OPENAI_MODEL,
    InterpretationFailure,
    interpret_notes,
    interpret_operator_notes,
    interpret_operator_notes_async,
)


BATTERY = {
    "capacity_kwh": 200,
    "initial_energy_kwh": 100,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50,
}


class MemberOneCompatibilityTests(unittest.TestCase):
    def test_interpret_notes_forwards_request_contract(self):
        request = SimpleNamespace(
            operator_notes=["Keep the battery ready."],
            battery=BATTERY,
        )
        expected = object()
        delegate = AsyncMock(return_value=expected)

        with patch(
            "app.llm_interpreter.interpret_operator_notes_async",
            new=delegate,
        ):
            actual = asyncio.run(interpret_notes(request))

        self.assertIs(actual, expected)
        delegate.assert_awaited_once_with(
            request.operator_notes,
            request.battery,
        )


def expected(index, directive, adjustment=None, applies=True):
    return {
        "note_index": index,
        "applies": applies,
        "directive_type": directive,
        "structured_adjustment": adjustment,
    }


HIDDEN_STYLE_CASES = [
    ("by-80", ["PV is reduced by 80% between 13:00 and 15:00."],
     [expected(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]),
    ("to-20", ["Solar will be reduced to 20% from one until three PM."],
     [expected(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]),
    ("fraction", ["One-fifth of forecast PV remains from 1 PM to 3 PM."],
     [expected(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]),
    ("half", ["Roughly half of forecast PV will remain from 10 AM until noon."],
     [expected(0, "solar_reduction", {"hours": [10, 11], "factor": 0.5})]),
    ("distractor", ["The energy seminar starts at 3 PM tomorrow."],
     [expected(0, "no_op", None, False)]),
    ("no-charge", ["Charging is unavailable from 11 AM through 1 PM."],
     [expected(0, "no_charge_window", {"hours": [11, 12]})]),
    ("no-discharge", ["The battery cannot supply power from 5 PM until 7 PM."],
     [expected(0, "no_discharge_window", {"hours": [17, 18]})]),
    ("feeder-cap", ["Feeder intake cannot exceed 155 kWh from 6 PM to 9 PM."],
     [expected(0, "max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 155})]),
    ("relative-reserve", ["Retain 50% of battery capacity from 6 PM to 9 PM."],
     [expected(0, "minimum_battery_reserve",
               {"hours": [18, 19, 20], "minimum_energy_kwh": 100})]),
    ("combined", [
        "Maintain at least 90 kWh from 6 PM until 10 PM.",
        "Transformer grid import is limited to 180 kWh from 7 PM to 9 PM.",
        "The electricity workshop was moved to 2 PM next week.",
    ], [
        expected(0, "minimum_battery_reserve",
                 {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 90}),
        expected(1, "max_grid_window", {"hours": [19, 20], "max_grid_kwh": 180}),
        expected(2, "no_op", None, False),
    ]),
    ("repeated-indexes", [
        "Do not charge from 2 PM to 4 PM.",
        "Do not charge from 2 PM to 4 PM.",
    ], [
        expected(0, "no_charge_window", {"hours": [14, 15]}),
        expected(1, "no_charge_window", {"hours": [14, 15]}),
    ]),
]


class FakeResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed
        self.status = "completed"


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        content = outcome if isinstance(outcome, str) else json.dumps(outcome)
        message = SimpleNamespace(content=content, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=message,
        )])


class FakeClient:
    def __init__(self, parsed=None, error=None, outcomes=None, backup_outcomes=None):
        primary = outcomes if outcomes is not None else [error if error else parsed]
        self.responses = FakeResponses(primary)
        self.chat = SimpleNamespace(completions=FakeCompletions(backup_outcomes or []))
        self.options = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self


class CountingResponses:
    def __init__(self, delegate, counter):
        self.delegate = delegate
        self.counter = counter

    async def parse(self, **kwargs):
        self.counter[0] += 1
        return await self.delegate.parse(**kwargs)


class CountingClient:
    def __init__(self, delegate, counter):
        self.delegate = delegate
        self.counter = counter
        self.responses = CountingResponses(delegate.responses, counter)

    def with_options(self, **kwargs):
        return CountingClient(self.delegate.with_options(**kwargs), self.counter)


class StatusError(Exception):
    def __init__(self, status_code, message="provider detail"):
        super().__init__(message)
        self.status_code = status_code


def interpretation(index, directive, adjustment, applies=True):
    return {
        "note_index": index,
        "applies": applies,
        "directive_type": directive,
        "structured_adjustment": adjustment,
        "explanation": "canonical interpretation",
    }


def find_public_samples():
    root = Path(__file__).resolve().parents[1]
    candidates = (
        root / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
        root / "tmp" / "pdfs" / "participant_docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    )
    return next((path for path in candidates if path.exists()), None)


class InterpreterUnitTests(unittest.TestCase):
    def test_sends_all_notes_in_exactly_one_request(self):
        parsed = {
            "interpretations": [
                interpretation(0, "no_charge_window", {"hours": [2, 3, 4]}),
                interpretation(1, "no_op", None, applies=False),
                interpretation(2, "max_grid_window", {"hours": [19, 20], "max_grid_kwh": 180}),
            ]
        }
        client = FakeClient(parsed)
        notes = ["charger unavailable", "unrelated note", "grid cap"]

        result = interpret_operator_notes(notes, BATTERY, client=client)

        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], DEFAULT_OPENAI_MODEL)
        payload = json.loads(call["input"][1]["content"])
        self.assertEqual([item["text"] for item in payload["operator_notes"]], notes)
        self.assertEqual([item["note_index"] for item in payload["operator_notes"]], [0, 1, 2])
        self.assertEqual(len(result), 3)

    def test_model_is_configurable(self):
        client = FakeClient(
            {"interpretations": [interpretation(0, "no_op", None, applies=False)]}
        )
        interpret_operator_notes(["irrelevant"], BATTERY, client=client, model="custom-model")
        self.assertEqual(client.responses.calls[0]["model"], "custom-model")

    def test_capacity_relative_reserve_result_passes_guardrails(self):
        client = FakeClient(
            {
                "interpretations": [
                    interpretation(
                        0,
                        "minimum_battery_reserve",
                        {"hours": [18, 19, 20], "minimum_energy_kwh": 100},
                    )
                ]
            }
        )
        result = interpret_operator_notes(
            ["Keep 50% of capacity from 6 PM to 9 PM"], BATTERY, client=client
        )
        self.assertEqual(result[0]["structured_adjustment"]["minimum_energy_kwh"], 100)

    def test_malformed_model_output_is_not_converted_to_no_op(self):
        client = FakeClient({"interpretations": []})
        with self.assertRaises(InterpretationFailure):
            interpret_operator_notes(["Do not charge from 2 PM to 4 PM"], BATTERY, client=client)

    def test_primary_timeout_retries_once_then_returns_success(self):
        valid = {"interpretations": [interpretation(0, "no_op", None, applies=False)]}
        client = FakeClient(outcomes=[TimeoutError("secret"), valid])
        result = interpret_operator_notes(["note"], BATTERY, client=client)
        self.assertEqual(result[0]["directive_type"], "no_op")
        self.assertEqual(len(client.responses.calls), 2)
        self.assertTrue(all(x["max_retries"] == 0 for x in client.options))

    def test_primary_500_retries_once_then_returns_success(self):
        valid = {"interpretations": [interpretation(0, "no_op", None, applies=False)]}
        client = FakeClient(outcomes=[StatusError(500), valid])
        result = interpret_operator_notes(["note"], BATTERY, client=client)
        self.assertEqual(result[0]["directive_type"], "no_op")
        self.assertEqual(len(client.responses.calls), 2)

    def test_rate_limit_goes_directly_to_backup(self):
        primary = FakeClient(outcomes=[StatusError(429)])
        backup_result = {"interpretations": [
            interpretation(0, "no_charge_window", {"hours": [2, 3]}, True)
        ]}
        backup = FakeClient(backup_outcomes=[backup_result])
        result = interpret_operator_notes(
            ["charger unavailable"], BATTERY,
            client=primary, backup_client=backup,
        )
        self.assertEqual(result[0]["directive_type"], "no_charge_window")
        self.assertEqual(len(primary.responses.calls), 1)
        self.assertEqual(len(backup.chat.completions.calls), 1)

    def test_malformed_primary_retries_then_backup_succeeds(self):
        primary = FakeClient(outcomes=[{"interpretations": []}, {"interpretations": []}])
        valid = {"interpretations": [interpretation(0, "no_op", None, applies=False)]}
        backup = FakeClient(backup_outcomes=[valid])
        result = interpret_operator_notes(
            ["unrelated"], BATTERY, client=primary, backup_client=backup,
        )
        self.assertEqual(result[0]["directive_type"], "no_op")
        self.assertEqual(len(primary.responses.calls), 2)

    def test_both_providers_fail_with_redacted_typed_error(self):
        client = FakeClient(error=TimeoutError("secret-bearing provider detail"))
        backup = FakeClient(backup_outcomes=[StatusError(500, "backup-secret")])
        with self.assertRaisesRegex(InterpretationFailure, "unavailable") as raised:
            interpret_operator_notes(
                ["note"], BATTERY, client=client, backup_client=backup,
            )
        self.assertNotIn("secret-bearing", str(raised.exception))
        self.assertNotIn("backup-secret", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_malformed_backup_json_is_a_typed_failure(self):
        primary = FakeClient(outcomes=[StatusError(429)])
        backup = FakeClient(backup_outcomes=["{not-json"])
        with self.assertRaises(InterpretationFailure):
            interpret_operator_notes(
                ["note"], BATTERY, client=primary, backup_client=backup,
            )

    def test_cancellation_propagates(self):
        client = FakeClient(error=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(interpret_operator_notes_async(
                ["note"], BATTERY, client=client,
            ))

    def test_rejects_invalid_note_input_before_model_call(self):
        client = FakeClient()
        for notes in ([], ["", "ok"], ["a", "b", "c", "d"]):
            with self.subTest(notes=notes):
                with self.assertRaises(ValueError):
                    interpret_operator_notes(notes, BATTERY, client=client)
        self.assertEqual(client.responses.calls, [])


@unittest.skipUnless(
    os.getenv("RUN_LIVE_LLM_TESTS") == "1" and bool(os.getenv("OPENAI_API_KEY")),
    "set RUN_LIVE_LLM_TESTS=1 and OPENAI_API_KEY to run official live semantics",
)
class OfficialPublicSemanticLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_path = find_public_samples()
        if sample_path is None:
            raise unittest.SkipTest("official public sample JSON is unavailable")
        cls.cases = json.loads(sample_path.read_text(encoding="utf-8"))["cases"]

    def test_all_ten_official_interpretations(self):
        latencies = []
        failures = []
        for case in self.cases:
            started = time.perf_counter()
            actual = interpret_operator_notes(
                case["input"]["operator_notes"], case["input"]["battery"]
            )
            latencies.append(time.perf_counter() - started)
            expected = case["expected_output"]["directive_interpretation"]
            actual_semantics = [
                {
                    "note_index": item["note_index"],
                    "applies": item["applies"],
                    "directive_type": item["directive_type"],
                    "structured_adjustment": item["structured_adjustment"],
                }
                for item in actual
            ]
            expected_semantics = [
                {
                    "note_index": item["note_index"],
                    "applies": item["applies"],
                    "directive_type": item["directive_type"],
                    "structured_adjustment": item["structured_adjustment"],
                }
                for item in expected
            ]
            if actual_semantics != expected_semantics:
                failures.append((case["id"], actual_semantics, expected_semantics))

        print(
            f"official_live_cases={len(self.cases)} "
            f"average_latency_seconds={sum(latencies) / len(latencies):.3f}"
        )
        self.assertEqual(failures, [])


@unittest.skipUnless(
    os.getenv("RUN_LIVE_LLM_TESTS") == "1" and bool(os.getenv("OPENAI_API_KEY")),
    "set RUN_LIVE_LLM_TESTS=1 and OPENAI_API_KEY to run hidden-style live semantics",
)
class HiddenStyleSemanticLiveTests(unittest.TestCase):
    def test_hidden_style_semantics_and_latency(self):
        failures, latencies, calls = asyncio.run(self._run_cases())
        ordered = sorted(latencies)
        p50 = statistics.median(ordered)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        retries = max(0, calls - len(HIDDEN_STYLE_CASES))
        print(
            f"hidden_live_cases={len(HIDDEN_STYLE_CASES)} "
            f"p50_seconds={p50:.3f} p95_seconds={p95:.3f} "
            f"primary_calls={calls} retries={retries} failures={len(failures)}"
        )
        self.assertEqual(failures, [])

    async def _run_cases(self):
        from openai import AsyncOpenAI

        failures = []
        latencies = []
        counter = [0]
        async with AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"], max_retries=0,
        ) as raw_client:
            client = CountingClient(raw_client, counter)
            for case_id, notes, expected_semantics in HIDDEN_STYLE_CASES:
                started = time.perf_counter()
                try:
                    actual = await interpret_operator_notes_async(
                        notes, BATTERY, client=client,
                    )
                    actual_semantics = [
                        {
                            "note_index": item["note_index"],
                            "applies": item["applies"],
                            "directive_type": item["directive_type"],
                            "structured_adjustment": item["structured_adjustment"],
                        }
                        for item in actual
                    ]
                    if actual_semantics != expected_semantics:
                        failures.append(
                            (case_id, actual_semantics, expected_semantics)
                        )
                except Exception as exc:
                    failures.append((case_id, type(exc).__name__))
                latencies.append(time.perf_counter() - started)
        return failures, latencies, counter[0]


if __name__ == "__main__":
    unittest.main()

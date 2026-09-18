import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from app.guardrails import (
    GuardrailViolation,
    validate_and_normalize,
    validate_interpretations,
)


def entry(index=0, directive="no_op", adjustment=None, applies=False):
    return {
        "note_index": index,
        "applies": applies,
        "directive_type": directive,
        "structured_adjustment": adjustment,
        "explanation": "test interpretation",
    }


class GuardrailTests(unittest.TestCase):
    def validate(self, entries, count=None, capacity=200):
        return validate_interpretations(
            entries,
            note_count=len(entries) if count is None else count,
            battery_capacity_kwh=capacity,
        )

    def test_member1_validate_and_normalize_contract(self):
        class DirectiveInterpretation:
            def __init__(self, values):
                self.note_index = values["note_index"]

            @classmethod
            def model_validate(cls, values):
                return cls(values)

        schemas = ModuleType("app.schemas")
        schemas.DirectiveInterpretation = DirectiveInterpretation
        request = SimpleNamespace(
            operator_notes=["first", "second"],
            battery=SimpleNamespace(capacity_kwh=200),
        )
        raw = [entry(0), entry(1)]

        with patch.dict(sys.modules, {"app.schemas": schemas}):
            result = validate_and_normalize(raw, request)

        self.assertTrue(all(isinstance(item, DirectiveInterpretation) for item in result))
        self.assertEqual([item.note_index for item in result], [0, 1])

    def test_accepts_all_six_directive_types(self):
        entries = [
            entry(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}, True),
            entry(1, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 100}, True),
            entry(2, "no_op", None, False),
        ]
        self.assertEqual(len(self.validate(entries)), 3)

        for directive, adjustment in (
            ("no_charge_window", {"hours": [2]}),
            ("no_discharge_window", {"hours": [18]}),
            ("max_grid_window", {"hours": [19], "max_grid_kwh": 155}),
        ):
            with self.subTest(directive=directive):
                result = self.validate([entry(0, directive, adjustment, True)])
                self.assertEqual(result[0]["directive_type"], directive)

    def test_sorts_valid_unsorted_hours_without_mutating_input(self):
        raw = [entry(0, "no_charge_window", {"hours": [4, 2, 3]}, True)]
        result = self.validate(raw)
        self.assertEqual(result[0]["structured_adjustment"]["hours"], [2, 3, 4])
        self.assertEqual(raw[0]["structured_adjustment"]["hours"], [4, 2, 3])

    def test_rejects_missing_duplicate_or_out_of_order_indexes(self):
        invalid_sets = (
            [entry(0), entry(0)],
            [entry(1), entry(0)],
            [entry(0)],
        )
        for entries in invalid_sets:
            with self.subTest(entries=entries):
                with self.assertRaises(GuardrailViolation):
                    self.validate(entries, count=2)

    def test_rejects_bad_no_op_and_bad_real_applies(self):
        bad = (
            [entry(0, "no_op", {"hours": [1]}, False)],
            [entry(0, "no_op", None, True)],
            [entry(0, "no_charge_window", {"hours": [1]}, False)],
        )
        for entries in bad:
            with self.subTest(entries=entries):
                with self.assertRaises(GuardrailViolation):
                    self.validate(entries)

    def test_rejects_unsupported_or_invented_fields(self):
        bad = entry(0)
        bad["demand_kwh"] = 50
        with self.assertRaises(GuardrailViolation):
            self.validate([bad])
        with self.assertRaises(GuardrailViolation):
            self.validate([entry(0, "change_tariff", None, True)])

    def test_rejects_wrong_adjustment_shapes(self):
        cases = (
            entry(0, "solar_reduction", {"hours": [1]}, True),
            entry(0, "no_charge_window", {"hours": [1], "factor": 0.5}, True),
            entry(0, "max_grid_window", {"hours": [1], "max_grid_kwh": 10, "tariff": 2}, True),
        )
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(GuardrailViolation):
                    self.validate([bad])

    def test_rejects_invalid_hours(self):
        bad_hours = ([1, 1], [-1], [24], [1.0], [True], [])
        for hours in bad_hours:
            with self.subTest(hours=hours):
                with self.assertRaises(GuardrailViolation):
                    self.validate([entry(0, "no_charge_window", {"hours": hours}, True)])

    def test_rejects_invalid_factor_reserve_and_grid_cap(self):
        cases = (
            entry(0, "solar_reduction", {"hours": [1], "factor": -0.1}, True),
            entry(0, "solar_reduction", {"hours": [1], "factor": 1.1}, True),
            entry(0, "solar_reduction", {"hours": [1], "factor": math.nan}, True),
            entry(0, "minimum_battery_reserve", {"hours": [1], "minimum_energy_kwh": 201}, True),
            entry(0, "minimum_battery_reserve", {"hours": [1], "minimum_energy_kwh": -1}, True),
            entry(0, "max_grid_window", {"hours": [1], "max_grid_kwh": -1}, True),
            entry(0, "max_grid_window", {"hours": [1], "max_grid_kwh": math.inf}, True),
        )
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(GuardrailViolation):
                    self.validate([bad])


if __name__ == "__main__":
    unittest.main()

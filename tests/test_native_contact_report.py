"""CPU-only checks of descriptive report arithmetic; no native environment."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import signal
import tempfile

import native_contact_comparison as diagnostic

import numpy as np

from native_contact_comparison import Capture, comparison_report, compare, contact_summary


def saved(value):
    array = np.asarray([value], dtype=np.float32)
    return Capture({"all_state_leaf": array}, {"all_state_leaf": {"shape": [1], "dtype": "float32"}},
                   ("complete-example-state",), {"root_position_m": array, "rng": np.asarray([7, 8], np.uint32)})


class ContactReportTests(unittest.TestCase):
    def test_reports_native_repeat_variation_and_cross_mode_deltas_without_a_verdict(self):
        result = comparison_report([saved(v) for v in (0, 1, 2)], [saved(v) for v in (1, 2, 3)])
        self.assertEqual([len(result[key]) for key in ("plain_pairs", "instrumented_pairs", "cross_mode_pairs")], [3, 3, 9])
        self.assertEqual(result["descriptive_envelopes"]["root_position_m"], {
            "plain_repeat_max_abs": 2., "cross_mode_max_abs": 3., "cross_minus_plain_envelope": 1.})
        self.assertEqual(result["descriptive_envelopes"]["rng"]["cross_mode_max_abs"], 0.)
        self.assertNotIn("passed", result)
        self.assertNotIn("tolerance", result)

    def test_full_tree_static_and_aval_changes_remain_visible(self):
        before, after = saved(1), saved(1)
        after.structure = ("different-static-model",)
        self.assertFalse(compare(before, after)["same_tree_and_avals"])
        after = saved(1)
        after.arrays["another_native_leaf"] = np.asarray([4])
        self.assertEqual(compare(before, after)["different_leaves"]["another_native_leaf"], {"missing": True})
        after = saved(1)
        after.avals["all_state_leaf"]["weak_type"] = True
        self.assertFalse(compare(before, after)["same_tree_and_avals"])

    def test_equal_nonfinite_cache_values_are_reported_outside_physical_quantities(self):
        left, right = saved(1), saved(1)
        left.arrays["native_cache"] = np.asarray([np.inf])
        right.arrays["native_cache"] = np.asarray([np.inf])
        with np.errstate(invalid="ignore"):
            result = compare(left, right)
        self.assertEqual(result["different_leaves"], {})
        self.assertEqual(result["nonfinite_left"], {"native_cache": 1})
        self.assertEqual(result["nonfinite_right"], {"native_cache": 1})

    def test_native_json_negative_exponent_weights_remain_floats(self):
        with tempfile.TemporaryDirectory() as scratch:
            config = Path(scratch) / "config.json"
            config.write_text('{"env_config":{"reward_config":{"scales":'
                              '{"penalty_torque":-2e-05,"smoothness_joint":-1e-06}}}}')
            scales = diagnostic.read_native_config(config)["env_config"]["reward_config"]["scales"]
        self.assertIsInstance(scales["penalty_torque"], float)
        self.assertIsInstance(scales["smoothness_joint"], float)
        self.assertEqual(scales, {"penalty_torque": -2e-5, "smoothness_joint": -1e-6})

    def test_external_interrupts_write_a_report_and_remain_interrupts(self):
        for number in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=number), tempfile.TemporaryDirectory() as scratch:
                output = Path(scratch) / "out"
                argv = ["native_contact_comparison.py", "--config", "unused", "--motion", "unused",
                        "--out", str(output), "--source-reference", "example-source",
                        "--motion-reference", "example-motion"]
                original = signal.getsignal(number)
                with patch("sys.argv", argv), patch.object(diagnostic, "run", side_effect=lambda *_: signal.raise_signal(number)):
                    with self.assertRaisesRegex(InterruptedError, signal.Signals(number).name):
                        diagnostic.main()
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "interrupted")
                self.assertEqual(report["error_type"], "InterruptedError")
                self.assertEqual(report["physical_calls"], [])
                self.assertIn("finished_at", report)
                self.assertEqual(signal.getsignal(number), original)

    def test_positive_force_requires_a_present_solved_slot_not_geometry_alone(self):
        samples = {"constraint_present": np.asarray([[[True, True, False], [True, True, False]]]),
                   "normal_n": np.asarray([[[0., 2., 900.], [0., 0., 900.]]]),
                   "geom": np.tile(np.asarray([[0, 1], [0, 2], [0, 3]]), (1, 2, 1, 1)),
                   "total_force_world_n": np.zeros((1, 2, 3, 3)),
                   "interval_start_s": np.asarray([[0., .002]]),
                   "interval_end_s": np.asarray([[.002, .004]])}
        result = contact_summary(samples, floor_geom_id=0)
        self.assertEqual(result["positive_normal_slots"], 1)
        self.assertEqual(result["positive_ground_normal_slots"], 1)
        self.assertEqual(result["maximum_normal_n"], 2.)
        self.assertEqual(result["positive_substeps"], 1)
        self.assertEqual(result["substeps"], 2)
        samples["normal_n"][:] = 0
        self.assertEqual(contact_summary(samples, 0)["positive_normal_slots"], 0)


if __name__ == "__main__":
    unittest.main()

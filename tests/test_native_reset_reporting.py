"""Pure NumPy checks for manual native reset diagnostic reporting; no physics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("native_reset_continuation.py")
spec = importlib.util.spec_from_file_location("native_reset_continuation", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def sample():
    qpos, qvel = np.zeros((2, 9)), np.zeros((2, 8))
    qpos[:, 3] = 1
    return {"data/qpos": qpos, "data/qvel": qvel, "done": np.array([False, False]),
            "data/derived_cache": np.ones((2, 1))}


class ReportingTest(unittest.TestCase):
    def test_all_leaves_including_derived_caches_and_done_remain_visible(self):
        a, b = sample(), sample()
        b["data/derived_cache"][0] = 3
        b["done"][1] = True
        changed = diagnostic.leaf_differences(a, b)
        self.assertEqual(set(changed), {"data/derived_cache", "done"})
        self.assertEqual(changed["data/derived_cache"], {"changed": 1, "max_abs": 2., "nonfinite_differences": 0})
        self.assertEqual(diagnostic.leaf_differences(a, a), {})

    def test_physical_coordinates_keep_worlds_names_and_units(self):
        a, b = sample(), sample()
        b["data/qpos"][0, 0] = .25
        b["data/qpos"][0, 7] = .5
        b["data/qvel"][0, 2] = .75
        b["data/qvel"][0, 4] = 1.25
        b["data/qvel"][0, 7] = 1.5
        rows = diagnostic.physical_differences(a, b, ["hip", "knee"], [0])
        indexed = {(r.get("field"), r["coordinate"]): r for r in rows}
        for field, coordinate, unit, difference in (("data/qpos", "root/x", "m", .25),
                ("data/qpos", "hip", "rad", .5), ("data/qvel", "root/linear/z", "m/s", .75),
                ("data/qvel", "root/angular_native/y", "rad/s", 1.25), ("data/qvel", "knee", "rad/s", 1.5)):
            self.assertEqual(indexed[field, coordinate]["unit"], unit)
            self.assertEqual(indexed[field, coordinate]["abs_difference"], difference)
        self.assertTrue(all(r["world"] == 0 for r in rows))
        self.assertEqual(len(rows), 18)  # Every qpos/qvel coordinate plus orientation angle.

    def test_orientation_angle_handles_sign_equivalence_and_known_rotation(self):
        a, b = sample(), sample()
        b["data/qpos"][0, 3:7] = [-1, 0, 0, 0]
        rows = diagnostic.physical_differences(a, b, ["hip", "knee"], [0])
        self.assertEqual(rows[-1]["difference"], 0.)
        b["data/qpos"][0, 3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        rows = diagnostic.physical_differences(a, b, ["hip", "knee"], [0])
        self.assertAlmostEqual(rows[-1]["difference"], np.pi / 2)

    def test_nonfinite_values_are_explicit_and_report_can_be_written(self):
        a, b = sample(), sample()
        b["data/qvel"][0, 0] = np.nan
        changed = diagnostic.leaf_differences(a, b)
        self.assertEqual(changed["data/qvel"]["nonfinite_differences"], 1)
        rows = diagnostic.physical_differences(a, b, ["hip", "knee"], [0])
        self.assertTrue(any(r.get("right") == "nan" for r in rows))
        json.dumps({"all_leaves": changed, "physical": rows}, allow_nan=False)

    def test_incompatible_layouts_fail_without_selecting_a_different_field_subset(self):
        a, b = sample(), sample()
        b.pop("done")
        with self.assertRaisesRegex(ValueError, "leaf paths"):
            diagnostic.leaf_differences(a, b)
        with self.assertRaisesRegex(ValueError, "physical coordinate shape"):
            diagnostic.physical_differences(a, a, ["wrong_width"], [0])

    def test_manual_help_is_available_without_importing_native_engines(self):
        code = """import runpy, sys
sys.argv = ['diagnostic', '--help']
try:
    runpy.run_path(%r, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('jax', 'mujoco', 'track_mj'))
""" % str(SCRIPT)
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--steps", result.stdout)


if __name__ == "__main__":
    unittest.main()

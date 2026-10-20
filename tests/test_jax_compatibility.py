"""CPU checks of the actual reward and render expressions across JAX versions."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class JaxCompatibilityTests(unittest.TestCase):
    def test_native_reward_caps_keep_negative_values_and_upper_bound(self):
        values = jp.array([-1000000., -3., 0., 1000000.])
        expected = np.array([-20000.01, -.07, -.01, 10000.], dtype=np.float32)
        paths = sorted(path for path in (ROOT / "track_mj/envs").glob("g1_tracking*/train/g1_env_tracking_general*.py")
                       if path.name in {"g1_env_tracking_general.py", "g1_env_tracking_general_dr.py"})
        for path in paths:
            assignments = [node for node in ast.walk(ast.parse(path.read_text()))
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "reward"
                                   for target in node.targets)
                           and isinstance(node.value, ast.Call)
                           and ast.unparse(node.value.func) == "jp.clip"]
            self.assertEqual(len(assignments), 1, str(path))
            expression = compile(ast.Expression(assignments[0].value), str(path), "eval")
            def reward(value):
                return eval(expression, {"jp": jp, "rewards": {"tracking": value, "penalty": -.5},
                                         "self": SimpleNamespace(dt=.02)})
            with self.subTest(environment=str(path.relative_to(ROOT))):
                np.testing.assert_allclose(jax.jit(reward)(values), expected, rtol=1e-6)
                self.assertAlmostEqual(float(jax.grad(reward)(jp.array(-3.))), .02)
                self.assertEqual(float(jax.grad(reward)(jp.array(1000000.))), 0.)

    def test_actual_render_tree_selection_preserves_structure(self):
        path = ROOT / "track_mj/learning/policy/dagger/utils.py"
        calls = [node for node in ast.walk(ast.parse(path.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and ast.unparse(node.func) == "jax.tree_util.tree_map"]
        self.assertEqual(len(calls), 1)
        expression = compile(ast.Expression(calls[0]), str(path), "eval")
        data = {"qpos": jp.arange(6).reshape(2, 3), "nested": (jp.array([4, 5]),)}
        selected = eval(expression, {"jax": jax, "mjx_data": data, "sim_id": 1})
        self.assertEqual(jax.tree_util.tree_structure(selected), jax.tree_util.tree_structure(data))
        np.testing.assert_array_equal(selected["qpos"], [3, 4, 5])
        self.assertEqual(int(selected["nested"][0]), 5)


if __name__ == "__main__":
    unittest.main()

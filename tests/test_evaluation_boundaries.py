"""CPU-only JAX checks. No MuJoCo environment or physics is constructed."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import ast
from functools import partial
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import unittest

import jax
import jax.numpy as jp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_evaluation", ROOT / "track_mj/envs/evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class State(NamedTuple):
    data: object
    obs: object
    reward: object
    done: object
    metrics: object
    info: object

    def replace(self, **changes):
        return self._replace(**changes)


class Data(NamedTuple):
    qpos: object
    qvel: object
    xpos: object
    xquat: object

    def replace(self, **changes):
        return self._replace(**changes)


class TrajState(NamedTuple):
    traj_no: object


class Carry(NamedTuple):
    traj_state: object


def assert_tree_equal(test, left, right):
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def native_methods(family, dr=False):
    """Execute the actual production step methods with a tiny fake physics port.

    Loading just these AST nodes avoids importing native MuJoCo and the package's
    eager environment registry. The motor-target/history/RNG code is unmodified.
    """
    path = ROOT / "track_mj/envs" / family / "train/g1_env_tracking_general.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1TrackingGeneralEnv")
    wanted = {"step", "step_no_autoreset", "_step", "_reset_and_update_state", "_get_motor_targets"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if dr:
        dr_path = path.with_name("g1_env_tracking_general_dr.py")
        dr_cls = next(node for node in ast.parse(dr_path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "G1TrackingGeneralDREnv")
        methods.extend(node for node in dr_cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *methods], type_ignores=[])
    ast.fix_missing_locations(module)

    def physics(rng, model, data, targets, **kwargs):
        # Same native per-substep key split, without any physical integration.
        for _ in range(kwargs["n_substeps"]):
            rng, _ = jax.random.split(rng)
        return rng, data.replace(qpos=data.qpos.at[7:].set(targets)), jp.zeros_like(targets)

    namespace = {"jax": jax, "jp": jp, "partial": partial, "torque_step": physics, "torque_step_dr": physics,
                 "copy_state_containers": evaluation.copy_state_containers, "mjx_impl": lambda cfg: "jax"}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in wanted if name in namespace}


def fake_native(family, residual, terminate, dr=False):
    methods = native_methods(family, dr)
    Env = type("NativeStepWithFakePhysics", (), methods)
    env = Env()
    env._config = SimpleNamespace(action_scale=2., student_use_residual_action=residual, history_len=0,
                                  episode_length=100, reward_config=SimpleNamespace(scales={"target_sum": 1.}),
                                  push_config=SimpleNamespace(enable=False, magnitude_range=[.1, 1.]))
    env.action_joint_ids = jp.arange(29)
    env._default_qpos = jp.zeros(29)
    env._kps = env._kds = env.torque_limit = jp.ones(29)
    env.mjx_model = None
    env.n_substeps, env.dt = 10, .02
    reference = SimpleNamespace(qpos=jp.concatenate([jp.zeros(7), jp.arange(29) / 10]))
    env.th = SimpleNamespace(get_current_traj_data_with_trajectory=lambda trajectory, carry: reference,
                             update_state_with_trajectory=lambda trajectory, carry: carry)
    env._get_reward = lambda data, reference, action, targets, torque, info: {"target_sum": targets.sum()}
    env._get_termination = lambda data, reference, info: jp.array(terminate)
    env._get_obs = lambda data, reference, info: ({"state": info["last_motor_targets"]}, jp.zeros(0))

    def fresh(key, trajectory=None):
        info = {"rng": key, "step": jp.array(0), "traj_no": jp.array(0),
                "traj_info": Carry(TrajState(jp.array(0))), "truncation": jp.array(0.),
                "last_motor_targets": jp.zeros(29), "last_action": jp.zeros(29), "last_root_pos": jp.zeros(3),
                "last_root_ori": jp.array([1., 0, 0, 0]), "last_dof_pos": jp.zeros(29),
                "last_rigid_body_pos": jp.zeros((1, 3)), "last_rigid_body_ori": jp.array([[1., 0, 0, 0]]),
                "last_joint_vel": jp.zeros(29)}
        if dr:
            info.update(push_step=jp.array(0), push_interval_steps=jp.array(10), push=jp.zeros(2),
                        kp_scale=jp.ones(29), kd_scale=jp.ones(29), rfi_lim_scale=jp.zeros(29))
        data = Data(jp.concatenate([jp.zeros(3), jp.array([1., 0, 0, 0]), jp.zeros(29)]), jp.zeros(35),
                    jp.zeros((1, 3)), jp.array([[1., 0, 0, 0]]))
        return State(data, {"state": jp.zeros(29)}, jp.array(0.), jp.array(0.), {"reward/target_sum": jp.array(0.)}, info)
    env.reset = fresh
    return env, fresh(jax.random.PRNGKey(7)), reference


def actual_vector_wrapper(family, env):
    path = ROOT / "track_mj/envs" / family / "utils/wrapper.py"
    wanted = {"VmapWrapper", "_annotate_worldid"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    class Wrapper:
        def __init__(self, wrapped):
            self.env = wrapped
    namespace = {"jax": jax, "jp": jp, "Wrapper": Wrapper, "MJX_WORLD_AXIS_NAME": "test_worlds",
                 "owned_data": evaluation.owned_data,
                 "mjx": SimpleNamespace(forward=lambda model, data: data.replace(xpos=data.xpos + 10))}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["VmapWrapper"](env)


class EvaluationBoundaryTests(unittest.TestCase):
        # Keeping the old world's efc__id here would mix it with a new global
        # contact order. Deliberately do not promise derived-cache identity.




    def test_actual_native_target_and_last_target_semantics(self):
        for family in ("g1_tracking", "g1_tracking_dagger"):
            for residual in (True, False):
                with self.subTest(family=family, residual=residual):
                    env, pre, reference = fake_native(family, residual, True)
                    old_info = dict(pre.info)
                    action = jp.arange(29) / 100
                    terminal = env.step_no_autoreset(pre, action, object())
                    expected = action * 2
                    if family == "g1_tracking" or residual:
                        expected += reference.qpos[7:]
                    np.testing.assert_allclose(terminal.info["last_motor_targets"], expected)
                    np.testing.assert_array_equal(terminal.obs["state"], terminal.info["last_motor_targets"])
                    np.testing.assert_array_equal(terminal.info["last_action"], action)
                    self.assertTrue(bool(terminal.done))
                    assert_tree_equal(self, pre.info, old_info)
                    training = env.step(pre, action, object())
                    np.testing.assert_array_equal(training.data.qpos[7:], jp.zeros(29))
                    np.testing.assert_array_equal(training.reward, terminal.reward)
                    np.testing.assert_array_equal(training.done, terminal.done)

    def test_actual_native_training_and_evaluation_match_before_reset(self):
        for family in ("g1_tracking", "g1_tracking_dagger"):
            for dr in (False, True):
                env, pre, _ = fake_native(family, False, False, dr=dr)
                assert_tree_equal(self, env.step(evaluation.copy_state_containers(pre), jp.ones(29), object()),
                                  env.step_no_autoreset(pre, jp.ones(29), object()))

    def test_actual_native_dr_keeps_terminal_targets_before_reset(self):
        for family in ("g1_tracking", "g1_tracking_dagger"):
            env, pre, reference = fake_native(family, False, True, dr=True)
            terminal = env.step_no_autoreset(pre, jp.ones(29), object())
            expected = jp.full(29, 2.) + (reference.qpos[7:] if family == "g1_tracking" else 0)
            np.testing.assert_allclose(terminal.info["last_motor_targets"], expected)
            training = env.step(pre, jp.ones(29), object())
            np.testing.assert_array_equal(training.data.qpos[7:], jp.zeros(29))
            np.testing.assert_array_equal(training.reward, terminal.reward)
            np.testing.assert_array_equal(training.done, terminal.done)



    def test_vector_terminal_and_physical_measurement_are_separate(self):
        for family in ("g1_tracking", "g1_tracking_dagger"):
            env, _, _ = fake_native(family, False, False)
            env._get_termination = lambda data, reference, info: data.qpos[7] > .5
            vector = actual_vector_wrapper(family, env)
            pre = vector.reset(jax.random.split(jax.random.PRNGKey(5), 2), object())
            saved = evaluation.copy_state_containers(pre)
            terminal = vector.step_no_autoreset(pre, jp.zeros((2, 29)).at[1, 0].set(1), object())
            physical = vector.physical_data(terminal)
            np.testing.assert_array_equal(terminal.done, [False, True])
            np.testing.assert_array_equal(terminal.info["last_motor_targets"][1, 0], 2)
            np.testing.assert_array_equal(physical.xpos, jp.full((2, 1, 3), 10))
            np.testing.assert_array_equal(terminal.data.xpos, jp.zeros((2, 1, 3)))
            assert_tree_equal(self, pre, saved)

    def test_physical_measurement_owns_its_array_storage(self):
        source = {"qpos": jp.ones((2, 3))}
        owned = evaluation.owned_data(source)
        np.testing.assert_array_equal(owned["qpos"], source["qpos"])
        self.assertNotEqual(owned["qpos"].unsafe_buffer_pointer(), source["qpos"].unsafe_buffer_pointer())
        self.assertFalse(evaluation.contact_measurement()["available"])


if __name__ == "__main__":
    unittest.main()

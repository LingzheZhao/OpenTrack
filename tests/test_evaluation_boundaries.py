"""CPU-only JAX checks. No MuJoCo environment or physics is constructed."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GLI_PATH", str(ROOT))
from track_mj.envs import evaluation

NATIVE = {family: import_module(f"track_mj.envs.{family}.train.g1_env_tracking_general")
          for family in ("g1_tracking", "g1_tracking_dagger")}
NATIVE_DR = {family: import_module(f"track_mj.envs.{family}.train.g1_env_tracking_general_dr")
             for family in NATIVE}
WRAPPERS = {family: import_module(f"track_mj.envs.{family}.utils.wrapper") for family in NATIVE}


def physics(rng, model, data, targets, **kwargs):
    """Keep native key cadence, substituting only physical integration."""
    for _ in range(kwargs["n_substeps"]):
        rng, _ = jax.random.split(rng)
    return rng, data.replace(qpos=data.qpos.at[7:].set(targets)), jp.zeros_like(targets)


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


def fake_native(family, residual, terminate, dr=False):
    module = NATIVE_DR[family] if dr else NATIVE[family]
    cls = module.G1TrackingGeneralDREnv if dr else module.G1TrackingGeneralEnv
    # Bypass construction: this is the real class with only its physics,
    # reference source, observation/reward reads and reset port substituted.
    env = object.__new__(cls)
    env._config = SimpleNamespace(ctrl_dt=.02, sim_dt=.002, action_scale=2., student_use_residual_action=residual, history_len=0,
                                  episode_length=100, reward_config=SimpleNamespace(scales={"target_sum": 1.}),
                                  push_config=SimpleNamespace(enable=False, magnitude_range=[.1, 1.]))
    env.action_joint_ids = jp.arange(29)
    env._default_qpos = jp.zeros(29)
    env._kps = env._kds = env.torque_limit = jp.ones(29)
    env._mjx_model = SimpleNamespace(nu=29)
    if family == "g1_tracking_dagger":
        env.obstacle_geom_ids, env._obstacle_centers, env.reference_mode = (), jp.zeros((0,3)), "tracking"
        env._warp_obstacle_mode = None
    env._ctrl_dt, env._sim_dt = .02, .002
    reference = SimpleNamespace(qpos=jp.concatenate([jp.zeros(7), jp.arange(29) / 10]))
    env.th = SimpleNamespace(get_current_traj_data_with_trajectory=lambda trajectory, carry: reference,
                             update_state_with_trajectory=lambda trajectory, carry: carry)
    env._get_reward = lambda data, reference, action, targets, torque, info: {"target_sum": targets.sum()}
    env._get_termination = lambda data, reference, info: jp.array(terminate)
    env._get_obs = lambda data, reference, info: ({"state": info["last_motor_targets"]}, jp.zeros(0))

    def fresh(key, trajectory=None, *, obstacle_centers=None):
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


class EvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.substitutions = ExitStack()
        self.addCleanup(self.substitutions.close)
        for module in (*NATIVE.values(), *NATIVE_DR.values()):
            port = "torque_step_dr" if module in NATIVE_DR.values() else "torque_step"
            self.substitutions.enter_context(patch.object(module, port, side_effect=physics))
            self.substitutions.enter_context(patch.object(module, "mjx_impl", return_value="jax"))
        self.substitutions.enter_context(patch.object(
            NATIVE["g1_tracking"].mjx, "forward",
            side_effect=lambda model, data: data.replace(xpos=data.xpos + 10)))

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
            vector = WRAPPERS[family].VmapWrapper(env)
            pre = vector.reset(jax.random.split(jax.random.PRNGKey(5), 2), object())
            saved = evaluation.copy_state_containers(pre)
            terminal = vector.step_no_autoreset(pre, jp.zeros((2, 29)).at[1, 0].set(1), object())
            physical = vector.physical_data(terminal)
            np.testing.assert_array_equal(terminal.done, [False, True])
            np.testing.assert_array_equal(terminal.info["last_motor_targets"][1, 0], 2)
            np.testing.assert_array_equal(physical.xpos, jp.full((2, 1, 3), 10))
            np.testing.assert_array_equal(terminal.data.xpos, jp.zeros((2, 1, 3)))
            assert_tree_equal(self, pre, saved)
            if family == "g1_tracking_dagger":
                reset = vector.reset_done(terminal, object())
                for original, continued in zip(jax.tree_util.tree_leaves(terminal), jax.tree_util.tree_leaves(reset)):
                    np.testing.assert_array_equal(original[0], continued[0])
                self.assertFalse(bool(reset.done[1]))
                np.testing.assert_array_equal(reset.info['truncation'], [0., 0.])
                self.assertFalse(np.array_equal(reset.info['rng'][1], terminal.info['rng'][1]))

    def test_physical_measurement_owns_its_array_storage(self):
        source = {"qpos": jp.ones((2, 3))}
        owned = evaluation.owned_data(source)
        np.testing.assert_array_equal(owned["qpos"], source["qpos"])
        self.assertNotEqual(owned["qpos"].unsafe_buffer_pointer(), source["qpos"].unsafe_buffer_pointer())
        self.assertFalse(evaluation.contact_measurement()["available"])

    def test_posture_control_precedes_native_done_counter_and_selective_reset(self):
        env, _, _ = fake_native('g1_tracking_dagger', False, False)
        env._evaluation_physical_stop = dict(root_link_height_min_m=.35, root_up_dot_min=.5, floor_height_w_m=0.)
        env._get_termination = type(env)._get_termination.__get__(env)
        vector = WRAPPERS['g1_tracking_dagger'].VmapWrapper(env)
        pre = vector.reset(jax.random.split(jax.random.PRNGKey(5), 2), object())
        pre = pre.replace(data=pre.data.replace(qpos=pre.data.qpos.at[:,2].set(.8)))
        action = jp.zeros((2,29))
        plain = vector.step_no_autoreset(pre, action, object())
        enabled = vector.step_no_autoreset(pre, action, object(), jp.array(True))
        assert_tree_equal(self, plain, enabled.replace(info={key:value for key,value in enabled.info.items()
            if key != 'posture_termination'}))
        mixed = pre.replace(data=pre.data.replace(qpos=pre.data.qpos.at[0,3:7].set(jp.array([0.,1.,0.,0.])),
            qvel=pre.data.qvel.at[1,-1].set(jp.nan)))
        terminal = vector.step_no_autoreset(mixed, action, object(), jp.array(False))
        np.testing.assert_array_equal(terminal.done, [False, True])
        np.testing.assert_array_equal(terminal.info['step'], [1,0])
        self.assertTrue(bool(terminal.info['posture_termination']['tilt'][0]))
        self.assertFalse(bool(terminal.info['posture_termination']['terminated'][0]))
        reset = vector.reset_done(terminal, object())
        for original, continued in zip(jax.tree_util.tree_leaves(terminal), jax.tree_util.tree_leaves(reset)):
            np.testing.assert_array_equal(original[0], continued[0])
        outside = vector.step_no_autoreset(reset, action, object(), jp.array(True))
        self.assertTrue(bool(outside.done[0]))
        self.assertEqual(int(outside.info['step'][0]), 0)
        timeout = pre.replace(info={**pre.info,'step':jp.full((2,),env._config.episode_length-1)})
        timed = vector.step_no_autoreset(timeout, action, object(), jp.array(False))
        np.testing.assert_array_equal(timed.done, [True, True])
        np.testing.assert_array_equal(timed.info['truncation'], [True, True])
        np.testing.assert_array_equal(timed.info['posture_termination']['terminated'], [False, False])


if __name__ == "__main__":
    unittest.main()

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


def batch_state(keys):
    n = len(keys)
    return State({"qpos": jp.arange(n, dtype=float)[:, None]}, {"policy": jp.zeros((n, 2))},
                 jp.arange(n, dtype=float), jp.zeros(n), {"terminal": jp.arange(n, dtype=float)},
                 {"rng": keys, "last_motor_targets": jp.zeros((n, 2))})


def reset_batch(keys):
    keys = jax.vmap(lambda key: jax.random.split(key)[1])(keys)
    state = batch_state(keys)
    return state.replace(data={"qpos": jp.full_like(state.data["qpos"], -10)},
                         obs={"policy": jp.full_like(state.obs["policy"], -20)})


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
    env._ctrl_dt, env._sim_dt = .02, .002
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

    def test_physical_measurement_owns_its_array_storage(self):
        source = {"qpos": jp.ones((2, 3))}
        owned = evaluation.owned_data(source)
        np.testing.assert_array_equal(owned["qpos"], source["qpos"])
        self.assertNotEqual(owned["qpos"].unsafe_buffer_pointer(), source["qpos"].unsafe_buffer_pointer())
        self.assertFalse(evaluation.contact_measurement()["available"])


    def test_shared_warp_contact_capacity_equal_to_world_count_is_not_world_axis(self):
        # MuJoCo 3.9 DATA_NON_VMAP declares both contact fields shared, even
        # when naconmax happens to equal num_envs. Constraint rows are per world.
        shared = {"contact__worldid", "contact__geom"}
        data = {"qpos": jp.array([[1.], [2.]]), "xpos": jp.array([[10.], [20.]]),
                "_impl": {"contact__worldid": jp.array([0, 1]),
                          "contact__geom": jp.array([[1, 2], [3, 4]]),
                          "efc__id": jp.array([[5, 6], [7, 8]])}}
        reset_data = {"qpos": jp.array([[-1.], [-2.]]), "xpos": jp.array([[-10.], [-20.]]),
                      "_impl": {"contact__worldid": jp.array([1, 0]),
                                "contact__geom": jp.array([[9, 10], [11, 12]]),
                                "efc__id": jp.array([[13, 14], [15, 16]])}}
        done = jp.array([False, True])
        merged = evaluation.merge_world_data(done, reset_data, data, non_vmap_fields=shared)
        np.testing.assert_array_equal(merged["qpos"], [[1], [-2]])
        np.testing.assert_array_equal(merged["_impl"]["contact__geom"], reset_data["_impl"]["contact__geom"])
        np.testing.assert_array_equal(merged["_impl"]["contact__worldid"], [1, 0])

        initial = batch_state(jax.random.split(jax.random.PRNGKey(5), 2)).replace(data=data, done=done)
        def reset(keys):
            return reset_batch(keys).replace(data=reset_data)
        def refresh(value):
            # Stand-in for a coherent native forward: every derived cache and
            # shared contact order is rebuilt together from the merged state.
            return {"qpos": value["qpos"], "xpos": value["qpos"] + 100,
                    "_impl": {"contact__worldid": jp.array([1, 0]),
                              "contact__geom": jp.array([[30, 31], [32, 33]]),
                              "efc__id": jp.array([[40, 41], [42, 43]])}}
        with patch.object(evaluation, "_native_non_vmap_fields", return_value=shared):
            actual = evaluation.reset_done_envs(initial, reset, refresh)
        np.testing.assert_array_equal(actual.data["qpos"][0], initial.data["qpos"][0])
        np.testing.assert_array_equal(actual.info["rng"][0], initial.info["rng"][0])
        np.testing.assert_array_equal(actual.data["xpos"], [[101], [98]])
        np.testing.assert_array_equal(actual.data["_impl"]["efc__id"], [[40, 41], [42, 43]])
        np.testing.assert_array_equal(actual.data["_impl"]["contact__geom"], [[30, 31], [32, 33]])


    def test_unknown_native_world_layout_is_refused(self):
        with self.assertRaisesRegex(ValueError, "lacks expected world prefix"):
            evaluation.merge_world_data(jp.array([False, True]), {"qpos": jp.zeros((3, 1))},
                                        {"qpos": jp.zeros((3, 1))})


    def test_peer_reset_preserves_ongoing_rng_and_persistent_state(self):
        state = batch_state(jax.random.split(jax.random.PRNGKey(8), 3))
        state = state.replace(done=jp.array([False, True, False]))
        reset = jax.jit(lambda s: evaluation.reset_done_envs(
            s, reset_batch, lambda data: {"qpos": data["qpos"] + 100}))(state)
        for world in (0, 2):
            np.testing.assert_array_equal(reset.info["rng"][world], state.info["rng"][world])
            np.testing.assert_array_equal(reset.data["qpos"][world], state.data["qpos"][world])
            np.testing.assert_array_equal(reset.obs["policy"][world], state.obs["policy"][world])
        reset_key = jax.random.split(state.info["rng"][1])[0]
        expected = jax.random.split(reset_key)[1]
        np.testing.assert_array_equal(reset.info["rng"][1], expected)
        assert_tree_equal(self, reset.metrics, state.metrics)
        np.testing.assert_array_equal(reset.done, state.done)
        np.testing.assert_array_equal(reset.reward, state.reward)


    def test_ongoing_random_draw_sequence_matches_isolated_world(self):
        keys = jax.random.split(jax.random.PRNGKey(99), 2)
        state, isolated = batch_state(keys), keys[0]
        for step in range(10):
            draws = jax.vmap(lambda key: jax.random.split(key, 2))(state.info["rng"])
            isolated, draw_key = jax.random.split(isolated)
            np.testing.assert_array_equal(jax.random.normal(draws[0, 1], (3,)), jax.random.normal(draw_key, (3,)))
            state = state.replace(info={**state.info, "rng": draws[:, 0]}, done=jp.array([False, step % 2 == 0]))
            state = evaluation.reset_done_envs(state, reset_batch, lambda data: data)
            np.testing.assert_array_equal(state.info["rng"][0], isolated)


    def test_actual_vector_wrappers_preserve_terminal_and_ongoing_world(self):
        for family in ("g1_tracking", "g1_tracking_dagger"):
            env, _, _ = fake_native(family, False, False)
            env._get_termination = lambda data, reference, info: data.qpos[7] > .5
            vector = WRAPPERS[family].VmapWrapper(env)
            keys = jax.random.split(jax.random.PRNGKey(5), 2)
            pre = vector.reset(keys, object())
            actions = jp.zeros((2, 29)).at[1, 0].set(1)
            transition = evaluation.evaluate_step(vector, pre, actions, object())
            np.testing.assert_array_equal(transition.reset_mask, [False, True])
            np.testing.assert_array_equal(transition.post_terminal.info["last_motor_targets"][1, 0], 2)
            np.testing.assert_array_equal(transition.next_state.info["last_motor_targets"][1, 0], 0)
            np.testing.assert_array_equal(transition.post_physical_data.xpos, jp.full((2, 1, 3), 10))
            np.testing.assert_array_equal(transition.post_terminal.data.xpos, jp.zeros((2, 1, 3)))
            np.testing.assert_array_equal(transition.next_state.data.xpos[0], transition.post_terminal.data.xpos[0] + 10)
            np.testing.assert_array_equal(transition.next_state.data.qpos[0], transition.post_terminal.data.qpos[0])
            np.testing.assert_array_equal(transition.next_state.info["rng"][0], transition.post_terminal.info["rng"][0])


    def test_measurement_copy_and_reset_keep_terminal_snapshot(self):
        initial = batch_state(jax.random.split(jax.random.PRNGKey(3), 2))
        class Env:
            def step_no_autoreset(self, state, action, trajectory):
                state = evaluation.copy_state_containers(state)
                state.info["last_motor_targets"] = action
                return state.replace(data={"qpos": state.data["qpos"] + 1}, done=jp.array([False, True]))
            def physical_data(self, state):
                copied = evaluation.owned_data(state.data)
                return {"qpos": copied["qpos"] + 2}
            def reset_done(self, state, trajectory):
                return evaluation.reset_done_envs(state, reset_batch, lambda data: data)
        transition = evaluation.evaluate_step(Env(), initial, jp.ones((2, 2)))
        np.testing.assert_array_equal(transition.pre.info["last_motor_targets"], jp.zeros((2, 2)))
        np.testing.assert_array_equal(transition.post_terminal.data["qpos"], [[1], [2]])
        np.testing.assert_array_equal(transition.post_physical_data["qpos"], [[3], [4]])
        np.testing.assert_array_equal(transition.next_state.data["qpos"], [[1], [-10]])
        copied = evaluation.owned_data(initial.data)
        self.assertNotEqual(copied["qpos"].unsafe_buffer_pointer(), initial.data["qpos"].unsafe_buffer_pointer())
        self.assertFalse(evaluation.contact_measurement()["available"])


if __name__ == "__main__":
    unittest.main()

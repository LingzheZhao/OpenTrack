"""CPU-only imports and algebra with fake physics; no model or simulator is made."""
import os
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('GLI_PATH', str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import support
from track_mj.envs.g1_tracking_dagger.train import g1_env_tracking_general as native


CONTACT = SimpleNamespace(
    efc_address=np.array([0, 3, -1, 6]),
    dist=jp.array([-.01, -.02, -.03, -.04]), includemargin=jp.zeros(4),
    geom=jp.array([[0, 1], [2, 1], [1, 2], [-1, -1]]),
    frame=jp.array([[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                    np.eye(3), np.eye(3), np.eye(3)]),
)
MODEL = SimpleNamespace(impl=mjx.Impl.JAX, nbody=3, geom_bodyid=np.array([0, 1, 2]), geom_type=np.array([0, 2, 2]),
                        opt=SimpleNamespace(integrator=mjx.IntegratorType.EULER))


class Data(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object
    time: object
    xpos: object
    xquat: object

    @property
    def impl(self):
        return mjx.Impl.JAX

    @property
    def _impl(self):
        return SimpleNamespace(contact=CONTACT)

    def replace(self, **values):
        return self._replace(**values)


class State(NamedTuple):
    data: object
    obs: object
    reward: object
    done: object
    metrics: object
    info: object

    def replace(self, **values):
        return self._replace(**values)


def initial_data():
    return Data(jp.concatenate([jp.zeros(3), jp.array([1., 0., 0., 0.]), jp.zeros(29)]),
                jp.zeros(35), jp.zeros(29), jp.array(0.), jp.zeros((3, 3)),
                jp.tile(jp.array([1., 0., 0., 0.]), (3, 1)))


def fake_physics(model, data):
    return data.replace(qpos=data.qpos.at[7:].add(data.ctrl * .002), time=data.time + .002)


def solved_force(model, data, index, to_world_frame=False):
    assert not to_world_frame
    return jp.array([[10., 2., 0., 0., 0., 0.], [5., -1., 3., 0., 0., 0.],
                     [999., 999., 999., 0., 0., 0.], [999., 999., 999., 0., 0., 0.]])[index]


class SubstepContactTests(unittest.TestCase):
    def test_world_frame_normal_total_and_signed_body_reduction(self):
        with patch.object(support, 'contact_force', side_effect=solved_force) as read, \
             patch.object(CONTACT, 'frame', CONTACT.frame.at[2:].set(jp.nan)):
            sample = native._solved_contact_sample(MODEL, initial_data(), jp.array(-.002))
        self.assertEqual([call.args[2] for call in read.call_args_list], [0, 1, 3])
        np.testing.assert_array_equal(sample.total_force_world_by_body_n,
                                      [[-2, 0, -10], [7, -1, 13], [-5, 1, -3]])
        np.testing.assert_array_equal(sample.normal_force_world_by_body_n,
                                      [[0, 0, -10], [5, 0, 10], [-5, 0, 0]])
        np.testing.assert_array_equal(sample.ground_normal_force_world_by_body_n,
                                      [[0, 0, -10], [0, 0, 10], [0, 0, 0]])
        np.testing.assert_array_equal(sample.ground_contact_found_by_body, [True, True, False])
        np.testing.assert_array_equal(sample.normal_n, [10, 5, 0, 0])
        np.testing.assert_array_equal(sample.constraint_present, [True, True, False, False])
        np.testing.assert_array_equal(sample.contact_found, [True, True, False, False])
        np.testing.assert_array_equal(sample.distance_m, CONTACT.dist)
        np.testing.assert_array_equal(sample.inclusion_margin_m, CONTACT.includemargin)
        np.testing.assert_array_equal(sample.total_force_world_n[2:], jp.zeros((2, 3)))
        np.testing.assert_array_equal(sample.total_force_world_by_body_n.sum(axis=0), [0, 0, 0])
        for value in sample:
            self.assertTrue(np.isfinite(value).all())
        # Native addresses/pair IDs remain allocated while geometry separates.
        # A zero-gap touch is geometric contact, but the native zero-margin
        # constraint is inactive at equality. Force is not the occurrence test.
        with patch.object(support, 'contact_force', return_value=jp.zeros(6)), \
             patch.object(CONTACT, 'dist', jp.array([.1, 0., -.1, -.1])):
            touch = native._solved_contact_sample(MODEL, initial_data(), jp.array(0.))
        np.testing.assert_array_equal(touch.constraint_present, [False, False, False, False])
        np.testing.assert_array_equal(touch.contact_found, [False, True, False, False])
        with patch.object(support, 'contact_force', return_value=jp.zeros(6)), \
             patch.object(CONTACT, 'dist', jp.array([.01, -.01, -.1, -.1])), \
             patch.object(CONTACT, 'includemargin', jp.array([.02, 0., 0., 0.])):
            margin = native._solved_contact_sample(MODEL, initial_data(), jp.array(0.))
        np.testing.assert_array_equal(margin.constraint_present, [True, True, False, False])
        np.testing.assert_array_equal(margin.contact_found, [False, True, False, False])
        np.testing.assert_array_equal(margin.ground_contact_found_by_body, [False, False, False])
        np.testing.assert_array_equal(margin.normal_n, jp.zeros(4))

    def test_no_contact_capacity_returns_zero_body_forces(self):
        empty = SimpleNamespace(efc_address=np.zeros(0, dtype=int), geom=jp.zeros((0, 2), dtype=int),
                                frame=jp.zeros((0, 3, 3)), dist=jp.zeros(0), includemargin=jp.zeros(0))
        with patch.object(CONTACT, 'efc_address', empty.efc_address), \
             patch.object(CONTACT, 'geom', empty.geom), patch.object(CONTACT, 'frame', empty.frame), \
             patch.object(CONTACT, 'dist', empty.dist), patch.object(CONTACT, 'includemargin', empty.includemargin):
            sample = native._solved_contact_sample(MODEL, initial_data(), jp.array(0.))
        self.assertEqual(sample.total_force_world_n.shape, (0, 3))
        np.testing.assert_array_equal(sample.total_force_world_by_body_n, jp.zeros((3, 3)))

    def test_capture_reuses_plain_physics_pd_and_random_key_sequence(self):
        key = jax.random.PRNGKey(7)
        args = (key, MODEL, initial_data(), jp.arange(29) / 10, jp.ones(29), jp.ones(29), jp.ones(29))
        with patch.object(native.mjx, 'step', side_effect=fake_physics), \
             patch.object(support, 'contact_force', side_effect=solved_force):
            plain = native.torque_step(*args, n_substeps=10)
            measured = native.torque_step_with_contacts(*args, n_substeps=10)
        for left, right in zip(jax.tree_util.tree_leaves(plain), jax.tree_util.tree_leaves(measured[:3])):
            np.testing.assert_array_equal(left, right)
        expected = key
        for _ in range(10):
            expected, _ = jax.random.split(expected, 2)
        np.testing.assert_array_equal(measured[0], expected)
        self.assertEqual(measured[3].total_force_world_by_body_n.shape, (10, 3, 3))
        np.testing.assert_array_equal(measured[3].interval_start_s[0], initial_data().time)
        np.testing.assert_array_equal(measured[3].interval_start_s[1:], measured[3].interval_end_s[:-1])
        np.testing.assert_array_equal(measured[3].interval_end_s[-1], measured[1].time)

    def test_warp_and_other_integrator_measurements_are_explicitly_rejected(self):
        for model in (SimpleNamespace(impl=mjx.Impl.WARP),
                      SimpleNamespace(impl=mjx.Impl.JAX, opt=SimpleNamespace(integrator=mjx.IntegratorType.RK4))):
            with self.assertRaises(NotImplementedError):
                native.torque_step_with_contacts(jax.random.PRNGKey(1), model, initial_data(),
                                                jp.zeros(29), jp.ones(29), jp.ones(29), jp.ones(29))

    def test_environment_keeps_terminal_state_and_input_containers(self):
        real = native.G1TrackingGeneralEnv
        Env = type('FakePhysicsEnv', (), {name: getattr(real, name) for name in
                   ('step', 'step_no_autoreset', 'step_with_contacts', '_step', '_get_motor_targets', 'model_for_state', 'model_for_centers', '_advance_reference')})
        env = Env()
        env._config = SimpleNamespace(student_use_residual_action=True, action_scale=1., history_len=0,
                                     episode_length=100, reward_config=SimpleNamespace(scales={'x': 1.}))
        env.mjx_model = MODEL
        env.obstacle_geom_ids, env._obstacle_centers, env.reference_mode = (), jp.zeros((0,3)), "tracking"
        env.action_joint_ids = jp.arange(29)
        env._default_qpos = jp.zeros(29)
        env._kps = env._kds = env.torque_limit = jp.ones(29)
        env.n_substeps, env.dt = 10, .02
        reference = SimpleNamespace(qpos=initial_data().qpos)
        env.th = SimpleNamespace(get_current_traj_data_with_trajectory=lambda trajectory, carry: reference,
                                 update_state_with_trajectory=lambda trajectory, carry: carry)
        env._get_reward = lambda *args: {'x': jp.array(1.)}
        env._get_termination = lambda *args: jp.array(True)
        env._get_obs = lambda data, reference, info: ({'state': info['last_motor_targets']}, jp.zeros(0))
        carry = SimpleNamespace(traj_state=SimpleNamespace(traj_no=jp.array(0)))
        info = {'rng': jax.random.PRNGKey(7), 'step': jp.array(0), 'traj_info': carry, 'traj_no': jp.array(0)}
        state = State(initial_data(), {'state': jp.zeros(29)}, jp.array(0.), jp.array(0.), {}, info)
        old_info_keys = set(info)
        with patch.object(native.mjx, 'step', side_effect=fake_physics), \
             patch.object(support, 'contact_force', side_effect=solved_force):
            terminal, samples = env.step_with_contacts(state, jp.ones(29), object())
            plain = env.step_no_autoreset(state, jp.ones(29), object())
        self.assertTrue(bool(terminal.done))
        self.assertEqual(set(state.info), old_info_keys)
        self.assertIsNot(terminal.info, state.info)
        for left, right in zip(jax.tree_util.tree_leaves(terminal.data), jax.tree_util.tree_leaves(plain.data)):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(terminal.info['rng'], plain.info['rng'])
        np.testing.assert_array_equal(samples.interval_end_s[-1], terminal.data.time)


if __name__ == '__main__':
    unittest.main()

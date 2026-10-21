"""CPU contract checks for additive native G1 physical-start injection."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
from track_mj.envs.g1_tracking_dagger.utils.wrapper import VmapWrapper


class Data(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object
    xpos: object
    xquat: object
    time: object

    def replace(self, **changes):
        return self._replace(**changes)


class State(NamedTuple):
    data: object
    obs: object
    reward: object
    done: object
    metrics: object
    info: object

    def replace(self, **changes):
        return self._replace(**changes)


def fixture():
    method_names = ("initialize_physical_start", "_reset_observation")
    Env = type("PhysicalStartFixture", (), {
        name: getattr(G1TrackingGeneralEnv, name) for name in method_names})
    env = Env()
    env.num_joints = 29
    env.action_joint_ids = jp.arange(29)
    env._config = SimpleNamespace(history_len=2)
    env._evaluation_physical_stop = None
    env.mj_model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=1),
        body_parentid=np.array([0, 0]), body_jntadr=np.array([0, 0]),
        body_jntnum=np.array([0, 1]), jnt_type=np.array([mujoco.mjtJoint.mjJNT_FREE]),
        jnt_qposadr=np.array([0]), jnt_dofadr=np.array([0]),
    )
    env.model_for_state = lambda state: object()
    env.th = SimpleNamespace(get_current_traj_data_with_trajectory=lambda data, carry:
                             SimpleNamespace(qpos=jp.array([carry, 0., 0.])))

    def observation(data, reference, info):
        info["rng"], _ = jax.random.split(info["rng"])
        value = jp.array([data.xpos[1, 0], reference.qpos[0]])
        return {"state": value, "privileged_state": value}, value

    env._get_obs = observation
    initial = jp.concatenate((jp.zeros(3), jp.array([1., 0., 0., 0.]), jp.zeros(29)))
    data = Data(initial, jp.zeros(35), jp.zeros(29), jp.zeros((2, 3)),
                jp.tile(jp.array([1., 0., 0., 0.]), (2, 1)), jp.array(0.))
    key = jax.random.PRNGKey(73)
    info = {"rng": jax.random.PRNGKey(9), "initial_obs_rng": key,
            "traj_info": jp.array(8), "initial_traj_info": jp.array(7),
            "step": jp.array(0), "last_motor_targets": jp.zeros(29),
            "last_action": jp.zeros(29), "last_root_pos": jp.zeros(3),
            "last_root_ori": jp.array([1., 0., 0., 0.]),
            "last_dof_pos": jp.zeros(29), "last_rigid_body_pos": jp.zeros((2, 3)),
            "last_rigid_body_ori": jp.tile(jp.array([1., 0., 0., 0.]), (2, 1)),
            "last_joint_vel": jp.zeros(29)}
    state = State(data, {"state": jp.zeros(6), "privileged_state": jp.zeros(6)},
                  jp.array(0.), jp.array(0.), {}, info)
    return env, state


def forward(model, data):
    xpos = data.xpos.at[1].set(data.qpos[:3] + jp.array([.1, 0., 0.]))
    xquat = data.xquat.at[1].set(data.qpos[3:7])
    return data.replace(xpos=xpos, xquat=xquat)


class InitialPhysicalStartTests(unittest.TestCase):
    def test_link_point_velocity_phase_and_reset_history(self):
        env, state = fixture()
        pose = jp.array([.5, .6, .7, 2**-.5, 0., 0., 2**-.5])
        velocity = jp.array([1., 2., 3., 1., 0., 2.])
        joints = jp.arange(29.) / 100
        joint_velocity = -joints
        original_rng = np.asarray(state.info["rng"]).copy()
        with patch("track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general.mjx.forward",
                   side_effect=forward) as simulated_forward:
            result = env.initialize_physical_start(
                state, object(), pose, velocity, joints, joint_velocity)
        self.assertEqual(simulated_forward.call_count, 2)
        np.testing.assert_allclose(result.data.xpos[1], pose[:3], atol=1e-6)
        np.testing.assert_allclose(result.data.qpos[:3], [.4, .6, .7], atol=1e-6)
        np.testing.assert_allclose(result.data.qvel[:3], [1., 1.8, 3.], atol=1e-6)
        np.testing.assert_allclose(result.data.qvel[3:6], [0., -1., 2.], atol=1e-5)
        np.testing.assert_allclose(result.data.qpos[7:], joints)
        np.testing.assert_allclose(result.data.qvel[6:], joint_velocity)
        np.testing.assert_allclose(result.info["last_joint_vel"], joint_velocity)
        self.assertEqual(float(result.data.time), 0.)
        self.assertEqual(int(result.info["traj_info"]), 8)
        self.assertEqual(int(result.info["initial_traj_info"]), 7)
        np.testing.assert_allclose(result.obs["state"], [.5, 8., .5, 7., .5, 7.])
        np.testing.assert_allclose(result.info["previous_obs"], [[.5, 7.], [.5, 8.]])
        np.testing.assert_array_equal(state.info["rng"], original_rng)
        expected, _ = jax.random.split(state.info["initial_obs_rng"])
        expected, _ = jax.random.split(expected)
        np.testing.assert_array_equal(result.info["rng"], expected)

    def test_wrapper_rejects_wrong_shape_before_vmap(self):
        env, state = fixture()
        batched = jax.tree_util.tree_map(lambda leaf: jp.stack((leaf, leaf)), state)
        wrapper = VmapWrapper(env)
        with self.assertRaisesRegex(ValueError, "physical start requires"):
            wrapper.initialize_physical_start(
                batched, object(), jp.zeros((2, 8)), jp.zeros((2, 6)),
                jp.zeros((2, 29)), jp.zeros((2, 29)))


if __name__ == "__main__":
    unittest.main()

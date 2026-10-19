"""Actual pure native methods on inert arrays; no environment, reset or step."""
import os
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('GLI_PATH', str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
import jax.numpy as jp
import numpy as np
from track_mj.envs.g1_tracking_dagger.train import g1_env_tracking_general as native


def env():
    value = native.G1TrackingGeneralEnv.__new__(native.G1TrackingGeneralEnv)
    value._evaluation_physical_stop = {'root_link_height_min_m': .35, 'root_up_dot_min': .5,
                                       'floor_height_w_m': 0.}
    return value


def data(height=.8, quaternion=(1., 0., 0., 0.)):
    return SimpleNamespace(qpos=jp.array([0., 0., height, *quaternion, 0.]), qvel=jp.zeros(7))


class PhysicalEvaluationTests(unittest.TestCase):
    def test_upright_is_reference_independent_and_thresholds_are_strict(self):
        value = env()
        # A reference is deliberately absent: this path must not inspect it.
        self.assertFalse(bool(value._get_termination(data(), None, {})))
        self.assertFalse(bool(value._physical_stop_flags(data(.35))['height']))
        self.assertTrue(bool(value._get_termination(data(.349), None, {})))
        flags = value._physical_stop_flags(data(quaternion=(np.sqrt(.75), .5, 0., 0.)))
        self.assertAlmostEqual(float(flags['up_dot']), .5, places=6)
        self.assertTrue(bool(value._get_termination(data(quaternion=(np.sqrt(.5), np.sqrt(.5), 0., 0.)), None, {})))

    def test_full_quaternion_tilt_is_yaw_invariant_and_normalized(self):
        value = env()
        upright_yaw = data(quaternion=(0., 0., 0., 1.))
        self.assertFalse(bool(value._get_termination(upright_yaw, None, {})))
        self.assertAlmostEqual(float(value._physical_stop_flags(data(quaternion=(2., 0., 0., 0.)))['up_dot']), 1.)
        self.assertTrue(bool(value._physical_stop_flags(data(quaternion=(0., 0., 0., 0.)))['invalid_quaternion']))
        info = {}
        self.assertFalse(bool(value._get_termination(data(quaternion=(0., 1., 0., 0.)), None, info, jp.array(False))))
        self.assertTrue(bool(info['posture_termination']['tilt']))
        self.assertFalse(bool(info['posture_termination']['terminated']))
        self.assertTrue(bool(value._get_termination(data(quaternion=(0., 0., 0., 0.)), None, {}, jp.array(False))))

    def test_infinity_and_nan_always_stop_without_a_reference(self):
        value = env()
        for bad in (jp.nan, jp.inf):
            current = data()
            current.qvel = current.qvel.at[-1].set(bad)
            self.assertTrue(bool(value._get_termination(current, None, {})))
            self.assertTrue(bool(value._get_termination(current, None, {}, jp.array(False))))
            self.assertTrue(bool(value._physical_stop_flags(current)['nonfinite']))

    def test_default_tracking_still_uses_reference_and_body_threshold(self):
        value = env()
        value._evaluation_physical_stop = None
        value._config = SimpleNamespace(termination_config=SimpleNamespace(root_height_threshold=.3, rigid_body_dif_threshold=.5))
        value.valid_body_ids = jp.array([1])
        current = data()
        current.xpos = jp.array([[0., 0., 0.], [0., 0., .8]])
        reference = data()
        reference.xpos = current.xpos
        self.assertFalse(bool(value._get_termination(current, reference, {})))
        reference.qpos = reference.qpos.at[2].set(1.2)
        self.assertTrue(bool(value._get_termination(current, reference, {})))

    def test_configuration_binds_root_link_and_floor_and_rejects_wrong_mapping(self):
        value = env()
        settings = {'root_link_height_min_m': .35, 'root_up_dot_min': .5, 'stop_on_nonfinite': True}
        value._config = {'evaluation_physical_stop': settings}
        model = SimpleNamespace(body=lambda name: SimpleNamespace(id=1), geom=lambda name: SimpleNamespace(id=0),
            body_jntadr=np.array([-1, 0]), body_parentid=np.array([0, 0]), body_jntnum=np.array([0, 1]),
            jnt_type=np.array([0]), jnt_qposadr=np.array([0]), jnt_dofadr=np.array([0]), jnt_pos=np.zeros((1, 3)),
            geom_bodyid=np.array([0]), geom_type=np.array([0]), geom_quat=np.array([[1., 0., 0., 0.]]),
            geom_pos=np.array([[0., 0., .1]]))
        value._mj_model = model
        value._configure_evaluation_physical_stop()
        self.assertEqual(value.evaluation_physical_stop_description()['floor_height_w_m'], .1)
        self.assertAlmostEqual(float(value._physical_stop_flags(data(.4))['height_m']), .3, places=6)
        model.jnt_qposadr[0] = 1
        with self.assertRaisesRegex(ValueError, 'pelvis LINK origin'):
            value._configure_evaluation_physical_stop()
        model.jnt_qposadr[0] = 0
        model.geom_quat[:] = [0., 1., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'horizontal world floor'):
            value._configure_evaluation_physical_stop()
        settings['stop_on_nonfinite'] = False
        with self.assertRaisesRegex(ValueError, 'mandatory nonfinite'):
            value._configure_evaluation_physical_stop()

    def test_missing_config_leaves_native_tracking_selected(self):
        value = env()
        value._config = {}
        value._configure_evaluation_physical_stop()
        self.assertIsNone(value.evaluation_physical_stop_description())


if __name__ == '__main__':
    unittest.main()

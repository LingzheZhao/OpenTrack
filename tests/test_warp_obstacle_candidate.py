"""CPU contract checks for candidate native Warp pillars; runtime remains gated."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from track_mj.utils.warp_obstacles import (
    add_mocap_pillar, add_pillar_pairs_and_sensors, decode_pair_sensor,
    pair_sensor_layout, select_warp_reset_data_and_forward, warp_measurement_invalid,
)


def model_with_sensor(slots=4):
    spec = mujoco.MjSpec.from_string('''<mujoco model="warp pillar probe"><worldbody>
        <body name="foot_body" pos="0.2 0 0.75"><freejoint name="root"/>
          <geom name="foot" type="box" size="0.1 0.1 0.1"/>
        </body></worldbody></mujoco>''')
    add_mocap_pillar(spec, name="pillar", center_xyz=(0, 0, .75), radius_m=.15, height_m=1.5)
    names = add_pillar_pairs_and_sensors(spec, robot_geom_names=("foot",),
                                         pillar_geom_names=("pillar",), slots=slots)
    return spec.compile(), names


class WarpPillarCandidateTests(unittest.TestCase):
    def test_full_g1_model_admits_two_exact_cylinders_and_all_pairs(self):
        from track_mj.envs.g1_tracking_dagger import g1_tracking_constants as constants

        spec = mujoco.MjSpec.from_file(constants.task_to_xml("flat_terrain").as_posix())
        for index, y in enumerate((-.72, .72)):
            add_mocap_pillar(spec, name=f"task_obstacle_{index:02d}",
                             center_xyz=(.75, y, .75), radius_m=.15, height_m=1.5)
        draft = spec.compile()
        root = int(draft.body("pelvis").id)
        robot = np.asarray(draft.body_rootid)[np.asarray(draft.geom_bodyid)] == root
        paired = np.concatenate((np.asarray(draft.pair_geom1), np.asarray(draft.pair_geom2)))
        enabled = (np.asarray(draft.geom_contype) != 0) | (np.asarray(draft.geom_conaffinity) != 0)
        enabled[np.asarray(paired, dtype=np.int64)] = True
        robot_names = tuple(draft.geom(int(i)).name for i in np.flatnonzero(robot & enabled))
        self.assertEqual(len(robot_names), 16)
        self.assertIn("left_foot", robot_names)
        self.assertIn("right_foot", robot_names)
        sensors = add_pillar_pairs_and_sensors(spec, robot_geom_names=robot_names,
            pillar_geom_names=("task_obstacle_00", "task_obstacle_01"))
        model = spec.compile()
        layout = pair_sensor_layout(model, sensors)
        self.assertEqual(model.nmocap, 2)
        self.assertEqual(len(layout), 32)
        self.assertEqual({pair.pillar_geom for pair in layout}, {"task_obstacle_00", "task_obstacle_01"})
        self.assertEqual(sum(pair.robot_geom in ("left_foot", "right_foot") for pair in layout), 4)
        self.assertEqual(int(model.geom_type[model.geom("task_obstacle_00").id]),
                         int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        # Model admission is only a structural check. Runtime remains gated.
        self.assertEqual(mjx.put_model(model, impl="warp").impl, mjx.Impl.WARP)

    def test_compiled_true_cylinder_and_public_sensor(self):
        model, names = model_with_sensor()
        pillar = int(model.geom("pillar").id)
        self.assertEqual(int(model.geom_type[pillar]), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        np.testing.assert_allclose(model.geom_size[pillar, :2], [.15, .75])
        self.assertEqual(int(model.body_mocapid[model.geom_bodyid[pillar]]), 0)
        self.assertEqual(model.npair, 1)
        layout = pair_sensor_layout(model, names)
        self.assertEqual(layout[0].slots, 4)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contact = decode_pair_sensor(data.sensordata, layout[0])
        self.assertEqual(float(contact["found_count"]), 2)
        self.assertTrue(bool(contact["geometric_contact"]))
        self.assertGreater(float(contact["max_normal_force_n"]), 0)
        self.assertFalse(bool(contact["invalid"]))
        data.mocap_pos[0] = [2, 0, .75]
        mujoco.mj_forward(model, data)
        far = decode_pair_sensor(data.sensordata, layout[0])
        self.assertEqual(float(far["found_count"]), 0)
        self.assertFalse(bool(far["geometric_contact"]))

    def test_full_found_count_exposes_sensor_slot_truncation(self):
        model, names = model_with_sensor(slots=1)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        reading = decode_pair_sensor(data.sensordata, pair_sensor_layout(model, names)[0])
        self.assertEqual(data.ncon, 2)
        self.assertEqual(float(reading["found_count"]), 2)
        self.assertTrue(bool(reading["invalid"]))
        warp = SimpleNamespace(impl=mjx.Impl.WARP, _impl=SimpleNamespace(overflow=jp.array(0)))
        self.assertTrue(bool(warp_measurement_invalid(warp, (reading,))))
        safe = dict(reading, invalid=jp.array(False))
        self.assertFalse(bool(warp_measurement_invalid(warp, (safe,))))
        warp._impl.overflow = jp.array(1)
        self.assertTrue(bool(warp_measurement_invalid(warp, (safe,))))

    @unittest.skipUnless(os.getenv("OT_TEST_WARP_CPU") == "1", "explicit pinned-image CPU Warp proof")
    def test_two_world_mocap_and_selective_data_where(self):
        model, names = model_with_sensor()
        layout = pair_sensor_layout(model, names)[0]
        warp_model = mjx.put_model(model, impl="warp")
        template = mjx.make_data(model, impl="warp", naconmax=32, njmax=64)
        make = jax.jit(jax.vmap(lambda center: mjx.forward(
            warp_model, template.replace(mocap_pos=center))))
        before = make(jp.array([[[0., 0., .75]], [[2., 0., .75]]]))
        reset = make(jp.array([[[2., 0., .75]], [[0., 0., .75]]]))
        self.assertEqual([float(x) for x in before.sensordata[:, layout.address]], [2, 0])
        # Data.where takes a scalar decision per world under vmap. The caller
        # then forwards every world outside a conditional reset branch.
        after = jax.jit(lambda live, fresh: select_warp_reset_data_and_forward(
            warp_model, live, fresh, jp.array([False, True])))(before, reset)
        np.testing.assert_allclose(after.mocap_pos[:, 0, 0], [0, 0])
        np.testing.assert_allclose(after.sensordata[:, layout.address], [2, 2])
        np.testing.assert_allclose(after.qpos[0], before.qpos[0])
        np.testing.assert_allclose(after.qvel[0], before.qvel[0])
        self.assertFalse(bool(jp.any(after._impl.overflow)))
        # A real native Warp step advances time while retaining solved sensor
        # samples from the step; the future recorder must capture this output
        # before any later endpoint forward or selective reset.
        stepped = jax.jit(jax.vmap(lambda world: mjx.step(warp_model, world)))(after)
        np.testing.assert_allclose(stepped.time, after.time + model.opt.timestep, atol=1e-7)
        self.assertTrue(bool(jp.all(stepped.sensordata[:, layout.address] > 0)))
        self.assertFalse(bool(jp.any(stepped._impl.overflow)))

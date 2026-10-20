"""CPU joint indexing with native, NumPy and JAX trajectory metadata."""
import os
from pathlib import Path
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("GLI_PATH", str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import mujoco
import numpy as np

from track_mj.utils.dataset.traj_class import TrajectoryData, TrajectoryInfo, TrajectoryModel
from track_mj.utils.dataset.traj_handler import TrajectoryHandler


def representations(values):
    return {
        "python": list(values),
        "numpy-int32": np.array(values, dtype=np.int32),
        "numpy-int64": np.array(values, dtype=np.int64),
        "jax-int32": jnp.array(values, dtype=jnp.int32),
    }


class TrajectoryJointTypesTests(unittest.TestCase):
    def test_supported_joint_types_keep_qpos_qvel_offsets(self):
        types = [int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE),
                 int(mujoco.mjtJoint.mjJNT_SLIDE)]
        for representation, values in representations(types).items():
            with self.subTest(representation=representation):
                info = TrajectoryInfo(["root", "hinge", "slide"],
                                      TrajectoryModel(njnt=3, jnt_type=values), frequency=50.)
                for name, qpos, qvel in [("root", range(7), range(6)),
                                        ("hinge", [7], [6]), ("slide", [8], [7])]:
                    np.testing.assert_array_equal(info.joint_name2ind_qpos[name], qpos)
                    np.testing.assert_array_equal(info.joint_name2ind_qvel[name], qvel)

    def test_ball_joint_remains_unsupported(self):
        for representation, values in representations([int(mujoco.mjtJoint.mjJNT_BALL)]).items():
            with self.subTest(representation=representation):
                with self.assertRaisesRegex(ValueError, "Unsupported joint type: 1 for joint ball"):
                    TrajectoryInfo(["ball"], TrajectoryModel(njnt=1, jnt_type=values), frequency=50.)

    def test_filter_keeps_free_hinge_slide_coordinates_with_jax_metadata(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body><freejoint name="root"/><geom type="sphere" size=".1"/>
            <body><joint name="hinge" type="hinge"/><geom type="sphere" size=".1"/>
              <body><joint name="slide" type="slide"/><geom type="sphere" size=".1"/></body>
            </body>
          </body></worldbody></mujoco>""")
        info = TrajectoryInfo(["root", "hinge", "slide"],
                              TrajectoryModel(njnt=3, jnt_type=jnp.array(model.jnt_type)), frequency=50.)
        data = TrajectoryData(qpos=jnp.arange(18).reshape(2, 9),
                              qvel=jnp.arange(16).reshape(2, 8))
        filtered_data, filtered_info = TrajectoryHandler.filter_and_extend(data, info, model)
        np.testing.assert_array_equal(filtered_data.qpos, data.qpos)
        np.testing.assert_array_equal(filtered_data.qvel, data.qvel)
        self.assertEqual(filtered_info.joint_names, ["root", "hinge", "slide"])


if __name__ == "__main__":
    unittest.main()

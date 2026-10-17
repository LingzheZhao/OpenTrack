"""Small CPU array tests; no environment, physics or policy is constructed."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GLI_PATH", str(ROOT))
from track_mj.utils.dataset import traj_class as traj, preparation as prep


def tiny_trajectory():
    qpos = np.zeros((4, 8), dtype=np.float32)
    qpos[:, 3] = 1
    qpos[:, 0] = np.arange(4) * .1
    qpos[:, 7] = np.arange(4) * .02
    data = traj.TrajectoryData(
        qpos=qpos, qvel=np.zeros((4, 7), dtype=np.float32),
        xpos=np.zeros((4, 1, 3)), xquat=np.tile([1., 0, 0, 0], (4, 1, 1)),
        cvel=np.zeros((4, 1, 6)), subtree_com=np.zeros((4, 1, 3)),
        site_xpos=np.zeros((4, 1, 3)), site_xmat=np.tile(np.eye(3).reshape(9), (4, 1, 1)),
        split_points=np.array([0, 4]),
    )
    info = traj.TrajectoryInfo(
        joint_names=["root", "hinge"],
        model=traj.TrajectoryModel(njnt=2, jnt_type=np.array([0, 3])),
        frequency=50., body_names=["world"], site_names=["foot"],
        metadata={"retarget": {"source": "original clip"}, "smooth_prefix_frames": 12},
    )
    return traj.Trajectory(info, data)


class MotionPreparationTests(unittest.TestCase):
    def test_preparation_preserves_source_and_native_velocity_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.npz", Path(directory) / "prepared.npz"
            tiny_trajectory().save(source)
            original = source.read_bytes()
            expected = traj.Trajectory.load(source, backend=jp)
            for operation in (traj.recalculate_traj_angular_velocity, traj.recalculate_traj_linear_velocity,
                              traj.recalculate_traj_joint_velocity):
                expected = operation(expected, frequency=50., backend=jp)
            env = SimpleNamespace(dt=.02, recalculate_vel_in_reference_motion=True)
            prep.prepare_motion(env, source, output, smooth_start_end=False, source_reference="dataset-release-1")
            loaded = prep.load_prepared_trajectory(output, backend=np, control_dt=.02, recalculate_velocity=True)
            self.assertEqual(source.read_bytes(), original)
            np.testing.assert_array_equal(loaded.data.qvel, np.asarray(expected.data.qvel))
            np.testing.assert_array_equal(loaded.data.qpos, tiny_trajectory().data.qpos)
            self.assertEqual(loaded.info.metadata["retarget"], {"source": "original clip"})
            self.assertEqual(loaded.info.metadata["smooth_prefix_frames"], 12)
            record = loaded.info.metadata[prep.PREPARATION_KEY]
            self.assertEqual(record["source_path"], str(source.resolve()))
            self.assertEqual(record["source_reference"], "dataset-release-1")
            self.assertFalse(record["smooth_start_end"])
            self.assertEqual(record["velocity_frequency"], 50.)
            with self.assertRaises(FileExistsError):
                prep.prepare_motion(env, source, output, smooth_start_end=False)
            with self.assertRaises(ValueError):
                prep.prepare_motion(env, source, source, smooth_start_end=False)

    def test_runtime_rejects_unprepared_or_wrong_rate_without_mutation(self):
        original = tiny_trajectory()
        with self.assertRaisesRegex(ValueError, "explicitly prepared reference velocities"):
            prep.require_prepared(original, control_dt=.02, recalculate_velocity=True, path="motion")
        with self.assertRaisesRegex(ValueError, "25 Hz"):
            prep.require_prepared(original, control_dt=.04, recalculate_velocity=False, path="motion")
        incomplete = replace(original, data=original.data.replace(xpos=np.empty(0)))
        with self.assertRaisesRegex(ValueError, "lacks prepared fields"):
            prep.require_prepared(incomplete, control_dt=.02, recalculate_velocity=False, path="motion")
        np.testing.assert_array_equal(original.data.qvel, np.zeros((4, 7)))
        self.assertEqual(original.info.metadata, tiny_trajectory().info.metadata)

    def test_smoothing_and_resampling_are_explicit_and_identity_is_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            first, second = Path(directory) / "first.npz", Path(directory) / "second.npz"
            tiny_trajectory().save(source)
            calls = []
            def extend(trajectory, smooth_start_end):
                calls.append(smooth_start_end)
                return replace(trajectory, info=replace(trajectory.info, frequency=25.))
            env = SimpleNamespace(dt=.04, recalculate_vel_in_reference_motion=False, extend_motion=extend)
            prep.prepare_motion(env, source, first, smooth_start_end=True)
            self.assertEqual(calls, [True])
            prep.prepare_motion(env, first, second, smooth_start_end=False)
            self.assertEqual(calls, [True])
            loaded = prep.load_prepared_trajectory(second, backend=np, control_dt=.04, recalculate_velocity=False)
            record = loaded.info.metadata[prep.PREPARATION_KEY]
            self.assertTrue(record["source_preparation"]["smooth_start_end"])
            self.assertEqual(record["source_preparation"]["source_path"], str(source.resolve()))
            self.assertEqual(record["source_path"], str(first.resolve()))

    def test_concatenation_retains_source_spans_and_rejects_different_model(self):
        a = tiny_trajectory()
        b = replace(a, info=replace(a.info, metadata={"source": "second"}))
        merged = prep.concatenate_prepared([a, b], backend=np)
        np.testing.assert_array_equal(merged.data.split_points, [0, 4, 8])
        sources = merged.info.metadata["opentrack_sources"]
        self.assertEqual([(r["sample_start"], r["sample_stop"]) for r in sources], [(0, 4), (4, 8)])
        self.assertEqual(sources[0]["metadata"], a.info.metadata)
        self.assertEqual(sources[1]["metadata"], b.info.metadata)
        wrong = replace(b, info=replace(b.info, joint_names=["root", "other_hinge"]))
        with self.assertRaises(AssertionError):
            prep.concatenate_prepared([a, wrong], backend=np)

    def test_batch_partition_and_flat_source_mapping(self):
        dataset = {"lafan1": ["a", "b", "c", "d", "e"]}
        captured = []
        def record(env, source, output, *, smooth_start_end, source_reference=None):
            captured.append((str(source), str(output), smooth_start_end))
            return output
        with patch.object(prep, "prepare_motion", side_effect=record):
            for batch in range(3):
                prep.preprocess_dataset(None, dataset, batch, 3, False,
                                        source_root="/raw", output_root="/prepared", flat_source=True)
        self.assertEqual([r[0] for r in captured], [f"/raw/{n}.npz" for n in dataset["lafan1"]])
        self.assertEqual(captured[-1][1], "/prepared/lafan1/UnitreeG1/e.npz")
        with self.assertRaises(ValueError):
            prep.preprocess_dataset(None, dataset, 0, 1, False, source_root="/raw", output_root="/raw")



if __name__ == "__main__":
    unittest.main()

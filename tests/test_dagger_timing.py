"""Exercise the actual timer with a fake clock; no physics or distributed run."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("GLI_PATH", str(Path(__file__).resolve().parents[1]))

from track_mj.learning.policy.dagger import dagger_horizon


class DaggerTimingTest(unittest.TestCase):
    def test_sections_record_their_own_elapsed_time(self):
        timer = dagger_horizon.PerformanceTimer()
        self.assertEqual(timer.get_stats(), {})
        self.assertIsNone(timer.get_stats("missing"))
        with patch.object(dagger_horizon.time, "perf_counter", side_effect=[1., 3., 5., 9.]):
            with timer.timer("preparation"):
                pass
            with timer.timer("barrier"):
                pass
        self.assertEqual(timer.get_stats("preparation")["mean"], 2.)
        self.assertEqual(timer.get_stats("barrier")["mean"], 4.)
        self.assertEqual(set(timer.get_stats()), {"preparation", "barrier"})

    def test_statistics_keep_only_the_requested_window(self):
        timer = dagger_horizon.PerformanceTimer(window_size=2)
        with patch.object(dagger_horizon.time, "perf_counter", side_effect=[0., 1., 2., 6., 7., 13.]):
            for _ in range(3):
                with timer.timer("update"):
                    pass
        self.assertEqual(timer.get_stats("update"), {"mean": 5., "std": 1., "min": 4., "max": 6., "count": 2})

    def test_failed_section_records_elapsed_time_and_preserves_the_exception(self):
        timer = dagger_horizon.PerformanceTimer()
        with patch.object(dagger_horizon.time, "perf_counter", side_effect=[10., 13.]):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                with timer.timer("update"):
                    raise RuntimeError("update failed")
        self.assertEqual(timer.get_stats("update")["mean"], 3.)
        self.assertEqual(timer.get_stats("update")["count"], 1)


if __name__ == "__main__":
    unittest.main()

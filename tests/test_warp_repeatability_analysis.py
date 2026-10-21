"""CPU checks for the diagnostic's independent-copy and leaf comparison rules."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

from collections import namedtuple
import unittest

import jax
import jax.numpy as jp
import numpy as np

from native_warp_pillar_repeatability import (
    check_independent_buffers, combine, device_copy, differences, host_copy,
    unselected_leaf_diffs,
)


State = namedtuple("State", "data obs reward done metrics info")
Data = namedtuple("Data", "qpos qvel")


class RepeatabilityAnalysisTests(unittest.TestCase):
    def test_three_copies_and_state_leaf_metrics(self):
        state = State(Data(jp.arange(4.).reshape(2, 2), jp.zeros((2, 2))),
                      {"state": jp.ones((2, 2))}, jp.zeros(2), jp.zeros(2),
                      {"reward": jp.zeros(2)}, {"rng": jp.arange(4).reshape(2, 2)})
        pre = host_copy(state)
        clones = [device_copy(pre, jax.devices("cpu")[0]) for _ in range(3)]
        check_independent_buffers(clones)
        first = host_copy(clones[0])
        second = host_copy(clones[1])
        changed = second._replace(data=second.data._replace(
            qpos=second.data.qpos + np.array([[0., .01], [0., 0.]])))
        baseline = combine([differences(first, second)] * 3)
        measured = combine([differences(first, changed)] * 3)
        key = next(name for name in measured if name.endswith(".data.qpos"))
        self.assertEqual(baseline[key]["max_abs"], 0.)
        self.assertAlmostEqual(measured[key]["max_abs"], .01)
        kept = unselected_leaf_diffs(first, changed)
        self.assertFalse(kept[key])
        self.assertTrue(all(same for name, same in kept.items() if name != key))


if __name__ == "__main__":
    unittest.main()

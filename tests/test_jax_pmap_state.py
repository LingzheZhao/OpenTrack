"""CPU checks of the trainers' mapped state axes and ordered device placement."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'track_mj/learning/policy/jax_sharding.py'
spec = importlib.util.spec_from_file_location('tested_jax_sharding', PATH)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def training_state(path):
    # Execute the actual Flax state declaration, without importing environments,
    # models, assets or the training entry point and its simulator dependencies.
    source = ast.parse(path.read_text())
    declaration = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'TrainingState')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              declaration], type_ignores=[])
    namespace = {'flax': flax, '__name__': __name__}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    params = {'policy': {'w': jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
              'value': jnp.array([1., -2.], dtype=jnp.float32)}
    optimizer = optax.adam(.001).init(params)
    values = {'params': params, 'normalizer_params': {'mean': jnp.zeros(3), 'count': jnp.array(4.)},
              'env_steps': {'hi': jnp.array(0, dtype=jnp.uint32), 'lo': jnp.array(7, dtype=jnp.uint32)}}
    values.update({node.target.id: optimizer for node in declaration.body if isinstance(node, ast.AnnAssign)
                   and 'optimizer_state' in node.target.id})
    return namespace['TrainingState'](**values)


class PmapStateTests(unittest.TestCase):
    def setUp(self):
        self.devices = jax.local_devices()
        self.assertGreaterEqual(len(self.devices), 2, 'run with two forced CPU devices')

    def assert_equal_tree(self, actual, expected):
        self.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
        for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(got, want)
            self.assertEqual(got.dtype, want.dtype)

    def test_actual_training_states_keep_leading_axis_shards_and_pmap_rank(self):
        paths = [ROOT / 'track_mj/learning/policy/ppo/train_tracking.py',
                 ROOT / 'track_mj/learning/policy/model_based_ppo/train_model_based_ppo.py']
        for path in paths:
            original = training_state(path)
            for devices in (self.devices[:1], self.devices[:2], self.devices[:2][::-1]):
                with self.subTest(trainer=str(path.relative_to(ROOT)), devices=[device.id for device in devices]):
                    placed = placement.replicate_for_pmap(original, devices)
                    expected = jax.tree_util.tree_map(lambda value: np.stack([np.asarray(value)] * len(devices)), original)
                    self.assert_equal_tree(placed, expected)
                    for leaf, value in zip(jax.tree_util.tree_leaves(placed), jax.tree_util.tree_leaves(original)):
                        self.assertEqual(leaf.shape, (len(devices), *value.shape))
                        self.assertEqual(leaf.sharding.device_set, set(devices))
                        indices = leaf.sharding.devices_indices_map(leaf.shape)
                        for offset, device in enumerate(devices):
                            self.assertEqual(indices[device][0].indices(leaf.shape[0]), (offset, offset + 1, 1))
                            shard = next(shard for shard in leaf.addressable_shards if shard.device == device)
                            np.testing.assert_array_equal(shard.data, np.asarray(value)[None])

                    def mapped(state):
                        self.assertEqual(state.params['policy']['w'].shape, (2, 3))
                        offset = jax.lax.axis_index('i')
                        shifted = jax.tree_util.tree_map(lambda value: value + offset.astype(value.dtype), state)
                        total = jax.tree_util.tree_map(lambda value: jax.lax.psum(value, 'i'), state)
                        return shifted, total

                    # First-N selections use the trainers' default pmap device
                    # selection. Reversed devices explicitly preserve ordering.
                    options = {} if devices == self.devices[:len(devices)] else {'devices': devices}
                    mapped = jax.pmap(mapped, axis_name='i', in_axes=0, out_axes=0, **options)
                    shifted, total = mapped(placed)
                    expected_shifted = jax.tree_util.tree_map(
                        lambda value: np.stack([np.asarray(value) + np.asarray(i, dtype=value.dtype)
                                                for i in range(len(devices))]), original)
                    expected_total = jax.tree_util.tree_map(lambda value: value * len(devices), expected)
                    self.assert_equal_tree(shifted, expected_shifted)
                    self.assert_equal_tree(total, expected_total)
                    historical = getattr(jax, 'device_put_replicated', None)
                    if historical is not None:
                        old = historical(original, devices)
                        self.assert_equal_tree(placed, old)
                        for new_leaf, old_leaf in zip(jax.tree_util.tree_leaves(placed), jax.tree_util.tree_leaves(old)):
                            old_indices = old_leaf.sharding.devices_indices_map(old_leaf.shape)
                            self.assertEqual(old_leaf.sharding.device_set, new_leaf.sharding.device_set)
                            for offset, device in enumerate(devices):
                                # PmapSharding may represent one leading-axis
                                # element with an integer instead of a slice.
                                selected = np.arange(len(devices))[old_indices[device][0]]
                                np.testing.assert_array_equal(np.atleast_1d(selected), [offset])
                        old_shifted, old_total = mapped(old)
                        self.assert_equal_tree(shifted, old_shifted)
                        self.assert_equal_tree(total, old_total)

    def test_python_scalar_leaves_and_empty_nodes_keep_historical_dtypes(self):
        original = {'int': 3, 'float': 1.5, 'numpy': np.array(2., dtype=np.float32), 'empty': (None, ())}
        placed = placement.replicate_for_pmap(original, self.devices[:2])
        historical = getattr(jax, 'device_put_replicated', None)
        if historical is not None:
            old = historical(original, self.devices[:2])
            self.assert_equal_tree(placed, old)
            for got, want in zip(jax.tree_util.tree_leaves(placed), jax.tree_util.tree_leaves(old)):
                self.assertEqual(got.weak_type, want.weak_type)
        for leaf in jax.tree_util.tree_leaves(placed):
            self.assertEqual(leaf.shape, (2,))

    def test_empty_device_selection_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'at least one device'):
            placement.replicate_for_pmap(jnp.array(1.), [])


if __name__ == '__main__':
    print(f'JAX {jax.__version__}; CPU devices: {jax.local_devices()}', flush=True)
    unittest.main()

"""Place training-state pytrees on the leading axis consumed by pmap."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def replicate_for_pmap(tree, devices):
    """Keep one identical slice per ordered device, including the mapped axis.

    A fully replicated PartitionSpec() would omit the leading axis required by
    the trainers' in_axes=0. This follows JAX's documented drop-in replacement
    for device_put_replicated, without changing the trainers' pmap functions.
    """
    devices = tuple(devices)
    if not devices:
        raise ValueError('pmap state placement requires at least one device')
    sharding = NamedSharding(Mesh(np.array(devices), ('i',)), PartitionSpec('i'))
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(jnp.stack([leaf] * len(devices)), sharding), tree)

"""Physical evaluation boundaries shared by the native tracking environments.

Policy observations keep their native timing and ordering. Endpoint kinematics
are measured on an owned copy, separately from the state used for continuation.
No normal- or total-contact-force history is exposed by this interface.
"""
from __future__ import annotations

import jax
import jax.numpy as jp


def copy_state_containers(state):
    """Protect the input snapshot from native Python info/metrics assignments.

    JAX leaves are immutable; rebuilding their containers is sufficient for the
    native functional step. Physical measurement copies the actual leaves too.
    """
    return state.replace(info=jax.tree_util.tree_map(lambda value: value, state.info),
                         metrics=jax.tree_util.tree_map(lambda value: value, state.metrics))


def owned_data(data):
    """Copy numerical leaves before calling native forward for measurement."""
    return jax.tree_util.tree_map(lambda value: jp.array(value, copy=True), data)


def contact_measurement() -> dict:
    return {"available": False, "reason": "native evaluation does not capture measured substep contact forces; "
            "collision geometry and endpoint forward are not force measurements"}

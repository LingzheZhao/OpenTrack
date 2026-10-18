"""Physical evaluation boundaries shared by the native tracking environments.

Policy observations keep their native timing and ordering. Endpoint kinematics
are measured on an owned copy, separately from the state used for continuation.
No normal- or total-contact-force history is exposed by this interface.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jp


class EvaluationTransition(NamedTuple):
    pre: Any
    post_terminal: Any
    post_physical_data: Any
    next_state: Any
    reset_mask: jax.Array


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


def where_done(done, reset_value, value):
    if not hasattr(value, "shape"):
        return reset_value
    if value.shape[:done.ndim] == done.shape:
        mask = done.reshape(done.shape + (1,) * (value.ndim - done.ndim))
        return jp.where(mask, reset_value, value)
    return jp.where(jp.any(done), reset_value, value)


PERSISTENT_DATA_FIELDS = frozenset({
    "time", "qpos", "qvel", "act", "history", "plugin_state", "qacc_warmstart",
    "ctrl", "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat", "userdata",
})


def _path_names(path):
    return tuple(getattr(key, "name", getattr(key, "key", None)) for key in path)


def _native_non_vmap_fields(data):
    impl = getattr(data, "_impl", None)
    if impl is not None and hasattr(impl, "contact__worldid"):
        # Authoritative native batching classification, not a leading-size
        # heuristic: global contact capacity can equal the number of worlds.
        from mujoco.mjx.warp.types import DATA_NON_VMAP
        return DATA_NON_VMAP
    return frozenset()


def merge_world_data(done, reset_data, current_data, *, non_vmap_fields=()):
    """Select per-world native data, with explicitly classified shared buffers.

    MuJoCo 3.9 DataWarp supplies DATA_NON_VMAP for its shared contact/work
    fields. Every other dynamic data leaf must have the vector world's prefix.
    An unexpected layout is refused instead of silently guessing an axis.
    """
    def select(path, reset, current):
        names = _path_names(path)
        shared = len(names) >= 2 and names[0] == "_impl" and names[1] in non_vmap_fields
        if shared:
            return jp.where(jp.any(done), reset, current)
        if not hasattr(current, "shape") or current.shape[:done.ndim] != done.shape:
            raise ValueError(f"native world data {names} lacks expected world prefix {done.shape}")
        mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
        return jp.where(mask, reset, current)
    return jax.tree_util.tree_map_with_path(select, reset_data, current_data)


def _restore_ongoing_persistent_data(done, refreshed, current_data):
    def select(path, fresh, current):
        names = _path_names(path)
        if names and names[0] in PERSISTENT_DATA_FIELDS:
            return where_done(done, fresh, current)
        return fresh
    return jax.tree_util.tree_map_with_path(select, refreshed, current_data)


def reset_done_envs(current_state, reset_fn: Callable, refresh_data_fn: Callable):
    """Reset isolation correction for ongoing RNG and persistent physical state.

    Reset candidates may be evaluated in parallel for all worlds, but their
    keys/state are installed only where done. Native forward refreshes derived
    caches for ALL worlds coherently with Warp's shared contact buffers; these
    caches are not promised identical to a run without a peer reset. Only the
    persistent integration/control fields of ongoing worlds are restored, so
    old per-world constraint indexes cannot be mixed with new shared contacts.
    This corrects vector-wrapper reset semantics; it is not a claim of bitwise
    parity with the previous vector wrapper. Terminal reward/done/metrics stay.
    """
    done = current_state.done.astype(bool)
    reset_rng = jax.vmap(lambda key: jax.random.split(key, 2)[0])(current_state.info["rng"])
    reset_state = reset_fn(reset_rng)
    select = lambda reset, current: where_done(done, reset, current)
    data = merge_world_data(done, reset_state.data, current_state.data,
                            non_vmap_fields=_native_non_vmap_fields(current_state.data))
    refreshed = refresh_data_fn(data)
    data = _restore_ongoing_persistent_data(done, refreshed, current_state.data)
    obs = jax.tree_util.tree_map(select, reset_state.obs, current_state.obs)
    info = dict(current_state.info)
    for key, reset_value in reset_state.info.items():
        info[key] = jax.tree_util.tree_map(select, reset_value, current_state.info[key])
    return current_state.replace(data=data, obs=obs, info=info)


def evaluate_step(env, state, action, trajectory_data=None) -> EvaluationTransition:
    """One native control action, true terminal observation, then explicit reset.

    ``env`` is the native vector wrapper, without an action-repeat wrapper. The
    forward used for ``post_physical_data`` is a measurement of copied endpoint
    data; it never replaces native policy observations or continuation data.
    """
    pre = copy_state_containers(state)
    terminal = env.step_no_autoreset(state, action, trajectory_data)
    physical = env.physical_data(terminal)
    next_state = env.reset_done(terminal, trajectory_data)
    return EvaluationTransition(pre, terminal, physical, next_state, terminal.done.astype(bool))


def contact_measurement() -> dict:
    return {"available": False, "reason": "native evaluation does not capture measured substep contact forces; "
            "collision geometry and endpoint forward are not force measurements"}

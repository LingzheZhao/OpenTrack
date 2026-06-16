from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jp
import mujoco.mjx as mjx

from brax.envs.base import Env, State, Wrapper
from mujoco_playground._src import mjx_env, wrapper
from track_mj.utils.mjx_backend import MJX_WORLD_AXIS_NAME, mjx_impl


def _is_warp_env(env: Env) -> bool:
    config = getattr(getattr(env, "unwrapped", env), "_config", None)
    return mjx_impl(config) == "warp"


def _where_done(done: jax.Array, reset_value, value):
    if not hasattr(value, "shape"):
        return reset_value
    if value.shape[: done.ndim] == done.shape:
        mask = done.reshape(done.shape + (1,) * (value.ndim - done.ndim))
        return jp.where(mask, reset_value, value)
    return jp.where(jp.any(done), reset_value, value)


def _annotate_worldid(state: State) -> State:
    info = dict(state.info)
    info["worldid"] = jp.arange(state.done.shape[0], dtype=jp.int32)
    return state.replace(info=info)


def _reset_done_envs(
    current_state: State,
    reset_fn: Callable[[jax.Array], State],
    refresh_data_fn: Callable[[mjx.Data], mjx.Data],
) -> State:
    split_rng = jax.vmap(lambda rng: jax.random.split(rng, 2))(current_state.info["rng"])
    reset_rng = split_rng[:, 0]
    next_rng = split_rng[:, 1]
    current_info = dict(current_state.info)
    current_info["rng"] = next_rng
    current_state = current_state.replace(info=current_info)

    reset_state = reset_fn(reset_rng)
    done = current_state.done.astype(bool)
    data = jax.tree_util.tree_map(lambda r, v: _where_done(done, r, v), reset_state.data, current_state.data)
    data = refresh_data_fn(data)
    obs = jax.tree_util.tree_map(lambda r, v: _where_done(done, r, v), reset_state.obs, current_state.obs)
    info = dict(current_state.info)
    for key, reset_value in reset_state.info.items():
        info[key] = jax.tree_util.tree_map(
            lambda r, v: _where_done(done, r, v),
            reset_value,
            current_state.info[key],
        )
    return current_state.replace(data=data, obs=obs, info=info)


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, trajectory_data) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        state = jax.vmap(self.env.reset, in_axes=(0, None), axis_name=MJX_WORLD_AXIS_NAME)(rng, trajectory_data)
        return _annotate_worldid(state)

    def step(self, state: State, action: jax.Array, trajectory_data) -> State:
        state = jax.vmap(self.env.step, in_axes=(0, 0, None), axis_name=MJX_WORLD_AXIS_NAME)(
            state, action, trajectory_data
        )
        if not _is_warp_env(self.env):
            return state

        reset_fn = lambda reset_rng: self.reset(reset_rng, trajectory_data)
        return jax.lax.cond(
            jp.any(state.done.astype(bool)),
            lambda x: _reset_done_envs(
                x,
                reset_fn,
                lambda data: jax.vmap(
                    lambda per_world_data: mjx.forward(self.env.mjx_model, per_world_data),
                    axis_name=MJX_WORLD_AXIS_NAME,
                )(data),
            ),
            lambda x: x,
            state,
        )


class ModifiedEpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array, trajectory_data) -> State:
        state = self.env.reset(rng, trajectory_data)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["truncation"] = jp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info["episode_done"] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics["sum_reward"] = jp.zeros(rng.shape[:-1])
        episode_metrics["average_sum_reward"] = jp.zeros(rng.shape[:-1])
        episode_metrics["length"] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
            episode_metrics["average_" + metric_name] = jp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array, trajectory_data) -> State:
        def f(state, _):
            nstate = self.env.step(state, action, trajectory_data)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        done = state.done
        state.info["steps"] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] += jp.sum(rewards, axis=0)
        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["average_sum_reward"] = (
            state.info["episode_metrics"]["sum_reward"] / state.info["episode_metrics"]["length"]
        )

        for metric_name in state.metrics.keys():
            if metric_name != "reward":
                state.info["episode_metrics"][metric_name] += state.metrics[metric_name]
                state.info["episode_metrics"]["average_" + metric_name] = (
                    state.info["episode_metrics"][metric_name] / state.info["episode_metrics"]["length"]
                )
                state.info["episode_metrics"][metric_name] *= 1 - prev_done

        state.info["episode_metrics"]["sum_reward"] *= 1 - prev_done
        state.info["episode_metrics"]["length"] *= 1 - prev_done
        state.info["episode_done"] = done
        return state


class ModifiedDomainRandomizationVmapWrapper(Wrapper):
    """Brax wrapper for domain randomization."""

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

    def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
        env = self.env
        env.unwrapped._mjx_model = mjx_model
        return env

    def reset(self, rng: jax.Array, trajectory_data) -> mjx_env.State:
        def reset(mjx_model, rng, trajectory_data):
            env = self._env_fn(mjx_model=mjx_model)
            return env.reset(rng, trajectory_data)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0, None], axis_name=MJX_WORLD_AXIS_NAME)(
            self._mjx_model_v, rng, trajectory_data
        )
        return _annotate_worldid(state)

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data) -> mjx_env.State:
        def step(mjx_model, s, a, trajectory_data):
            env = self._env_fn(mjx_model=mjx_model)
            return env.step(s, a, trajectory_data)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0, None], axis_name=MJX_WORLD_AXIS_NAME)(
            self._mjx_model_v, state, action, trajectory_data
        )
        if not _is_warp_env(self.env):
            return res

        reset_fn = lambda reset_rng: self.reset(reset_rng, trajectory_data)
        res = jax.lax.cond(
            jp.any(res.done.astype(bool)),
            lambda x: _reset_done_envs(
                x,
                reset_fn,
                lambda data: jax.vmap(
                    lambda mjx_model, per_world_data: mjx.forward(mjx_model, per_world_data),
                    in_axes=[self._in_axes, 0],
                    axis_name=MJX_WORLD_AXIS_NAME,
                )(self._mjx_model_v, data),
            ),
            lambda x: x,
            res,
        )
        return res


def wrap_fn(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]] | None = None,
) -> wrapper.Wrapper:
    """Common wrapper pattern for all brax training agents.

    Args:
      env: environment to be wrapped
      vision: whether the environment will be vision based
      num_vision_envs: number of environments the renderer should generate,
        should equal the number of batched envs
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized model
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """

    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = ModifiedDomainRandomizationVmapWrapper(env, randomization_fn)
    env = ModifiedEpisodeWrapper(env, episode_length, action_repeat)
    return env

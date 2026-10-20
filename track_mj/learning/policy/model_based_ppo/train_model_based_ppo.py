"""Training loop for model-based PPO."""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Sequence

from absl import logging
from brax import base, envs
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint
from brax.training.types import Params, PRNGKey, Metrics, Policy, Transition
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from track_mj.learning.policy.ppo import acting_tracking
from track_mj.learning.policy.ppo import metrics_aggregator as metric_logger
from track_mj.learning.policy.jax_sharding import replicate_for_pmap
from track_mj.learning.policy.model_based_ppo import model_based_ppo_networks as ppo_networks
from track_mj.learning.policy.model_based_ppo import model_based_ppo_losses as ppo_losses


def recursive_update(base_dict: dict, update_dict: dict):
    for key, value in update_dict.items():
        base_value = base_dict.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            recursive_update(base_value, value)
        else:
            base_dict[key] = value


def actor_step(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
    trajectory_data=None,
) -> Tuple[envs.State, Transition]:
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions, trajectory_data)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


def generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    trajectory_data=None,
) -> Tuple[envs.State, Transition]:
    @jax.jit
    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(env, state, policy, current_key, extra_fields=extra_fields, trajectory_data=trajectory_data)
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, data


_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    policy_optimizer_state: optax.OptState
    world_model_optimizer_state: optax.OptState
    params: ppo_losses.ModelBasedPPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
):
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    wrap_for_training = wrap_env_fn if wrap_env_fn is not None else envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )
    return env


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    wrap_env: bool = True,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[ppo_networks.ModelBasedPPONetworks] = ppo_networks.make_model_based_ppo_networks,
    seed: int = 0,
    num_evals: int = 1,
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    trajectory_data: Optional[Any] = None,
    restore_ppo_checkpoint_path: Optional[str] = None,
    policy_learning_rate: float = 1e-4,
    world_model_learning_rate: float = 1e-4,
    use_inverse_dynamics_model: bool = False,
    use_adapter: bool = True,
    train_history_encoder_in_policy: bool = False,
    train_world_model: bool = True,
    supervised_loss_weight: float = 1.0,
    ppo_loss_weight: float = 1.0,
    world_model_gyro_weight: float = 1.0,
    world_model_gvec_weight: float = 1.0,
    world_model_joint_pos_weight: float = 1.0,
    world_model_joint_vel_weight: float = 1.0,
    world_model_root_height_weight: float = 1.0,
    world_model_autoregressive: bool = False,
    **kwargs,
):
    logging.info(f"Unused kwargs from loaded config: {kwargs}")
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * process_count

    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, _ = jax.random.split(local_key, 3)
    key_policy, key_value, key_history_encoder, key_world_model = jax.random.split(global_key, 4)

    assert num_envs % device_count == 0
    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        local_device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    if trajectory_data is None:
        trajectory_data = env.unwrapped.th.traj.data
    reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    env_state = reset_fn(key_envs, trajectory_data)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = (lambda x, y: x) if not normalize_observations else running_statistics.normalize
    ppo_network = network_factory(
        obs_shape,
        env.action_size,
        preprocess_observations_fn=normalize,
        use_inverse_dynamics_model=use_inverse_dynamics_model,
        use_adapter=use_adapter,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    def label_fn(params):
        def get_label(path, _):
            for key in path:
                if "adapter" in key:
                    return "train"
                if "hidden" in key:
                    return "freeze"
            raise ValueError(f"Parameter path {path} did not match any labelling rule.")

        if len(params) == 2:
            return [flax.traverse_util.path_aware_map(get_label, params[0]), "train"]
        return [flax.traverse_util.path_aware_map(get_label, params[0]), "train", "train"]

    if use_adapter:
        policy_optimizer = optax.multi_transform(
            {"train": optax.adam(learning_rate=policy_learning_rate), "freeze": optax.set_to_zero()},
            label_fn,
        )
    else:
        policy_optimizer = optax.adam(learning_rate=policy_learning_rate)
    world_model_optimizer = optax.adam(learning_rate=world_model_learning_rate)
    if max_grad_norm is not None:
        policy_optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), policy_optimizer)
        world_model_optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), world_model_optimizer)

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss_with_world_model,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        use_inverse_dynamics_model=use_inverse_dynamics_model,
        train_history_encoder_in_policy=train_history_encoder_in_policy,
        supervised_loss_weight=supervised_loss_weight,
        ppo_loss_weight=ppo_loss_weight,
        world_model_gyro_weight=world_model_gyro_weight,
        world_model_gvec_weight=world_model_gvec_weight,
        world_model_joint_pos_weight=world_model_joint_pos_weight,
        world_model_joint_vel_weight=world_model_joint_vel_weight,
        world_model_root_height_weight=world_model_root_height_weight,
    )

    if use_inverse_dynamics_model:
        world_model_loss_fn = functools.partial(ppo_losses.compute_inverse_dynamics_model_loss, ppo_network=ppo_network)
    else:
        world_model_loss_fn = functools.partial(
            ppo_losses.compute_world_model_loss,
            ppo_network=ppo_network,
            world_model_gyro_weight=world_model_gyro_weight,
            world_model_gvec_weight=world_model_gvec_weight,
            world_model_joint_pos_weight=world_model_joint_pos_weight,
            world_model_joint_vel_weight=world_model_joint_vel_weight,
            world_model_root_height_weight=world_model_root_height_weight,
            world_model_autoregressive=world_model_autoregressive,
        )

    gradient_update_fn = gradients.gradient_update_fn(loss_fn, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    world_model_gradient_update_fn = gradients.gradient_update_fn(
        world_model_loss_fn,
        world_model_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
    )

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        devices=local_devices_to_use,
        steps_between_logging=training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    ckpt_config = checkpoint.network_config(
        observation_size=obs_shape,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
    )

    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, trained_params, fixed_params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), trained_params, optimizer_state = gradient_update_fn(
            trained_params,
            fixed_params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state,
        )
        return (optimizer_state, trained_params, fixed_params, key), metrics

    def world_model_minibatch_step(carry, data, normalizer_params):
        optimizer_state, trained_params, fixed_params, key = carry
        (_, metrics), trained_params, optimizer_state = world_model_gradient_update_fn(
            trained_params,
            normalizer_params,
            data,
            optimizer_state=optimizer_state,
        )
        return (optimizer_state, trained_params, fixed_params, key), metrics

    def sgd_step(carry, _, data, normalizer_params):
        optimizer_state, trained_params, fixed_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, trained_params, fixed_params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, trained_params, fixed_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, trained_params, fixed_params, key), metrics

    def world_model_sgd_step(carry, _, data, normalizer_params):
        optimizer_state, trained_params, fixed_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, trained_params, fixed_params, _), metrics = jax.lax.scan(
            functools.partial(world_model_minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, trained_params, fixed_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, trained_params, fixed_params, key), metrics

    def training_step(carry, _):
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
                training_state.params.world_model,
                training_state.params.history_encoder,
            )
        )

        def f(carry, _):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
                trajectory_data=trajectory_data,
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        policy_optimizer_state = training_state.policy_optimizer_state
        world_model_optimizer_state = training_state.world_model_optimizer_state
        params_policy = training_state.params.policy
        params_value = training_state.params.value
        params_history_encoder = training_state.params.history_encoder
        params_world_model = training_state.params.world_model

        world_model_metrics = {}
        if train_world_model:
            (world_model_optimizer_state, trained_params, _, _), world_model_metrics = jax.lax.scan(
                functools.partial(world_model_sgd_step, data=data, normalizer_params=normalizer_params),
                (world_model_optimizer_state, [params_world_model, params_history_encoder], None, key_sgd),
                (),
                length=num_updates_per_batch,
            )
            params_world_model, params_history_encoder = trained_params

        metrics = {}
        if train_history_encoder_in_policy:
            (policy_optimizer_state, trained_params, _, _), metrics = jax.lax.scan(
                functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
                (
                    policy_optimizer_state,
                    [params_policy, params_value, params_history_encoder],
                    [params_world_model],
                    key_sgd,
                ),
                (),
                length=num_updates_per_batch,
            )
            params_policy, params_value, params_history_encoder = trained_params
        else:
            (policy_optimizer_state, trained_params, _, _), metrics = jax.lax.scan(
                functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
                (
                    policy_optimizer_state,
                    [params_policy, params_value],
                    [params_world_model, params_history_encoder],
                    key_sgd,
                ),
                (),
                length=num_updates_per_batch,
            )
            params_policy, params_value = trained_params

        if log_training_metrics:
            gpu_id = jax.lax.axis_index(_PMAP_AXIS_NAME)
            jax.lax.cond(
                gpu_id == 0,
                lambda: jax.debug.callback(
                    metrics_aggregator.update_episode_metrics,
                    data.extras["state_extras"]["episode_metrics"],
                    data.extras["state_extras"]["episode_done"],
                    {**metrics, **world_model_metrics},
                ),
                lambda: None,
            )

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            world_model_optimizer_state=world_model_optimizer_state,
            params=ppo_losses.ModelBasedPPONetworkParams(
                policy=params_policy,
                value=params_value,
                world_model=params_world_model,
                history_encoder=params_history_encoder,
            ),
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), {**metrics, **world_model_metrics}

    def training_epoch(training_state, state, key):
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(training_state, env_state, key):
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (num_training_steps_per_epoch * env_step_per_training_step * max(num_resets_per_eval, 1)) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

    init_params = ppo_losses.ModelBasedPPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        history_encoder=ppo_network.history_encoder.init(key_history_encoder),
        world_model=ppo_network.world_model.init(key_world_model),
    )

    obs_shape_specs = jax.tree_util.tree_map(lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs)
    if train_history_encoder_in_policy:
        training_state = TrainingState(
            policy_optimizer_state=policy_optimizer.init([init_params.policy, init_params.value, init_params.history_encoder]),
            world_model_optimizer_state=world_model_optimizer.init([init_params.world_model, init_params.history_encoder]),
            params=init_params,
            normalizer_params=running_statistics.init_state(obs_shape_specs),
            env_steps=types.UInt64(hi=0, lo=0),
        )
    else:
        training_state = TrainingState(
            policy_optimizer_state=policy_optimizer.init([init_params.policy, init_params.value]),
            world_model_optimizer_state=world_model_optimizer.init([init_params.world_model, init_params.history_encoder]),
            params=init_params,
            normalizer_params=running_statistics.init_state(obs_shape_specs),
            env_steps=types.UInt64(hi=0, lo=0),
        )

    if restore_params is not None:
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1],
                value=restore_params[2],
                world_model=restore_params[3],
                history_encoder=restore_params[4],
            ),
        )
    elif restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(
                policy=params[1],
                value=params[2],
                world_model=params[3],
                history_encoder=params[4],
            ),
        )
    elif restore_ppo_checkpoint_path is not None:
        pretrained_policy_params = checkpoint.load(restore_ppo_checkpoint_path)[1]
        recursive_update(init_params.policy, pretrained_policy_params)

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
                training_state.params.world_model,
                training_state.params.history_encoder,
            ),
            {},
        )

    training_state = replicate_for_pmap(training_state, jax.local_devices()[:local_devices_to_use])

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for _ in range(num_evals_after_init):
        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics = training_epoch_with_timing(training_state, env_state, epoch_keys)
            current_step = int(_unpmap(training_state.env_steps))
            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(key_envs, key_envs.shape[1])
            env_state = reset_fn(key_envs, trajectory_data) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
                training_state.params.world_model,
                training_state.params.history_encoder,
            )
        )
        policy_params_fn(current_step, make_policy, params)
        if save_checkpoint_path is not None:
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

    if current_step < num_timesteps:
        raise AssertionError(f"Total steps {current_step} is less than num_timesteps={num_timesteps}.")

    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
            training_state.params.world_model,
            training_state.params.history_encoder,
        )
    )
    logging.info("total steps: %s", current_step)
    pmap.synchronize_hosts()
    return make_policy, params, training_metrics

import inspect
import functools
import time
import os
import json
import pytz

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from absl import logging
import tyro
import wandb
import numpy as np
import jax
import jax.numpy as jp

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

from brax.training.agents.ppo.networks import make_ppo_networks

import track_mj as tmj
from track_mj import update_file_handler
from track_mj.constant import WANDB_PATH_LOG
from track_mj.envs.g1_tracking.train.base_env import G1Env
from track_mj.learning.policy.ppo import train_tracking as ppo
from track_mj.learning.policy.model_based_ppo import train_model_based_ppo as mbppo
from track_mj.learning.policy.model_based_ppo.model_based_ppo_networks import make_model_based_ppo_networks
from track_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from track_mj.dr.domain_randomize_tracking import (
    domain_randomize,
    domain_randomize_terrain,
)
from track_mj.utils.mjx_backend import configure_mjx


@dataclass
class Args:
    task: str
    load_exp_name: str
    exp_name: str = "debug"
    exp_tags: str = None
    exp_notes: str = None
    seed: int = 42
    convert_onnx: bool = True

    # ====== fine-tune overrides ======
    num_timesteps: int = 2_000_000_000
    num_envs: Optional[int] = 16384
    batch_size: Optional[int] = 512
    learning_rate: Optional[float] = None
    restore_value_fn: bool = True
    use_adapter: bool = True
    use_world_model: bool = True
    policy_lr: Optional[float] = None
    world_model_lr: Optional[float] = None

    # ====== env overrides ======
    obs_noise_level: Optional[float] = None
    history_len: Optional[int] = None
    trajectory_name: Optional[str] = None


def _prepare_exp_name(task: str, exp_name: str) -> str:
    r"""
    timestamp_task_expname
    """
    cst_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    timestamp = cst_time.strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"


def _parse_exp_tags(tags):
    r"""
    Parse tags like `"'[tag1, tag2]'" into a list.
    """
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        cleaned = tags.strip()
        if (cleaned.startswith("[") and cleaned.endswith("]")) or \
           (cleaned.startswith("(") and cleaned.endswith(")")) or \
           (cleaned.startswith("\"") and cleaned.endswith("\"")) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        result = []
        for tag in cleaned.split(","):
            tag = tag.strip()
            if tag.startswith("\"") and tag.endswith("\"") or \
               tag.startswith("'") and tag.endswith("'"):
                tag = tag[1:-1]
            if tag:
                result.append(tag)
        return result
    return [str(tags)]


def _enable_debug_mode():
    jax.config.update("jax_traceback_filtering", "off")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)


def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = Path(WANDB_PATH_LOG) / "adapter" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path


def _log_checkpoint_path(ckpt_path: Path):
    logging.info(f"Checkpoint path: {ckpt_path}")


def _prepare_training_params(cfg, ckpt_path: Path):
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_fn
    network_fn = make_ppo_networks
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory) if hasattr(cfg, "network_factory") else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def _get_latest_ckpt(ckpt_root: Path) -> Optional[Path]:
    ckpts = [p for p in ckpt_root.glob("*") if p.is_dir() and p.name.isdigit()]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name))
    return ckpts[-1]


def _prepare_mbppo_training_params(policy_cfg, mbppo_cfg, ckpt_path: Path):
    params = policy_cfg.to_dict()
    params.update(mbppo_cfg.to_dict())
    params.pop("network_factory", None)
    # Adapter MBPPO fine-tune should not restore full MBPPO params from policy_cfg.
    # We only restore PPO policy weights via `restore_ppo_checkpoint_path`.
    params.pop("restore_checkpoint_path", None)
    params.pop("restore_params", None)
    params["wrap_env_fn"] = wrap_fn
    params["network_factory"] = functools.partial(
        make_model_based_ppo_networks,
        **policy_cfg.network_factory,
        **mbppo_cfg.network_factory,
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def _load_pretrained_for_finetune(args: Args, policy_cfg) -> None:
    load_root = Path(WANDB_PATH_LOG) / "track" / args.load_exp_name / "checkpoints"
    if not load_root.exists():
        raise FileNotFoundError(f"Pretrained checkpoint root does not exist: {load_root}")

    config_path = load_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Pretrained config does not exist: {config_path}")

    with open(config_path, "r") as f:
        config_json = json.load(f)

    loaded_policy_cfg = config_json.get("policy_config", {})
    if "network_factory" in loaded_policy_cfg:
        policy_cfg.network_factory.update(loaded_policy_cfg["network_factory"])

    latest_ckpt = _get_latest_ckpt(load_root)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {load_root}")
    policy_cfg.restore_checkpoint_path = str(latest_ckpt)
    policy_cfg.restore_value_fn = args.restore_value_fn
    logging.info(f"Fine-tune restore checkpoint path: {latest_ckpt}")


def _load_pretrained_for_mbppo_finetune(args: Args, policy_cfg, mbppo_policy_cfg) -> None:
    load_root = Path(WANDB_PATH_LOG) / "track" / args.load_exp_name / "checkpoints"
    if not load_root.exists():
        raise FileNotFoundError(f"Pretrained checkpoint root does not exist: {load_root}")

    config_path = load_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Pretrained config does not exist: {config_path}")

    with open(config_path, "r") as f:
        config_json = json.load(f)

    loaded_policy_cfg = config_json.get("policy_config", {})
    if "network_factory" in loaded_policy_cfg:
        policy_cfg.network_factory.update(loaded_policy_cfg["network_factory"])

    latest_ckpt = _get_latest_ckpt(load_root)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {load_root}")
    # Clear full-parameter restore path to avoid loading incompatible MBPPO heads/encoders.
    policy_cfg.restore_checkpoint_path = None
    mbppo_policy_cfg.restore_ppo_checkpoint_path = str(latest_ckpt)
    mbppo_policy_cfg.use_adapter = args.use_adapter
    mbppo_policy_cfg.train_history_encoder_in_policy = not args.use_world_model
    mbppo_policy_cfg.train_world_model = args.use_world_model
    if args.policy_lr is not None:
        mbppo_policy_cfg.policy_learning_rate = args.policy_lr
    if args.world_model_lr is not None:
        mbppo_policy_cfg.world_model_learning_rate = args.world_model_lr
    logging.info(f"Fine-tune restore PPO checkpoint path: {latest_ckpt}")


def _apply_policy_args_to_config(args: Args, cfg, debug: bool):
    cfg.num_timesteps = args.num_timesteps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    if debug:
        cfg.training_metrics_steps = 1000
        cfg.num_evals = 0
        cfg.batch_size = 8
        cfg.num_minibatches = 2
        cfg.num_envs = cfg.batch_size * cfg.num_minibatches
        cfg.episode_length = 200
        cfg.unroll_length = 10
        cfg.num_updates_per_batch = 1
        cfg.action_repeat = 1
        cfg.num_timesteps = 100_000
        cfg.num_resets_per_eval = 1


def _apply_env_args_to_config(args: Args, cfg):
    if (args.use_adapter or args.use_world_model) and args.history_len is None and cfg.history_len == 0:
        cfg.history_len = 79
    if args.history_len is not None:
        cfg.history_len = args.history_len
    if args.obs_noise_level is not None:
        cfg.noise_config.level = args.obs_noise_level
    if args.trajectory_name is not None:
        cfg.reference_traj_config.name = {"lafan1": args.trajectory_name.replace(" ", "").split(",")}

    cfg.obs_keys = sorted(list(set(cfg.obs_keys)))
    cfg.privileged_obs_keys = sorted(list(set(cfg.privileged_obs_keys)))

    print("Final obs keys:", cfg.obs_keys)
    print("Final privileged obs keys:", cfg.privileged_obs_keys)


def _init_wandb(args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"):
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group="Adapter",
        config={
            "num_timesteps": args.num_timesteps,
            "task": args.task,
            "group": "Adapter",
            "load_exp_name": args.load_exp_name,
        },
        dir=os.path.join(WANDB_PATH_LOG, "adapter"),
        tags=_parse_exp_tags(args.exp_tags),
        notes=args.exp_notes,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _progress(num_steps, metrics, times, total_steps, debug_mode):
    r"""
    Log metrcis to wandb. Estimate remaining time.
    """
    now = time.monotonic()
    times.append(now)
    if metrics and not debug_mode:
        try:
            wandb.log(metrics, step=num_steps)
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

    if len(times) < 2 or num_steps == 0:
        return
    step_times = np.diff(times)
    median_step_time = np.median(step_times)
    if median_step_time <= 0:
        return
    steps_logged = num_steps / len(step_times)
    est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
    logging.info(f"NumSteps {num_steps} - EstTimeLeft {est_seconds_left:.1f}[s]")


def _report_training_time(times):
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def get_trajectory_handler(env, args: Args):
    trajectory_data = env.prepare_trajectory(env._config.reference_traj_config.name)
    obs_size = env.observation_size
    act_size = env.action_size
    env.th.traj = None

    print("=" * 50)
    print(
        f"Tracking {len(trajectory_data.split_points) - 1} trajectories with {trajectory_data.qpos.shape[0]} timesteps, fps={1 / env.dt:.1f}"
    )
    print(f"Observation: {env._config.obs_keys}")
    print(f"Privileged state: {env._config.privileged_obs_keys}")
    print("=" * 50)

    return trajectory_data, obs_size, act_size


def train(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_adapter_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_adapter_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config
    mbppo_policy_cfg = getattr(task_cfg, "mbppo_policy_config", None)

    exp_name = _prepare_exp_name(args.task, args.exp_name)
    debug_mode = "debug" in exp_name

    if debug_mode:
        _enable_debug_mode()

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    _apply_policy_args_to_config(args, policy_cfg, debug_mode)
    _apply_env_args_to_config(args, env_cfg)
    configure_mjx(env_cfg, policy_cfg.num_envs)
    if mbppo_policy_cfg is not None:
        mbppo_policy_cfg.network_factory.history_len = env_cfg.history_len

    if args.task == "G1TrackingGeneralTerrainDR":
        hfield_data = jp.asarray(np.load("storage/data/hfield/terrain.npz")["hfield_data"])
        policy_cfg.randomization_fn = functools.partial(domain_randomize_terrain, all_hfield_data=hfield_data)
        del hfield_data
        assert env_cfg.terrain_type == "rough_terrain"
    elif args.task == "G1TrackingGeneralDR":
        assert policy_cfg.randomization_fn == domain_randomize
    elif args.task == "G1TrackingGeneral":
        assert policy_cfg.randomization_fn is None

    use_mbppo = mbppo_policy_cfg is not None and (args.use_adapter or args.use_world_model)
    if use_mbppo:
        _load_pretrained_for_mbppo_finetune(args, policy_cfg, mbppo_policy_cfg)
        train_fn = functools.partial(mbppo.train, **_prepare_mbppo_training_params(policy_cfg, mbppo_policy_cfg, ckpt_path))
    else:
        _load_pretrained_for_finetune(args, policy_cfg)
        train_fn = functools.partial(ppo.train, **_prepare_training_params(policy_cfg, ckpt_path))

    if not debug_mode:
        _init_wandb(args, exp_name, env_class, task_cfg, ckpt_path)

    times = [time.monotonic()]

    env: G1Env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    trajectory_data, obs_size, act_size = get_trajectory_handler(env, args)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        trajectory_data=trajectory_data,
        progress_fn=lambda s, m: _progress(
            num_steps=s,
            metrics=m,
            times=times,
            total_steps=policy_cfg.num_timesteps,
            debug_mode=debug_mode,
        ),
        policy_params_fn=lambda *unused_args: None,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    logging.info(f"Run {exp_name} Fine-tune done.")

    if args.convert_onnx:
        from track_mj.eval.adapter.brax2onnx import (
            get_latest_ckpt,
            convert_jax2onnx,
            convert_jax2onnx_with_history,
        )

        env.prepare_trajectory(env._config.reference_traj_config.name)

        try:
            ckpt_dir = get_latest_ckpt(ckpt_path)
            output_path = f"{ckpt_dir}/policy.onnx"
            if use_mbppo:
                convert_jax2onnx_with_history(
                    output_path=output_path,
                    inference_fn=inference_fn,
                    policy_network_cfg=policy_cfg.network_factory,
                    mbppo_network_cfg=mbppo_policy_cfg.network_factory,
                    obs_size=obs_size,
                    action_size=act_size,
                    history_len=env_cfg.history_len,
                    jax_params=params,
                    use_adapter=args.use_adapter,
                    activation="swish",
                )
            else:
                policy_obs_key = policy_cfg.network_factory.policy_obs_key
                convert_jax2onnx(
                    ckpt_dir=ckpt_dir,
                    output_path=output_path,
                    inference_fn=inference_fn,
                    hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                    obs_size=obs_size,
                    action_size=act_size,
                    policy_obs_key=policy_obs_key,
                    jax_params=params,
                    activation="swish",
                )
        except ImportError:
            logging.warning("TensorFlow is not installed. Please install TensorFlow to use ONNX conversion.")


if __name__ == "__main__":
    train(tyro.cli(Args))

import argparse
import os

from track_mj.learning.train.train_ppo_track import Args, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded PPO tracking job for MJX backend comparison.")
    parser.add_argument("--backend", choices=["jax", "warp"], required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--task", default="G1TrackingGeneral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-timesteps", type=int, default=327680)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--unroll-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-updates-per-batch", type=int, default=1)
    parser.add_argument("--training-metrics-steps", type=int, default=327680)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["OPENTRACK_MJX_IMPL"] = args.backend
    train(
        Args(
            task=args.task,
            exp_name=args.exp_name,
            seed=args.seed,
            convert_onnx=False,
            disable_wandb=True,
            num_timesteps=args.num_timesteps,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            unroll_length=args.unroll_length,
            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            num_evals=0,
            training_metrics_steps=args.training_metrics_steps,
            max_devices_per_host=1,
            save_checkpoints=args.save_checkpoints,
        )
    )


if __name__ == "__main__":
    main()

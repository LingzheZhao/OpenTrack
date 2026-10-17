"""Prepare immutable motion artifacts in a new tree (run as a compute job)."""
from dataclasses import dataclass
from pathlib import Path
import tyro
import track_mj as tmj


@dataclass
class Args:
    task: str
    source_root: Path
    output_root: Path
    smooth_start_end: bool = False
    num_batches: int = 1
    batch_idx: int = 0
    flat_source: bool = False
    source_reference: str | None = None


def motion_preprocess(args: Args):
    if args.source_root.resolve() == args.output_root.resolve():
        raise ValueError("Preparation output root must differ from source root.")
    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    return env.preprocess_trajectory(
        env_cfg.reference_traj_config.name, args.batch_idx, args.num_batches,
        args.smooth_start_end, source_root=args.source_root,
        output_root=args.output_root, flat_source=args.flat_source,
        source_reference=args.source_reference,
    )


if __name__ == "__main__":
    motion_preprocess(tyro.cli(Args))

"""Explicit motion preparation; runtime loaders never rewrite or transform inputs.

The numerical processing remains in the existing trajectory utilities.  Preparation
records its source reference and operations in TrajectoryInfo.metadata, retaining the
source metadata (including any upstream retargeting or smoothing provenance).
"""
from dataclasses import replace
import math
from pathlib import Path


PREPARATION_KEY = "opentrack_preparation"


def motion_path(reference_config, dataset_name, trajectory_name):
    """Resolve the configured prepared-data root with the native dataset layout."""
    root = reference_config.get("root", None) or Path.cwd() / "storage/data/mocap"
    return Path(root) / dataset_name / "UnitreeG1" / f"{trajectory_name}.npz"


def require_prepared(trajectory, *, control_dt, recalculate_velocity, path):
    """Validate prepared fields without extending, resampling or recalculating."""
    frequency = 1.0 / control_dt
    if not trajectory.data.is_complete:
        raise ValueError(f"Motion {path} lacks prepared fields; run preprocess_motion with a new output root.")
    if not math.isclose(float(trajectory.info.frequency), frequency, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Motion {path} must be prepared at {frequency:g} Hz before runtime loading.")
    preparation = (trajectory.info.metadata or {}).get(PREPARATION_KEY, {})
    if recalculate_velocity and preparation.get("velocity_frequency") != frequency:
        raise ValueError(f"Motion {path} requires explicitly prepared reference velocities at {frequency:g} Hz.")
    return trajectory


def load_prepared_trajectory(path, *, backend, control_dt, recalculate_velocity):
    from track_mj.utils.dataset.traj_class import Trajectory
    return require_prepared(
        Trajectory.load(path, backend=backend), control_dt=control_dt,
        recalculate_velocity=recalculate_velocity, path=path,
    )


def prepare_motion(env, source_path, output_path, *, smooth_start_end, source_reference=None):
    """Write one new artifact using native resampling, smoothing and FK helpers."""
    import jax.numpy as jp
    import numpy as np
    from track_mj.utils.dataset.traj_class import (
        Trajectory, recalculate_traj_angular_velocity,
        recalculate_traj_linear_velocity, recalculate_traj_joint_velocity,
    )
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Preparation output must differ from the source motion.")
    if output_path.exists():
        raise FileExistsError(f"Preparation refuses existing artifact {output_path}.")
    trajectory = Trajectory.load(source_path, backend=jp)
    source_frequency = float(trajectory.info.frequency)
    frequency = 1.0 / env.dt
    extended = (not trajectory.data.is_complete or smooth_start_end or
                not math.isclose(source_frequency, frequency, rel_tol=0, abs_tol=1e-8))
    if extended:
        trajectory = env.extend_motion(trajectory, smooth_start_end=smooth_start_end)
    recalculate = env.recalculate_vel_in_reference_motion
    if recalculate:
        for operation in (recalculate_traj_angular_velocity,
                          recalculate_traj_linear_velocity, recalculate_traj_joint_velocity):
            trajectory = operation(trajectory, frequency=frequency, backend=jp)
    metadata = dict(trajectory.info.metadata or {})
    # Keep a prior preparation record nested: resampling/smoothing is never erased.
    prior_preparation = metadata.get(PREPARATION_KEY)
    metadata[PREPARATION_KEY] = {
        "version": 1, "source_path": str(source_path.resolve()),
        "source_reference": source_reference, "source_frequency": source_frequency,
        "frequency": frequency, "extended": extended,
        "smooth_start_end": bool(smooth_start_end),
        "velocity_frequency": frequency if recalculate else None,
        "source_preparation": prior_preparation,
    }
    trajectory = replace(trajectory, info=replace(trajectory.info, metadata=metadata))
    require_prepared(trajectory, control_dt=env.dt, recalculate_velocity=recalculate, path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents np.savez's normal overwrite behavior, including races.
    with output_path.open("xb") as stream:
        np.savez(stream, **trajectory.to_dict())
    return output_path


def preprocess_dataset(env, dataset_dict, batch_idx, num_batches, smooth_start_end,
                       *, source_root, output_root, flat_source=False, source_reference=None):
    """Prepare a deterministic contiguous shard into a separate output tree."""
    if num_batches < 1 or not 0 <= batch_idx < num_batches:
        raise ValueError("Require num_batches >= 1 and 0 <= batch_idx < num_batches.")
    source_root, output_root = Path(source_root), Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("Preparation source and output roots must differ.")
    entries = [(dataset, name) for dataset, names in dataset_dict.items() for name in names]
    size, remainder = divmod(len(entries), num_batches)
    start = batch_idx * size + min(batch_idx, remainder)
    end = start + size + (batch_idx < remainder)
    outputs = []
    for dataset, name in entries[start:end]:
        source = source_root / f"{name}.npz" if flat_source else source_root / dataset / "UnitreeG1" / f"{name}.npz"
        output = output_root / dataset / "UnitreeG1" / f"{name}.npz"
        outputs.append(prepare_motion(env, source, output, smooth_start_end=smooth_start_end,
                                      source_reference=source_reference))
    return outputs


def concatenate_prepared(trajectories, *, backend):
    """Concatenate aligned models while retaining each input's metadata and span."""
    from track_mj.utils.dataset.traj_class import Trajectory, TrajectoryData
    if len(trajectories) == 1:
        return trajectories[0]
    # The native structural equality includes metadata. Source provenance differs
    # by design; compare all model/shape information and keep metadata per source.
    infos = [replace(t.info, metadata=None) for t in trajectories]
    data, info = TrajectoryData.concatenate([t.data for t in trajectories], infos, backend=backend)
    sources, start = [], 0
    for trajectory in trajectories:
        end = start + int(trajectory.data.n_samples)
        sources.append({"sample_start": start, "sample_stop": end,
                        "metadata": trajectory.info.metadata})
        start = end
    return Trajectory(replace(info, metadata={"opentrack_sources": sources}), data)

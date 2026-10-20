"""One-shot MJX-Warp G1 repeatability diagnostic; never promotes a task result.

Each of eight controls runs three independent plain/plain/instrumented triples
from identical copied PRE state. Every call is fully blocked and copied to host
before the next call. The original 1e-6 physics-neutrality gate is reported,
not redefined. Raw diagnostics belong under a new shared output root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import jax
import jax.numpy as jp
from ml_collections import ConfigDict
import mujoco
import numpy as np

from native_warp_pillar_probe import (
    cpu_reference, scene, shallow_touch_center,
)
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
from track_mj.envs.g1_tracking_dagger.utils.wrapper import VmapWrapper
from track_mj.utils.dataset.preparation import load_prepared_trajectory
from track_mj.utils.mjx_backend import MJX_WORLD_AXIS_NAME, configure_mjx, mjx_impl


def host_copy(tree):
    return jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), tree)


def device_copy(host, device):
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(np.array(leaf, copy=True), device), host)


def leaf_map(tree):
    return {jax.tree_util.keystr(path): np.asarray(value)
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]}


def differences(left, right):
    a, b = leaf_map(left), leaf_map(right)
    if a.keys() != b.keys():
        raise ValueError("native State tree changed during repeatability comparison")
    result = {}
    for name in a:
        if a[name].shape != b[name].shape:
            raise ValueError(f"native State leaf changed shape: {name}")
        x, y = a[name].astype(np.float64), b[name].astype(np.float64)
        difference = x - y
        finite = bool(np.isfinite(difference).all())
        result[name] = {
            "shape": list(x.shape), "exact": bool(np.array_equal(a[name], b[name])),
            "max_abs": float(np.max(np.abs(difference))) if difference.size and finite else (0. if not difference.size else None),
            "rms": float(np.sqrt(np.mean(difference ** 2))) if difference.size and finite else (0. if not difference.size else None),
            "nonfinite": not finite,
        }
    return result


def check_independent_buffers(clones):
    flattened = [jax.tree_util.tree_flatten(clone)[0] for clone in clones]
    if not all(len(leaves) == len(flattened[0]) for leaves in flattened):
        raise ValueError("copied starting-state trees differ")
    # Verify every PRE leaf, then reject device-buffer aliasing. The host
    # readback also completes every copy before any physics call starts.
    paths = jax.tree_util.tree_flatten_with_path(clones[0])[0]
    for index, (path, original) in enumerate(paths):
        values = [np.asarray(leaves[index]) for leaves in flattened]
        if not all(np.array_equal(values[0], value) for value in values[1:]):
            raise ValueError(f"copied PRE values differ: {jax.tree_util.keystr(path)}")
        if not hasattr(original, "unsafe_buffer_pointer") or not original.size:
            continue
        pointers = [leaves[index].unsafe_buffer_pointer() for leaves in flattened]
        if len(set(pointers)) != len(pointers):
            raise ValueError(f"PRE device buffers alias across clones: {jax.tree_util.keystr(path)}")


def combine(comparisons):
    names = comparisons[0].keys()
    result = {}
    for name in names:
        items = [comparison[name] for comparison in comparisons]
        total = sum(int(np.prod(item["shape"])) for item in items)
        rms = (sum((item["rms"] ** 2) * int(np.prod(item["shape"])) for item in items) / total) ** .5 \
            if total and all(item["rms"] is not None for item in items) else (0. if total == 0 else None)
        result[name] = {"shape": items[0]["shape"],
                        "all_exact": all(item["exact"] for item in items),
                        "max_abs": max(item["max_abs"] for item in items) if all(item["max_abs"] is not None for item in items) else None,
                        "rms": rms, "any_nonfinite": any(item["nonfinite"] for item in items)}
    return result


def unselected_leaf_diffs(before, after):
    result = {}
    after_leaves = leaf_map(after)
    for name, old in leaf_map(before).items():
        new = after_leaves[name]
        if old.shape != new.shape:
            raise ValueError(f"selective reset changed State leaf shape: {name}")
        old_part = old[0] if old.ndim and old.shape[0] == 2 else old
        new_part = new[0] if new.ndim and new.shape[0] == 2 else new
        result[name] = bool(np.array_equal(old_part, new_part))
    return result


def run(args, output):
    document = json.loads(args.config.read_text())
    config = ConfigDict(document.get("env_config", document))
    config.reference_traj_config.random_start = False
    config.reference_traj_config.fixed_start_frame = 0
    config.noise_config.level = 0.
    config.push_config.enable = False
    config.evaluation_physical_stop = {
        "root_link_height_min_m": .30, "root_up_dot_min": .50,
        "stop_on_nonfinite": True,
    }
    configure_mjx(config, 2)
    config.mjx.impl = "warp"
    config.mjx.num_envs = 2
    config.mjx.warp_graph_mode = args.graph_mode
    config.mjx.warp_nconmax_per_env = 128
    if mjx_impl(config) != "warp" or len(jax.devices(args.device)) != 1:
        raise ValueError("diagnostic requires one visible selected device and MJX-Warp")
    device = jax.devices(args.device)[0]
    spec, pillars, sensors, robot = scene(20, 20)
    with jax.default_device(device):
        env = G1TrackingGeneralEnv(config=config, model_spec=spec,
            obstacle_geom_names=pillars, warp_obstacle_sensor_names=sensors,
            reference_mode="initialization_only")
        if len(robot) != 16 or len(env._warp_pair_sensors) != 32:
            raise ValueError("full current G1 robot–pillar collision contract changed")
        motion = load_prepared_trajectory(args.motion, backend=jp, control_dt=env.dt,
            recalculate_velocity=config.recalculate_vel_in_reference_motion)
        env.th = env.load_trajectory(motion, warn=False)
        trajectory = env.th.traj.data

        initial = mujoco.MjData(env.mj_model)
        initial.qpos[:] = np.asarray(trajectory.qpos[0])
        initial.qvel[:] = np.asarray(trajectory.qvel[0])
        mujoco.mj_forward(env.mj_model, initial)
        foot = np.asarray(initial.geom_xpos[env.mj_model.geom("left_foot").id])
        touch, distance = shallow_touch_center(env.mj_model, initial,
            int(env.mj_model.geom("left_foot").id), int(env.mj_model.geom(pillars[0]).id))
        clear = [float(foot[0] + 3.), float(foot[1] + 3.), .75]
        centers = jp.asarray([[touch, clear], [clear, clear]])
        reset = jax.jit(jax.vmap(
            lambda key, per_world_centers: env.reset(
                key, trajectory, obstacle_centers=per_world_centers),
            in_axes=(0, 0), axis_name=MJX_WORLD_AXIS_NAME))
        vector = VmapWrapper(env)
        plain_step = jax.jit(vector.step_no_autoreset)
        measured_step = jax.jit(jax.vmap(
            env.step_with_pillar_sensors, in_axes=(0, 0, None),
            axis_name=MJX_WORLD_AXIS_NAME))
        selected_reset = jax.jit(vector.reset_done)
        state = reset(jax.random.split(jax.random.PRNGKey(521), 2), centers)
        jax.block_until_ready(state)
        state = state.replace(info={**state.info, "worldid": jp.arange(2, dtype=jp.int32)})
        cpu = cpu_reference(env.mj_model, np.asarray(state.data.qpos),
            np.asarray(state.data.qvel), np.asarray(centers), env._warp_pair_sensors)
        if cpu[0]["pillar_pair_found_sum"] <= 0 or cpu[1]["pillar_pair_found_sum"] != 0:
            raise ValueError("native CPU touch/clear geometry changed")
        output.update({"code_revision": os.environ.get("COMPUTE_CODE_REVISION"),
            "image_environment": "OpenTrack pinned MJX-Warp", "status": "running",
            "interpretation": "repeatability diagnostic, not paper task or capability admission",
            "frozen_prior_gate_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "controls": args.controls, "repetitions_per_control": 3,
            "calls_per_repetition": ["plain_a", "plain_b", "instrumented"],
            "substeps_per_control": 10, "physics_dt_s": .002,
            "device_platform": args.device, "graph_mode": args.graph_mode,
            "solver_iterations": 20, "solver_linesearch_iterations": 20,
            "centers_xyz_m": np.asarray(centers).tolist(),
            "native_cpu_initial_pair_distance_m": distance,
            "native_cpu_reset_contacts": cpu, "pair_sensor_count": len(sensors),
            "force_semantics": "maximum individual matching contact-frame normal component in N, not summed body or pillar force",
            "control_results": []})
        action = jp.zeros((2, env.action_size), dtype=state.data.qpos.dtype)

        for control in range(args.controls):
            pre = host_copy(state)
            triples = []
            plain_plain = []
            plain_instr = []
            plain_cross = []
            sensed_readouts = []
            for repetition in range(3):
                clones = [device_copy(pre, device) for _ in range(3)]
                check_independent_buffers(clones)
                outputs = []
                for kind, clone in zip(("plain_a", "plain_b", "instrumented"), clones):
                    if kind == "instrumented":
                        next_state, samples = measured_step(clone, action, trajectory)
                        jax.block_until_ready((next_state, samples))
                        sensed = host_copy(samples)
                        sensed_readouts.append({
                            "repetition": repetition,
                            "overflow_codes": np.asarray(sensed.overflow_code).tolist(),
                            "sensor_invalid_any": bool(np.any(sensed.pair_sensor_invalid)),
                            "invalid_any": bool(np.any(sensed.invalid)),
                            "pillar_contact_substeps":
                                (np.asarray(sensed.pair_found_count) > 0).any(axis=2).sum(axis=1).tolist(),
                            "max_individual_contact_normal_n":
                                np.asarray(sensed.pair_max_normal_force_n).max(axis=(1, 2)).tolist(),
                            "pre_time_s": np.asarray(sensed.interval_start_s)[:, 0].tolist(),
                            "post_time_s": np.asarray(sensed.interval_end_s)[:, -1].tolist(),
                        })
                    else:
                        next_state = plain_step(clone, action, trajectory)
                        jax.block_until_ready(next_state)
                    outputs.append(host_copy(next_state))
                plain_plain.append(differences(outputs[0], outputs[1]))
                plain_instr.append(differences(outputs[0], outputs[2]))
                triples.append(outputs)
            for repetition in (1, 2):
                plain_cross.append(differences(triples[0][0], triples[repetition][0]))
            baseline = combine(plain_plain + plain_cross)
            instrumented = combine(plain_instr)
            per_leaf = {name: {
                "plain_plain_max_abs": baseline[name]["max_abs"],
                "plain_plain_rms": baseline[name]["rms"],
                "plain_instrumented_max_abs": instrumented[name]["max_abs"],
                "plain_instrumented_rms": instrumented[name]["rms"],
                "instrumented_exceeds_plain_max":
                    (instrumented[name]["max_abs"] > baseline[name]["max_abs"]
                     if instrumented[name]["max_abs"] is not None and baseline[name]["max_abs"] is not None else None),
                "all_plain_exact": baseline[name]["all_exact"],
                "all_instrumented_exact": instrumented[name]["all_exact"],
                "nonfinite": baseline[name]["any_nonfinite"] or instrumented[name]["any_nonfinite"],
                "shape": baseline[name]["shape"],
            } for name in baseline}
            record = {"control": control,
                "pre_time_s": np.asarray(pre.data.time).tolist(),
                "post_time_s": np.asarray(triples[0][0].data.time).tolist(),
                "terminal_before_reset": np.asarray(triples[0][0].done).tolist(),
                "plain_final_overflow": [np.asarray(triple[0].data._impl.overflow).tolist() for triple in triples],
                "instrumented_final_overflow": [np.asarray(triple[2].data._impl.overflow).tolist() for triple in triples],
                "sensor_readouts": sensed_readouts,
                "leaf_comparison": per_leaf}
            state = device_copy(triples[0][0], device)
            if control == args.controls // 2 - 1:
                before = host_copy(state)
                forced = state.replace(done=jp.asarray([0., 1.], dtype=state.done.dtype))
                after = selected_reset(forced, trajectory)
                jax.block_until_ready(after)
                after_host = host_copy(after)
                unchanged = unselected_leaf_diffs(before, after_host)
                record["selected_reset"] = {
                    "forced_done": [False, True],
                    "unselected_all_leaves_exact": all(unchanged.values()),
                    "unselected_changed_leaves": [name for name, same in unchanged.items() if not same],
                    "selected_post_time_s": float(np.asarray(after_host.data.time)[1]),
                    "unselected_post_time_s": float(np.asarray(after_host.data.time)[0]),
                }
                state = after
            output["control_results"].append(record)
            (args.out / "progress.json").write_text(json.dumps(output, indent=2) + "\n")
        output["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--graph-mode", choices=("WARP", "JAX"), default="WARP")
    parser.add_argument("--controls", type=int, default=8, choices=range(2, 9))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    output = {"status": "starting"}
    try:
        run(args, output)
    except Exception as exc:
        output["status"] = "failed"
        output["exception"] = repr(exc)
        output["traceback"] = traceback.format_exc()
        (args.out / "result.json").write_text(json.dumps(output, indent=2) + "\n")
        raise
    (args.out / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "controls": len(output["control_results"])}))


if __name__ == "__main__":
    main()

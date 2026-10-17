"""Manual, bounded native JAX positive-contact comparison; excluded from test discovery.

Run only as a compute-service job. No policy training, reference replay, native
forward refresh, tolerance verdict, or full-clip qualification is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import traceback

import numpy as np


@dataclass
class Capture:
    arrays: dict
    avals: dict
    structure: object
    physical: dict


def capture(value, jax, *, state=False):
    pairs, structure = jax.tree_util.tree_flatten_with_path(value)
    arrays, avals = {}, {}
    for path, leaf in pairs:
        name = jax.tree_util.keystr(path)
        arrays[name] = np.array(jax.device_get(leaf), copy=True)
        aval = jax.core.get_aval(leaf)
        avals[name] = {"shape": list(aval.shape), "dtype": str(aval.dtype),
                       "weak_type": bool(getattr(aval, "weak_type", False))}
    physical = {}
    if state:
        qpos, qvel = (np.array(jax.device_get(v), copy=True) for v in (value.data.qpos, value.data.qvel))
        physical = {"root_position_m": qpos[..., :3], "root_quaternion_wxyz": qpos[..., 3:7],
                    "joint_position_rad": qpos[..., 7:], "root_linear_velocity_mps": qvel[..., :3],
                    "root_angular_velocity_radps": qvel[..., 3:6], "joint_velocity_radps": qvel[..., 6:]}
        physical.update({name: np.array(jax.device_get(v), copy=True) for name, v in (
            ("actuator_force_nm", value.data.actuator_force), ("qfrc_actuator", value.data.qfrc_actuator),
            ("ctrl", value.data.ctrl), ("rng", value.info["rng"]),
            ("last_motor_targets_rad", value.info["last_motor_targets"]), ("done", value.done))})
    return Capture(arrays, avals, structure, physical)


def array_difference(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "shapes": [list(a.shape), list(b.shape)],
                "dtypes": [str(a.dtype), str(b.dtype)]}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"compatible": True, "exact_equal": bool(np.array_equal(a, b)),
            "changed": int(np.count_nonzero(a != b)),
            "nonfinite": int(np.count_nonzero(~np.isfinite(delta))),
            "max_abs": float(np.max(delta)) if delta.size and np.isfinite(delta).all() else (0. if not delta.size else None)}


def nonfinite_counts(arrays):
    return {name: int(np.count_nonzero(~np.isfinite(value))) for name, value in arrays.items()
            if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all()}


def compare(left, right):
    same = left.structure == right.structure and left.avals == right.avals
    differences = {}
    for name in left.arrays.keys() | right.arrays.keys():
        difference = (array_difference(left.arrays[name], right.arrays[name])
                      if name in left.arrays and name in right.arrays else {"missing": True})
        if not difference.get("exact_equal", False):
            differences[name] = difference
    return {"same_tree_and_avals": bool(same), "different_leaves": differences,
            "nonfinite_left": nonfinite_counts(left.arrays), "nonfinite_right": nonfinite_counts(right.arrays),
            "physical": {name: array_difference(left.physical[name], right.physical[name])
                         for name in left.physical.keys() & right.physical.keys()}}


def comparison_report(plain, instrumented):
    """Retain every pair and descriptive envelopes, including native repeat variation."""
    baseline = [compare(a, b) for i, a in enumerate(plain) for b in plain[i + 1:]]
    measured = [compare(a, b) for i, a in enumerate(instrumented) for b in instrumented[i + 1:]]
    cross = [compare(a, b) for a in plain for b in instrumented]
    envelope = {}
    for name in plain[0].physical:
        native = [entry["physical"][name].get("max_abs") for entry in baseline]
        extra = [entry["physical"][name].get("max_abs") for entry in cross]
        if all(value is not None for value in native + extra):
            base, instrument = max(native), max(extra)
            envelope[name] = {"plain_repeat_max_abs": base, "cross_mode_max_abs": instrument,
                              "cross_minus_plain_envelope": instrument - base}
    return {"plain_pairs": baseline, "instrumented_pairs": measured, "cross_mode_pairs": cross,
            "descriptive_envelopes": envelope}


def contact_summary(samples, floor_geom_id):
    present = np.asarray(samples["constraint_present"], bool)
    normal = np.asarray(samples["normal_n"])
    total = np.asarray(samples["total_force_world_n"])
    geom = np.asarray(samples["geom"])
    positive = present & (normal > 0.)
    ground = positive & np.any(geom == floor_geom_id, axis=-1)
    for name, values in samples.items():
        if np.asarray(values).dtype.kind in "fc" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite native contact samples: {name}")
    return {"positive_normal_slots": int(positive.sum()), "positive_ground_normal_slots": int(ground.sum()),
            "positive_substeps": int(np.any(positive, axis=-1).sum()),
            "substeps": int(np.asarray(samples["interval_start_s"]).size),
            "maximum_normal_n": float(np.where(present, normal, 0.).max(initial=0)),
            "maximum_total_norm_n": float(np.where(present, np.linalg.norm(total, axis=-1), 0.).max(initial=0)),
            "positive_indices": np.argwhere(positive).tolist(),
            "interval_start_s": np.asarray(samples["interval_start_s"]).tolist(),
            "interval_end_s": np.asarray(samples["interval_end_s"]).tolist()}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save_capture(root, name, value):
    with (root / (name + ".npz")).open("xb") as stream:
        np.savez(stream, **value.arrays)
    write_json(root / (name + ".json"), {"avals": value.avals, "tree_description": str(value.structure),
                                                       "nonfinite_leaves": nonfinite_counts(value.arrays)})


def run(args, report):
    # Actual native imports, deferred so --help and reporting unit tests construct no physics.
    import jax
    import jax.numpy as jp
    import mujoco
    from ml_collections import ConfigDict
    import yaml
    from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
    from track_mj.envs.g1_tracking_dagger.utils.wrapper import VmapWrapper
    from track_mj.utils.dataset.preparation import load_prepared_trajectory
    from track_mj.utils.mjx_backend import MJX_WORLD_AXIS_NAME, configure_mjx, mjx_impl

    document = yaml.safe_load(args.config.read_text())
    config = ConfigDict(document.get("env_config", document))
    config.reference_traj_config.random_start = False
    config.reference_traj_config.fixed_start_frame = args.start_frame
    config.push_config.enable = False
    config.mjx = ConfigDict({"impl": "jax", "num_envs": 1})
    configure_mjx(config, 1)
    if mjx_impl(config) != "jax":
        raise ValueError("diagnostic requires native JAX; remove conflicting backend environment overrides")
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise ValueError("diagnostic requires exactly one allocated visible GPU")
    with jax.default_device(devices[0]):
        env = G1TrackingGeneralEnv(terrain_type=config.terrain_type, config=config)
        motion = load_prepared_trajectory(args.motion, backend=jp, control_dt=env.dt,
                                          recalculate_velocity=config.recalculate_vel_in_reference_motion)
        env.th = env.load_trajectory(motion, warn=False)
        trajectory = env.th.traj.data
        vector = VmapWrapper(env)
        plain_step = jax.jit(vector.step_no_autoreset)
        contact_step = jax.jit(jax.vmap(env.step_with_contacts, in_axes=(0, 0, None), axis_name=MJX_WORLD_AXIS_NAME))
        state = jax.jit(vector.reset)(jax.random.split(jax.random.PRNGKey(args.seed), 1), trajectory)
        jax.block_until_ready(state)
        action = (jp.zeros((1, env.action_size), dtype=state.data.qpos.dtype) if args.action is None
                  else jp.asarray(np.load(args.action, allow_pickle=False)))
        if action.shape != (1, env.action_size) or action.dtype.kind != "f" or not np.isfinite(np.asarray(action)).all():
            raise ValueError("fixed action must be finite [1, native action_size]")
        write_json(args.out / "resolved-config.json", config.to_dict())
        report.update(runtime={"jax": jax.__version__, "mujoco": mujoco.__version__, "device": str(devices[0]),
                               "floor_geom_id": int(env._floor_geom_id),
                               "control_dt": float(env.dt), "physics_dt": float(env.sim_dt)},
                      action="zero fixed action" if args.action is None else str(args.action),
                      warmup=[], contact_repeats=[])
        for name, value in (("model", env.mjx_model), ("trajectory", trajectory), ("fixed-action", action)):
            save_capture(args.out, name, capture(value, jax))

        def clone(value):
            # Own every numerical leaf of the actual complete native State, preserving its static tree.
            return jax.tree_util.tree_map(lambda leaf: jp.array(leaf, copy=True), value)

        def advance(label, function, value):
            entry = {"kind": label, "completed": False}
            report["physical_calls"].append(entry)
            result = function(value, action, trajectory)
            jax.block_until_ready(result)
            entry["completed"] = True
            return result

        fixed = None
        for control in range(1, args.warmup_controls + 1):
            state = advance("warmup_plain", plain_step, clone(state))
            jax.block_until_ready(state)
            if bool(np.any(jax.device_get(state.done))):
                report["status"] = "warmup_terminated_before_comparison"
                return
            candidate = clone(state)
            probe_state, samples = advance("warmup_contact_probe", contact_step, candidate)
            jax.block_until_ready((probe_state, samples))
            measured = {name: np.array(jax.device_get(value), copy=True) for name, value in samples._asdict().items()}
            reading = contact_summary(measured, env._floor_geom_id)
            report["warmup"].append({"plain_controls": control, "probe": reading})
            print(f"warmup {control}: positive normal slots {reading['positive_normal_slots']}", flush=True)
            if reading["positive_normal_slots"]:
                fixed = clone(state)
                break
        if fixed is None:
            report["status"] = "no_positive_contact_within_budget"
            return
        fixed_capture = capture(fixed, jax, state=True)
        save_capture(args.out, "fixed-input", fixed_capture)
        by_mode = {mode: [[] for _ in range(args.continuation_controls + 1)] for mode in ("plain", "instrumented")}
        report["input_checks"] = []
        for mode in by_mode:
            for repeat in range(3):
                owned = clone(fixed)
                before = capture(owned, jax, state=True)
                if compare(fixed_capture, before)["different_leaves"] or not compare(fixed_capture, before)["same_tree_and_avals"]:
                    raise ValueError("owned native input differs from the fixed complete input")
                if mode == "plain":
                    output = advance("comparison_plain", plain_step, owned)
                else:
                    output, samples = advance("comparison_instrumented", contact_step, owned)
                    jax.block_until_ready((output, samples))
                    measured = {name: np.array(jax.device_get(value), copy=True) for name, value in samples._asdict().items()}
                    with (args.out / f"contacts-{repeat}.npz").open("xb") as stream:
                        np.savez(stream, **measured)
                    report["contact_repeats"].append(contact_summary(measured, env._floor_geom_id))
                jax.block_until_ready(output)
                immediate = capture(output, jax, state=True)
                save_capture(args.out, f"{mode}-{repeat}-step-0", immediate)
                by_mode[mode][0].append(immediate)
                unchanged = compare(before, capture(owned, jax, state=True))
                report["input_checks"].append(unchanged)
                if unchanged["different_leaves"] or not unchanged["same_tree_and_avals"]:
                    raise ValueError("native call mutated its owned input")
                # Capture immediately before another call. Even a terminal state is retained;
                # these bounded no-reset continuations are diagnostics, not task completions.
                for step in range(1, args.continuation_controls + 1):
                    output = advance("continuation_plain", plain_step, clone(output))
                    jax.block_until_ready(output)
                    saved = capture(output, jax, state=True)
                    save_capture(args.out, f"{mode}-{repeat}-step-{step}", saved)
                    by_mode[mode][step].append(saved)
                print(f"captured {mode} repeat {repeat}", flush=True)
        report["comparisons_by_next_plain_control"] = [comparison_report(a, b) for a, b in zip(by_mode["plain"], by_mode["instrumented"])]
        report["status"] = ("descriptive_positive_contact_comparison" if all(row["positive_normal_slots"] for row in report["contact_repeats"])
                            else "paired_repeats_did_not_all_show_positive_contact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True, help="immutable prepared native trajectory")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--motion-reference", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--warmup-controls", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--continuation-controls", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 600:
        parser.error("wall-seconds must be between 1 and 600")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"physical_calls": [], "source_reference": args.source_reference, "motion_reference": args.motion_reference,
              "config": str(args.config), "motion": str(args.motion), "started_at": datetime.now(timezone.utc).isoformat(),
              "budget": {"warmup_controls": args.warmup_controls, "repeats_per_mode": 3,
                         "next_plain_controls": args.continuation_controls, "wall_seconds": args.wall_seconds,
                         "maximum_native_controls": 2 * args.warmup_controls + 6 * (1 + args.continuation_controls)},
              "scope": "Full owned native State; fixed model, trajectory, action and static tree. All deltas are descriptive; no tolerance pass, numerical-neutrality, or full-clip claim."}
    def expired(signum, frame):
        raise TimeoutError("manual contact diagnostic wall budget exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        run(args, report)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()

"""Manual native JAX reset diagnostic; numerical differences are descriptive.

Select this checkout through PYTHONPATH=. in the chosen native environment,
with GLI_PATH selecting its assets:
  python tests/native_reset_continuation.py --config /inputs/config.json \
    --motion /inputs/prepared.npz --out /new/output --steps 5
Optional --actions is a float32 .npy [steps,2,29] in native action order; otherwise
all prescribed actions are zero. Every repeat uses the same sequence. Each reset
arm is evaluated once, then its owned output starts each repeated sequence.
Sequences stop at the first native done, retaining that terminal sample. No policy,
training, automatic continuation reset, tolerance or equivalence claim is included.
The proposed reset arm requires the draft reset_done API, not the safe terminal API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def leaf_differences(left, right):
    """Describe every changed leaf, without ignoring caches or nonfinite values."""
    if left.keys() != right.keys():
        raise ValueError("captured leaf paths differ")
    result = {}
    for name, a in left.items():
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"captured shape/dtype differs: {name}")
        if not np.array_equal(a, b):
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            finite = delta[np.isfinite(delta)]
            result[name] = {"changed": int(np.count_nonzero(a != b)),
                            "max_abs": float(finite.max()) if finite.size else None,
                            "nonfinite_differences": int((~np.isfinite(delta)).sum())}
    return result


def _number(value):
    return float(value) if np.isfinite(value) else str(float(value))


def physical_differences(left, right, joint_names, worlds):
    """Each physical coordinate keeps its unit; native angular qvel keeps its frame."""
    labels = [("data/qpos", [*(f"root/{a}" for a in "xyz"),
               *(f"root/quaternion/{a}" for a in "wxyz"), *joint_names], ["m"] * 3 + ["1"] * 4 + ["rad"] * len(joint_names)),
              ("data/qvel", [*(f"root/linear/{a}" for a in "xyz"),
               *(f"root/angular_native/{a}" for a in "xyz"), *joint_names], ["m/s"] * 3 + ["rad/s"] * (3 + len(joint_names)))]
    rows = []
    for world in worlds:
        for field, names, units in labels:
            a, b = left[field][world], right[field][world]
            if a.shape != (len(names),) or b.shape != a.shape:
                raise ValueError(f"unsupported physical coordinate shape: {field}")
            for name, unit, x, y in zip(names, units, a, b):
                rows.append({"world": world, "field": field, "coordinate": name, "unit": unit,
                             "left": _number(x), "right": _number(y), "abs_difference": _number(abs(x - y))})
        a, b = left["data/qpos"][world, 3:7], right["data/qpos"][world, 3:7]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        angle = None if not norm else float(2 * np.arccos(np.clip(abs(np.dot(a, b)) / norm, 0, 1)))
        rows.append({"world": world, "coordinate": "root/orientation_angle", "unit": "rad", "difference": None if angle is None else _number(angle)})
    return rows


def run(args):
    # Imports are lazy: numeric reporting tests need no JAX, MuJoCo or native env.
    import jax
    import jax.numpy as jp
    from ml_collections import ConfigDict
    from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
    from track_mj.envs.g1_tracking_dagger.utils.wrapper import VmapWrapper
    from track_mj.utils.dataset.preparation import load_prepared_trajectory
    from track_mj.utils.mjx_backend import configure_mjx, MJX_WORLD_AXIS_NAME

    cfg = ConfigDict(json.loads(args.config.read_text())["env_config"])
    cfg.mjx = ConfigDict({"impl": "jax", "num_envs": 2})
    configure_mjx(cfg, 2)
    if cfg.mjx.impl != "jax" or cfg.mjx.num_envs != 2:
        raise ValueError("this diagnostic uses native JAX with two worlds")
    device = jax.devices("gpu")[0]
    with jax.default_device(device):
        env = G1TrackingGeneralEnv(terrain_type=cfg.terrain_type, config=cfg)
        trajectory = load_prepared_trajectory(args.motion, backend=jp, control_dt=env.dt,
                          recalculate_velocity=cfg.recalculate_vel_in_reference_motion)
        env.th = env.load_trajectory(trajectory, warn=False)
        trajectory, vector = env.th.traj.data, VmapWrapper(env)
        actions = np.zeros((args.steps, 2, env.action_size), np.float32) if args.actions is None else np.load(args.actions, allow_pickle=False)
        if actions.shape != (args.steps, 2, env.action_size) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("actions must be finite float32 [steps,2,action_size]")
        actions = jp.array(actions)
        owned = lambda value: jax.tree_util.tree_map(lambda leaf: jp.array(leaf, copy=True), value)
        step_jit = jax.jit(vector.step_no_autoreset)
        state = jax.jit(vector.reset)(jax.random.split(jax.random.PRNGKey(args.seed), 2), trajectory)
        # The first native step adds truncation. Freeze the executable afterwards.
        state = owned(step_jit(state, actions[0], trajectory))
        jax.block_until_ready(state)
        if np.asarray(jax.device_get(state.done)).any():
            raise ValueError("schema warmup terminated; choose a valid diagnostic start")
        executable = step_jit.lower(state, actions[0], trajectory).compile()

        def native_reset(current, motion):
            def world_reset(world):
                return jax.lax.cond(world.done.astype(bool),
                    lambda value: env._reset_and_update_state(value, motion), lambda value: value, world)
            return jax.vmap(world_reset, axis_name=MJX_WORLD_AXIS_NAME)(current)
        resets = {"native": jax.jit(native_reset).lower(owned(state), trajectory).compile(),
                  "proposed": jax.jit(vector.reset_done).lower(owned(state), trajectory).compile()}

        def capture(name, value):
            jax.block_until_ready(value)
            leaves, tree = jax.tree_util.tree_flatten_with_path(value)
            path_name = lambda path: "/".join(str(getattr(k, "name", getattr(k, "key", getattr(k, "idx", k)))) for k in path)
            arrays = {path_name(path): np.array(jax.device_get(leaf), copy=True) for path, leaf in leaves}
            signature = (tree, [(a.shape, str(a.dtype), bool(getattr(a, "weak_type", False))) for _, a in leaves])
            with (args.out / (name + ".npz")).open("xb") as stream:
                np.savez(stream, **arrays)
            return arrays, signature

        base, signature = capture("fixed-state", state)
        fixed_model, _ = capture("fixed-model", env.mjx_model)
        fixed_trajectory, _ = capture("fixed-trajectory", trajectory)
        capture("prescribed-actions", actions)
        joint_names = [env.mj_model.joint(i).name for i in range(1, env.mj_model.njnt)]
        report = {"config": cfg.to_dict(), "seed": args.seed, "steps": args.steps, "repeats": args.repeats,
                  "control_dt": float(env.dt), "physics_dt": float(env.sim_dt), "jax_version": jax.__version__,
                  "joint_names": joint_names, "action_joint_names": list(env.action_joint_names),
                  "interpretation": "Descriptive repeated native execution; no tolerance or continuation qualification.",
                  "sequences": {}, "boundaries": {}, "comparisons": []}
        groups = {}
        def sequence(name, start):
            groups[name], report["sequences"][name] = [], []
            for repeat in range(args.repeats):
                input_state = current = owned(start)
                before, current_signature = capture(f"{name}-r{repeat}-input", current)
                if current_signature != signature:
                    raise ValueError("input signature differs from the fixed compiled callable")
                outputs, done = [], []
                for step in range(args.steps):
                    current = executable(current, actions[step], trajectory)
                    arrays, _ = capture(f"{name}-r{repeat}-s{step + 1}", current)
                    outputs.append(arrays)
                    done.append(np.asarray(jax.device_get(current.done), dtype=bool).tolist())
                    if any(done[-1]):
                        break
                after, _ = capture(f"{name}-r{repeat}-input-after", input_state)
                report["sequences"][name].append({"done": done, "completed_steps": len(outputs),
                                                  "input_leaf_differences": leaf_differences(before, after)})
                groups[name].append(outputs)

        sequence("baseline-before", state)
        boundaries = {}
        for mask_name, mask in (("zero", [0, 0]), ("peer", [0, 1]), ("all", [1, 1])):
            for reset_name, reset in resets.items():
                name = reset_name + "-" + mask_name
                current = owned(state).replace(done=jp.array(mask, dtype=state.done.dtype))
                before, _ = capture(name + "-before-reset", current)
                following = reset(current, trajectory)
                after, _ = capture(name + "-after-reset", following)
                input_after, _ = capture(name + "-reset-input-after", current)
                boundaries[name] = after
                report["boundaries"][name] = {"reset_input_leaf_differences": leaf_differences(before, input_after), "mask": mask, "all_leaf_differences": leaf_differences(before, after),
                    "ongoing_physical": physical_differences(before, after, joint_names, [w for w in range(2) if not mask[w]])}
                sequence(name, following)
        sequence("baseline-after", state)

        def compare(left_name, right_name, worlds):
            for i, left in enumerate(groups[left_name]):
                for j, right in enumerate(groups[right_name]):
                    if left_name == right_name and i >= j:
                        continue
                    for step, (a, b) in enumerate(zip(left, right), 1):
                        report["comparisons"].append({"left": left_name, "right": right_name, "repeats": [i, j],
                            "step": step, "left_steps": len(left), "right_steps": len(right), "physical_worlds": worlds, "all_leaf_differences_all_worlds": leaf_differences(a, b),
                            "physical": physical_differences(a, b, joint_names, worlds)})
        for name in groups:
            compare(name, name, [0, 1])
        compare("baseline-before", "baseline-after", [0, 1])
        for mask, worlds in (("zero", [0, 1]), ("peer", [0]), ("all", [])):
            native, proposed = "native-" + mask, "proposed-" + mask
            report["boundaries"][mask + "-native-vs-proposed"] = {
                "all_leaf_differences": leaf_differences(boundaries[native], boundaries[proposed]),
                "physical": physical_differences(boundaries[native], boundaries[proposed], joint_names, [0, 1])}
            compare(native, proposed, [0, 1])
            if worlds:
                compare("baseline-before", native, worlds)
                compare("baseline-before", proposed, worlds)
        final, _ = capture("fixed-state-final", state)
        report["fixed_state_leaf_differences"] = leaf_differences(base, final)
        report["fixed_model_leaf_differences"] = leaf_differences(fixed_model, capture("fixed-model-final", env.mjx_model)[0])
        report["fixed_trajectory_leaf_differences"] = leaf_differences(fixed_trajectory, capture("fixed-trajectory-final", trajectory)[0])
        with (args.out / "comparison.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "motion", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 2:
        parser.error("steps must be positive and repeats at least two")
    root = Path(__file__).resolve().parents[1]
    args.out = args.out.resolve()
    if args.out.is_relative_to(root):
        parser.error("raw diagnostic captures must be outside the source repository")
    args.out.mkdir(parents=True, exist_ok=False)
    run(args)  # Actual execution errors propagate; exact numerical differences do not change exit status.


if __name__ == "__main__":
    main()

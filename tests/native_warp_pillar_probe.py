"""Bounded, opt-in two-world G1 MJX-Warp pillar feasibility probe.

Run in a pinned OpenTrack environment with one GPU. This script is not a
paper-task evaluator: it checks exact geometry, native solved contacts,
per-world placement/reset and unchanged physics when sensing is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from ml_collections import ConfigDict
import numpy as np

from track_mj.envs.g1_tracking_dagger import g1_tracking_constants as consts
from track_mj.envs.g1_tracking_dagger.train.g1_env_tracking_general import G1TrackingGeneralEnv
from track_mj.envs.g1_tracking_dagger.utils.wrapper import VmapWrapper
from track_mj.utils.dataset.preparation import load_prepared_trajectory
from track_mj.utils.mjx_backend import MJX_WORLD_AXIS_NAME, configure_mjx, mjx_impl
from track_mj.utils.warp_obstacles import (
    add_mocap_pillar, add_pillar_pairs_and_sensors,
    physical_robot_geom_names,
)


def scene(iterations: int, ls_iterations: int):
    spec = mujoco.MjSpec.from_file(consts.task_to_xml("flat_terrain").as_posix())
    # Record any deliberate solver budget increase. Native overflow remains
    # fail closed; it is never hidden by disabling warnings.
    spec.option.iterations = iterations
    spec.option.ls_iterations = ls_iterations
    pillars = ("task_obstacle_00", "task_obstacle_01")
    for name, y in zip(pillars, (-.72, .72)):
        add_mocap_pillar(spec, name=name, center_xyz=(.75, y, .75),
                         radius_m=.15, height_m=1.5)
    draft = spec.compile()
    robot = physical_robot_geom_names(draft, int(draft.body("pelvis").id))
    sensors = add_pillar_pairs_and_sensors(spec, robot_geom_names=robot,
                                            pillar_geom_names=pillars)
    return spec, pillars, sensors, robot


def cpu_reference(model, qpos, qvel, centers, sensor_layout):
    readings = []
    for world in range(2):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[world]
        data.qvel[:] = qvel[world]
        for index, center in enumerate(centers[world]):
            data.mocap_pos[index] = center
        mujoco.mj_forward(model, data)
        pair_found = [float(data.sensordata[pair.address]) for pair in sensor_layout]
        readings.append({"contact_count": int(data.ncon),
                         "pillar_pair_found_max": max(pair_found),
                         "pillar_pair_found_sum": sum(pair_found)})
    return readings


def shallow_touch_center(model, initial, foot_geom_id, pillar_geom_id):
    """Use MuJoCo's true box–cylinder distance to pick a 2 mm overlap."""
    foot = np.asarray(initial.geom_xpos[foot_geom_id])
    forward = np.asarray(initial.geom_xmat[foot_geom_id]).reshape(3, 3)[:2, 0]
    forward /= np.linalg.norm(forward)
    probe = mujoco.MjData(model)
    probe.qpos[:] = initial.qpos
    probe.qvel[:] = initial.qvel
    probe.mocap_pos[1] = [foot[0] + 3., foot[1] + 3., .75]
    best = None
    for offset in np.linspace(.15, .40, 101):
        center = [float(foot[0] + forward[0] * offset),
                  float(foot[1] + forward[1] * offset), .75]
        probe.mocap_pos[0] = center
        mujoco.mj_forward(model, probe)
        distances = [float(contact.dist) for contact in probe.contact
                     if {int(contact.geom1), int(contact.geom2)} ==
                     {foot_geom_id, pillar_geom_id}]
        if distances and min(distances) <= 0:
            candidate = (abs(min(distances) + .002), center, min(distances))
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None or best[0] > .01:
        raise ValueError("native CPU box–cylinder geometry cannot select a shallow contact")
    return best[1], best[2]


def run(motion_path: Path, config_path: Path, controls: int = 8,
        graph_mode: str = "WARP", iterations: int = 20,
        ls_iterations: int = 20):
    document = json.loads(config_path.read_text())
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
    config.mjx.warp_graph_mode = graph_mode
    config.mjx.warp_nconmax_per_env = 128
    if mjx_impl(config) != "warp":
        raise ValueError("environment override does not select native MJX-Warp")
    spec, pillars, sensors, robot = scene(iterations, ls_iterations)
    env = G1TrackingGeneralEnv(config=config, model_spec=spec,
        obstacle_geom_names=pillars, warp_obstacle_sensor_names=sensors,
        reference_mode="initialization_only")
    if len(robot) != 16 or len(env._warp_pair_sensors) != 32:
        raise ValueError("the full current G1 robot–pillar collision contract changed")
    motion = load_prepared_trajectory(motion_path, backend=jp, control_dt=env.dt,
                                      recalculate_velocity=config.recalculate_vel_in_reference_motion)
    env.th = env.load_trajectory(motion, warn=False)
    trajectory = env.th.traj.data

    initial = mujoco.MjData(env.mj_model)
    initial.qpos[:] = np.asarray(trajectory.qpos[0])
    initial.qvel[:] = np.asarray(trajectory.qvel[0])
    mujoco.mj_forward(env.mj_model, initial)
    foot = np.asarray(initial.geom_xpos[env.mj_model.geom("left_foot").id])
    touch, cpu_initial_pair_distance = shallow_touch_center(
        env.mj_model, initial, int(env.mj_model.geom("left_foot").id),
        int(env.mj_model.geom(pillars[0]).id))
    clear = [float(foot[0] + 3.), float(foot[1] + 3.), .75]
    centers = jp.asarray([[touch, clear], [clear, clear]])
    keys = jax.random.split(jax.random.PRNGKey(521), 2)

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
    state = reset(keys, centers)
    state = state.replace(info={**state.info, "worldid": jp.arange(2, dtype=jp.int32)})
    jax.block_until_ready(state)
    np.testing.assert_allclose(state.data.mocap_pos[:, :2], centers, atol=1e-6)
    cpu = cpu_reference(env.mj_model, np.asarray(state.data.qpos),
        np.asarray(state.data.qvel), np.asarray(centers), env._warp_pair_sensors)
    if cpu[0]["pillar_pair_found_sum"] <= 0 or cpu[1]["pillar_pair_found_sum"] != 0:
        raise AssertionError("independent native CPU scene does not have touching/clear worlds")

    result = {"backend": "native_mjx_warp", "graph_mode": graph_mode,
              "controls": controls, "substeps_per_control": 10,
              "sim_dt_s": float(config.sim_dt), "world_count": 2,
              "solver_iterations": int(env.mj_model.opt.iterations),
              "solver_linesearch_iterations": int(env.mj_model.opt.ls_iterations),
              "obstacle_geom_type": "true_cylinder", "robot_geom_count": len(robot),
              "pair_sensor_count": len(sensors), "centers_xyz_m": np.asarray(centers).tolist(),
              "native_cpu_initial_pair_distance_m": cpu_initial_pair_distance,
              "cpu_reference_at_reset": cpu, "steps": []}
    action = jp.zeros((2, env.action_size), dtype=state.data.qpos.dtype)
    for control in range(controls):
        pre_time = np.asarray(state.data.time).copy()
        plain = plain_step(state, action, trajectory)
        measured, substeps = measured_step(state, action, trajectory)
        jax.block_until_ready((plain, measured, substeps))
        for field in ("qpos", "qvel", "time", "mocap_pos"):
            np.testing.assert_allclose(np.asarray(getattr(plain.data, field)),
                                       np.asarray(getattr(measured.data, field)),
                                       rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(plain.info["rng"]),
                                      np.asarray(measured.info["rng"]))
        np.testing.assert_allclose(np.asarray(plain.done), np.asarray(measured.done))
        if np.any(np.asarray(substeps.invalid)):
            raise AssertionError("Warp contact or constraint capacity/slot invalidity: "
                                 f"sample={np.asarray(substeps.invalid).tolist()}, "
                                 f"overflow_codes={np.asarray(substeps.overflow_code).tolist()}, "
                                 f"sensor_invalid_any={bool(np.any(np.asarray(substeps.pair_sensor_invalid)))}, "
                                 f"post_nacon={np.asarray(measured.data._impl.nacon).tolist()}, "
                                 f"post_nefc={np.asarray(measured.data._impl.nefc).tolist()}, "
                                 f"naconmax={config.mjx.warp_naconmax or (2 * config.mjx.warp_nconmax_per_env)}, "
                                 f"njmax={config.mjx.warp_njmax}")
        np.testing.assert_allclose(np.asarray(substeps.interval_start_s[:, 0]),
                                   pre_time, atol=1e-6)
        np.testing.assert_allclose(np.asarray(substeps.interval_end_s[:, -1]),
                                   np.asarray(measured.data.time), atol=1e-6)
        np.testing.assert_allclose(np.asarray(substeps.interval_start_s[:, 1:]),
                                   np.asarray(substeps.interval_end_s[:, :-1]), atol=1e-6)
        found = np.asarray(substeps.pair_found_count)
        normal = np.asarray(substeps.pair_max_normal_force_n)
        result["steps"].append({"control": control,
            "pre_time_s": pre_time.tolist(),
            "post_time_s": np.asarray(measured.data.time).tolist(),
            "terminal_before_reset": np.asarray(measured.done).tolist(),
            "pair_found_substeps": (found > 0).any(axis=2).sum(axis=1).tolist(),
            "geometric_contact_substeps":
                np.asarray(substeps.pair_geometric_contact).any(axis=2).sum(axis=1).tolist(),
            "max_pillar_normal_force_n": normal.max(axis=(1, 2)).tolist()})
        state = measured
        if control == controls // 2 - 1:
            before = state
            forced = state.replace(done=jp.asarray([0., 1.], dtype=state.done.dtype))
            after = selected_reset(forced, trajectory)
            jax.block_until_ready(after)
            np.testing.assert_array_equal(np.asarray(after.data.qpos[0]),
                                          np.asarray(before.data.qpos[0]))
            np.testing.assert_array_equal(np.asarray(after.data.qvel[0]),
                                          np.asarray(before.data.qvel[0]))
            np.testing.assert_array_equal(np.asarray(after.info["rng"][0]),
                                          np.asarray(before.info["rng"][0]))
            for original, retained in zip(jax.tree_util.tree_leaves(before),
                                          jax.tree_util.tree_leaves(after)):
                a, b = np.asarray(original), np.asarray(retained)
                np.testing.assert_array_equal(a[0] if a.ndim and a.shape[0] == 2 else a,
                                              b[0] if b.ndim and b.shape[0] == 2 else b)
            np.testing.assert_allclose(np.asarray(after.data.mocap_pos[:, :2]),
                                       np.asarray(centers), atol=1e-6)
            result["selected_reset_after_control"] = control
            result["selected_reset_mask"] = [False, True]
            result["terminal_state_read_before_reset"] = True
            state = after
    if result["steps"][0]["max_pillar_normal_force_n"][0] <= 0:
        raise AssertionError("touching world did not produce a solved pillar normal force")
    if any(step["pair_found_substeps"][1] for step in result["steps"]):
        raise AssertionError("clear world acquired a pillar contact")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=8, choices=range(2, 9))
    parser.add_argument("--graph-mode", choices=("WARP", "JAX"), default="WARP")
    parser.add_argument("--solver-iterations", type=int, default=20)
    parser.add_argument("--solver-ls-iterations", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.motion, args.config, controls=args.controls,
                     graph_mode=args.graph_mode, iterations=args.solver_iterations,
                     ls_iterations=args.solver_ls_iterations)
        result["status"] = "complete"
    except Exception as exc:
        result = {"status": "failed", "exception": repr(exc)}
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "max_touch_force_n": result["steps"][0]["max_pillar_normal_force_n"][0]}))


if __name__ == "__main__":
    main()

"""Candidate native Warp pillar geometry and contact-sensor contract.

This module does not enable Warp obstacles in the tracking environment. The
current JAX-only guard remains until per-world reset, substep timing and
physical readback have been verified in the full G1 environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


CONTACT_SLOTS = 4
CONTACT_DATA = sum(1 << int(field) for field in (
    mujoco.mjtConDataField.mjCONDATA_FOUND,
    mujoco.mjtConDataField.mjCONDATA_FORCE,
    mujoco.mjtConDataField.mjCONDATA_DIST,
))
# MuJoCo's contact-sensor reduction enum: maxforce. The compiled contract is
# checked below, including the five floats per selected contact slot.
MAXFORCE_REDUCTION = 2
FLOATS_PER_SLOT = 5  # found, contact-frame force xyz, signed distance


@dataclass(frozen=True)
class PairSensor:
    name: str
    robot_geom: str
    pillar_geom: str
    address: int
    slots: int


def add_mocap_pillar(spec: mujoco.MjSpec, *, name: str, center_xyz: tuple[float, float, float],
                     radius_m: float, height_m: float):
    """Add one true vertical cylinder whose world center belongs to MjData."""
    if (not name or len(center_xyz) != 3 or not all(math.isfinite(float(x)) for x in center_xyz)
            or not math.isfinite(radius_m) or radius_m <= 0
            or not math.isfinite(height_m) or height_m <= 0):
        raise ValueError("pillar requires a name, finite center, positive radius and height")
    body = spec.worldbody.add_body(name=f"{name}_mocap", mocap=True, pos=center_xyz)
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=(radius_m, height_m / 2), pos=(0, 0, 0),
                  contype=1, conaffinity=1)


def add_pillar_pairs_and_sensors(spec: mujoco.MjSpec, *, robot_geom_names: tuple[str, ...],
                                 pillar_geom_names: tuple[str, ...], slots: int = CONTACT_SLOTS) -> tuple[str, ...]:
    """Declare every exact robot–pillar collision pair and its native contact sensor.

    ``num=slots`` bounds stored contact details. MuJoCo's sensor ``found``
    reports the full matching count even when it exceeds ``num``; readback
    rejects that case instead of treating omitted contacts as no contact.
    """
    if not robot_geom_names or not pillar_geom_names or slots < 1:
        raise ValueError("nonempty robot/pillar geom sets and positive slots are required")
    if len(set(robot_geom_names)) != len(robot_geom_names) or len(set(pillar_geom_names)) != len(pillar_geom_names):
        raise ValueError("robot and pillar geom names must be unique")
    sensors = []
    for pillar in pillar_geom_names:
        for robot in robot_geom_names:
            prefix = f"{pillar}_{robot}"
            spec.add_pair(name=f"{prefix}_pair", geomname1=robot, geomname2=pillar)
            name = f"{prefix}_contact"
            spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_CONTACT,
                            objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=robot,
                            reftype=mujoco.mjtObj.mjOBJ_GEOM, refname=pillar,
                            intprm=[CONTACT_DATA, MAXFORCE_REDUCTION, slots])
            sensors.append(name)
    return tuple(sensors)


def pair_sensor_layout(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[PairSensor, ...]:
    """Bind sensor slices to compiled geom identities and reject another dialect."""
    result = []
    for name in names:
        sensor = model.sensor(name)
        index = int(sensor.id)
        config = tuple(int(x) for x in model.sensor_intprm[index])
        if (int(model.sensor_type[index]) != int(mujoco.mjtSensor.mjSENS_CONTACT)
                or int(model.sensor_objtype[index]) != int(mujoco.mjtObj.mjOBJ_GEOM)
                or int(model.sensor_reftype[index]) != int(mujoco.mjtObj.mjOBJ_GEOM)
                or config[:2] != (CONTACT_DATA, MAXFORCE_REDUCTION)
                or int(model.sensor_dim[index]) != config[2] * FLOATS_PER_SLOT):
            raise ValueError(f"{name}: expected a maxforce geom-pair sensor with found, force and distance")
        result.append(PairSensor(name, model.geom(int(model.sensor_objid[index])).name,
                                 model.geom(int(model.sensor_refid[index])).name,
                                 int(model.sensor_adr[index]), config[2]))
    return tuple(result)


def decode_pair_sensor(sensordata, sensor: PairSensor):
    """Read one world's sensor without treating truncated slots as complete."""
    values = jp.asarray(sensordata)[sensor.address:sensor.address + sensor.slots * FLOATS_PER_SLOT]
    values = values.reshape(sensor.slots, FLOATS_PER_SLOT)
    found = values[0, 0]
    occupied = jp.arange(sensor.slots) < found
    force = values[:, 1:4]  # native contact-frame normal, tangent, tangent
    distance = values[:, 4]
    invalid = (~jp.isfinite(values).all() | (found < 0) | (found != jp.floor(found))
               | (found > sensor.slots))
    return {
        "found_count": found,
        "geometric_contact": jp.any(occupied & (distance <= 0)),
        "max_normal_force_n": jp.max(jp.where(occupied, jp.maximum(force[:, 0], 0), 0)),
        "max_total_force_n": jp.max(jp.where(occupied, jp.linalg.norm(force, axis=-1), 0)),
        "invalid": invalid,
    }


def warp_measurement_invalid(data, pair_readings) -> object:
    """Fail closed on a native Warp buffer overflow or sensor slot truncation.

    The source's recorder must check the returned JAX boolean after every
    actual physics substep, before publishing a force or no-contact claim.
    """
    if data.impl != mjx.Impl.WARP or not hasattr(data._impl, "overflow"):
        raise ValueError("the declared Warp overflow channel is unavailable")
    return jp.any(jp.asarray(data._impl.overflow) != 0) | jp.any(
        jp.asarray([reading["invalid"] for reading in pair_readings]))


def select_warp_reset_data_and_forward(model, live, fresh, done):
    """Select data in each world, then refresh outside any reset branch.

    ``Data.where`` takes a *scalar* decision inside ``vmap`` in this pinned
    MJX-Warp implementation. A single call with a vector decision broadcasts
    incorrectly. The caller must publish each terminal state before using
    this helper and must merge non-physics state with the same ``done`` mask.
    """
    if model.impl != mjx.Impl.WARP or live.impl != mjx.Impl.WARP or fresh.impl != mjx.Impl.WARP:
        raise ValueError("selective Warp reset requires Warp model and data")
    if jp.asarray(done).ndim != 1:
        raise ValueError("selective Warp reset requires one decision per world")
    selected = jax.vmap(lambda current, reset, flag: current.where(flag, reset))(live, fresh, done)
    return jax.vmap(lambda world: mjx.forward(model, world))(selected)

"""Helpers for selecting and initializing the MJX backend."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx


MJX_IMPL_ENV = "OPENTRACK_MJX_IMPL"
WARP_GRAPH_MODE_ENV = "OPENTRACK_MJX_WARP_GRAPH_MODE"
WARP_NUM_ENVS_ENV = "OPENTRACK_MJX_WARP_NUM_ENVS"
WARP_NCONMAX_PER_ENV_ENV = "OPENTRACK_MJX_WARP_NCONMAX_PER_ENV"
WARP_NACONMAX_ENV = "OPENTRACK_MJX_WARP_NACONMAX"
WARP_NACCDMAX_ENV = "OPENTRACK_MJX_WARP_NACCDMAX"
WARP_NJMAX_ENV = "OPENTRACK_MJX_WARP_NJMAX"

DEFAULT_IMPL = "jax"
DEFAULT_WARP_NCONMAX_PER_ENV = 48
DEFAULT_WARP_NJMAX = 192


@dataclass(frozen=True)
class ContactView:
    geom: jax.Array
    dist: jax.Array
    frame: jax.Array


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _mjx_cfg(config: Optional[config_dict.ConfigDict]) -> Any:
    if config is None:
        return None
    return getattr(config, "mjx", None)


def configure_mjx(
    env_config: config_dict.ConfigDict,
    num_envs: Optional[int],
) -> None:
    """Populates env_config.mjx from defaults and environment overrides."""
    if not hasattr(env_config, "mjx"):
        env_config.mjx = config_dict.create()

    mjx_cfg = env_config.mjx
    default_num_envs = int(num_envs or _cfg_get(mjx_cfg, "num_envs", 1))
    impl = _env_str(MJX_IMPL_ENV, _cfg_get(mjx_cfg, "impl", DEFAULT_IMPL)).lower()

    mjx_cfg.impl = impl
    mjx_cfg.num_envs = _env_int(WARP_NUM_ENVS_ENV, default_num_envs)
    mjx_cfg.warp_graph_mode = _env_str(WARP_GRAPH_MODE_ENV, _cfg_get(mjx_cfg, "warp_graph_mode", "WARP"))
    mjx_cfg.warp_nconmax_per_env = _env_int(
        WARP_NCONMAX_PER_ENV_ENV,
        _cfg_get(mjx_cfg, "warp_nconmax_per_env", DEFAULT_WARP_NCONMAX_PER_ENV),
    )
    mjx_cfg.warp_naconmax = _env_int(WARP_NACONMAX_ENV, _cfg_get(mjx_cfg, "warp_naconmax", 0))
    mjx_cfg.warp_naccdmax = _env_int(WARP_NACCDMAX_ENV, _cfg_get(mjx_cfg, "warp_naccdmax", 0))
    mjx_cfg.warp_njmax = _env_int(WARP_NJMAX_ENV, _cfg_get(mjx_cfg, "warp_njmax", DEFAULT_WARP_NJMAX))


def mjx_impl(config: Optional[config_dict.ConfigDict]) -> str:
    cfg = _mjx_cfg(config)
    return _env_str(MJX_IMPL_ENV, _cfg_get(cfg, "impl", DEFAULT_IMPL)).lower()


def _supports_mjx_impl_arg(fn: Any) -> bool:
    return "impl" in inspect.signature(fn).parameters


def _warp_graph_mode(config: Optional[config_dict.ConfigDict]) -> Any:
    cfg = _mjx_cfg(config)
    graph_mode_name = _env_str(WARP_GRAPH_MODE_ENV, _cfg_get(cfg, "warp_graph_mode", "WARP")).upper()
    try:
        import mujoco.mjx.warp as mjxw
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("MJX-Warp graph modes require mujoco-mjx with warp support.") from exc
    return getattr(mjxw.types.GraphMode, graph_mode_name)


def put_model(
    mj_model: mujoco.MjModel,
    config: Optional[config_dict.ConfigDict],
) -> mjx.Model:
    """Places an MjModel on the configured MJX backend."""
    impl = mjx_impl(config)
    if impl == "jax":
        if _supports_mjx_impl_arg(mjx.put_model):
            return mjx.put_model(mj_model, impl="jax")
        return mjx.put_model(mj_model)
    if impl != "warp":
        raise ValueError(f"Unsupported OPENTRACK_MJX_IMPL={impl!r}; expected 'jax' or 'warp'.")
    if not _supports_mjx_impl_arg(mjx.put_model):
        raise RuntimeError(
            "OPENTRACK_MJX_IMPL=warp requires mujoco-mjx>=3.9.0. "
            "Install with `uv sync` after updating dependencies."
        )
    return mjx.put_model(mj_model, impl="warp", graph_mode=_warp_graph_mode(config))


def _warp_buffer_sizes(config: Optional[config_dict.ConfigDict]) -> dict[str, Optional[int]]:
    cfg = _mjx_cfg(config)
    num_envs = _env_int(WARP_NUM_ENVS_ENV, _cfg_get(cfg, "num_envs", 1)) or 1
    nconmax_per_env = _env_int(
        WARP_NCONMAX_PER_ENV_ENV,
        _cfg_get(cfg, "warp_nconmax_per_env", DEFAULT_WARP_NCONMAX_PER_ENV),
    )
    naconmax = _env_int(WARP_NACONMAX_ENV, _cfg_get(cfg, "warp_naconmax", 0))
    naccdmax = _env_int(WARP_NACCDMAX_ENV, _cfg_get(cfg, "warp_naccdmax", 0))
    njmax = _env_int(WARP_NJMAX_ENV, _cfg_get(cfg, "warp_njmax", DEFAULT_WARP_NJMAX))

    if not naconmax:
        naconmax = int(num_envs) * int(nconmax_per_env or DEFAULT_WARP_NCONMAX_PER_ENV)
    if not naccdmax:
        naccdmax = naconmax

    return {"naconmax": int(naconmax), "naccdmax": int(naccdmax), "njmax": int(njmax) if njmax else None}


def make_data(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    config: Optional[config_dict.ConfigDict],
) -> mjx.Data:
    """Creates data for the configured backend."""
    impl = mjx_impl(config)
    if impl == "jax":
        if _supports_mjx_impl_arg(mjx.make_data):
            return mjx.make_data(mjx_model, impl="jax")
        return mjx.make_data(mjx_model)
    if impl != "warp":
        raise ValueError(f"Unsupported OPENTRACK_MJX_IMPL={impl!r}; expected 'jax' or 'warp'.")
    if not _supports_mjx_impl_arg(mjx.make_data):
        raise RuntimeError(
            "OPENTRACK_MJX_IMPL=warp requires mujoco-mjx>=3.9.0. "
            "Install with `uv sync` after updating dependencies."
        )
    return mjx.make_data(mj_model, impl="warp", **_warp_buffer_sizes(config))


def init_data(
    mj_model: mujoco.MjModel,
    mjx_model: mjx.Model,
    config: Optional[config_dict.ConfigDict],
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
) -> mjx.Data:
    """Initializes and forwards MJX data on the configured backend."""
    data = make_data(mj_model, mjx_model, config)
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(mjx_model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(mjx_model.nmocap, -1))
    return mjx.forward(mjx_model, data)


def contact_view(data: mjx.Data) -> ContactView:
    """Returns a minimal contact view for JAX and Warp MJX data."""
    impl_data = getattr(data, "_impl", None)
    if impl_data is not None and hasattr(impl_data, "contact__geom"):
        return ContactView(
            geom=impl_data.contact__geom,
            dist=impl_data.contact__dist,
            frame=impl_data.contact__frame,
        )
    if impl_data is not None and hasattr(impl_data, "contact"):
        contact = impl_data.contact
        return ContactView(geom=contact.geom, dist=contact.dist, frame=contact.frame)
    contact = data.contact
    return ContactView(geom=contact.geom, dist=contact.dist, frame=contact.frame)


def get_collision_info(data: mjx.Data, geom1: int, geom2: int) -> tuple[jax.Array, jax.Array]:
    contact = contact_view(data)
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def geoms_colliding(data: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    return get_collision_info(data, geom1, geom2)[0] < 0

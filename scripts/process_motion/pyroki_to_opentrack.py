#!/usr/bin/env python
"""Direct PyRoki-retarget npz -> OpenTrack mjData (Deliverable 2). Frame-native: builds qpos straight from the
PyRoki output in the OpenTrack G1 frame, grounds + wraps, computes velocities with OpenTrack's multi-horizon-min
recalc, then FK's cvel and extend_motion. NO ProtoMotions .motion / cross-frame conversion.

Quaternion conventions (see quat-convention-hygiene): PyRoki base_frame_wxyz = WXYZ; MuJoCo qpos/xquat = WXYZ;
OpenTrack qpos = WXYZ. No scipy here, so no WXYZ<->XYZW reindex is needed.

  python pyroki_to_opentrack.py --npz-list <file: 'src//name' per line> --template <lafan1 npz> --out-dir <UnitreeG1 dir>
"""
from __future__ import annotations
import argparse, os
from dataclasses import replace
import numpy as np
import jax.numpy as jp
import mujoco
import track_mj as tmj
from track_mj.utils.dataset.traj_class import Trajectory, TrajectoryData, _mhm_backward, _mhm_angular

def _hinge_names(mjm):
    return [mjm.joint(i).name for i in range(mjm.njnt) if mjm.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]

def wrap_pi(a):                                   # joint angles -> [-pi, pi] (matches ProtoMotions extract_qpos wrap)
    return (a + np.pi) % (2 * np.pi) - np.pi

def ground(mjm, qpos, foot_offset=0.02):
    """Per-frame grounding: shift root z so the ROBOT's lowest geom sits at foot_offset. Excludes worldbody
    geoms (floor/terrain at z=0). Pure z-translation (frame-agnostic), keeps the mjData at the same
    feet-on-floor baseline as the .motion deliverable."""
    mjd = mujoco.MjData(mjm); qpos = qpos.copy()
    robot = mjm.geom_bodyid > 0                       # exclude worldbody (floor) geoms
    for t in range(qpos.shape[0]):
        mjd.qpos[:] = qpos[t]; mujoco.mj_forward(mjm, mjd)
        qpos[t, 2] += foot_offset - float(mjd.geom_xpos[robot, 2].min())
    return qpos

def build_qvel(qpos, fps, H):
    """Frame-native qvel from qpos via multi-horizon-min (frame 0 = 0). Linear (global), angular (LOCAL, WXYZ
    quat-delta), joint — the exact scheme OpenTrack's recalc uses, so training-time recalc is consistent."""
    T = qpos.shape[0]
    lv = _mhm_backward(qpos[:, :3], fps, H, np)          # [T-1,3] global linear
    av = _mhm_angular(qpos[:, 3:7], fps, H, np)          # [T-1,3] local angular (qpos quat is WXYZ)
    jv = _mhm_backward(qpos[:, 7:], fps, H, np)          # [T-1,29] joint
    qvel = np.zeros((T, 6 + jv.shape[1]), np.float32)
    qvel[1:] = np.concatenate([lv, av, jv], axis=1)
    return qvel

def forward_kinematics(mjm, qpos, qvel):
    mjd = mujoco.MjData(mjm); nb, ns = mjm.nbody, mjm.nsite; T = qpos.shape[0]
    xpos=np.zeros((T,nb,3),np.float32); xquat=np.zeros((T,nb,4),np.float32); cvel=np.zeros((T,nb,6),np.float32)
    stc=np.zeros((T,nb,3),np.float32); sxp=np.zeros((T,ns,3),np.float32); sxm=np.zeros((T,ns,9),np.float32)
    for t in range(T):
        mjd.qpos[:]=qpos[t]; mjd.qvel[:]=qvel[t]; mujoco.mj_forward(mjm,mjd)
        xpos[t]=mjd.xpos; xquat[t]=mjd.xquat; cvel[t]=mjd.cvel; stc[t]=mjd.subtree_com
        if ns: sxp[t]=mjd.site_xpos; sxm[t]=mjd.site_xmat
    return dict(xpos=xpos,xquat=xquat,cvel=cvel,subtree_com=stc,site_xpos=sxp,site_xmat=sxm)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz-list",required=True); ap.add_argument("--template",required=True)
    ap.add_argument("--out-dir",required=True); ap.add_argument("--task",default="G1TrackingGeneralDR")
    ap.add_argument("--velocity-horizon",type=int,default=3)
    a=ap.parse_args()
    env_class=tmj.registry.get(a.task,"tracking_train_env_class")
    env_cfg=tmj.registry.get(a.task,"tracking_config").env_config
    env=env_class(terrain_type=env_cfg.terrain_type,config=env_cfg); mjm=env._mj_model
    assert len(_hinge_names(mjm))==29, "OpenTrack model hinge count != 29"
    template=Trajectory.load(a.template,backend=jp)
    os.makedirs(a.out_dir,exist_ok=True)
    for ln in open(a.npz_list):
        ln=ln.strip()
        if not ln: continue
        src,name=(ln.split("//",1)+[None])[:2]; src=src.strip()
        name=(name or os.path.splitext(os.path.basename(src))[0]).strip()
        out=os.path.join(a.out_dir,f"{name}.npz")
        if os.path.exists(out): continue                           # skip-existing (resume-safe)
        try:
            z=np.load(src,allow_pickle=True)
            root_pos=np.asarray(z["base_frame_pos"],np.float64)        # world position
            root_quat=np.asarray(z["base_frame_wxyz"],np.float64)      # WXYZ (== MuJoCo qpos order; no reindex)
            dof=wrap_pi(np.asarray(z["joint_angles"],np.float64))      # [-pi,pi]; PyRoki order == OpenTrack hinge order
            fps=float(np.asarray(z["fps"]).reshape(-1)[0])
            qpos=np.concatenate([root_pos, root_quat, dof],axis=1).astype(np.float32)   # [T,36] WXYZ root
            qpos=ground(mjm, qpos)
            qvel=build_qvel(qpos, fps, a.velocity_horizon)
            kin=forward_kinematics(mjm, qpos, qvel)                     # cvel now consistent with qvel
            T=qpos.shape[0]
            data=TrajectoryData(qpos=jp.asarray(qpos),qvel=jp.asarray(qvel),xpos=jp.asarray(kin["xpos"]),
                xquat=jp.asarray(kin["xquat"]),cvel=jp.asarray(kin["cvel"]),subtree_com=jp.asarray(kin["subtree_com"]),
                site_xpos=jp.asarray(kin["site_xpos"]),site_xmat=jp.asarray(kin["site_xmat"]),split_points=jp.asarray([0,T]))
            traj=Trajectory(info=replace(template.info,frequency=fps),data=data); traj.save(out)
            traj=env.extend_motion(traj,smooth_start_end=False); traj.save(out)   # interp to env dt + replay
            print(f"[p2ot] {name}: T={T} fps={fps:.1f} -> {out} complete={traj.data.is_complete}",flush=True)
        except Exception as e:
            print(f"[p2ot-ERR] {name}: {type(e).__name__} {str(e)[:150]}",flush=True)

if __name__=="__main__": main()

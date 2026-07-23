"""Canonical observation-key order shared by G1 tracking policies."""

# Keep this tuple in the order serialized into training configs and consumed by
# policy checkpoints. Callers convert it to a list before assigning obs_keys.
G1_TRACKING_OBS_KEYS: tuple[str, ...] = (
    "dif_joint_pos",
    "dif_joint_vel",
    "gvec_pelvis",
    "gyro_pelvis",
    "joint_pos",
    "joint_vel",
    "last_motor_targets",
    "ref_feet_height",
    "ref_root_height",
)

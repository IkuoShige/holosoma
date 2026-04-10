"""Locomotion reward presets for the K1 robot."""

from dataclasses import replace as _replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.loco.flashsac_transform import (
    K1_UPPER_BODY_POSE_INDICES,
    make_flashsac_reward,
)

k1_22dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    # Upper body (10 DOFs)
                    50.0,  # AAHead_yaw
                    50.0,  # Head_pitch
                    50.0,  # ALeft_Shoulder_Pitch
                    50.0,  # Left_Shoulder_Roll
                    50.0,  # Left_Elbow_Pitch
                    50.0,  # Left_Elbow_Yaw
                    50.0,  # ARight_Shoulder_Pitch
                    50.0,  # Right_Shoulder_Roll
                    50.0,  # Right_Elbow_Pitch
                    50.0,  # Right_Elbow_Yaw
                    # Left leg (6 DOFs)
                    0.01,  # Left_Hip_Pitch
                    1.0,  # Left_Hip_Roll
                    5.0,  # Left_Hip_Yaw
                    0.01,  # Left_Knee_Pitch
                    5.0,  # Left_Ankle_Pitch
                    5.0,  # Left_Ankle_Roll
                    # Right leg (6 DOFs)
                    0.01,  # Right_Hip_Pitch
                    1.0,  # Right_Hip_Roll
                    5.0,  # Right_Hip_Yaw
                    0.01,  # Right_Knee_Pitch
                    5.0,  # Right_Ankle_Pitch
                    5.0,  # Right_Ankle_Roll
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

k1_22dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    # Upper body (10 DOFs)
                    50.0,  # AAHead_yaw
                    50.0,  # Head_pitch
                    50.0,  # ALeft_Shoulder_Pitch
                    50.0,  # Left_Shoulder_Roll
                    50.0,  # Left_Elbow_Pitch
                    50.0,  # Left_Elbow_Yaw
                    50.0,  # ARight_Shoulder_Pitch
                    50.0,  # Right_Shoulder_Roll
                    50.0,  # Right_Elbow_Pitch
                    50.0,  # Right_Elbow_Yaw
                    # Left leg (6 DOFs)
                    0.01,  # Left_Hip_Pitch
                    1.0,  # Left_Hip_Roll
                    5.0,  # Left_Hip_Yaw
                    0.01,  # Left_Knee_Pitch
                    5.0,  # Left_Ankle_Pitch
                    5.0,  # Left_Ankle_Roll
                    # Right leg (6 DOFs)
                    0.01,  # Right_Hip_Pitch
                    1.0,  # Right_Hip_Roll
                    5.0,  # Right_Hip_Yaw
                    0.01,  # Right_Knee_Pitch
                    5.0,  # Right_Ankle_Pitch
                    5.0,  # Right_Ankle_Roll
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

# FlashSAC reward derived from K1 PPO preset via canonical v5 recipe.
#
# K1 joint order: upper body (indices 0-9) FIRST, then legs (10-21).
# This differs from G1 (legs first, upper body at 12-28), so we pass
# ``K1_UPPER_BODY_POSE_INDICES`` explicitly.
#
# Head joints (AAHead_yaw, Head_pitch) are boosted to 150 along with
# arm joints. G1 has no head DOFs so this is a deliberate K1 extension
# of the v5 recipe — the head should stay stable during locomotion.
#
# K1-specific tuning history:
#   v1 20260410_043117: G1 v5 defaults (sigma=0.15) → shuffle gait.
#   v2 20260410_054053: feet_phase 7, swing 0.065, action_rate -0.001
#       → marginal improvement, still shuffling.
#   v3 20260410_065007: feet_phase 12, swing 0.04, tracking 1.0
#       → stopped walking. Marching in place.
#   v4 20260410_072839: tracking 2.0, feet_phase 10, swing 0.04
#       → still no walk. temp=0.0004 in ALL v1-v4 runs.
#   v5 20260410_080903: sigma 0.15→0.25, reward=v2
#       → back to shuffle (temp higher but still shuffling).
#   v6 (current): Codex report showed upstream FlashSAC uses a MUCH
#       simpler reward (5 terms only). Key insight: holosoma's
#       ``feet_phase`` (clock-based height match) is exploitable by a
#       deterministic policy via oscillation — unlike upstream's
#       ``feet_air_time`` (time airborne). Also, extra shaping terms
#       (pose, orientation, close_feet, feet_ori) all create local
#       optima that a narrow policy gets stuck in.
#       Strategy: strip to upstream-like minimal reward. Walk first,
#       add shaping later. Keep sigma=0.25 (K1 needs exploration).
k1_22dof_loco_flashsac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # Core task: velocity tracking (upstream weights)
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        # Gait: feet_phase at LOW weight. Not upstream's feet_air_time
        # (unavailable in holosoma), but weak enough to not dominate.
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=1.0,  # v2: 7.0. Upstream feet_air_time=0.75. Weak to avoid clock exploitation.
            params={"swing_height": 0.065, "tracking_sigma": 0.008},
        ),
        # Minimal penalties (upstream-matched magnitudes)
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.05,  # matches upstream ang_vel_xy_l2
            params={},
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.005,  # matches upstream action_rate_l2
            params={},
        ),
    },
)

__all__ = ["k1_22dof_loco", "k1_22dof_loco_fast_sac", "k1_22dof_loco_flashsac"]

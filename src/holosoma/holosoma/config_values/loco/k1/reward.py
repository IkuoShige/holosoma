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
#   v6 20260410_084759: upstream-minimal 5-term reward, sigma=0.25
#       → still shuffle. entropy OK but shuffle persists.
#   v7 20260410_094504: PD Kp 200→80 (physics fix), 5-term reward
#       → no visible change. PD alone not enough.
#   v8 20260410_103035: G1 v5 9-term, G1 obs scales, friction fix
#       → improvement. Walking but step pitch too fast.
#   v9 20260410_111104: gait_period 1.2s + randomization
#       → eval: fwd=0.266m/s (53% tracking), LEFT leg 0.78Hz (correct),
#       RIGHT leg 5.86Hz (vibration). L-R asymmetry is primary issue.
#       FlashSAC has no symmetry mechanism (PPO has use_symmetry=True).
#   v10 20260410_125717: action_rate -0.05 + symmetry augmentation
#       → L-R asymmetry FIXED (both legs 0.78Hz). fwd=0.29m/s.
#       But hip amplitude still small (0.15rad vs 0.56rad available).
#   v11 20260410_142039: action_rate reverted to -0.005. Symmetry
#       holds. fwd=0.354m/s (+22%). But hip amp still 0.12rad (small).
#       Policy uses knee/ankle instead of hip swing.
#   v12 (current): tighten tracking_sigma 0.25→0.1 to demand closer
#       velocity tracking. At sigma=0.25, 0.354m/s already gives 91%
#       reward. sigma=0.1 makes that only 81% → policy must stride more.
_k1_base = make_flashsac_reward(
    k1_22dof_loco,
    upper_body_pose_indices=K1_UPPER_BODY_POSE_INDICES,
    weight_overrides={
        "penalty_ang_vel_xy": -0.05,
        "penalty_orientation": -1.0,
        "penalty_action_rate": -0.005,  # G1 value. Symmetry fixes vibration.
        "pose": -0.2,
        # v24: revert feet_phase to 4.0 (v22 value). v23 halved it to 2.0
        # which hurt G_r_max (29.16→23.00). feet_phase is the gait clock
        # anchor; removing it leaves feet_air_time without structure.
        "feet_phase": 4.0,
        "penalty_feet_ori": -0.5,
        "penalty_close_feet_xy": -1.0,
    },
)


# v22: added feet_air_time at weight 2.0 (T1 default). Evidence of
#     effect: G_r_max 21.6→29.2, mean_bias magnitude up 29%. But not
#     enough gradient signal to overcome feet_phase (weight 4.0 was
#     drowning it out). User: "変わらん" (no visible change).
# v23: match T1's reward ratio. T1 has feet_air_time=2.0, feet_phase=1.0
#     (2:1 air:phase). Our v22 was 2.0:4.0 (0.5:1). v23 uses 4.0:2.0
#     to exactly match T1's 2:1 ratio while keeping absolute magnitudes
#     similar to the v22 total gait signal.
k1_22dof_loco_flashsac = _replace(
    _k1_base,
    terms={
        **_k1_base.terms,
        # v25: FeetAirTime surgical fix after v24 diagnostics.
        #
        # v24 TB breakdown (at ~50M steps):
        #   feet_phase       : +3.02  (54% of positive reward, DOMINANT)
        #   tracking_lin_vel : +1.77  (32%)
        #   tracking_ang_vel : +0.77  (14%)
        #   feet_air_time    : +0.0000  ← ZERO CONTRIBUTION
        #   Hip_Pitch |action|: 0.64 (legs move aggressively)
        #   episode_length: 987/1000 (no termination bottleneck)
        #
        # Diagnosis: the policy converged to a shuffle gait local minimum
        # where legs oscillate fast enough to satisfy feet_phase (clocked
        # foot-height) and tracking_lin_vel (via forward shuffle motion),
        # but swing time per step is < 0.2 s. With threshold_min=0.2,
        # FeetAirTime hit its deadband and provided ZERO gradient to
        # escape the shuffle.
        #
        # v25 surgical fix (only FeetAirTime params change):
        #   - threshold_min: 0.2 → 0.0 (every swing gets reward)
        #   - threshold_max: 0.5 → 1.0 (longer swings rewarded more)
        #   - weight:        2.0 → 4.0 (match feet_phase strength)
        "feet_air_time": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:FeetAirTime",
            weight=4.0,
            params={
                "threshold_min": 0.0,
                "threshold_max": 1.0,
                "contact_force_threshold": 1.0,
                "command_norm_threshold": 0.1,
            },
        ),
    },
)

__all__ = ["k1_22dof_loco", "k1_22dof_loco_fast_sac", "k1_22dof_loco_flashsac"]

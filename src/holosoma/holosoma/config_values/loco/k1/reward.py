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
# v36: PPO-compatible FlashSAC reward.
#
# PPO reward STRUCTURE preserved (same terms, same positive weights).
# Penalties reduced to 1/10 of PPO — tolerable for FlashSAC's bounded
# exploration while still providing the same optimization objective.
# NO FlashSAC-specific terms (no stride_progress, no feet_air_time).
# alive kept (PPO has it; v5's drop was a FlashSAC workaround).
#
# PPO values → v36 values:
# v43: revert to v39 reward base (proven to walk).
_k1_base = make_flashsac_reward(
    k1_22dof_loco,
    upper_body_pose_indices=K1_UPPER_BODY_POSE_INDICES,
    weight_overrides={
        "penalty_ang_vel_xy": -0.05,
        "penalty_orientation": -1.0,
        # v46: revert to -0.005 (v39 value). v44's -0.02 didn't fix
        # standing oscillation and v45 with stand_still=-2.0 killed walking.
        "penalty_action_rate": -0.005,
        "pose": -0.2,
        # v49: 2.5→3.5. Full command range needs stronger clock to
        # enforce alternation (v48: 46% feet_phase but still skipping).
        "feet_phase": 3.5,
        "penalty_feet_ori": -0.5,
        "penalty_close_feet_xy": -1.0,
    },
)


def _patch_knee_pose(base: RewardManagerCfg) -> RewardManagerCfg:
    """v34: knee pose_weight 0.01→1.0 to enforce bent-knee walking."""
    pose_term = base.terms["pose"]
    pw = list(pose_term.params["pose_weights"])
    pw[13] = 1.0  # Left_Knee_Pitch
    pw[19] = 1.0  # Right_Knee_Pitch
    return _replace(
        base,
        terms={**base.terms, "pose": _replace(pose_term, params={**pose_term.params, "pose_weights": pw})},
    )


# v52: v5 recipe + K1 short-leg stride fix.
#
# v51 (``k1_22dof_flash_sac_v5``, run 20260418_153529) walked, but:
#   - feet grazed the ground (didn't lift)
#   - cadence was rapid "tan-tan" rather than sinusoidal
#   - standstill showed hip-forward arched posture
#
# Root cause for the feet + cadence issues: the G1 v5 feet_phase kernel
# (``swing_height=0.09m``, ``tracking_sigma=0.008m``) is geometrically
# unreachable for K1's shorter legs with the default-pose kinematics,
# AND the kernel is so sharp that any imprecision gives near-zero reward
# (exp(-(0.02/0.008)^2) ~= 0.002). The narrow FlashSAC policy accepts
# feet_phase = 0 and optimizes the remaining 8 terms, which removes the
# clock-synchronization gradient — cadence defaults to the policy's
# natural oscillation frequency and stride stays tiny.
#
# This preset relaxes the feet_phase kernel specifically:
#   swing_height    0.09 → 0.065  (K1 v11–v15 short-leg finding)
#   tracking_sigma  0.008 → 0.015 (partial reward restores gradient)
# Everything else is identical to v5. If stride increases, cadence
# should lock to the clock and standstill posture may also improve
# indirectly; stand_still + gait_period 1.2s remain in reserve as v53.
k1_22dof_loco_flashsac_v5_stride = _patch_knee_pose(_replace(
    _k1_base,
    terms={
        **_k1_base.terms,
        "feet_phase": _replace(
            _k1_base.terms["feet_phase"],
            weight=4.0,
            params={"swing_height": 0.065, "tracking_sigma": 0.015},
        ),
    },
))


# v51 / Plan A+: G1 v5 recipe (9 terms) applied literally to K1.
#
# Plan A (``k1_22dof_flash_sac_stripped``, run 20260418_124413) converged
# smoothly but to a static-splits local optimum (right foot forward, left
# foot backward, body low). The 5-term reward has no term forcing foot
# lift or alternation, and no term penalizing legs spread apart, so a
# narrow policy finds the stretched static pose as a tracking-reward
# maximizer.
#
# The G1 v5 recipe fixes this exact failure mode:
#   feet_phase (4.0)             — forces alternating foot lift → breaks static
#   pose (-0.2, ub=150)          — weak leg constraint, strong upper-body keep
#   penalty_close_feet_xy (-1.0) — prevents feet overlap (and indirectly splits)
#   penalty_feet_ori (-0.5)      — gentle foot orientation guidance
#
# No feet_air_time / stride_progress / stand_still / alive — those are
# the K1-specific additions that v28–v49 layered on, and none produced
# PPO-quality gait. This preset is the "stop adding shaping, just use
# what worked on G1" control experiment.
#
# Note: _k1_base uses feet_phase=3.5 (v49 override); restore 4.0 to match
# G1 v5 exactly.
k1_22dof_loco_flashsac_v5 = _patch_knee_pose(_replace(
    _k1_base,
    terms={
        **_k1_base.terms,
        "feet_phase": _replace(
            _k1_base.terms["feet_phase"],
            weight=4.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
    },
))


k1_22dof_loco_flashsac = _patch_knee_pose(_replace(
    _k1_base,
    terms={
        **_k1_base.terms,
        # v49: tighten feet_phase tracking_sigma 0.008→0.004.
        # Forces precise clock-following. At 0.008, sloppy timing
        # still gets ~80% reward. At 0.004, deviation is punished
        # more sharply → periodic, rhythmic gait.
        "feet_phase": _replace(
            _k1_base.terms["feet_phase"],
            params={"swing_height": 0.09, "tracking_sigma": 0.004},
        ),
        "feet_air_time": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:FeetAirTime",
            weight=2.0,
            params={
                "threshold_max": 0.5,
                "contact_force_threshold": 5.0,
                "command_norm_threshold": 0.1,
            },
        ),
        "stride_progress": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:StrideProgress",
            weight=2.0,
            params={
                "target_stride": 0.15,
                "contact_force_threshold": 5.0,
                "command_norm_threshold": 0.1,
            },
        ),
        # v47: -0.5 improved but not enough. -2.0 killed walking. Try -1.0.
        "penalty_stand_still": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stand_still",
            weight=-1.0,
            params={"command_threshold": 0.1},
        ),
    },
))

# v50 (Plan A / stripped): IsaacLab-stock-style minimal reward.
#
# Motivation: v1-v49 (50 iterations) all use the full PPO reward structure
# (tracking + feet_phase + pose + orientation + ang_vel_xy + action_rate +
# close_feet_xy + feet_ori, with K1-specific additions feet_air_time /
# stride_progress / stand_still). Every iteration re-weights these terms;
# none remove them. The G1 port's only clean-walking recipe (attempt #6,
# ``20260408_142832``: 0.28 m/s, stable) is the 5-term stripped IsaacLab-
# stock mirror, which was NEVER tried on K1 under the post-v13 correct
# action-scale regime (``target_action_scale_rad=1.0``).
#
# This preset is that missing experiment: drop every shaping term that
# competes with forward walking and leave only the IsaacLab-stock core.
#
# Removed vs k1_22dof_loco_flashsac (v49):
#   feet_phase, feet_air_time, stride_progress, penalty_stand_still,
#   pose, penalty_feet_ori, penalty_close_feet_xy
#
# Kept (IsaacLab stock + FlashSAC-safe penalty weights):
#   tracking_lin_vel    +2.0  sigma=0.25
#   tracking_ang_vel    +1.5  sigma=0.25
#   penalty_ang_vel_xy  -0.05   (PPO: -1.0; IsaacLab stock: -0.05)
#   penalty_orientation -1.0    (PPO: -10.0; 10× weaker)
#   penalty_action_rate -0.005  (PPO: -2.0; 400× weaker — required)
#
# Observation must be ``k1_22dof_loco_single_flashsac`` (no sin/cos phase
# clock): without feet_phase in the reward, the clock observation is dead
# weight and only contributes a feature the policy cannot exploit usefully.
k1_22dof_loco_flashsac_stripped = RewardManagerCfg(
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
            weight=-0.05,
            params={},
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-1.0,
            params={},
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.005,
            params={},
        ),
    },
)

__all__ = [
    "k1_22dof_loco",
    "k1_22dof_loco_fast_sac",
    "k1_22dof_loco_flashsac",
    "k1_22dof_loco_flashsac_stripped",
    "k1_22dof_loco_flashsac_v5",
    "k1_22dof_loco_flashsac_v5_stride",
]

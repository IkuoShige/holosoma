"""Locomotion reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_loco = RewardManagerCfg(
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
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
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
            weight=1.0,
            params={},
        ),
    },
)

g1_29dof_loco_fast_sac = RewardManagerCfg(
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
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
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

g1_29dof_loco_fpo = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        **g1_29dof_loco.terms,
        "termination": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-2000.0,
            params={},
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["action_rate_warmup"],
        ),
    },
)

# Reward preset for FlashSAC tuned to closely mirror IsaacLab stock
# ``Isaac-Velocity-Flat-G1-v0`` (the reference task FlashSAC's algorithm
# hyperparameters were trained on). The differences from
# ``g1_29dof_loco`` are deliberate ablations of holosoma-specific terms
# that create degenerate local optima for FlashSAC's narrow
# deterministic policy:
#
# * ``feet_phase`` REMOVED.   Holosoma's feet_phase is a pure foot-height
#   match against a clock signal — it has no coupling to forward
#   velocity, COM progression, or stance impulse. A near-deterministic
#   policy can harvest it by stepping in place. With weight=5.0 and
#   sigma=0.008 it dominated the per-term reward decomposition (+36 ep
#   sum vs tracking_lin_vel +14) on prior runs.
# * ``alive`` REMOVED.        IsaacLab stock has no alive bonus at all.
#   Even at weight=1.0 the constant +1/step competes with tracking
#   gradients. Removing it forces the policy to maximize tracking.
# * ``pose`` REMOVED.         Holosoma's pose penalty has 50.0 weight
#   per joint on the upper body, which strongly discourages any torso
#   sway. PPO's high-entropy policy explores past it; FlashSAC's narrow
#   deterministic policy locks the torso to default and never moves.
# * ``penalty_feet_ori`` REMOVED.
# * ``penalty_close_feet_xy`` REMOVED.
# * Penalties scaled to IsaacLab stock magnitudes: ``ang_vel_xy`` from
#   -1.0 to -0.05, ``orientation`` from -10.0 to -1.0.
# * ``tracking_*`` weights kept at holosoma values (they're already
#   reasonable; the issue is that other terms dominated, not that
#   tracking was too small).
#
# The full per-term ablation rationale and Codex consultation that led
# to this preset live in the commit message.
g1_29dof_loco_flashsac = RewardManagerCfg(
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
        # ---- Shaping terms below: all 10× weaker than PPO default ----
        # Run #10 (pose=-0.05 only) showed forward walking improved but
        # backward became two-foot jumping (no gait constraint). Runs
        # #6/#7 (no shaping at all) walked with bent posture. Run #9
        # (full PPO weights) collapsed. Strategy: add back ALL PPO
        # shaping terms at 10× weaker weights to provide gentle guidance
        # without triggering FlashSAC's zero-action attractor.
        # ``alive`` is the only PPO term intentionally omitted — it
        # creates a lazy-attractor even at reduced weight.
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.05,  # PPO: -0.5 (10× weaker)
            params={
                "pose_weights": [
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,    # left leg
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,    # right leg
                    50.0, 50.0, 50.0, 50.0, 50.0,       # upper body
                    50.0, 50.0, 50.0, 50.0, 50.0,
                    50.0, 50.0, 50.0, 50.0, 50.0,
                    50.0, 50.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=2.5,  # PPO: 5.0 (2× weaker). 0.5 was too weak for backward gait.
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-0.5,  # PPO: -5.0 (10× weaker)
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-1.0,  # PPO: -10.0 (10× weaker)
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
    },
)

__all__ = ["g1_29dof_loco", "g1_29dof_loco_fast_sac", "g1_29dof_loco_fpo", "g1_29dof_loco_flashsac"]

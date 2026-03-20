"""Kick reward presets for the K1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

k1_22dof_ball_kick = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # === Task rewards ===
        "ball_velocity_target_direction": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:ball_velocity_target_direction",
            weight=10.0,
            params={"decay_time": 0.1, "max_reward": 10.0},
        ),
        "ball_acceleration_toward_target": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:ball_acceleration_toward_target",
            weight=0.25,
            params={"scale": 80.0, "max_reward": 80.0},
        ),
        "kicking_foot_approach_ball": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:kicking_foot_approach_ball",
            weight=10.0,
            params={
                "proximity_sigma": 0.1,
                "stationary_threshold": 0.1,
                "max_reward": 50.0,
            },
        ),
        "body_alignment_for_kick": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:body_alignment_for_kick",
            weight=1.0,
            params={"sigma": 0.5, "max_reward": 1.0},
        ),
        "waiting_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:waiting_penalty",
            weight=-1.0,
            params={"max_still_time": 2.0},
        ),
        # === Locomotion rewards ===
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.25,
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-200.0,
            params={"desired_base_height": 0.54},
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-20.0,
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-1.5,
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.1,
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=1.0,
            params={"swing_height": 0.08, "tracking_sigma": 0.25},
        ),
    },
)

"""Dribble reward presets for the K1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

k1_22dof_ball_dribble = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # === Task rewards ===
        "ball_velocity_tracking": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:ball_velocity_tracking",
            weight=2.0,
            params={"sigma": 1.0, "min_speed": 0.1},
        ),
        "ball_distance_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:ball_distance_penalty",
            weight=-0.25,
            params={"sigma": 1.0, "max_dist": 3.0},
        ),
        "look_at_ball": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:look_at_ball",
            weight=0.5,
            params={"sigma": 0.5},
        ),
        # === Locomotion rewards ===
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.25,
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-20.0,
            params={"desired_base_height": 0.54},
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-1.5,
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=2.0,
            params={"swing_height": 0.08, "tracking_sigma": 0.25},
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
        ),
    },
)

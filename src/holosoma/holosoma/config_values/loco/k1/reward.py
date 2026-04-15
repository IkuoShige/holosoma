"""Locomotion reward presets for the K1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

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

k1_22dof_agile_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=4.0,
            params={
                "tracking_sigma": 0.20,
                "push_compensation_tau_s": 0.18,
                "push_compensation_cutoff_s": 0.45,
                "push_compensation_max_speed": 1.2,
                "push_compensation_max_cmd_ratio": 0.8,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.0,
            params={"tracking_sigma": 0.20},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={
                "swing_height": 0.07,
                "tracking_sigma": 0.008,
                "dynamic_swing_height_from_lin_speed": 0.0,
                "dynamic_swing_height_from_yaw_speed": 0.0,
                "dynamic_swing_height_from_gait_freq": 0.0,
                "dynamic_swing_height_max": 0.18,
            },
        ),
        "stride_pitch_coupling": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:stride_pitch_coupling",
            weight=3.0,
            params={
                "tracking_sigma": 0.04,
                "min_cmd_speed": 0.2,
                "base_stride": 0.12,
                "stride_from_lin_speed": 0.13,
                "stride_from_yaw_speed": 0.04,
                "stride_from_gait_freq": 0.0,
                "min_stride": 0.10,
                "max_stride": 0.38,
            },
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_head_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_head_ang_vel_xy",
            weight=-1.2,
            params={"head_body_name": "Head_2", "fallback_contains": "Head", "deadzone": 0.6},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-15.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_far_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_far_feet_xy",
            weight=-0.8,
            params={"far_feet_threshold": 0.38},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stumble": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stumble",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-5.0,
            params={"desired_base_height": 0.68},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
            tags=["penalty_curriculum"],
        ),
        "termination": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-50.0,
            params={},
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=2.0,
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

k1_22dof_agile_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=8.0,
            params={
                "tracking_sigma": 0.50,
                "push_compensation_tau_s": 0.18,
                "push_compensation_cutoff_s": 0.45,
                "push_compensation_max_speed": 1.2,
                "push_compensation_max_cmd_ratio": 0.8,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=5.0,
            params={"tracking_sigma": 0.30},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_head_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_head_ang_vel_xy",
            weight=-1.2,
            params={"head_body_name": "Head_2", "fallback_contains": "Head", "deadzone": 0.6},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-15.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.4,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase_gated": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase_gated",
            weight=3.0,
            params={
                "swing_height": 0.07,
                "tracking_sigma": 0.008,
                "dynamic_swing_height_from_lin_speed": 0.0,
                "dynamic_swing_height_from_yaw_speed": 0.0,
                "dynamic_swing_height_from_gait_freq": 0.0,
                "dynamic_swing_height_max": 0.18,
                "cmd_speed_gate_threshold": 0.15,
                "cmd_speed_gate_ramp": 0.15,
                "cmd_speed_high_threshold": 1.2,
                "cmd_speed_high_ramp": 0.6,
            },
        ),
        "reward_standstill": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:reward_standstill",
            weight=3.0,
            params={
                "cmd_speed_threshold": 0.15,
                "vel_sigma": 0.08,
                "dof_vel_sigma": 4.0,
            },
        ),
        "stride_pitch_coupling": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:stride_pitch_coupling",
            weight=1.5,
            params={
                "tracking_sigma": 0.04,
                "min_cmd_speed": 0.2,
                "base_stride": 0.12,
                "stride_from_lin_speed": 0.16,
                "stride_from_yaw_speed": 0.04,
                "stride_from_gait_freq": 0.0,
                "min_stride": 0.10,
                "max_stride": 0.42,
                "actual_speed_gate_ratio": 0.35,
                "actual_speed_gate_threshold": 0.30,
                "actual_speed_gate_vy_weight": 0.35,
                "high_speed_fade_threshold": 1.2,
                "high_speed_fade_ramp": 0.6,
            },
        ),
        "penalty_stall_when_commanded": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stall_when_commanded",
            weight=-4.0,
            params={"command_speed_threshold": 0.35, "min_speed_ratio": 0.35, "vy_speed_weight": 0.35},
            tags=["penalty_curriculum"],
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
        "penalty_action_rate_l1": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate_l1",
            weight=-0.3,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_far_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_far_feet_xy",
            weight=-0.8,
            params={"far_feet_threshold": 0.38},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stumble": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stumble",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-5.0,
            params={"desired_base_height": 0.68},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
            tags=["penalty_curriculum"],
        ),
        "termination": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-50.0,
            params={},
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=4.0,
            params={},
        ),
    },
)

k1_22dof_soccer_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=5.0,
            params={
                "tracking_sigma": 0.20,
                "push_compensation_tau_s": 0.18,
                "push_compensation_cutoff_s": 0.45,
                "push_compensation_max_speed": 1.0,
                "push_compensation_max_cmd_ratio": 0.8,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=4.5,
            params={"tracking_sigma": 0.20},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={
                "swing_height": 0.05,
                "tracking_sigma": 0.008,
                "dynamic_swing_height_from_lin_speed": 0.02,
                "dynamic_swing_height_from_yaw_speed": 0.0,
                "dynamic_swing_height_from_gait_freq": 0.0,
                "dynamic_swing_height_max": 0.10,
            },
        ),
        "stride_pitch_coupling": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:stride_pitch_coupling",
            weight=3.0,
            params={
                "tracking_sigma": 0.04,
                "min_cmd_speed": 0.2,
                "base_stride": 0.06,
                "stride_from_lin_speed": 0.25,
                "stride_from_yaw_speed": 0.05,
                "stride_from_gait_freq": 0.0,
                "min_stride": 0.04,
                "max_stride": 0.45,
            },
        ),
        "reward_standstill": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:reward_standstill",
            weight=2.0,
            params={
                "cmd_speed_threshold": 0.15,
                "vel_sigma": 0.08,
                "dof_vel_sigma": 4.0,
            },
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.5,
            params={},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_head_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_head_ang_vel_xy",
            weight=-1.2,
            params={"head_body_name": "Head_2", "fallback_contains": "Head", "deadzone": 0.6},
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
            weight=-1.5,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate_l1": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate_l1",
            weight=-0.3,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-12.0,
            params={"close_feet_threshold": 0.18},
            tags=["penalty_curriculum"],
        ),
        "penalty_far_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_far_feet_xy",
            weight=-1.0,
            params={"far_feet_threshold": 0.35},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-8.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stumble": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stumble",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stall_when_commanded": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stall_when_commanded",
            weight=-5.0,
            params={"command_speed_threshold": 0.30, "min_speed_ratio": 0.30, "vy_speed_weight": 0.35},
            tags=["penalty_curriculum"],
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-5.0,
            params={"desired_base_height": 0.68},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
            tags=["penalty_curriculum"],
        ),
        "termination": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-50.0,
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

k1_22dof_soccer_run_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=6.0,
            params={
                "tracking_sigma": 0.20,
                "push_compensation_tau_s": 0.18,
                "push_compensation_cutoff_s": 0.45,
                "push_compensation_max_speed": 1.2,
                "push_compensation_max_cmd_ratio": 0.8,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=3.0,
            params={"tracking_sigma": 0.25},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={
                "swing_height": 0.08,
                "tracking_sigma": 0.008,
                "dynamic_swing_height_from_lin_speed": 0.03,
                "dynamic_swing_height_from_yaw_speed": 0.0,
                "dynamic_swing_height_from_gait_freq": 0.0,
                "dynamic_swing_height_max": 0.14,
            },
        ),
        "stride_pitch_coupling": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:stride_pitch_coupling",
            weight=4.0,
            params={
                "tracking_sigma": 0.04,
                "min_cmd_speed": 0.5,
                "base_stride": 0.15,
                "stride_from_lin_speed": 0.30,
                "stride_from_yaw_speed": 0.03,
                "stride_from_gait_freq": 0.0,
                "min_stride": 0.10,
                "max_stride": 0.55,
            },
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.5,
            params={},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_head_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_head_ang_vel_xy",
            weight=-3.0,
            params={"head_body_name": "Head_2", "fallback_contains": "Head", "deadzone": 0.3},
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
            weight=-1.5,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate_l1": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate_l1",
            weight=-0.3,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-15.0,
            params={"close_feet_threshold": 0.20},
            tags=["penalty_curriculum"],
        ),
        "penalty_far_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_far_feet_xy",
            weight=-1.0,
            params={"far_feet_threshold": 0.40},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stumble": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stumble",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_stall_when_commanded": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_stall_when_commanded",
            weight=-6.0,
            params={"command_speed_threshold": 0.50, "min_speed_ratio": 0.30, "vy_speed_weight": 0.2},
            tags=["penalty_curriculum"],
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-5.0,
            params={"desired_base_height": 0.68},
            tags=["penalty_curriculum"],
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
            tags=["penalty_curriculum"],
        ),
        "termination": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-50.0,
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
                    2.0,  # Left_Knee_Pitch — discourage over-bending
                    5.0,  # Left_Ankle_Pitch
                    5.0,  # Left_Ankle_Roll
                    # Right leg (6 DOFs)
                    0.01,  # Right_Hip_Pitch
                    1.0,  # Right_Hip_Roll
                    5.0,  # Right_Hip_Yaw
                    2.0,  # Right_Knee_Pitch — discourage over-bending
                    5.0,  # Right_Ankle_Pitch
                    5.0,  # Right_Ankle_Roll
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

__all__ = [
    "k1_22dof_agile_loco",
    "k1_22dof_agile_loco_fast_sac",
    "k1_22dof_loco",
    "k1_22dof_loco_fast_sac",
    "k1_22dof_soccer_loco",
    "k1_22dof_soccer_run_loco",
]

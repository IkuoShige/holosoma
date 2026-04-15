"""Locomotion command presets for the K1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

k1_22dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.0, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
)

k1_22dof_agile_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 8.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                # Base cadence ~1.82 Hz; scales up with speed for smooth low-to-high velocity transitions.
                "gait_period": 0.55,
                "gait_period_randomization_width": 0.05,
                "adaptive_gait_freq_from_lin_speed": 0.6,
                "adaptive_gait_freq_from_yaw_speed": 0.2,
                "adaptive_gait_freq_min": 1.2,
                "adaptive_gait_freq_max": 3.0,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.5, 2.5],
                    "lin_vel_y": [-1.2, 1.2],
                    "ang_vel_yaw": [-1.5, 1.5],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
                # High-speed soccer-like cases should emphasize forward + yaw while keeping vy available.
                "combined_motion_prob": 0.3,
                "combined_lin_vel_x_range": [1.4, 2.5],
                "combined_lin_vel_y_range": [-0.6, 0.6],
                "combined_ang_vel_yaw_abs_range": [0.6, 1.5],
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
)

k1_22dof_soccer_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 7.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                # Walk gait: ~2.22 Hz base, scales to ~2.72 Hz at 1.0 m/s.
                "gait_period": 0.45,
                "gait_period_randomization_width": 0.03,
                "adaptive_gait_freq_from_lin_speed": 0.5,
                "adaptive_gait_freq_from_yaw_speed": 0.15,
                "adaptive_gait_freq_min": 1.8,
                "adaptive_gait_freq_max": 3.5,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.5, 1.5],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.15,
                "combined_motion_prob": 0.35,
                "combined_lin_vel_x_range": [0.5, 1.0],
                "combined_lin_vel_y_range": [-0.8, 0.8],
                "combined_ang_vel_yaw_abs_range": [0.5, 1.5],
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
)

k1_22dof_soccer_run_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 8.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                # Run gait: ~2.5 Hz base, scales to ~3.25 Hz at 1.5 m/s.
                "gait_period": 0.40,
                "gait_period_randomization_width": 0.03,
                "adaptive_gait_freq_from_lin_speed": 0.5,
                "adaptive_gait_freq_from_yaw_speed": 0.1,
                "adaptive_gait_freq_min": 2.0,
                "adaptive_gait_freq_max": 4.0,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [0.8, 1.5],
                    "lin_vel_y": [-0.5, 0.5],
                    "ang_vel_yaw": [-0.8, 0.8],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.0,
                "combined_motion_prob": 0.2,
                "combined_lin_vel_x_range": [1.0, 1.5],
                "combined_lin_vel_y_range": [-0.4, 0.4],
                "combined_ang_vel_yaw_abs_range": [0.3, 0.8],
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
)

__all__ = [
    "k1_22dof_agile_command",
    "k1_22dof_command",
    "k1_22dof_soccer_command",
    "k1_22dof_soccer_run_command",
]

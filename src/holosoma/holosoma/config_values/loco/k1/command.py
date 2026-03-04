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
                "adaptive_gait_freq_from_lin_speed": 0.5,
                "adaptive_gait_freq_from_yaw_speed": 0.2,
                "adaptive_gait_freq_min": 1.2,
                "adaptive_gait_freq_max": 2.5,
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

__all__ = ["k1_22dof_agile_command", "k1_22dof_command"]

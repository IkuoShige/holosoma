"""Dribble command presets for the K1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

k1_22dof_ball_dribble_command = CommandManagerCfg(
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "dribble_command": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:DribbleCommand",
            params={
                "speed_range": [0.1, 3.0],
                "resampling_time_range": [3.0, 8.0],
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "dribble_command": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:DribbleCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "dribble_command": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:DribbleCommand",
        ),
    },
)

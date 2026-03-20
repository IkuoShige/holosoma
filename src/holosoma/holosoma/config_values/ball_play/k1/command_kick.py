"""Kick command presets for the K1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

k1_22dof_ball_kick_command = CommandManagerCfg(
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "kick_target": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:KickTargetCommand",
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "kick_target": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:KickTargetCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
        ),
        "kick_target": CommandTermCfg(
            func="holosoma.managers.command.terms.ball_play:KickTargetCommand",
        ),
    },
)

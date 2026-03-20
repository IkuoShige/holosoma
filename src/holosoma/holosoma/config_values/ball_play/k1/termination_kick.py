"""Kick termination presets for the K1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

k1_22dof_ball_kick_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "low_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold",
            params={"min_height": 0.45},
        ),
        "ball_still_too_long": TerminationTermCfg(
            func="holosoma.managers.termination.terms.ball_play:ball_still_too_long",
            params={"max_still_time": 2.0},
        ),
        "ball_moving_too_long": TerminationTermCfg(
            func="holosoma.managers.termination.terms.ball_play:ball_moving_too_long",
            params={"max_moving_time": 5.0},
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

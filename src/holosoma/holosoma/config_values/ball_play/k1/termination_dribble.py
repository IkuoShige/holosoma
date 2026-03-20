"""Dribble termination presets for the K1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

k1_22dof_ball_dribble_termination = TerminationManagerCfg(
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
            params={"min_height": 0.35},
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

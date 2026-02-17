"""Locomotion termination presets for the K1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

k1_22dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

k1_22dof_agile_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "gravity_tilt": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:gravity_tilt_exceeded",
            params={
                "threshold_x": 0.7,
                "threshold_y": 0.7,
            },
        ),
        "low_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold",
            params={
                "min_height": 0.3,
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["k1_22dof_agile_termination", "k1_22dof_termination"]

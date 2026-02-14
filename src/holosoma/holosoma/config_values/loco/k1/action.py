"""Locomotion action presets for the K1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

k1_22dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["k1_22dof_joint_pos"]

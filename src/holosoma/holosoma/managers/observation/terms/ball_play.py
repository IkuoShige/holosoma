"""Ball play observation terms.

Observation functions for ball kick and dribble tasks.
Each function takes the environment instance and returns a tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.envs.ball_play.ball_kick_task import BallKickTask
    from holosoma.envs.ball_play.ball_dribble_task import BallDribbleTask


def ball_pos_relative(env: BallKickTask | BallDribbleTask) -> torch.Tensor:
    """Ball XY position in robot local frame.

    Returns:
        Tensor of shape [num_envs, 2]
    """
    return env.ball_pos_relative[:, :2]


def ball_pos_relative_perceived(env: BallDribbleTask) -> torch.Tensor:
    """Perceived ball XY position in robot local frame (delayed detection).

    Returns:
        Tensor of shape [num_envs, 2]
    """
    return env.perceived_ball_pos_relative


def last_ball_pos_relative_perceived(env: BallDribbleTask) -> torch.Tensor:
    """Previous perceived ball XY position in robot local frame.

    Returns:
        Tensor of shape [num_envs, 2]
    """
    return env.last_perceived_ball_pos_relative


def kick_target_direction(env: BallKickTask) -> torch.Tensor:
    """Kick target direction as (cos, sin).

    Returns:
        Tensor of shape [num_envs, 2]
    """
    return env.kick_target_dir


def _get_dribble_commands(env: BallDribbleTask) -> torch.Tensor:
    """Get dribble command tensor from DribbleCommand term.

    Returns:
        Tensor of shape [num_envs, 2] (vx, vy in world frame)
    """
    state = env.command_manager.get_state("dribble_command")
    if state is not None and hasattr(state, "commands") and state.commands is not None:
        return state.commands
    return torch.zeros(env.num_envs, 2, device=env.device)


def dribble_target_direction_local(env: BallDribbleTask) -> torch.Tensor:
    """Dribble target direction in robot local frame.

    Converts world-frame dribble command to robot-local frame.

    Returns:
        Tensor of shape [num_envs, 2]
    """
    base_quat = env.base_quat
    commands = _get_dribble_commands(env)
    world_dir = torch.zeros(env.num_envs, 3, device=env.device)
    world_dir[:, 0] = commands[:, 0]
    world_dir[:, 1] = commands[:, 1]
    local_dir = quat_rotate_inverse(base_quat, world_dir, w_last=True)
    return local_dir[:, :2]


def ball_vel_world(env: BallKickTask | BallDribbleTask) -> torch.Tensor:
    """Ball XY velocity in world frame (privileged observation for critic).

    Returns:
        Tensor of shape [num_envs, 2]
    """
    return env.ball_vel[:, :2]

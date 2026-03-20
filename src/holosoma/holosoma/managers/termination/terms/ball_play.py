"""Ball play termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from holosoma.envs.ball_play.ball_kick_task import BallKickTask
    from holosoma.envs.ball_play.ball_dribble_task import BallDribbleTask


def ball_still_too_long(
    env: BallKickTask,
    max_still_time: float = 2.0,
) -> torch.Tensor:
    """Terminate if ball has been stationary for too long.

    Args:
        env: The environment instance
        max_still_time: Maximum allowed stationary time (seconds)

    Returns:
        Boolean mask [num_envs]
    """
    return env.time_ball_still > max_still_time


def ball_moving_too_long(
    env: BallKickTask,
    max_moving_time: float = 5.0,
) -> torch.Tensor:
    """Terminate if ball has been moving for too long after kick.

    Args:
        env: The environment instance
        max_moving_time: Maximum allowed moving time (seconds)

    Returns:
        Boolean mask [num_envs]
    """
    return env.time_ball_moving > max_moving_time


def ball_too_far(
    env: BallDribbleTask,
    max_distance: float = 3.0,
) -> torch.Tensor:
    """Flag when ball is too far from robot.

    Note: This is used for ball-only reset, not full episode termination.
    The environment class handles the ball-only reset logic.

    Args:
        env: The environment instance
        max_distance: Maximum allowed distance (m)

    Returns:
        Boolean mask [num_envs]
    """
    robot_pos_xy = env.simulator.robot_root_states[:, :2]
    ball_pos_xy = env.ball_pos[:, :2]
    distance = torch.norm(ball_pos_xy - robot_pos_xy, dim=-1)
    return distance > max_distance

"""Ball play reward terms.

Reward functions for ball kick and dribble tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_apply

if TYPE_CHECKING:
    from holosoma.envs.ball_play.ball_kick_task import BallKickTask
    from holosoma.envs.ball_play.ball_dribble_task import BallDribbleTask


# =============================================================================
# Kick Task Rewards
# =============================================================================


def ball_velocity_target_direction(
    env: BallKickTask,
    decay_time: float = 0.1,
    max_reward: float = 10.0,
) -> torch.Tensor:
    """Reward ball velocity projected onto kick target direction.

    Applies exponential time decay after the ball starts moving to avoid
    rewarding residual rolling.

    Args:
        env: The environment instance
        decay_time: Time constant for exponential decay (seconds)
        max_reward: Maximum clamp value

    Returns:
        Reward tensor [num_envs]
    """
    ball_vel_xy = env.ball_vel[:, :2]
    target_dir = env.kick_target_dir

    vel_toward_target = torch.sum(ball_vel_xy * target_dir, dim=-1)

    decay_factor = torch.exp(-env.time_ball_moving / decay_time)

    reward = vel_toward_target * decay_factor
    return torch.clamp(reward, min=0.0, max=max_reward)


def ball_acceleration_toward_target(
    env: BallKickTask,
    scale: float = 80.0,
    max_reward: float = 80.0,
) -> torch.Tensor:
    """Reward ball acceleration in the kick target direction.

    Args:
        env: The environment instance
        scale: Scaling factor for tanh
        max_reward: Maximum clamp value

    Returns:
        Reward tensor [num_envs]
    """
    current_vel = env.ball_vel[:, :2]
    prev_vel = env.last_ball_vel[:, :2]

    dt = max(env.dt, 1e-6)
    ball_accel = (current_vel - prev_vel) / dt
    target_dir = env.kick_target_dir

    accel_toward = torch.sum(ball_accel * target_dir, dim=-1)
    accel_lateral = torch.norm(
        ball_accel - accel_toward.unsqueeze(-1) * target_dir, dim=-1
    )
    effective_accel = accel_toward - torch.abs(accel_lateral)

    reward = torch.tanh(torch.clamp(effective_accel, min=0.0) / scale) * max_reward
    return reward


def kicking_foot_approach_ball(
    env: BallKickTask,
    proximity_sigma: float = 0.1,
    stationary_threshold: float = 0.1,
    max_reward: float = 50.0,
) -> torch.Tensor:
    """Reward foot proximity to stationary ball.

    Only active when ball speed is below stationary_threshold.

    Args:
        env: The environment instance
        proximity_sigma: Sigma for exponential proximity reward
        stationary_threshold: Max ball speed to be considered stationary (m/s)
        max_reward: Maximum clamp value

    Returns:
        Reward tensor [num_envs]
    """
    ball_pos_world = env.ball_pos
    ball_speed = torch.norm(env.ball_vel[:, :2], dim=-1)

    left_foot_pos = env.simulator._rigid_body_pos[:, env.feet_indices[0], :]
    right_foot_pos = env.simulator._rigid_body_pos[:, env.feet_indices[1], :]

    dist_left = torch.norm(left_foot_pos - ball_pos_world, dim=-1)
    dist_right = torch.norm(right_foot_pos - ball_pos_world, dim=-1)
    foot_ball_dist = torch.min(dist_left, dist_right)

    proximity = torch.exp(-foot_ball_dist / proximity_sigma)

    is_stationary = (ball_speed < stationary_threshold).float()
    reward = proximity * is_stationary

    return torch.clamp(reward, min=0.0, max=max_reward)


def body_alignment_for_kick(
    env: BallKickTask,
    sigma: float = 0.5,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Reward alignment of robot forward direction toward kick target.

    Args:
        env: The environment instance
        sigma: Sigma for exponential alignment reward
        max_reward: Maximum clamp value

    Returns:
        Reward tensor [num_envs]
    """
    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    robot_forward = quat_apply(env.base_quat, forward_vec, w_last=True)

    alignment = torch.sum(robot_forward[:, :2] * env.kick_target_dir, dim=-1)

    reward = torch.exp((alignment - 1.0) / sigma)
    return torch.clamp(reward, min=0.0, max=max_reward)


def waiting_penalty(
    env: BallKickTask,
    max_still_time: float = 2.0,
) -> torch.Tensor:
    """Quadratic time penalty to encourage early kicks.

    Args:
        env: The environment instance
        max_still_time: Time normalization constant (seconds)

    Returns:
        Reward tensor [num_envs] (positive values, applied with negative weight)
    """
    max_steps = max_still_time / env.dt
    progress = env.episode_length_buf.float() / max_steps
    return progress * progress


# =============================================================================
# Dribble Task Rewards
# =============================================================================


def ball_velocity_tracking(
    env: BallDribbleTask,
    sigma: float = 1.0,
    min_speed: float = 0.1,
) -> torch.Tensor:
    """Reward ball velocity tracking: direction (cosine sim) * speed matching.

    Args:
        env: The environment instance
        sigma: Sigma for speed matching exponential
        min_speed: Minimum ball speed to get reward

    Returns:
        Reward tensor [num_envs]
    """
    actual_vel = env.ball_vel[:, :2]
    target_vel = env.command_manager.commands[:, :2]

    actual_speed = torch.norm(actual_vel, dim=-1)
    target_speed = torch.norm(target_vel, dim=-1)

    dot_product = torch.sum(actual_vel * target_vel, dim=-1)
    cos_sim = dot_product / (actual_speed * target_speed + 1e-8)

    speed_error = torch.abs(actual_speed - target_speed)
    speed_reward = torch.exp(-speed_error / sigma)

    reward = cos_sim * speed_reward

    is_moving = (actual_speed > min_speed).float()
    has_target = (target_speed > 1e-6).float()

    return reward * is_moving * has_target


def ball_distance_penalty(
    env: BallDribbleTask,
    sigma: float = 1.0,
    max_dist: float = 3.0,
) -> torch.Tensor:
    """Penalty for distance from robot to optimal dribble position.

    Args:
        env: The environment instance
        sigma: Sigma for exponential penalty
        max_dist: Max distance clamp (m)

    Returns:
        Reward tensor [num_envs] (positive values, applied with negative weight)
    """
    robot_pos_xy = env.simulator.robot_root_states[:, :2]
    ball_pos_xy = env.ball_pos[:, :2]

    commands = env.command_manager.commands[:, :2]
    command_norm = torch.norm(commands, dim=-1, keepdim=True)
    normed_dir = commands / (command_norm + 1e-8)
    target_pos = ball_pos_xy - 0.175 * normed_dir

    distance = torch.norm(target_pos - robot_pos_xy, dim=-1)
    distance = torch.clamp(distance, min=0.0, max=max_dist)

    return torch.exp(distance / sigma) - 1.0


def look_at_ball(
    env: BallDribbleTask,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Reward for robot yaw aligned toward ball.

    Args:
        env: The environment instance
        sigma: Sigma for exponential reward

    Returns:
        Reward tensor [num_envs]
    """
    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    robot_forward = quat_apply(env.base_quat, forward_vec, w_last=True)
    robot_yaw = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])

    ball_dir = env.ball_pos[:, :2] - env.simulator.robot_root_states[:, :2]
    angle_to_ball = torch.atan2(ball_dir[:, 1], ball_dir[:, 0])

    angle_error = (robot_yaw - angle_to_ball + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.exp(-torch.square(angle_error) / sigma)

"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _resolve_body_index(
    env: LeggedRobotLocomotionManager,
    *,
    body_name: str,
    fallback_contains: str | None = None,
    cache_key: str,
) -> int:
    cache_attr = f"_reward_cached_body_index_{cache_key}"
    cached = getattr(env, cache_attr, None)
    if isinstance(cached, int) and cached >= 0:
        return cached

    chosen_name: str | None = None
    if body_name in env.body_names:
        chosen_name = body_name
    elif fallback_contains is not None:
        for candidate in env.body_names:
            if fallback_contains in candidate:
                chosen_name = candidate
                break

    if chosen_name is None:
        raise ValueError(
            f"Could not resolve body index for body_name='{body_name}' (fallback_contains={fallback_contains})."
        )

    index = int(env.simulator.find_rigid_body_indice(chosen_name))
    setattr(env, cache_attr, index)
    return index


def _expected_foot_height(phi: torch.Tensor, swing_height: float | torch.Tensor) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    if isinstance(swing_height, torch.Tensor):
        swing_height_tensor = torch.ones_like(x) * swing_height.to(device=phi.device, dtype=phi.dtype)
    else:
        swing_height_tensor = torch.full_like(x, swing_height)

    stance = cubic_bezier_interpolation(torch.zeros_like(x), swing_height_tensor, 2 * x)
    swing = cubic_bezier_interpolation(swing_height_tensor, torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_action_rate_l1(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions using L1 norm.

    Unlike L2, L1 provides constant gradient near zero, effectively
    suppressing small oscillations that L2 ignores.
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.abs(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def penalty_feet_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def _get_push_velocity_compensation_xy(
    env,
    *,
    tau_s: float,
    cutoff_s: float,
) -> torch.Tensor:
    if tau_s <= 0.0:
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    if not hasattr(env, "record_push_robot_vel_buf"):
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)

    push_world_xy = env.record_push_robot_vel_buf
    if (
        not isinstance(push_world_xy, torch.Tensor)
        or push_world_xy.shape[0] != env.num_envs
        or push_world_xy.shape[1] < 2
    ):
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)

    # Push impulses are injected as world-frame base linear velocities in XY.
    push_world = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    push_world[:, :2] = push_world_xy[:, :2]
    push_base = quat_rotate_inverse(env.base_quat, push_world, w_last=True)[:, :2]

    decay = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    if hasattr(env, "randomization_manager") and env.randomization_manager is not None:
        state = env.randomization_manager.get_state("push_randomizer_state")
        if state is not None and getattr(state, "push_robot_counter", None) is not None:
            elapsed_s = state.push_robot_counter.float() * env.dt
            decay = torch.exp(-elapsed_s / tau_s)
            if cutoff_s > 0.0:
                decay = torch.where(elapsed_s <= cutoff_s, decay, torch.zeros_like(decay))

    return push_base * decay.unsqueeze(1)


def tracking_lin_vel(
    env,
    tracking_sigma: float = 0.25,
    push_compensation_tau_s: float = 0.0,
    push_compensation_cutoff_s: float = 0.0,
    push_compensation_max_speed: float = 0.0,
    push_compensation_max_cmd_ratio: float = 0.0,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling
        push_compensation_tau_s: Exponential decay constant for disturbance velocity compensation (0 disables)
        push_compensation_cutoff_s: Max elapsed time window after push to apply compensation (0 disables cutoff)
        push_compensation_max_speed: Absolute cap for compensated speed magnitude
        push_compensation_max_cmd_ratio: Additional cap proportional to command speed

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    measured_lin_vel = get_base_lin_vel(env)[:, :2]
    if push_compensation_tau_s > 0.0:
        compensated = _get_push_velocity_compensation_xy(
            env,
            tau_s=push_compensation_tau_s,
            cutoff_s=push_compensation_cutoff_s,
        )
        if push_compensation_max_speed > 0.0 or push_compensation_max_cmd_ratio > 0.0:
            comp_norm = torch.linalg.norm(compensated, dim=1).clamp(min=1e-6)
            cmd_norm = torch.linalg.norm(commands[:, :2], dim=1)
            max_allowed = push_compensation_max_speed + push_compensation_max_cmd_ratio * cmd_norm
            scale = torch.clamp(max_allowed / comp_norm, max=1.0)
            compensated = compensated * scale.unsqueeze(1)
        measured_lin_vel = measured_lin_vel - compensated

    lin_vel_error = torch.sum(torch.square(commands[:, :2] - measured_lin_vel), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalty_head_ang_vel_xy(
    env,
    head_body_name: str = "Head_2",
    fallback_contains: str = "Head",
    deadzone: float = 0.0,
) -> torch.Tensor:
    """Penalize head pitch/roll angular velocity (camera blur proxy).

    Args:
        env: The environment instance
        head_body_name: Preferred rigid body name for the camera/head link
        fallback_contains: Fallback substring used if exact body name is not found
        deadzone: Ignore angular velocity magnitudes below this threshold (rad/s)

    Returns:
        Penalty tensor [num_envs]
    """
    head_idx = _resolve_body_index(
        env,
        body_name=head_body_name,
        fallback_contains=fallback_contains,
        cache_key="head_ang_vel_xy",
    )
    head_ang_vel_world = env.simulator._rigid_body_ang_vel[:, head_idx, :2]
    head_ang_speed_xy = torch.linalg.norm(head_ang_vel_world, dim=1)
    if deadzone > 0.0:
        head_ang_speed_xy = (head_ang_speed_xy - deadzone).clip(min=0.0)
    return torch.square(head_ang_speed_xy)


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def penalty_far_feet_xy(env, far_feet_threshold: float = 0.35) -> torch.Tensor:
    """Penalize excessively wide foot spacing in the lateral direction.

    Args:
        env: The environment instance
        far_feet_threshold: Maximum desired lateral distance between feet

    Returns:
        Penalty tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Penalize width above threshold (hinge-squared)
    far_error = (feet_distance - far_feet_threshold).clip(min=0.0)
    return torch.square(far_error)


def base_height(
    env, desired_base_height: float = 0.89, zero_vel_penalty_scale: float = 1.0, stance_penalty_scale: float = 1.0
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    base_height_penalty = torch.square(
        env.terrain_manager.get_state("locomotion_terrain").base_heights - desired_base_height
    )

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(
    env,
    swing_height: float = 0.08,
    tracking_sigma: float = 0.25,
    dynamic_swing_height_from_lin_speed: float = 0.0,
    dynamic_swing_height_from_yaw_speed: float = 0.0,
    dynamic_swing_height_from_gait_freq: float = 0.0,
    dynamic_swing_height_min: float | None = None,
    dynamic_swing_height_max: float | None = None,
) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Base maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling
        dynamic_swing_height_from_lin_speed: Extra swing height gain from |v_xy| command
        dynamic_swing_height_from_yaw_speed: Extra swing height gain from |yaw| command
        dynamic_swing_height_from_gait_freq: Extra swing height gain from gait frequency above nominal
        dynamic_swing_height_min: Optional lower clamp for dynamic swing height
        dynamic_swing_height_max: Optional upper clamp for dynamic swing height

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Build per-env swing height (optionally speed/frequency adaptive)
    swing_height_tensor = torch.full_like(foot_z_left, float(swing_height))
    if dynamic_swing_height_from_lin_speed > 0.0 or dynamic_swing_height_from_yaw_speed > 0.0:
        commands = env.command_manager.commands
        lin_speed = torch.linalg.norm(commands[:, :2], dim=1)
        yaw_speed = torch.abs(commands[:, 2])
        swing_height_tensor += (
            dynamic_swing_height_from_lin_speed * lin_speed + dynamic_swing_height_from_yaw_speed * yaw_speed
        )

    if dynamic_swing_height_from_gait_freq > 0.0:
        gait_state = env.command_manager.get_state("locomotion_gait")
        if gait_state is not None and getattr(gait_state, "gait_freq", None) is not None:
            gait_freq = gait_state.gait_freq.squeeze(1)
            nominal_freq = float(getattr(gait_state, "mean_gait_freq", 0.0))
            swing_height_tensor += dynamic_swing_height_from_gait_freq * torch.clamp(gait_freq - nominal_freq, min=0.0)

    min_height = swing_height if dynamic_swing_height_min is None else dynamic_swing_height_min
    if dynamic_swing_height_max is not None:
        swing_height_tensor = torch.clamp(swing_height_tensor, min=min_height, max=dynamic_swing_height_max)
    else:
        swing_height_tensor = torch.clamp(swing_height_tensor, min=min_height)

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height_tensor)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height_tensor)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def feet_phase_gated(
    env,
    swing_height: float = 0.08,
    tracking_sigma: float = 0.25,
    dynamic_swing_height_from_lin_speed: float = 0.0,
    dynamic_swing_height_from_yaw_speed: float = 0.0,
    dynamic_swing_height_from_gait_freq: float = 0.0,
    dynamic_swing_height_min: float | None = None,
    dynamic_swing_height_max: float | None = None,
    cmd_speed_gate_threshold: float = 0.15,
    cmd_speed_gate_ramp: float = 0.15,
) -> torch.Tensor:
    """Reward for tracking foot height, gated by command speed.

    When commanded speed is near zero, the reward is clamped to 1.0 so
    the agent has no incentive to keep stepping in place.  As commanded
    speed ramps above *cmd_speed_gate_threshold*, the normal feet_phase
    reward activates smoothly over *cmd_speed_gate_ramp*.

    All feet_phase parameters are forwarded unchanged.
    """
    base_reward = feet_phase(
        env,
        swing_height=swing_height,
        tracking_sigma=tracking_sigma,
        dynamic_swing_height_from_lin_speed=dynamic_swing_height_from_lin_speed,
        dynamic_swing_height_from_yaw_speed=dynamic_swing_height_from_yaw_speed,
        dynamic_swing_height_from_gait_freq=dynamic_swing_height_from_gait_freq,
        dynamic_swing_height_min=dynamic_swing_height_min,
        dynamic_swing_height_max=dynamic_swing_height_max,
    )

    commands = env.command_manager.commands
    cmd_speed = torch.linalg.norm(commands[:, :2], dim=1) + 0.5 * torch.abs(commands[:, 2])

    gate = torch.clamp((cmd_speed - cmd_speed_gate_threshold) / max(cmd_speed_gate_ramp, 1e-6), 0.0, 1.0)

    return gate * base_reward + (1.0 - gate) * 1.0


def reward_standstill(
    env,
    cmd_speed_threshold: float = 0.15,
    vel_sigma: float = 0.25,
    dof_vel_sigma: float = 4.0,
) -> torch.Tensor:
    """Reward low base and joint velocities when command is near zero.

    Active only when commanded speed < *cmd_speed_threshold*.
    Returns 0 when the robot is commanded to move.

    The reward is the product of two exp-decay terms:
    - base velocity error (linear XY + angular yaw)
    - joint velocity magnitude
    """
    commands = env.command_manager.commands
    cmd_speed = torch.linalg.norm(commands[:, :2], dim=1) + 0.5 * torch.abs(commands[:, 2])
    standstill_mask = cmd_speed < cmd_speed_threshold

    base_lin_vel = get_base_lin_vel(env)[:, :2]
    base_ang_vel = get_base_ang_vel(env)[:, 2]
    vel_error = torch.sum(torch.square(base_lin_vel), dim=1) + torch.square(base_ang_vel)
    vel_reward = torch.exp(-vel_error / vel_sigma)

    dof_vel = env.simulator.dof_vel
    dof_vel_error = torch.sum(torch.square(dof_vel), dim=1)
    dof_vel_reward = torch.exp(-dof_vel_error / dof_vel_sigma)

    reward = vel_reward * dof_vel_reward
    return torch.where(standstill_mask, reward, torch.zeros_like(reward))


def stride_pitch_coupling(
    env,
    tracking_sigma: float = 0.02,
    min_cmd_speed: float = 0.2,
    base_stride: float = 0.10,
    stride_from_lin_speed: float = 0.08,
    stride_from_yaw_speed: float = 0.03,
    stride_from_gait_freq: float = 0.05,
    min_stride: float = 0.08,
    max_stride: float = 0.35,
    actual_speed_gate_ratio: float = 0.0,
    actual_speed_gate_threshold: float = 0.2,
    actual_speed_gate_vy_weight: float = 0.35,
) -> torch.Tensor:
    """Reward speed-dependent stride with phase-locked fore-aft foot cycling.

    This term encourages:
    1) larger stride as commanded translational speed increases
    2) actual leg cycling (foot x in base frame follows gait phase), not static front/back split
    """
    commands = env.command_manager.commands
    # Favor forward component for stride targets while keeping some lateral contribution.
    lin_speed = torch.abs(commands[:, 0]) + 0.35 * torch.abs(commands[:, 1])
    yaw_speed = torch.abs(commands[:, 2])

    left_foot = env.simulator._rigid_body_pos[:, env.feet_indices[0], :]
    right_foot = env.simulator._rigid_body_pos[:, env.feet_indices[1], :]
    base_pos = env.simulator.robot_root_states[:, :3]
    left_rel_base = quat_rotate_inverse(env.base_quat, left_foot - base_pos, w_last=True)
    right_rel_base = quat_rotate_inverse(env.base_quat, right_foot - base_pos, w_last=True)
    left_x = left_rel_base[:, 0]
    right_x = right_rel_base[:, 0]

    desired_stride = torch.full_like(left_x, float(base_stride))
    desired_stride += stride_from_lin_speed * lin_speed + stride_from_yaw_speed * yaw_speed

    if stride_from_gait_freq > 0.0:
        gait_state = env.command_manager.get_state("locomotion_gait")
        if gait_state is not None and getattr(gait_state, "gait_freq", None) is not None:
            gait_freq = gait_state.gait_freq.squeeze(1)
            nominal_freq = float(getattr(gait_state, "mean_gait_freq", 0.0))
            desired_stride += stride_from_gait_freq * torch.clamp(gait_freq - nominal_freq, min=0.0)

    desired_stride = torch.clamp(desired_stride, min=min_stride, max=max_stride)

    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is not None and getattr(gait_state, "phase", None) is not None:
        phase = gait_state.phase
        # Each foot should oscillate around the base with half of desired stride amplitude.
        half_stride = 0.5 * desired_stride
        left_target_x = half_stride * torch.sin(phase[:, 0])
        right_target_x = half_stride * torch.sin(phase[:, 1])
        stride_error = torch.square(left_x - left_target_x) + torch.square(right_x - right_target_x)
    else:
        # Fallback if gait state is unavailable.
        stride = torch.abs(left_x - right_x)
        stride_error = torch.square(stride - desired_stride)

    moving_mask = (lin_speed + 0.5 * yaw_speed) > min_cmd_speed
    reward = torch.exp(-stride_error / tracking_sigma)
    reward = torch.where(moving_mask, reward, torch.ones_like(reward))

    if actual_speed_gate_ratio > 0.0:
        base_lin_vel = get_base_lin_vel(env)
        actual_speed = torch.abs(base_lin_vel[:, 0]) + actual_speed_gate_vy_weight * torch.abs(base_lin_vel[:, 1])
        desired_min_speed = actual_speed_gate_ratio * lin_speed
        speed_gate = torch.ones_like(reward)
        active_mask = lin_speed > actual_speed_gate_threshold
        gated = torch.clamp(actual_speed / (desired_min_speed + 1e-6), min=0.0, max=1.0)
        speed_gate = torch.where(active_mask, gated, speed_gate)
        reward = reward * speed_gate

    return reward


def penalty_stall_when_commanded(
    env,
    command_speed_threshold: float = 0.35,
    min_speed_ratio: float = 0.35,
    vy_speed_weight: float = 0.35,
) -> torch.Tensor:
    """Penalize failing to translate when significant translational command is present."""
    commands = env.command_manager.commands
    commanded_speed = torch.abs(commands[:, 0]) + vy_speed_weight * torch.abs(commands[:, 1])
    actual_lin = get_base_lin_vel(env)
    actual_speed = torch.abs(actual_lin[:, 0]) + vy_speed_weight * torch.abs(actual_lin[:, 1])

    desired_min_speed = min_speed_ratio * commanded_speed
    speed_shortfall = (desired_min_speed - actual_speed).clip(min=0.0)
    penalty = torch.square(speed_shortfall)
    active_mask = commanded_speed > command_speed_threshold
    return torch.where(active_mask, penalty, torch.zeros_like(penalty))


def pose(
    env,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


def penalty_stumble(env) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, env.feet_indices, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def penalty_foothold(env, foothold_epsilon: float = 0.01) -> torch.Tensor:
    """Sampling-based foothold penalty.

    For each foot in contact, sample a grid of points on the sole, transform to world,
    read terrain height at those XY, compute depth d_ij = z_sample - terrain_z, and count
    samples with d_ij < epsilon. Sum over both feet.

    Args:
        env: The environment instance
        foothold_epsilon: Threshold for foothold depth penalty

    Returns:
        Reward tensor [num_envs]
    """
    # Contact mask per foot
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E,2]
    if not (contact.any()):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Accumulator
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for foot_idx_local in range(2):
        # Skip if no env has contact on this foot to save work
        if not contact[:, foot_idx_local].any():
            continue
        rb_idx = env.feet_indices[foot_idx_local]
        foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]  # [E,3]
        foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]  # [E,4]

        # Use precomputed sample points in the foot frame
        pts_local = env.foot_samples_local[foot_idx_local].unsqueeze(0).repeat(env.num_envs, 1, 1)

        # Rotate to world and translate
        pts_world = quat_rotate_batched(foot_quat_w, pts_local) + foot_pos_w.unsqueeze(1)

        # Query terrain height at those XY positions
        terrain_h = env._get_terrain_heights_at_points_world(pts_world)

        # Depth: world z minus terrain height
        depth = pts_world[:, :, 2] - terrain_h  # [E,S]

        # Indicator for d_ij > epsilon, only for envs with this foot in contact
        bad = (depth > foothold_epsilon).float()
        bad *= contact[:, foot_idx_local].unsqueeze(1).float()

        penalty += torch.sum(bad, dim=1)

    return penalty / env.num_foot_samples


def alive(env) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)

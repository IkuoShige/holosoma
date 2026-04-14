"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.reward import RewardTermCfg
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
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


def tracking_lin_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
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


def feet_phase(env, swing_height: float = 0.08, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


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


def penalty_stand_still(env, command_threshold: float = 0.1) -> torch.Tensor:
    """Penalize joint deviation from default pose when command is near zero.

    v45: directly tells the policy "don't move when commanded to stand."
    Only active when ||cmd[:3]|| < command_threshold. Walking is unaffected.
    Ported from mujoco_playground T1's _cost_stand_still.

    Args:
        env: The environment instance
        command_threshold: Command norm below which penalty applies

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    stand_mask = (cmd_norm < command_threshold).float()
    qpos_error = torch.sum(torch.abs(env.simulator.dof_pos - env.default_dof_pos), dim=1)
    return qpos_error * stand_mask


# ================================================================================================
# Stateful reward terms
# ================================================================================================


class FeetAirTime(RewardTermBase):
    """Reward swing-leg air time per step (continuous).

    v26: switched from the T1-style discrete first_contact reward to a
    continuous per-step reward. The discrete version fired only at
    landing moments (~2 Hz), which gave ~1/120 the effective signal
    strength of feet_phase (~50 Hz continuous) at the same weight.
    v24/v25 diagnostics showed feet_air_time episode reward was 118x
    smaller than feet_phase even with weight 4.0 vs 4.0 — the discrete
    firing rate dominated.

    v26 per-step reward: while a foot is airborne, reward the current
    swing duration (clipped at threshold_max). The reward grows linearly
    with air_time while the foot is off the ground, giving continuous
    gradient for "stay airborne longer". When foot lands, reward drops
    to 0 and air_time resets.

    Params:
        threshold_max: Maximum air time rewarded per step. Air time
            above this is clipped. Default 0.5.
        contact_force_threshold: Normal-force threshold (Newtons, z-axis)
            for detecting ground contact on a foot. Default 5.0. v27
            switched from full-3D-norm to z-component only to match the
            convention used by ``feet_phase`` and ``penalty_foothold``.
        command_norm_threshold: Minimum command magnitude below which no
            reward is given. Default 0.1.

    (threshold_min from the v22-v25 discrete version is removed; v25
    already set it to 0.0, and per-step rewards have no deadband
    semantics.)
    """

    def __init__(self, cfg: "RewardTermCfg", env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.threshold_max = float(params.get("threshold_max", 0.5))
        self.contact_force_threshold = float(params.get("contact_force_threshold", 5.0))
        self.command_norm_threshold = float(params.get("command_norm_threshold", 0.1))

        num_envs = env.num_envs
        num_feet = int(env.feet_indices.shape[0])
        device = env.device

        # Per-env, per-foot air time in seconds (grows while airborne).
        self._air_time = torch.zeros(num_envs, num_feet, dtype=torch.float32, device=device)

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        # v27: use z-component (normal force) for contact detection, matching
        # feet_phase / penalty_foothold convention elsewhere in holosoma.
        # v26 used `norm(3D)` which summed lateral forces from swing motion
        # (torque reactions, air drag) into the threshold, causing spurious
        # contact triggers during swing phases. This reset air_time frequently
        # and suppressed the reward signal (observed 0.06 vs predicted 0.35).
        normal_force = env.simulator.contact_forces[:, env.feet_indices, 2]  # [E, F]
        current_contact = normal_force > self.contact_force_threshold  # [E, F] bool
        airborne = (~current_contact).float()  # [E, F]

        # Continuous per-step reward: while airborne, reward = clipped
        # air_time. Grows as swing progresses; drops to 0 on landing.
        clipped_air_time = torch.clamp(self._air_time, min=0.0, max=self.threshold_max)
        per_foot_reward = clipped_air_time * airborne
        reward = torch.sum(per_foot_reward, dim=1)  # [E]

        # Gate by command norm: no reward when standing still command.
        commands = env.command_manager.commands
        cmd_norm = torch.norm(commands[:, :3], dim=1)  # lin_x, lin_y, ang_z
        reward = reward * (cmd_norm > self.command_norm_threshold).float()

        # Advance state for next step: air time increments while airborne,
        # resets on contact. Use env.dt for physical seconds.
        self._air_time = torch.where(
            current_contact,
            torch.zeros_like(self._air_time),
            self._air_time + float(env.dt),
        )

        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            self._air_time.zero_()
        else:
            self._air_time[env_ids] = 0.0


class StrideProgress(RewardTermBase):
    """Reward forward foot displacement during swing (ankle-flick-proof).

    v28: Codex-recommended reward that directly measures how far each
    foot moves forward (in the base heading frame) relative to the body
    during the swing phase. Unlike ``FeetAirTime`` which rewards airborne
    duration (gameable by ankle flicks), this rewards actual stride
    displacement — the foot must physically advance in the walk direction.

    At liftoff (transition from contact to airborne), saves the foot
    and base world positions. Each step while airborne, computes the
    foot's forward displacement relative to the base displacement,
    projected onto the current heading direction.

    Per-step reward = sum_feet(clip(fore_aft_progress / target_stride, 0, 1) * airborne)

    Ankle flick: foot stays near hip → fore_aft ≈ 0 → reward ≈ 0.
    Walking stride: foot moves ~15cm forward relative to body → reward ≈ 1.

    Params:
        target_stride: Forward displacement (meters) at which reward
            clips to 1.0. Default 0.15 (reasonable for K1 hip height ~0.68m).
        contact_force_threshold: Normal-force threshold (z-axis, Newtons).
            Default 5.0.
        command_norm_threshold: Minimum command magnitude. Default 0.1.
    """

    def __init__(self, cfg: "RewardTermCfg", env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.target_stride = float(params.get("target_stride", 0.15))
        self.contact_force_threshold = float(params.get("contact_force_threshold", 5.0))
        self.command_norm_threshold = float(params.get("command_norm_threshold", 0.1))

        num_envs = env.num_envs
        num_feet = int(env.feet_indices.shape[0])
        device = env.device

        # Per-foot liftoff positions (world frame).
        self._liftoff_foot_pos = torch.zeros(num_envs, num_feet, 3, dtype=torch.float32, device=device)
        # Per-foot liftoff base position (world frame, needed for relative displacement).
        self._liftoff_base_pos = torch.zeros(num_envs, num_feet, 3, dtype=torch.float32, device=device)
        # Previous contact state.
        self._prev_contact = torch.ones(num_envs, num_feet, dtype=torch.bool, device=device)

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        # Contact detection (z-component, matching FeetAirTime v27).
        normal_force = env.simulator.contact_forces[:, env.feet_indices, 2]  # [E, F]
        current_contact = normal_force > self.contact_force_threshold
        airborne = (~current_contact).float()

        # Detect liftoff: was in contact, now airborne.
        liftoff = (~current_contact) & self._prev_contact  # [E, F]

        # Current positions.
        foot_pos = env.simulator._rigid_body_pos[:, env.feet_indices, :]  # [E, F, 3]
        base_pos = env.simulator.robot_root_states[:, :3]  # [E, 3]

        # At liftoff: save foot and base world positions.
        for f_idx in range(foot_pos.shape[1]):
            mask = liftoff[:, f_idx]
            if mask.any():
                self._liftoff_foot_pos[mask, f_idx] = foot_pos[mask, f_idx]
                self._liftoff_base_pos[mask, f_idx] = base_pos[mask]

        # Relative foot displacement: how far foot moved minus how far
        # base moved since liftoff. Positive = foot advanced further.
        foot_delta = foot_pos - self._liftoff_foot_pos  # [E, F, 3]
        base_delta = base_pos.unsqueeze(1) - self._liftoff_base_pos  # [E, F, 3]
        relative_delta = foot_delta - base_delta  # [E, F, 3]

        # Project onto current base heading direction.
        fwd = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)  # [E, 3]
        heading_x = fwd[:, 0].unsqueeze(1)  # [E, 1]
        heading_y = fwd[:, 1].unsqueeze(1)  # [E, 1]
        fore_aft = relative_delta[:, :, 0] * heading_x + relative_delta[:, :, 1] * heading_y  # [E, F]

        # Reward: normalized progress clipped to [0, 1], only while airborne.
        progress = torch.clamp(fore_aft / self.target_stride, min=0.0, max=1.0)
        per_foot_reward = progress * airborne
        reward = torch.sum(per_foot_reward, dim=1)  # [E]

        # Command gate.
        commands = env.command_manager.commands
        cmd_norm = torch.norm(commands[:, :3], dim=1)
        reward = reward * (cmd_norm > self.command_norm_threshold).float()

        # Update state.
        self._prev_contact = current_contact

        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            self._liftoff_foot_pos.zero_()
            self._liftoff_base_pos.zero_()
            self._prev_contact[:] = True
        else:
            self._liftoff_foot_pos[env_ids] = 0.0
            self._liftoff_base_pos[env_ids] = 0.0
            self._prev_contact[env_ids] = True

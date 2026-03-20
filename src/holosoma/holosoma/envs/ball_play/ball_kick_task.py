"""Ball kick task environment.

Extends locomotion to ball approach + directional kick.
The robot must walk toward a ball, align with a target direction, and kick.
"""

from __future__ import annotations

import torch
from loguru import logger

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse
from holosoma.utils.torch_utils import torch_rand_float


class BallKickTask(LeggedRobotLocomotionManager):
    """Locomotion + ball approach + directional kick."""

    BALL_RADIUS = 0.075
    BALL_STATIONARY_THRESHOLD = 0.1  # m/s

    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)

    def _get_task_name(self) -> str:
        return "ball_kick"

    def _init_buffers(self):
        super()._init_buffers()

        # Ball state buffers
        self.ball_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.ball_vel = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.ball_pos_relative = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.last_ball_vel = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Kick target direction (cos, sin) — set by KickTargetCommand
        self.kick_target_dir = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

        # Ball activity timers
        self.time_ball_still = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.time_ball_moving = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        self._update_ball_state()

    def _update_ball_state(self):
        """Fetch ball state from simulator and update derived quantities."""
        # Save previous velocity for acceleration reward
        self.last_ball_vel[:] = self.ball_vel

        # Get ball state via unified actor API
        ball_states = self.simulator.get_actor_states(
            ["object"],
            torch.arange(self.num_envs, device=self.device),
        )
        self.ball_pos[:] = ball_states[:, :3]
        self.ball_vel[:] = ball_states[:, 7:10]

        # Transform ball position to robot local frame
        robot_pos = self.simulator.robot_root_states[:, :3]
        ball_pos_world_relative = self.ball_pos - robot_pos
        self.ball_pos_relative[:] = quat_rotate_inverse(
            self.base_quat, ball_pos_world_relative, w_last=True
        )

        # Update ball activity timers
        ball_speed = torch.norm(self.ball_vel[:, :2], dim=-1)
        is_moving = ball_speed > self.BALL_STATIONARY_THRESHOLD

        self.time_ball_still = torch.where(
            is_moving,
            torch.zeros_like(self.time_ball_still),
            self.time_ball_still + self.dt,
        )
        self.time_ball_moving = torch.where(
            ~is_moving,
            torch.zeros_like(self.time_ball_moving),
            self.time_ball_moving + self.dt,
        )

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states)
        self._reset_ball_at_robot_front(env_ids)

    def _reset_ball_at_robot_front(self, env_ids):
        """Reset ball in front of robot with randomized offset."""
        if len(env_ids) == 0:
            return

        robot_pos = self.simulator.robot_root_states[env_ids, :3]
        robot_quat = self.simulator.robot_root_states[env_ids, 3:7]

        # Forward direction in world frame
        forward_local = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).expand(len(env_ids), -1)
        forward_world = quat_apply(robot_quat, forward_local, w_last=True)

        # Random offset
        offset_x = torch_rand_float(
            0.25, 0.4, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        offset_y = torch_rand_float(
            -0.2, 0.2, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)

        # Ball position in world frame
        ball_xy = robot_pos[:, :2] + forward_world[:, :2] * offset_x.unsqueeze(-1)
        # Add lateral offset
        lateral = torch.stack([-forward_world[:, 1], forward_world[:, 0]], dim=-1)
        ball_xy = ball_xy + lateral * offset_y.unsqueeze(-1)

        # Build new ball states [num_envs, 13]
        new_states = torch.zeros(len(env_ids), 13, device=self.device)
        new_states[:, 0] = ball_xy[:, 0]
        new_states[:, 1] = ball_xy[:, 1]
        new_states[:, 2] = self.BALL_RADIUS  # on ground
        new_states[:, 6] = 1.0  # identity quaternion w component
        # velocities stay zero

        self.simulator.set_actor_states(["object"], env_ids, new_states)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)

        # Reset ball timers
        self.time_ball_still[env_ids] = 0.0
        self.time_ball_moving[env_ids] = 0.0
        self.last_ball_vel[env_ids] = 0.0

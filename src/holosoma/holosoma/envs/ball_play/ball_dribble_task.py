"""Ball dribble task environment.

Extends locomotion to ball tracking + directional dribbling.
The robot must follow a ball and push it in a commanded direction.
"""

from __future__ import annotations

import math

import torch
from loguru import logger

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse
from holosoma.utils.torch_utils import torch_rand_float


class BallDribbleTask(LeggedRobotLocomotionManager):
    """Locomotion + ball tracking + directional dribbling."""

    BALL_RADIUS = 0.075
    DETECTION_FPS = 30.0
    DETECTION_JITTER = 0.15
    MAX_BALL_DISTANCE = 3.0

    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)

    def _get_task_name(self) -> str:
        return "ball_dribble"

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

        # Perceived ball position (delayed, simulating camera detection)
        self.perceived_ball_pos_relative = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.last_perceived_ball_pos_relative = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

        # Detection timing
        detection_interval = 1.0 / self.DETECTION_FPS / self.dt
        self.ball_detection_interval = detection_interval
        self.ball_detection_timer = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # Per-env jittered thresholds
        self._detection_thresholds = torch.full(
            (self.num_envs,), detection_interval, device=self.device
        )

        # Ball-only reset flag
        self.reset_ball_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        self._update_ball_state()

    def _update_ball_state(self):
        """Fetch ball state from simulator and update derived quantities."""
        # Get ball state via simulator's object state API
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        ball_states = self.simulator._get_object_states("object", all_env_ids)
        self.ball_pos[:] = ball_states[:, :3]
        self.ball_vel[:] = ball_states[:, 7:10]

        # Transform to robot local frame
        robot_pos = self.simulator.robot_root_states[:, :3]
        ball_pos_world_relative = self.ball_pos - robot_pos
        self.ball_pos_relative[:] = quat_rotate_inverse(
            self.base_quat, ball_pos_world_relative, w_last=True
        )

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._update_ball_detection()
        self._check_ball_distance()

    def _update_ball_detection(self):
        """Simulate camera-rate ball detection with jitter."""
        self.ball_detection_timer += 1.0

        # Check which envs should update perception
        should_update = self.ball_detection_timer >= self._detection_thresholds

        if should_update.any():
            update_ids = should_update.nonzero(as_tuple=False).flatten()

            # Save previous perceived position
            self.last_perceived_ball_pos_relative[update_ids] = (
                self.perceived_ball_pos_relative[update_ids].clone()
            )

            # Update perceived position with current actual position
            self.perceived_ball_pos_relative[update_ids] = (
                self.ball_pos_relative[update_ids, :2]
            )

            # Reset timer and resample jittered threshold
            self.ball_detection_timer[update_ids] = 0.0
            jitter = torch_rand_float(
                1.0 - self.DETECTION_JITTER,
                1.0 + self.DETECTION_JITTER,
                (len(update_ids), 1),
                device=str(self.device),
            ).squeeze(-1)
            self._detection_thresholds[update_ids] = (
                self.ball_detection_interval * jitter
            )

    def _check_ball_distance(self):
        """Flag envs where ball is too far for ball-only reset."""
        robot_pos_xy = self.simulator.robot_root_states[:, :2]
        ball_pos_xy = self.ball_pos[:, :2]
        distance = torch.norm(ball_pos_xy - robot_pos_xy, dim=-1)
        self.reset_ball_buf = distance > self.MAX_BALL_DISTANCE

        # Perform ball-only reset for flagged envs
        ball_reset_ids = self.reset_ball_buf.nonzero(as_tuple=False).flatten()
        if ball_reset_ids.numel() > 0:
            self._reset_ball_at_robot_front(ball_reset_ids)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states)
        self._reset_ball_at_robot_front(env_ids)

    def _reset_ball_at_robot_front(self, env_ids):
        """Reset ball around robot within spawn radius."""
        if len(env_ids) == 0:
            return

        robot_pos = self.simulator.robot_root_states[env_ids, :3]

        # Random angle and distance
        angles = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        distances = torch_rand_float(
            0.3, 2.0, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)

        # Ball position
        ball_x = robot_pos[:, 0] + torch.cos(angles) * distances
        ball_y = robot_pos[:, 1] + torch.sin(angles) * distances

        # Build new ball states [num_envs, 13]
        new_states = torch.zeros(len(env_ids), 13, device=self.device)
        new_states[:, 0] = ball_x
        new_states[:, 1] = ball_y
        new_states[:, 2] = self.BALL_RADIUS
        new_states[:, 6] = 1.0  # identity quaternion w

        # Optional initial velocity randomization
        new_states[:, 7] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)
        new_states[:, 8] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 1), device=str(self.device)
        ).squeeze(-1)

        self.simulator._write_object_state_unified("object", new_states, env_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)

        # Reset detection state
        self.perceived_ball_pos_relative[env_ids] = 0.0
        self.last_perceived_ball_pos_relative[env_ids] = 0.0
        self.ball_detection_timer[env_ids] = 0.0
        self.reset_ball_buf[env_ids] = False

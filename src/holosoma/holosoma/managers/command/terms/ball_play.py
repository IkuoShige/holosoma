"""Ball play command terms.

Command generators for kick target direction and dribble target velocity.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.torch_utils import torch_rand_float


class KickTargetCommand(CommandTermBase):
    """Generates a random kick target direction at each episode reset.

    Stores kick_target_dir as (cos, sin) of a random angle.
    No resampling within episode. No locomotion velocity commands.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self.kick_target_dir: torch.Tensor | None = None

    def setup(self) -> None:
        self.kick_target_dir = torch.zeros(
            self.env.num_envs, 2, dtype=torch.float32, device=self.env.device
        )
        self.env.kick_target_dir = self.kick_target_dir

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.kick_target_dir is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if env_ids.numel() == 0:
            return

        angles = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=str(self.env.device)
        ).squeeze(-1)
        self.kick_target_dir[env_ids, 0] = torch.cos(angles)
        self.kick_target_dir[env_ids, 1] = torch.sin(angles)

    def step(self) -> None:
        pass


class DribbleCommand(CommandTermBase):
    """Generates a random dribble target velocity vector in world frame.

    Resampled at random intervals (resampling_time_min to resampling_time_max seconds).
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.speed_range: tuple[float, float] = tuple(params.get("speed_range", [0.1, 3.0]))
        self.resampling_time_range: tuple[float, float] = tuple(
            params.get("resampling_time_range", [3.0, 8.0])
        )
        self.commands: torch.Tensor | None = None
        self._resampling_steps: torch.Tensor | None = None

    def setup(self) -> None:
        self.commands = torch.zeros(
            self.env.num_envs, 2, dtype=torch.float32, device=self.env.device
        )
        self._resampling_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.commands is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if env_ids.numel() == 0:
            return
        self._resample(env_ids)

    def step(self) -> None:
        if self.commands is None or self._resampling_steps is None:
            return
        if self.env.is_evaluating:
            return

        env_ids = (
            self.env.episode_length_buf >= self._resampling_steps
        ).nonzero(as_tuple=False).flatten()

        if env_ids.numel() > 0:
            self._resample(env_ids)

    def _resample(self, env_ids: torch.Tensor) -> None:
        if self.commands is None or self._resampling_steps is None:
            return
        n = len(env_ids)
        device = str(self.env.device)

        angles = torch_rand_float(-math.pi, math.pi, (n, 1), device=device).squeeze(-1)
        speeds = torch_rand_float(
            self.speed_range[0], self.speed_range[1], (n, 1), device=device
        ).squeeze(-1)

        self.commands[env_ids, 0] = torch.cos(angles) * speeds
        self.commands[env_ids, 1] = torch.sin(angles) * speeds

        resample_time = torch_rand_float(
            self.resampling_time_range[0],
            self.resampling_time_range[1],
            (n, 1),
            device=device,
        ).squeeze(-1)
        self._resampling_steps[env_ids] = (
            self.env.episode_length_buf[env_ids] + (resample_time / self.env.dt).long()
        )

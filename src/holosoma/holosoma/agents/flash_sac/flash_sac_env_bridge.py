"""Adapter that exposes a holosoma ``BaseTask`` as a gymnasium-style VectorEnv.

The vendored FlashSAC training loop (and replay buffer) consume the same
interface that ``flash_rl/envs/isaaclab.py`` defines for IsaacLab tasks:

- ``num_envs``, ``observation_space``, ``action_space``, ``single_*`` props
- ``reset(*, seed=None, options=None, random_start_init=True) -> (obs, infos)``
- ``step(actions) -> (obs, rewards, terminated, truncated, infos)``
- numpy outputs (the upstream wrapper has ``to_numpy=True`` by default)
- ``infos["final_obs"]`` shaped ``(num_envs, obs_dim)`` containing the
  pre-reset observation for envs that just terminated
- ``infos["actor_observation_size"] == (obs_dim,)`` returned from ``reset()``

Holosoma's :class:`holosoma.envs.base_task.base_task.BaseTask` exposes a
*different* contract:

- ``reset_all() -> obs_buf_dict`` (no info dict)
- ``step({"actions": actions_tensor}) -> (obs_buf_dict, rew_buf, reset_buf, extras)``
- observations are dict[str, torch.Tensor] keyed by *observation group name*
  (typically ``"actor_obs"`` and ``"critic_obs"`` for locomotion tasks)
- ``reset_buf`` already includes timeouts (``reset_buf = terminated | time_out``)
- ``extras["time_outs"]`` is a per-env bool/float tensor
- ``extras["final_observations"]`` (when present) is dict[obs_group, tensor]
  containing the pre-reset observation for envs that just reset

The :class:`FlashSACGymBridge` translates between the two contracts. It does
NOT mutate the underlying env state; it is a pure read/translate layer that
the vendored FlashSAC agent can drive without knowing about holosoma.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from holosoma.envs.base_task.base_task import BaseTask


class FlashSACGymBridge(VectorEnv):
    """Wrap a holosoma ``BaseTask`` so it looks like FlashSAC's IsaacLabVectorEnv."""

    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str] = ("actor_obs",),
        critic_obs_keys: Sequence[str] = ("critic_obs",),
        action_bounds: float = 1.0,
        to_numpy: bool = True,
    ):
        self._env = env
        self._actor_obs_keys = list(actor_obs_keys)
        self._critic_obs_keys = list(critic_obs_keys)
        self._action_bounds = float(action_bounds)
        self._to_numpy = bool(to_numpy)

        self.num_envs = int(env.num_envs)
        self.device = getattr(env, "device", "cuda:0")
        self.max_episode_steps = int(getattr(env, "max_episode_length", 0)) or None

        # Compute the actor / critic observation dimensions by snapshotting one
        # ``compute()`` from the observation manager. We rely on the env having
        # been ``setup()``-ed by the holosoma training entrypoint before the
        # bridge is constructed (otherwise the obs buffers don't exist yet).
        env.reset_all()
        actor_dim = sum(int(env.obs_buf_dict[k].shape[-1]) for k in self._actor_obs_keys)
        if self._critic_obs_keys and all(k in env.obs_buf_dict for k in self._critic_obs_keys):
            critic_dim = sum(int(env.obs_buf_dict[k].shape[-1]) for k in self._critic_obs_keys)
            self._has_critic_obs = True
        else:
            critic_dim = 0
            self._has_critic_obs = False
        self._actor_obs_dim = actor_dim
        self._critic_obs_dim = critic_dim

        # FlashSAC's asymmetric_observation path concatenates actor + critic obs
        # into a single flat tensor and exposes the actor portion via
        # ``infos["actor_observation_size"]``. We follow that convention.
        self.asymmetric_obs = self._has_critic_obs and critic_dim > 0
        total_obs_dim = actor_dim + critic_dim if self.asymmetric_obs else actor_dim

        self.obs_size = (actor_dim,)
        self.critic_obs_size = (critic_dim,) if self.asymmetric_obs else (0,)

        action_shape = (int(getattr(env.robot_config, "actions_dim", env.dim_actions)),)
        self.action_size = action_shape

        self.single_observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(total_obs_dim,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.single_action_space = gym.spaces.Box(
            low=-1.0 * self._action_bounds,
            high=1.0 * self._action_bounds,
            shape=action_shape,
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ------------------------------------------------------------------
    # gymnasium vector env API
    # ------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        obs_buf_dict = self._env.reset_all()
        actor_obs = self._concat_groups(obs_buf_dict, self._actor_obs_keys)
        if self.asymmetric_obs:
            critic_obs = self._concat_groups(obs_buf_dict, self._critic_obs_keys)
            obs = torch.cat((actor_obs, critic_obs), dim=-1)
        else:
            obs = actor_obs

        if random_start_init and self.max_episode_steps:
            episode_buf = getattr(self._env, "episode_length_buf", None)
            if isinstance(episode_buf, torch.Tensor):
                episode_buf.copy_(
                    torch.randint_like(episode_buf, high=int(self.max_episode_steps))
                )

        infos: dict[str, Any] = {
            "actor_observation_size": self.obs_size,
            "asymmetric_obs": self.asymmetric_obs,
        }
        if self._to_numpy:
            obs = obs.detach().cpu().numpy()
        return obs, infos

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:  # type: ignore[override]
        if isinstance(actions, np.ndarray):
            torch_actions = torch.from_numpy(actions).to(self.device)
        elif isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.as_tensor(actions, device=self.device)

        torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self._action_bounds

        obs_buf_dict, rew_buf, reset_buf, extras = self._env.step({"actions": torch_actions})

        actor_obs = self._concat_groups(obs_buf_dict, self._actor_obs_keys)
        if self.asymmetric_obs:
            critic_obs = self._concat_groups(obs_buf_dict, self._critic_obs_keys)
            obs = torch.cat((actor_obs, critic_obs), dim=-1)
        else:
            critic_obs = None
            obs = actor_obs

        # Holosoma's reset_buf is the OR of termination and timeout. FlashSAC
        # wants them split (so the agent knows whether to bootstrap from
        # next_obs). Recover the split from extras["time_outs"].
        time_outs = extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(reset_buf)
        time_outs_bool = time_outs.bool()
        reset_bool = reset_buf.bool()
        truncated = time_outs_bool
        terminated = reset_bool & ~time_outs_bool

        # Synthesize the per-env "final observation" tensor that FlashSAC's
        # train loop reads to repair the next-obs entry on episode-end.
        final_actor_obs = actor_obs.clone()
        final_obs_dict = extras.get("final_observations") or {}
        if reset_bool.any() and final_obs_dict:
            try:
                stacked = self._concat_groups(final_obs_dict, self._actor_obs_keys)
                env_ids = reset_bool.nonzero(as_tuple=False).flatten()
                # Holosoma stores final_observations only for the envs that reset,
                # in the order returned by ``reset_buf.nonzero()`` (see BaseTask).
                final_actor_obs[env_ids] = stacked
            except KeyError:
                # Fall back to using the post-reset obs as the bootstrap target.
                pass

        if self.asymmetric_obs and self._critic_obs_dim > 0:
            final_critic_obs = critic_obs.clone() if critic_obs is not None else None
            if final_critic_obs is not None and reset_bool.any() and final_obs_dict:
                try:
                    stacked_c = self._concat_groups(final_obs_dict, self._critic_obs_keys)
                    env_ids = reset_bool.nonzero(as_tuple=False).flatten()
                    final_critic_obs[env_ids] = stacked_c
                except KeyError:
                    pass
            full_final = torch.cat((final_actor_obs, final_critic_obs), dim=-1) if final_critic_obs is not None else final_actor_obs
        else:
            full_final = final_actor_obs

        infos: dict[str, Any] = {
            "time_outs": truncated,
            "observations": {"critic": critic_obs},
            "final_obs": full_final,
        }

        if self._to_numpy:
            obs = obs.detach().cpu().numpy()
            rew = rew_buf.detach().cpu().numpy()
            terminated_np = terminated.detach().cpu().numpy()
            truncated_np = truncated.detach().cpu().numpy()
            infos["time_outs"] = truncated_np
            infos["final_obs"] = full_final.detach().cpu().numpy()
            return obs, rew, terminated_np, truncated_np, infos

        return obs, rew_buf, terminated, truncated, infos

    def close(self, **kwargs: Any) -> None:  # type: ignore[override]
        # The underlying BaseTask is owned by the holosoma trainer; closing
        # here would tear down IsaacSim while the trainer still expects it
        # to be alive. We deliberately make this a no-op, mirroring the
        # vendored ``IsaacLabVectorEnv.close``.
        return None

    def render(self) -> None:  # type: ignore[override]
        # Same rationale as the upstream IsaacLab wrapper: rendering through
        # IsaacSim from inside the bridge is not supported. Smoke runs disable
        # video recording so this is never reached.
        raise NotImplementedError("FlashSACGymBridge does not support render()")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _concat_groups(
        self,
        obs_buf_dict: dict[str, torch.Tensor],
        keys: Sequence[str],
    ) -> torch.Tensor:
        if not keys:
            return torch.zeros((self.num_envs, 0), device=self.device)
        if len(keys) == 1:
            return obs_buf_dict[keys[0]]
        return torch.cat([obs_buf_dict[k] for k in keys], dim=-1)

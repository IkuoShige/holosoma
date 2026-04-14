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
from loguru import logger

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
        target_action_scale_rad: float | None = 0.5,
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

        # FlashSAC's actor outputs ``tanh(mean) ∈ [-1, 1]``. Holosoma's
        # ``JointPositionActionTerm`` then multiplies by
        # ``robot.control.action_scale`` (0.25 for G1/K1) to produce the
        # joint position target offset from default.
        #
        # Two scaling modes:
        #
        # 1. UNIFORM (target_action_scale_rad != None, use_per_joint_scaling=False):
        #    All joints get the same multiplier. Original FlashSAC behavior.
        #
        # 2. PER-JOINT (use_per_joint_scaling=True):
        #    Each joint's multiplier = max_ROM_from_default / action_scale.
        #    Ported from FastSAC's _compute_action_boundaries(). This gives
        #    each joint full authority over its ROM. K1 Hip_Pitch gets 11.2x
        #    (vs uniform 4.0x), enabling PPO-level hip swing amplitude.
        #
        #    Previously removed due to "thrashing" but that was with
        #    different reward/exploration settings. v42 re-tests with
        #    v39's proven reward + exploration.
        self._action_scale_multiplier: torch.Tensor | float | None = None
        use_per_joint = getattr(env.robot_config.control, '_use_per_joint_scaling', False)
        # Check algo config for the flag (set via experiment config)
        if hasattr(env, '_flash_sac_per_joint_scaling'):
            use_per_joint = env._flash_sac_per_joint_scaling

        if target_action_scale_rad is not None and target_action_scale_rad == -1.0:
            # Special sentinel: -1.0 means per-joint scaling
            use_per_joint = True

        if use_per_joint:
            # FastSAC-style per-joint scaling
            robot_config = env.robot_config
            env_action_scale = float(robot_config.control.action_scale)
            dof_lower = torch.tensor(robot_config.dof_pos_lower_limit_list, device=self.device)
            dof_upper = torch.tensor(robot_config.dof_pos_upper_limit_list, device=self.device)
            default_pos = torch.zeros(len(robot_config.dof_names), device=self.device)
            for i, name in enumerate(robot_config.dof_names):
                if name in robot_config.init_state.default_joint_angles:
                    default_pos[i] = robot_config.init_state.default_joint_angles[name]
            range_lower = torch.abs(dof_lower - default_pos)
            range_upper = torch.abs(dof_upper - default_pos)
            max_range = torch.maximum(range_lower, range_upper)
            self._action_scale_multiplier = (max_range / env_action_scale).float()
            logger.info(
                f"[FlashSAC bridge] Per-joint scaling enabled. "
                f"Multipliers: min={self._action_scale_multiplier.min():.1f}, "
                f"max={self._action_scale_multiplier.max():.1f}, "
                f"hip_pitch={self._action_scale_multiplier[10]:.1f}"
            )
        elif target_action_scale_rad is not None:
            env_action_scale = float(env.robot_config.control.action_scale)
            if env_action_scale <= 0:
                raise ValueError(
                    f"robot.control.action_scale must be positive, got {env_action_scale}"
                )
            self._action_scale_multiplier = float(target_action_scale_rad) / env_action_scale
            self._target_action_scale_rad = float(target_action_scale_rad)

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
        # Uniform action scaling so the actor's ``tanh ∈ [-1, 1]`` output
        # produces the same effective joint target range (±0.5 rad by
        # default) that FlashSAC's upstream algorithm hyperparameters were
        # tuned against on IsaacLab's stock G1 task. MUST be applied
        # identically in training and eval — this is part of the env
        # transition the policy learns against.
        if self._action_scale_multiplier is not None:
            torch_actions = torch_actions * self._action_scale_multiplier

        # Diagnostics: capture per-env episode length BEFORE step() runs.
        # Holosoma resets episode_length_buf[env_ids] to 0 inside
        # _post_physics_step (via reset_envs_idx), so by the time we see
        # the post-step state, the length for terminating envs is already
        # lost. We snapshot it before step and compute the final length
        # as snapshot + 1 (for the step that just happened).
        prev_episode_length: torch.Tensor | None = None
        env_ep_buf = getattr(self._env, "episode_length_buf", None)
        if isinstance(env_ep_buf, torch.Tensor):
            prev_episode_length = env_ep_buf.detach().clone()

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
        #
        # Holosoma's ``extras["final_observations"][obs_key]`` is a persistent
        # full-batch tensor of shape ``(num_envs, obs_dim)`` — ``BaseTask.
        # _store_final_observations`` pre-allocates it via
        # ``torch.zeros_like(obs_buf_dict[obs_key])`` and only updates rows that
        # match the *current* step's ``env_ids``. Rows for non-resetting envs
        # carry stale values from earlier resets (or zero). We therefore pull
        # only the current ``env_ids`` rows out of the full tensor and write
        # them into the matching rows of ``final_actor_obs``.
        final_actor_obs = actor_obs.clone()
        final_obs_dict = extras.get("final_observations") or {}
        env_ids = reset_bool.nonzero(as_tuple=False).flatten() if reset_bool.any() else None
        if env_ids is not None and env_ids.numel() > 0 and final_obs_dict:
            try:
                stacked = self._concat_groups(final_obs_dict, self._actor_obs_keys)
                final_actor_obs[env_ids] = stacked[env_ids]
            except KeyError:
                # Fall back to using the post-reset obs as the bootstrap target.
                pass

        if self.asymmetric_obs and self._critic_obs_dim > 0:
            final_critic_obs = critic_obs.clone() if critic_obs is not None else None
            if (
                final_critic_obs is not None
                and env_ids is not None
                and env_ids.numel() > 0
                and final_obs_dict
            ):
                try:
                    stacked_c = self._concat_groups(final_obs_dict, self._critic_obs_keys)
                    final_critic_obs[env_ids] = stacked_c[env_ids]
                except KeyError:
                    pass
            full_final = (
                torch.cat((final_actor_obs, final_critic_obs), dim=-1)
                if final_critic_obs is not None
                else final_actor_obs
            )
        else:
            full_final = final_actor_obs

        infos: dict[str, Any] = {
            "time_outs": truncated,
            "observations": {"critic": critic_obs},
            "final_obs": full_final,
        }

        # ------------------------------------------------------------------
        # Diagnostic forwarding (added for v24 debug cycle).
        #
        # Forward holosoma's per-term episode reward breakdown and
        # environment log_dict through infos so the FlashSAC training loop
        # can log them to TensorBoard. Without this, FlashSAC sees only the
        # aggregate reward scalar and has no visibility into which reward
        # terms are actually accumulating — the root cause of the
        # 20-iteration hyperparameter tuning dead end.
        #
        # `extras["episode"]` is populated only for env_ids that reset this
        # step. Each value is a tensor over reset envs (shape [num_resets]).
        # We aggregate to a single scalar mean for this step and let the
        # agent maintain a running average across logging windows.
        # ------------------------------------------------------------------
        if env_ids is not None and env_ids.numel() > 0:
            episode_rewards = extras.get("episode") or {}
            raw_episode_rewards = extras.get("raw_episode") or {}

            def _reset_env_mean(tensor_map: dict[str, Any]) -> dict[str, float]:
                out: dict[str, float] = {}
                for term_name, tensor in tensor_map.items():
                    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        continue
                    out[term_name] = float(tensor.float().mean().item())
                return out

            episode_scalars = _reset_env_mean(episode_rewards)
            raw_episode_scalars = _reset_env_mean(raw_episode_rewards)
            if episode_scalars:
                infos["episode"] = episode_scalars
            if raw_episode_scalars:
                infos["raw_episode"] = raw_episode_scalars

            # Episode length at termination: pre-step length + 1 (for the
            # step that pushed the episode into termination).
            if prev_episode_length is not None:
                final_lengths = prev_episode_length[env_ids].float() + 1.0
                infos["episode_length_mean"] = float(final_lengths.mean().item())
                infos["episode_length_max"] = float(final_lengths.max().item())

            # Split terminated vs truncated (fall vs timeout) for this step.
            terminated_count = int((terminated & reset_bool)[env_ids].sum().item())
            truncated_count = int(time_outs_bool[env_ids].sum().item())
            total = terminated_count + truncated_count
            if total > 0:
                infos["termination_fraction"] = terminated_count / total

            infos["num_resets"] = int(env_ids.numel())
        else:
            infos["num_resets"] = 0

        # Forward env log_dict (average_episode_length etc.) always.
        to_log = extras.get("to_log") or {}
        to_log_scalars: dict[str, float] = {}
        for k, v in to_log.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                to_log_scalars[k] = float(v.item())
            elif isinstance(v, (int, float)):
                to_log_scalars[k] = float(v)
        if to_log_scalars:
            infos["to_log"] = to_log_scalars

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

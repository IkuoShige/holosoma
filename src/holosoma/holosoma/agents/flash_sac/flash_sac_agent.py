"""Holosoma BaseAlgo wrapper around the vendored FlashSAC implementation.

This module bridges holosoma's training entrypoint (``train_agent.py`` →
``BaseAlgo.learn()``) to the vendored FlashSAC algorithm
(``holosoma._vendored.flash_rl.agents.flashSAC.agent.FlashSACAgent``).

Design notes
------------

- We do NOT call the vendored ``train.py``; instead we re-host its training
  loop in-line so we can plug in holosoma's logger, checkpoint paths, and
  curriculum hooks. The loop body (sample → process_transition → update) is
  bit-equivalent to upstream lines 113-200.
- We do NOT route the holosoma config through OmegaConf — the vendored
  :class:`FlashSACConfig` is a plain dataclass, so we just call
  ``FlashSACConfig(**asdict(self.config.config))`` and let pydantic's
  generated dataclass conversion do the field marshalling.
- ``FlashSACGymBridge`` is responsible for translating between holosoma's
  dict-action / dict-observation env contract and the gymnasium VectorEnv
  surface that the vendored agent expects.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable

import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from holosoma._vendored.flash_rl.agents.flashSAC.agent import (
    FlashSACAgent as VendoredFlashSACAgent,
)
from holosoma._vendored.flash_rl.agents.flashSAC.agent import FlashSACConfig as VendoredFlashSACConfig
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge
from holosoma.config_types.algo import FlashSACVendorConfig
from holosoma.envs.base_task.base_task import BaseTask


class FlashSACAgent(BaseAlgo):
    """Holosoma BaseAlgo wrapper around vendored FlashSAC."""

    config: FlashSACVendorConfig

    def __init__(
        self,
        env: BaseTask,
        config: FlashSACVendorConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict | None = None,
    ):
        # Wrap the holosoma env in the bridge BEFORE calling super().__init__,
        # so that BaseAlgo.env is the gym-shaped wrapper. We still keep a
        # reference to the unwrapped env for checkpoint metadata.
        bridge = FlashSACGymBridge(
            env,
            actor_obs_keys=tuple(config.actor_obs_keys),
            critic_obs_keys=tuple(config.critic_obs_keys),
        )
        super().__init__(env=bridge, config=config, device=device, multi_gpu_cfg=multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.bridge = bridge
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self._inner_agent: VendoredFlashSACAgent | None = None
        self._update_counter: float = 0.0

    # ------------------------------------------------------------------
    # BaseAlgo lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        logger.info("Setting up FlashSAC (vendored adapter)")

        if self.is_multi_gpu and self.has_curricula_enabled():
            logger.info(
                f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs"
            )

        # Translate the holosoma dataclass into the vendored FlashSACConfig.
        # The vendored FlashSACConfig dataclass requires several fields that
        # the upstream YAML computes via Hydra's ${eval: ...} resolver
        # (notably ``learning_rate_warmup_step`` and ``learning_rate_decay_step``).
        # We materialize them here from the corresponding *_rate fields and the
        # holosoma loop length.
        cfg_dict = dataclasses.asdict(self.config)
        adapter_only_fields = {
            "num_learning_iterations",
            "logging_interval",
            "save_interval",
            "actor_obs_keys",
            "critic_obs_keys",
            "updates_per_interaction_step",
            "eval_callbacks",
        }
        vendored_kwargs = {k: v for k, v in cfg_dict.items() if k not in adapter_only_fields}

        loop_length = int(self.config.num_learning_iterations)
        updates_per_step = float(self.config.updates_per_interaction_step)
        vendored_kwargs["learning_rate_warmup_step"] = int(
            self.config.learning_rate_warmup_rate * loop_length * updates_per_step
        )
        vendored_kwargs["learning_rate_decay_step"] = int(
            self.config.learning_rate_decay_rate * loop_length * updates_per_step
        )

        # The vendored constructor accepts ``temp_target_entropy`` as a float
        # OR ``None`` (in which case agent.__init__ computes the heuristic
        # ``-0.5 * action_dim``). Treat ``None`` from the holosoma side
        # consistently with that contract.
        if vendored_kwargs.get("temp_target_entropy") is None:
            vendored_kwargs["temp_target_entropy"] = None

        # Reset to obtain the env_info dict the vendored agent expects.
        _, env_info = self.bridge.reset(random_start_init=False)

        vendored_cfg = VendoredFlashSACConfig(**vendored_kwargs)
        self._inner_agent = VendoredFlashSACAgent(
            observation_space=self.bridge.observation_space,
            action_space=self.bridge.action_space,
            env_info=env_info,
            cfg=vendored_cfg,
        )

    def learn(self) -> None:
        if self._inner_agent is None:
            self.setup()
        assert self._inner_agent is not None

        env = self.bridge
        agent = self._inner_agent
        cfg = self.config

        observations, env_infos = env.reset()
        actions: np.ndarray | None = None
        transition: dict[str, Any] | None = None
        update_info: dict[str, Any] = {}
        update_counter: float = 0.0

        pbar = tqdm.tqdm(
            range(1, int(cfg.num_learning_iterations + 1)),
            smoothing=0.1,
            mininterval=0.5,
            disable=not self.is_main_process,
        )
        for interaction_step in pbar:
            self.global_step = interaction_step
            env_step = interaction_step * env.num_envs

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            if agent.can_start_training() and transition is not None:
                actions = agent.sample_actions(
                    interaction_step,
                    prev_transition=transition,
                    training=True,
                )
            else:
                actions = env.action_space.sample()

            assert actions is not None
            actions = np.asarray(actions)
            next_observations, rewards, terminateds, truncateds, env_infos = env.step(actions)
            next_buffer_observations = next_observations.copy()
            for env_idx in range(env.num_envs):
                if terminateds[env_idx] or truncateds[env_idx]:
                    next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

            transition = {
                "observation": observations,
                "action": actions,
                "reward": rewards,
                "terminated": terminateds,
                "truncated": truncateds,
                "next_observation": next_buffer_observations,
            }
            agent.process_transition(transition)
            transition["next_observation"] = next_observations
            observations = next_observations

            if agent.can_start_training():
                update_counter += cfg.updates_per_interaction_step
                while update_counter >= 1:
                    update_info = agent.update()
                    update_counter -= 1

                if cfg.logging_interval and interaction_step % cfg.logging_interval == 0:
                    self._log_metrics(update_info, env_step)

                if cfg.save_interval and interaction_step % cfg.save_interval == 0:
                    self._save_checkpoint(interaction_step)

        # Final flush + checkpoint.
        self._log_metrics(update_info, env_step=int(cfg.num_learning_iterations) * env.num_envs)
        self._save_checkpoint(int(cfg.num_learning_iterations))
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        if self._inner_agent is None:
            self.setup()
        assert self._inner_agent is not None
        # The vendored agent's ``load`` expects a directory or file path that
        # ``torch.save`` previously wrote. We delegate verbatim.
        self._inner_agent.load(ckpt_path)

    def get_inference_policy(
        self, device: str | None = None
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        if self._inner_agent is None:
            raise RuntimeError("FlashSACAgent.setup() must run before get_inference_policy()")

        actor_obs_keys = list(self.config.actor_obs_keys)
        bridge = self.bridge
        inner = self._inner_agent

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            # Concatenate the same actor observation groups the bridge uses.
            actor_obs = obs[actor_obs_keys[0]] if len(actor_obs_keys) == 1 else torch.cat(
                [obs[k] for k in actor_obs_keys], dim=-1
            )
            actor_obs_np = actor_obs.detach().cpu().numpy()
            # The vendored agent's ``sample_actions`` requires a prev_transition
            # so it can resample noise; for deterministic inference we feed an
            # all-zero placeholder which it ignores when training=False.
            placeholder_transition = {"next_observation": actor_obs_np}
            actions_np = inner.sample_actions(
                interaction_step=0,
                prev_transition=placeholder_transition,
                training=False,
            )
            actions = torch.from_numpy(np.asarray(actions_np)).to(actor_obs.device)
            return actions

        return policy_fn

    @property
    def actor_onnx_wrapper(self) -> Any:
        # ONNX export is out of scope for the smoke proof; raise rather than
        # silently exporting an incomplete model.
        raise NotImplementedError("FlashSAC ONNX export is not yet implemented")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _log_metrics(self, update_info: dict[str, Any], env_step: int) -> None:
        if not self.is_main_process:
            return
        for key, value in update_info.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"flashsac/{key}", float(value), env_step)
        self.writer.flush()

    def _save_checkpoint(self, interaction_step: int) -> None:
        if not self.is_main_process or self._inner_agent is None:
            return
        save_path = os.path.join(self.log_dir, f"flashsac_step{interaction_step}")
        os.makedirs(save_path, exist_ok=True)
        try:
            self._inner_agent.save(save_path)
        except Exception as exc:  # pragma: no cover - persistence is best-effort during smoke
            logger.warning(f"FlashSACAgent checkpoint save failed: {exc}")

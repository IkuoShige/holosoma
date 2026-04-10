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
import itertools
import os
from pathlib import Path
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

# Keys used to pack the five vendored per-component state files into the
# single-file ``.ckpt`` format holosoma's ``eval_agent.py`` expects. Each
# value is the on-disk file name the vendored agent writes inside its
# checkpoint directory; the same names are used for the inline payload
# dict keys so round-tripping is trivially reversible.
_FLASHSAC_COMPONENT_FILES: tuple[str, ...] = (
    "actor.pt",
    "critic.pt",
    "target_critic.pt",
    "temperature.pt",
    "reward_normalizer.pt",
    "agent_state.pt",
)


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
            target_action_scale_rad=config.target_action_scale_rad,
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
            "target_action_scale_rad",
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

        # Upstream ``FlashSACAgent.load`` asserts that ``reward_normalizer is
        # not None`` whenever ``load_reward_normalizer`` is True, which fires
        # a misleading AssertionError when someone trained with
        # ``normalize_reward=False``. Force the two flags to agree so the
        # adapter's load path is always consistent: if we never created a
        # normalizer, we never try to load one.
        if not vendored_kwargs.get("normalize_reward", True):
            vendored_kwargs["load_reward_normalizer"] = False

        # Reset to obtain the env_info dict the vendored agent expects.
        _, env_info = self.bridge.reset(random_start_init=False)

        vendored_cfg = VendoredFlashSACConfig(**vendored_kwargs)
        self._inner_agent = VendoredFlashSACAgent(
            observation_space=self.bridge.observation_space,
            action_space=self.bridge.action_space,
            env_info=env_info,
            cfg=vendored_cfg,
        )

        # Attach symmetry augmentation if the robot has symmetry config.
        if getattr(self.unwrapped_env.robot_config, "symmetry_joint_names", None):
            self._setup_symmetry_augmentation()

    def _setup_symmetry_augmentation(self) -> None:
        """Attach a batch-level symmetry augmenter to the vendored agent.

        Reuses holosoma's ``SymmetryUtils`` (same as PPO/FastSAC) to mirror
        observations and actions in sampled replay batches. This doubles the
        effective batch size and enforces left-right consistency without
        modifying the replay buffer itself.
        """
        from holosoma.agents.modules.augmentation_utils import SymmetryUtils

        sym = SymmetryUtils(self.unwrapped_env)
        actor_obs_keys = list(self.config.actor_obs_keys)
        critic_obs_keys = list(self.config.critic_obs_keys)
        actor_dim = self.bridge._actor_obs_dim
        env = self.unwrapped_env

        def augment_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            obs = batch["observation"]
            next_obs = batch["next_observation"]
            actions = batch["action"]

            # Split concatenated obs into actor/critic parts, mirror each.
            actor_obs = obs[:, :actor_dim]
            actor_next = next_obs[:, :actor_dim]
            aug_actor_obs = sym.augment_observations(actor_obs, env, actor_obs_keys)
            aug_actor_next = sym.augment_observations(actor_next, env, actor_obs_keys)
            aug_actions = sym.augment_actions(actions)

            if obs.shape[-1] > actor_dim:
                # Asymmetric: critic obs is the remainder.
                critic_obs = obs[:, actor_dim:]
                critic_next = next_obs[:, actor_dim:]
                aug_critic_obs = sym.augment_observations(critic_obs, env, critic_obs_keys)
                aug_critic_next = sym.augment_observations(critic_next, env, critic_obs_keys)
                aug_obs = torch.cat([aug_actor_obs, aug_critic_obs], dim=-1)
                aug_next_obs = torch.cat([aug_actor_next, aug_critic_next], dim=-1)
            else:
                aug_obs = aug_actor_obs
                aug_next_obs = aug_actor_next

            n_aug = aug_obs.shape[0] // obs.shape[0]
            return {
                "observation": aug_obs,
                "next_observation": aug_next_obs,
                "action": aug_actions,
                "reward": batch["reward"].repeat(n_aug),
                "terminated": batch["terminated"].repeat(n_aug),
                "truncated": batch["truncated"].repeat(n_aug),
            }

        assert self._inner_agent is not None
        self._inner_agent._batch_augment_fn = augment_batch
        logger.info("[FlashSAC] Symmetry augmentation enabled (batch-level mirroring)")

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
        # ``agent.update()`` returns different key sets depending on
        # ``_update_step % actor_update_period``: even steps carry
        # actor/temperature/critic keys, odd steps carry only critic keys.
        # Overwriting a single dict per update therefore loses actor metrics
        # when the last update of a logging window happens to be an odd
        # step. With the stock ``updates_per_interaction_step=2.0`` and
        # ``logging_interval=100``, the last update of every window is odd,
        # so actor/entropy/temperature metrics would NEVER land in
        # TensorBoard. Accumulate into a running dict instead and reset it
        # once we flush to the writer.
        window_update_info: dict[str, Any] = {}
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
                    step_update_info = agent.update()
                    window_update_info.update(step_update_info)
                    update_counter -= 1

                if cfg.logging_interval and interaction_step % cfg.logging_interval == 0:
                    self._log_metrics(window_update_info, env_step)
                    window_update_info = {}

                if cfg.save_interval and interaction_step % cfg.save_interval == 0:
                    self._save_checkpoint(interaction_step)

        # Final flush + checkpoint.
        self._log_metrics(
            window_update_info, env_step=int(cfg.num_learning_iterations) * env.num_envs
        )
        self._save_checkpoint(int(cfg.num_learning_iterations))
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------
    # Evaluation (holosoma ``eval_agent.py`` entry-point)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        """Roll out the deterministic inference policy through the bridge.

        Mirrors ``FastSACAgent.evaluate_policy`` in shape so that holosoma's
        ``eval_agent.py`` can drive the FlashSAC adapter without any
        algorithm-specific branching. The eval loop uses ``training=False``
        (deterministic action, no entropy noise) and treats the bridge as a
        numpy gymnasium VectorEnv.
        """

        if self._inner_agent is None:
            self.setup()
        assert self._inner_agent is not None

        env = self.bridge
        agent = self._inner_agent

        observations, _ = env.reset(random_start_init=False)
        prev_transition: dict[str, Any] = {"next_observation": observations}

        iterator = itertools.count() if max_eval_steps is None else range(max_eval_steps)
        for _ in iterator:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_transition=prev_transition,
                training=False,
            )
            actions = np.asarray(actions)
            next_observations, _rewards, _terminateds, _truncateds, _infos = env.step(actions)
            prev_transition = {"next_observation": next_observations}
            observations = next_observations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None, name: str = "model.ckpt") -> None:
        """Write both the vendored directory format and a single-file ``.ckpt``.

        - The directory contains the five per-component ``.pt`` files the
          vendored ``FlashSACAgent.load`` reads (actor, critic,
          target_critic, temperature, reward_normalizer, agent_state).
          This preserves byte-exact compatibility with upstream
          ``play_isaaclab.py``.
        - The single-file ``.ckpt`` written next to the directory packs the
          holosoma-side ``experiment_config`` / ``wandb_run_path`` metadata
          alongside the *contents* of those five files, so
          ``holosoma.eval_agent`` can ingest it through its standard
          ``load_checkpoint`` → ``algo.load`` path.

        The ``path`` argument mirrors ``BaseAlgo.save``: if ``None``, use
        ``self.log_dir``; otherwise treat ``path`` as the target directory.
        The single-file checkpoint is written at ``<path>/<name>``.
        """

        if not self.is_main_process or self._inner_agent is None:
            return

        save_dir = str(path) if path is not None else self.log_dir
        os.makedirs(save_dir, exist_ok=True)

        # 1) Vendored directory format (for play_isaaclab.py compatibility).
        self._inner_agent.save(save_dir)

        # 2) Single-file ``.ckpt`` (for holosoma eval_agent.py).
        components: dict[str, Any] = {}
        for fname in _FLASHSAC_COMPONENT_FILES:
            fpath = os.path.join(save_dir, fname)
            if os.path.exists(fpath):
                components[fname] = torch.load(fpath, map_location="cpu", weights_only=False)

        metadata = self._checkpoint_metadata(iteration=self.global_step)

        save_dict: dict[str, Any] = {
            **metadata,
            "flashsac_components": components,
            "global_step": int(self.global_step),
        }

        ckpt_path = os.path.join(save_dir, name)
        torch.save(save_dict, ckpt_path)
        logger.info(f"[FlashSAC] wrote single-file checkpoint at {ckpt_path}")

    def load(self, ckpt_path: str | None) -> None:
        """Load weights from either a directory or a single ``.ckpt`` file.

        The directory form delegates verbatim to the vendored
        ``FlashSACAgent.load`` (reads actor.pt / critic.pt / ...).

        The single-file form unpacks ``flashsac_components`` back into a
        temporary directory, then reuses the vendored loader so the on-disk
        format of each component file is honored without re-implementing
        the inner ``Network.load`` logic here.
        """

        if not ckpt_path:
            return
        if self._inner_agent is None:
            self.setup()
        assert self._inner_agent is not None

        path = Path(ckpt_path)
        if path.is_dir():
            self._inner_agent.load(str(path))
            return

        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or "flashsac_components" not in payload:
            raise ValueError(
                f"Checkpoint {path} is neither a vendored FlashSAC directory nor a "
                f"holosoma single-file FlashSAC checkpoint (missing "
                f"'flashsac_components' key; got keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)})."
            )

        components: dict[str, Any] = payload["flashsac_components"]
        import tempfile

        with tempfile.TemporaryDirectory(prefix="flashsac_load_") as tmpdir:
            for fname, contents in components.items():
                torch.save(contents, os.path.join(tmpdir, fname))
            self._inner_agent.load(tmpdir)

        self.global_step = int(payload.get("global_step", self.global_step))

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
            # Delegates to ``FlashSACAgent.save`` which writes both the
            # vendored directory format and the holosoma-side single-file
            # ``.ckpt`` that ``eval_agent.py`` can ingest.
            self.save(save_path, name=f"flashsac_step{interaction_step}.ckpt")
        except Exception as exc:  # pragma: no cover - persistence is best-effort during smoke
            logger.warning(f"FlashSACAgent checkpoint save failed: {exc}")

from __future__ import annotations

import copy
from typing import Callable

import torch
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.module_utils import (
    setup_flow_policy_module,
    setup_ppo_critic_module,
)
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import FPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from loguru import logger
from torch import nn


class FPOAgent(PPO):
    """Flow Policy Optimization (FPO) agent.

    FPO is an on-policy RL algorithm that uses conditional flow matching (CFM)
    to learn a flow-based policy capable of expressing multimodal action distributions.
    It reuses the PPO framework for critic, GAE, and rollout logic, but replaces the
    Gaussian actor with a FlowPolicy and the probability-ratio surrogate with a
    CFM-loss-based ratio.
    """

    config: FPOConfig

    def __init__(self, env: BaseTask, config: FPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        # PPO.__init__ calls _init_config, sets up logging, etc.
        super().__init__(env, config, log_dir, device, multi_gpu_cfg)
        self._ema_cfm_loss: float | None = None
        self._div_consecutive_count: int = 0

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim

    def _setup_models_and_optimizer(self):
        logger.info("Setting up FPO models")
        self.actor = setup_flow_policy_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            device=self.device,
            history_length=self.algo_history_length_dict,
            time_embed_dim=self.config.time_embed_dim,
            use_ada_ln=self.config.use_ada_ln,
            num_flow_steps=self.config.num_flow_steps,
            action_bound=self.config.action_bound,
            flow_param_mode=self.config.flow_param_mode,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )

        # Observation normalization
        if self.config.obs_normalization:
            actor_obs_dim = sum(self.algo_obs_dim_dict[k] for k in self.actor_obs_keys)
            critic_obs_dim = sum(self.algo_obs_dim_dict[k] for k in self.critic_obs_keys)
            self.obs_normalizer = EmpiricalNormalization(shape=actor_obs_dim, device=self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )

    def _setup_storage(self):
        super()._setup_storage()
        # Register FPO-specific MC sample buffers
        k = self.config.num_mc_samples
        self.storage.register("flow_eps", shape=(k, self.num_act), dtype=torch.float)
        self.storage.register("flow_t", shape=(k, 1), dtype=torch.float)
        self.storage.register("flow_old_loss", shape=(k, 1), dtype=torch.float)

    def _compute_flow_loss_chunked(self, obs, action, eps, t):
        """Compute CFM loss with optional chunking along the K (MC sample) dimension.

        This avoids CUDA OOM when K is large by processing chunks sequentially.
        """
        chunk_size = self.config.mc_chunk_size
        reduction = self.config.cfm_loss_reduction
        k = eps.shape[1]
        if chunk_size is None or k <= chunk_size:
            return self.actor.compute_flow_loss(obs, action, eps, t, reduction=reduction)
        chunks = []
        for start in range(0, k, chunk_size):
            end = min(start + chunk_size, k)
            chunks.append(
                self.actor.compute_flow_loss(obs, action, eps[:, start:end], t[:, start:end], reduction=reduction)
            )
        return torch.cat(chunks, dim=1)

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                # Environment step
                actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)

                # Normalize observations (updates running stats in train mode)
                actor_obs = self.obs_normalizer(actor_obs)
                critic_obs = self.critic_obs_normalizer(critic_obs)

                # Generate actions via ODE integration
                actions = self.actor.act({"actor_obs": actor_obs})
                values = self.critic.evaluate({"critic_obs": critic_obs}).detach()

                # Sample MC noise and time for old-policy loss
                batch_size = actor_obs.shape[0]
                eps, t = self.actor.sample_mc_noise_and_time(batch_size, self.config.num_mc_samples, actor_obs.device)
                # Compute old-policy CFM loss: [B, K, 1]
                old_loss = self._compute_flow_loss_chunked(actor_obs, actions, eps, t)

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                    if self.config.obs_normalization:
                        final_critic_obs = self.critic_obs_normalizer(final_critic_obs, update=False)
                    final_values = self.critic.evaluate({"critic_obs": final_critic_obs}).detach()
                    final_rewards += self.config.gamma * torch.squeeze(
                        final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                    )

                # Add transition to storage
                # For FPO, we store dummy values for PPO-specific fields
                self.storage.add(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    actions=actions,
                    values=values,
                    actions_log_prob=torch.zeros(batch_size, 1, device=self.device),  # unused in FPO
                    action_mean=torch.zeros(batch_size, self.num_act, device=self.device),  # unused in FPO
                    action_sigma=torch.zeros(batch_size, self.num_act, device=self.device),  # unused in FPO
                    rewards=(rewards + final_rewards).view(-1, 1),
                    dones=dones.view(-1, 1),
                    flow_eps=eps.detach(),
                    flow_t=t.detach(),
                    flow_old_loss=old_loss.detach(),
                )

                # Reset actor and critic for completed envs
                self.actor.reset(dones)
                self.critic.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            if self.config.obs_normalization:
                last_critic_obs = self.critic_obs_normalizer(last_critic_obs, update=False)
            last_values = self.critic.evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_fpo_ratio(self, old_loss: torch.Tensor, new_loss: torch.Tensor) -> torch.Tensor:
        """Compute FPO surrogate ratio from CFM losses.

        Parameters
        ----------
        old_loss : Tensor
            Old-policy CFM loss, shape [B, K, 1]
        new_loss : Tensor
            New-policy CFM loss, shape [B, K, 1]

        Returns
        -------
        Tensor
            If ratio_mode == "per_sample": [B, K] per-sample ratios (NOT averaged).
            Otherwise: [B] averaged ratio.
        """
        clip_val = self.config.ratio_log_clip

        if self.config.ratio_mode == "per_sample":
            # Two-stage clamp (stage 2): clamp the difference
            log_r = torch.clamp(old_loss - new_loss, -clip_val, clip_val)  # [B, K, 1]
            ratio = torch.exp(log_r).squeeze(-1)  # [B, K]
        else:
            # Legacy avg: average losses first, then compute single ratio
            old_loss_avg = old_loss.mean(dim=1)  # [B, 1]
            new_loss_avg = new_loss.mean(dim=1)  # [B, 1]
            log_r = torch.clamp(old_loss_avg - new_loss_avg, -clip_val, clip_val)  # [B, 1]
            ratio = torch.exp(log_r).squeeze(-1)  # [B]

        return ratio

    def _compute_ppo_loss(self, minibatch):
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]

        # FPO-specific: retrieve stored MC samples
        flow_eps = minibatch["flow_eps"]  # [B, K, A]
        flow_t = minibatch["flow_t"]  # [B, K, 1]
        flow_old_loss = minibatch["flow_old_loss"]  # [B, K, 1]

        # Note: observations in storage are already normalized during rollout.
        # No additional normalization needed here.

        # Symmetry augmentation
        original_batch_size = actions_batch.shape[0]
        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["actor_obs"],
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"],
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_batch = self.symmetry_utils.augment_actions(actions=actions_batch)
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
            # Repeat MC samples for symmetry-augmented batch
            flow_eps = flow_eps.repeat(num_aug, 1, 1)
            flow_t = flow_t.repeat(num_aug, 1, 1)
            flow_old_loss = flow_old_loss.repeat(num_aug, 1, 1)
        else:
            actor_obs = minibatch["actor_obs"]
            critic_obs = minibatch["critic_obs"]

        # Compute new-policy CFM loss with gradients
        new_loss = self._compute_flow_loss_chunked(actor_obs, actions_batch, flow_eps, flow_t)  # [B, K, 1]

        # Compute critic values
        value_batch = self.critic.evaluate({"critic_obs": critic_obs})

        # Compute FPO ratio
        ratio = self._compute_fpo_ratio(flow_old_loss, new_loss)
        if not torch.isfinite(ratio).all():
            logger.warning("Non-finite ratio detected, replacing with 1.0")
            ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
        eps = self.config.clip_param

        if self.config.ratio_mode == "per_sample":
            # ratio: [B, K], advantages: [B, 1] -> expand to [B, K]
            adv = torch.squeeze(advantages_batch)  # [B]
            adv_k = adv.unsqueeze(1).expand_as(ratio)  # [B, K]

            if self.config.trust_region_mode == "aspo":
                # ASPO: A>=0 uses PPO clip, A<0 uses SPO quadratic penalty
                pos_mask = adv_k >= 0  # [B, K]
                ppo_obj = torch.min(adv_k * ratio, adv_k * ratio.clamp(1 - eps, 1 + eps))
                spo_obj = adv_k * ratio - (adv_k.abs() / (2 * eps)) * (ratio - 1).pow(2)
                objective = torch.where(pos_mask, ppo_obj, spo_obj)  # [B, K]
            else:
                # ppo_clip fallback
                objective = torch.min(adv_k * ratio, adv_k * ratio.clamp(1 - eps, 1 + eps))

            # Mean over both K (MC samples) and batch dimensions
            if not torch.isfinite(objective).all():
                logger.warning("Non-finite objective detected, replacing with 0.0")
                objective = torch.where(torch.isfinite(objective), objective, torch.zeros_like(objective))
            surrogate_loss = -objective.mean()
        else:
            # legacy_avg: ratio is [B], standard PPO clip
            adv = torch.squeeze(advantages_batch)
            surrogate = -adv * ratio
            surrogate_clipped = -adv * ratio.clamp(1 - eps, 1 + eps)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss (same as PPO)
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        # Symmetry losses
        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            with torch.no_grad():
                mean_actions_batch = self.actor.act_inference({"actor_obs": actor_obs.detach().clone()})
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = torch.nn.functional.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )
            symmetry_critic_loss = torch.nn.functional.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        # CFM loss regularization: prevents velocity field divergence
        cfm_loss_mean = new_loss.mean()
        cfm_loss_val = cfm_loss_mean.item()
        if self._ema_cfm_loss is None:
            self._ema_cfm_loss = cfm_loss_val
        else:
            self._ema_cfm_loss = (
                self.config.cfm_reg_ema_beta * self._ema_cfm_loss + (1.0 - self.config.cfm_reg_ema_beta) * cfm_loss_val
            )
        normalized_cfm_loss = cfm_loss_mean / max(self._ema_cfm_loss, 1.0)
        cfm_reg_loss = self.config.cfm_reg_coef * normalized_cfm_loss

        actor_loss = surrogate_loss + cfm_reg_loss + self.config.symmetry_actor_coef * symmetry_actor_loss
        if not torch.isfinite(actor_loss):
            logger.warning("Non-finite actor_loss detected, replacing with zero")
            actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        # Diagnostic metrics for 2x2 ablation (no grad impact)
        with torch.no_grad():
            ratio_flat = ratio.detach().flatten()
            ratio_p10 = torch.quantile(ratio_flat, 0.1)
            ratio_p50 = torch.quantile(ratio_flat, 0.5)
            ratio_p90 = torch.quantile(ratio_flat, 0.9)

            # Clamp firing rates
            old_l = flow_old_loss.detach()
            new_l = new_loss.detach()
            if self.config.cfm_loss_clip is not None:
                stage1_rate = (
                    ((old_l >= self.config.cfm_loss_clip) | (new_l >= self.config.cfm_loss_clip)).float().mean()
                )
                old_l = old_l.clamp(max=self.config.cfm_loss_clip)
                new_l = new_l.clamp(max=self.config.cfm_loss_clip)
            else:
                stage1_rate = torch.tensor(0.0, device=self.device)
            diff = old_l - new_l
            clip_val = self.config.ratio_log_clip
            stage2_rate = (diff.abs() >= clip_val).float().mean()

            # Cov(A, log_ratio) and sign agreement
            log_r_per = torch.log(ratio.detach().clamp(min=1e-8))  # [B,K] or [B]
            if log_r_per.dim() == 2:
                log_r_avg = log_r_per.mean(dim=1)  # [B]
            else:
                log_r_avg = log_r_per  # [B]
            adv_d = adv.detach()
            cov_a_logr = ((adv_d - adv_d.mean()) * (log_r_avg - log_r_avg.mean())).mean()
            sign_agree = ((adv_d > 0) == (log_r_avg > 0)).float().mean()

            # --- ratio停滞の因果識別用 診断指標 ---
            # KL(old, new): KL ≈ (ratio - 1) - log(ratio), always >= 0
            fpo_kl = (ratio.detach() - 1.0 - log_r_per).mean()

            # ΔL std: old_loss - new_loss の標準偏差 (ratio感度の直接指標)
            delta_loss = (flow_old_loss - new_loss).detach()  # [B, K, 1]
            fpo_delta_loss_std = delta_loss.std()

            # logr std: log-ratio の標準偏差 (ratio分散の直接指標)
            fpo_logr_std = log_r_per.std()

            # clipfrac: ratio が [1-eps, 1+eps] の外にある割合
            fpo_clipfrac = ((ratio.detach() - 1.0).abs() > eps).float().mean()

        # CFM loss ratio for divergence monitoring
        cfm_loss_ratio = cfm_loss_val / max(self._ema_cfm_loss, 1.0)

        # Return a dict compatible with PPO's _update_algo_step expectations
        # kl_mean is required by the parent; we return 0 since FPO doesn't use KL-based LR scheduling
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": torch.tensor(0.0, device=self.device),
            "kl_mean": torch.tensor(0.0, device=self.device),
            "cfm_loss": cfm_loss_mean,
            "cfm_reg_loss": cfm_reg_loss,
            "cfm_loss_ratio": cfm_loss_ratio,
            "fpo_ratio_mean": ratio.mean(),
            "fpo_ratio_p10": ratio_p10,
            "fpo_ratio_p50": ratio_p50,
            "fpo_ratio_p90": ratio_p90,
            "fpo_cov_adv_logr": cov_a_logr,
            "fpo_sign_agree": sign_agree,
            "fpo_clamp_stage1_rate": stage1_rate,
            "fpo_clamp_stage2_rate": stage2_rate,
            "fpo_kl_old_new": fpo_kl,
            "fpo_delta_loss_std": fpo_delta_loss_std,
            "fpo_logr_std": fpo_logr_std,
            "fpo_clipfrac": fpo_clipfrac,
        }

    def _update_algo_step(self, minibatch, loss_dict):
        ppo_loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        ppo_loss.backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Accumulate metrics (same as PPO base)
        loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
        loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
        loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
        loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
        for key, loss in ppo_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_value = loss.item() if torch.is_tensor(loss) else loss
            loss_dict[key] += loss_value

        # FPO-specific: actor grad norm (pre-clipping)
        if "fpo_actor_grad_norm" not in loss_dict:
            loss_dict["fpo_actor_grad_norm"] = 0.0
        loss_dict["fpo_actor_grad_norm"] += actor_grad_norm.item()

        return loss_dict

    def _post_epoch_logging(self, it, loss_dict):
        extra_log_dicts = {
            "Policy": {
                "cfm_loss": loss_dict.get("cfm_loss", 0.0),
                "cfm_reg_loss": loss_dict.get("cfm_reg_loss", 0.0),
                "cfm_loss_ratio": loss_dict.get("cfm_loss_ratio", 0.0),
                "fpo_ratio_mean": loss_dict.get("fpo_ratio_mean", 0.0),
                "fpo_ratio_p10": loss_dict.get("fpo_ratio_p10", 0.0),
                "fpo_ratio_p50": loss_dict.get("fpo_ratio_p50", 0.0),
                "fpo_ratio_p90": loss_dict.get("fpo_ratio_p90", 0.0),
                "fpo_cov_adv_logr": loss_dict.get("fpo_cov_adv_logr", 0.0),
                "fpo_sign_agree": loss_dict.get("fpo_sign_agree", 0.0),
                "fpo_clamp_stage1_rate": loss_dict.get("fpo_clamp_stage1_rate", 0.0),
                "fpo_clamp_stage2_rate": loss_dict.get("fpo_clamp_stage2_rate", 0.0),
                "fpo_kl_old_new": loss_dict.get("fpo_kl_old_new", 0.0),
                "fpo_delta_loss_std": loss_dict.get("fpo_delta_loss_std", 0.0),
                "fpo_logr_std": loss_dict.get("fpo_logr_std", 0.0),
                "fpo_clipfrac": loss_dict.get("fpo_clipfrac", 0.0),
                "fpo_actor_grad_norm": loss_dict.get("fpo_actor_grad_norm", 0.0),
            },
        }
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    # ===== Save / Load =====

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.config.obs_normalization:
            checkpoint_dict["obs_normalizer_state"] = self.obs_normalizer.state_dict()
            checkpoint_dict["critic_obs_normalizer_state"] = self.critic_obs_normalizer.state_dict()
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.config.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            if self.config.obs_normalization and "obs_normalizer_state" in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_normalizer_state"])
                self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer_state"])
                logger.info("Loaded observation normalizer states from checkpoint")
            self.current_learning_iteration = loaded_dict["iter"]
            self._restore_env_state(loaded_dict.get("env_state"))
            return loaded_dict.get("infos")
        return None

    # ===== Divergence Guard =====

    def _should_early_stop(self, it: int, loss_dict: dict) -> bool:
        """Check for velocity field divergence and trigger early stopping if needed."""
        if not self.config.divergence_guard_enabled:
            return False
        if it < self.config.divergence_warmup_iters:
            return False

        cfm_loss_ratio = loss_dict.get("cfm_loss_ratio", 1.0)
        stage2_rate = float(loss_dict.get("fpo_clamp_stage2_rate", 0.0))
        cov_adv_logr = float(loss_dict.get("fpo_cov_adv_logr", 0.0))

        # Immediate stop: any single hard threshold breach
        if cfm_loss_ratio >= self.config.div_hard_cfm_ratio or stage2_rate >= self.config.div_hard_stage2_rate:
            logger.error(f"HARD STOP: cfm_loss_ratio={cfm_loss_ratio:.1f}, stage2_rate={stage2_rate:.3f}")
            return True

        # Warning: log at warn level but don't stop
        warn_count = sum(
            [
                cfm_loss_ratio >= self.config.div_warn_cfm_ratio,
                stage2_rate >= self.config.div_warn_stage2_rate,
                cov_adv_logr <= self.config.div_warn_cov_adv_logr,
            ]
        )
        if warn_count > 0:
            logger.warning(
                f"Divergence warning ({warn_count}/3): cfm_loss_ratio={cfm_loss_ratio:.1f}, "
                f"stage2_rate={stage2_rate:.3f}, cov_adv_logr={cov_adv_logr:.4f}"
            )

        # Stop condition: 2+ of 3 stop thresholds met
        stop_count = sum(
            [
                cfm_loss_ratio >= self.config.div_stop_cfm_ratio,
                stage2_rate >= self.config.div_stop_stage2_rate,
                cov_adv_logr <= self.config.div_stop_cov_adv_logr,
            ]
        )
        if stop_count >= 2:
            self._div_consecutive_count += 1
            logger.warning(
                f"Divergence stop condition met ({self._div_consecutive_count}/"
                f"{self.config.div_stop_consecutive}): {stop_count}/3 thresholds breached"
            )
            if self._div_consecutive_count >= self.config.div_stop_consecutive:
                logger.error(
                    f"EARLY STOP: {self.config.div_stop_consecutive} consecutive divergence detections. "
                    f"cfm_loss_ratio={cfm_loss_ratio:.1f}, stage2_rate={stage2_rate:.3f}, "
                    f"cov_adv_logr={cov_adv_logr:.4f}"
                )
                return True
        else:
            self._div_consecutive_count = 0

        return False

    # ===== ONNX / Inference =====

    @property
    def actor_onnx_wrapper(self):
        """Return an ONNX-exportable wrapper with fixed K steps unrolled.

        Uses zeros as the initial state instead of torch.randn to ensure
        deterministic ONNX export (avoids device mismatch during constant folding).
        At inference time on a real robot, the ONNX model produces a deterministic
        policy (the ODE-integrated "mean" from the zero-noise starting point).
        """
        num_steps = self.config.onnx_export_num_flow_steps or self.config.num_flow_steps
        actor = self.actor
        use_obs_norm = self.config.obs_normalization
        obs_norm_copy = copy.deepcopy(self.obs_normalizer).cpu() if use_obs_norm else None

        class FlowPolicyONNXWrapper(nn.Module):
            def __init__(self, flow_actor, export_num_flow_steps, obs_normalizer):
                super().__init__()
                self.velocity_field = flow_actor.velocity_field
                self.num_actions = flow_actor.num_actions
                self.export_num_flow_steps = export_num_flow_steps
                self._obs_dim = flow_actor.obs_dim
                self._action_bound = flow_actor.action_bound
                self._flow_param_mode = flow_actor.flow_param_mode
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                if self.obs_normalizer is not None:
                    actor_obs = self.obs_normalizer(actor_obs, update=False)
                k = self.export_num_flow_steps
                dt = 1.0 / k
                # Start from zeros (deterministic) instead of randn.
                # Use new_zeros to inherit device/dtype from input tensor.
                x = actor_obs.new_zeros(actor_obs.shape[0], self.num_actions)
                for j in range(k, 0, -1):
                    t_val = j / k
                    t_tensor = actor_obs.new_full((actor_obs.shape[0], 1), t_val)
                    output = self.velocity_field(actor_obs, x, t_tensor)
                    if self._flow_param_mode == "velocity":
                        x = x + dt * output
                    else:
                        t_prev = (j - 1) / k
                        x = x * (t_prev / t_val) + output * (dt / t_val)
                return self._action_bound * torch.tanh(x)

        return FlowPolicyONNXWrapper(actor, num_steps, obs_norm_copy)

    def export(self, onnx_file_path: str):
        """Export FPO policy as ONNX.

        Overrides PPO.export to bypass _OnnxMotionPolicyExporter which cannot
        decompose the FlowPolicy ODE-loop wrapper. Always uses the simple
        export_policy_as_onnx path directly.
        """
        was_training = self.actor.training
        self._eval_mode()

        wrapper = self.actor_onnx_wrapper
        # Move wrapper to CPU for ONNX export (all tensors on same device)
        wrapper = wrapper.cpu()
        example_obs = torch.zeros(1, wrapper._obs_dim)

        export_policy_as_onnx(
            wrapper=wrapper,
            onnx_file_path=onnx_file_path,
            example_obs_dict={"actor_obs": example_obs},
        )

        # Move wrapper back (it shares velocity_field with self.actor)
        wrapper.to(self.device)

        # Attach metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=metadata)

        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    def get_inference_policy(
        self, device: str | None = None, num_flow_steps: int | None = None
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        self.actor.eval()
        if device is not None:
            self.actor.to(device)

        flow_steps = num_flow_steps or self.config.num_flow_steps

        if self.config.obs_normalization:
            normalizer = self.obs_normalizer
            normalizer.eval()
            if device is not None:
                normalizer.to(device)

            def policy_fn(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
                normalized_obs = normalizer(obs_dict["actor_obs"], update=False)
                return self.actor.act_inference({"actor_obs": normalized_obs}, num_flow_steps=flow_steps)
        else:

            def policy_fn(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
                return self.actor.act_inference(obs_dict, num_flow_steps=flow_steps)

        return policy_fn

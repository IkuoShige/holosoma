"""Verify that g1_29dof_fpo_pp_repro preset pins all tuned values."""

from __future__ import annotations

from holosoma.config_values.experiment import DEFAULTS


def test_fpo_pp_repro_preset_values():
    cfg = DEFAULTS["g1_29dof_fpo_pp_repro"].algo.config

    # Tuned hyperparameters (diverged from paper during G1 experiments)
    assert cfg.num_learning_epochs == 8
    assert cfg.num_steps_per_env == 24
    assert cfg.num_mini_batches == 4
    assert cfg.lam == 0.95
    assert cfg.clip_param == 0.05
    assert cfg.cfm_reg_coef == 0.0
    assert cfg.module_dict.actor.layer_config.hidden_dims == [256, 256, 256]
    assert cfg.module_dict.critic.layer_config.hidden_dims == [768, 768, 768]

    # Explicitly pinned reproduction parameters
    assert cfg.num_flow_steps == 64
    assert cfg.num_mc_samples == 16
    assert cfg.ratio_mode == "per_sample"
    assert cfg.ratio_log_clip == 1.0
    assert cfg.trust_region_mode == "aspo"
    assert cfg.action_bound == 1.5
    assert cfg.max_grad_norm == 0.5
    assert cfg.flow_param_mode == "velocity"
    assert cfg.cfm_loss_reduction == "mean"
    assert cfg.cfm_loss_clip == 10000.0
    assert cfg.obs_normalization is True
    assert cfg.divergence_guard_enabled is True

    # Optimizer settings (inherited from fpo default)
    assert cfg.actor_optimizer.betas == (0.9, 0.95)
    assert cfg.actor_optimizer.weight_decay == 1e-4
    assert cfg.critic_optimizer.betas == (0.9, 0.95)
    assert cfg.critic_optimizer.weight_decay == 1e-4


def test_fpo_pp_warmstart_probe_preset_values():
    cfg = DEFAULTS["g1_29dof_fpo_pp_warmstart_probe"].algo.config

    # Same FPO hyperparams as pp_repro
    assert cfg.num_learning_epochs == 8
    assert cfg.action_bound == 1.5
    assert cfg.cfm_loss_reduction == "mean"
    assert cfg.ratio_log_clip == 1.0

    # BC warm-start specific
    assert cfg.warm_start_mode == "bc_teacher"
    assert cfg.warm_start_checkpoint is None  # Must be set at runtime
    assert cfg.warm_start_bc_steps == 200
    assert cfg.warm_start_load_critic is False

    # Teacher module matches PPO default architecture
    assert cfg.warm_start_teacher_module is not None
    assert cfg.warm_start_teacher_module.layer_config.hidden_dims == [512, 256, 128]

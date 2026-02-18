"""Verify that g1_29dof_fpo_pp_repro preset pins all paper-critical values."""

from __future__ import annotations

from holosoma.config_values.experiment import DEFAULTS


def test_fpo_pp_repro_preset_values():
    cfg = DEFAULTS["g1_29dof_fpo_pp_repro"].algo.config

    # Paper-specified hyperparameters
    assert cfg.num_learning_epochs == 32
    assert cfg.num_steps_per_env == 96
    assert cfg.num_mini_batches == 16
    assert cfg.lam == 0.95
    assert cfg.clip_param == 0.05
    assert cfg.cfm_reg_coef == 0.0
    assert cfg.module_dict.actor.layer_config.hidden_dims == [256, 256, 256]
    assert cfg.module_dict.critic.layer_config.hidden_dims == [768, 768, 768]

    # Explicitly pinned reproduction parameters
    assert cfg.num_flow_steps == 64
    assert cfg.num_mc_samples == 16
    assert cfg.ratio_mode == "per_sample"
    assert cfg.ratio_log_clip == 0.5
    assert cfg.trust_region_mode == "aspo"
    assert cfg.action_bound == 3.0
    assert cfg.max_grad_norm == 0.5
    assert cfg.flow_param_mode == "velocity"
    assert cfg.cfm_loss_reduction == "sum"
    assert cfg.obs_normalization is True
    assert cfg.divergence_guard_enabled is True

    # Optimizer settings (inherited from fpo default)
    assert cfg.actor_optimizer.betas == (0.9, 0.95)
    assert cfg.actor_optimizer.weight_decay == 1e-4
    assert cfg.critic_optimizer.betas == (0.9, 0.95)
    assert cfg.critic_optimizer.weight_decay == 1e-4

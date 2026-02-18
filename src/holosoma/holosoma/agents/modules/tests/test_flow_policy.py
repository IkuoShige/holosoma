"""Tests for FlowPolicy with velocity and data parameterization modes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from holosoma.agents.modules.flow_policy import FlowPolicy
from holosoma.config_types.algo import LayerConfig, ModuleConfig


@pytest.fixture
def module_config():
    """Create a simple MLP module configuration for FlowPolicy."""
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
    )
    return ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        layer_config=layer_config,
    )


@pytest.fixture
def obs_dim_dict():
    return {"actor_obs": 48}


@pytest.fixture
def history_length():
    return {"actor_obs": 1}


NUM_ACTIONS = 12
BATCH_SIZE = 8
NUM_MC_SAMPLES = 4


def _make_flow_policy(obs_dim_dict, module_config, history_length, flow_param_mode="velocity"):
    return FlowPolicy(
        obs_dim_dict=obs_dim_dict,
        module_config=module_config,
        num_actions=NUM_ACTIONS,
        history_length=history_length,
        time_embed_dim=32,
        use_ada_ln=True,
        num_flow_steps=8,
        action_bound=3.0,
        flow_param_mode=flow_param_mode,
    )


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_compute_flow_loss_shape(obs_dim_dict, module_config, history_length, mode):
    """compute_flow_loss returns [B, K, 1] for both modes."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
    obs = torch.randn(BATCH_SIZE, 48)
    action = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    eps = torch.randn(BATCH_SIZE, NUM_MC_SAMPLES, NUM_ACTIONS)
    t = torch.rand(BATCH_SIZE, NUM_MC_SAMPLES, 1)

    loss = policy.compute_flow_loss(obs, action, eps, t)
    assert loss.shape == (BATCH_SIZE, NUM_MC_SAMPLES, 1)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_act_output_shape(obs_dim_dict, module_config, history_length, mode):
    """act() returns [B, A] within action_bound for both modes."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
    obs_dict = {"actor_obs": torch.randn(BATCH_SIZE, 48)}

    actions = policy.act(obs_dict)
    assert actions.shape == (BATCH_SIZE, NUM_ACTIONS)
    assert (actions.abs() <= policy.action_bound).all()


def test_data_mode_last_step_equals_d_theta(obs_dim_dict, module_config, history_length):
    """With K=1 in data mode, final output x_0 = d_theta(x_1, 1; obs)."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, "data")
    obs = torch.randn(BATCH_SIZE, 48)
    obs_dict = {"actor_obs": obs}

    # K=1: single step from t=1 to t=0
    # x_0 = x_1 * (0/1) + d_theta * (1/1) = d_theta
    torch.manual_seed(42)
    actions_k1 = policy.act(obs_dict, num_flow_steps=1)

    # Manually compute: start from same noise, get d_theta at t=1
    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    t_tensor = obs.new_full((BATCH_SIZE, 1), 1.0)
    with torch.no_grad():
        d_theta = policy.velocity_field(obs, x, t_tensor)
    expected = policy.action_bound * torch.tanh(d_theta)

    torch.testing.assert_close(actions_k1, expected)


def test_velocity_field_never_receives_t_zero(obs_dim_dict, module_config, history_length):
    """In both modes, the velocity field is never called with t=0."""
    for mode in ("velocity", "data"):
        policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
        obs_dict = {"actor_obs": torch.randn(BATCH_SIZE, 48)}

        original_forward = policy.velocity_field.forward
        t_values_seen = []

        def _capture_t(obs, x_t, t, _orig=original_forward, _seen=t_values_seen):
            _seen.append(t.clone())
            return _orig(obs, x_t, t)

        with patch.object(policy.velocity_field, "forward", side_effect=_capture_t):
            policy.act(obs_dict)

        assert len(t_values_seen) > 0, "velocity_field should be called at least once"
        for t_tensor in t_values_seen:
            assert (t_tensor > 0).all(), f"t=0 detected in mode '{mode}': {t_tensor}"


def test_invalid_flow_param_mode(obs_dim_dict, module_config, history_length):
    """Invalid flow_param_mode raises ValueError."""
    with pytest.raises(ValueError, match="flow_param_mode must be"):
        _make_flow_policy(obs_dim_dict, module_config, history_length, "invalid")


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_act_inference_is_deterministic(obs_dim_dict, module_config, history_length, mode):
    """act_inference() produces identical output across multiple calls (deterministic)."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
    policy.eval()
    obs_dict = {"actor_obs": torch.randn(BATCH_SIZE, 48)}

    with torch.no_grad():
        result1 = policy.act_inference(obs_dict)
        result2 = policy.act_inference(obs_dict)

    torch.testing.assert_close(result1, result2)


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_act_inference_differs_from_stochastic_act(obs_dim_dict, module_config, history_length, mode):
    """act_inference() (zeros start) differs from act() (randn start)."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
    policy.eval()
    obs_dict = {"actor_obs": torch.randn(BATCH_SIZE, 48)}

    with torch.no_grad():
        deterministic = policy.act_inference(obs_dict)
        stochastic = policy.act(obs_dict)

    # With overwhelming probability, random noise start != zeros start
    assert not torch.allclose(deterministic, stochastic, atol=1e-3)


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_backward_computes_gradients(obs_dim_dict, module_config, history_length, mode):
    """Gradients flow through compute_flow_loss for both modes."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, mode)
    obs = torch.randn(BATCH_SIZE, 48)
    action = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    eps = torch.randn(BATCH_SIZE, NUM_MC_SAMPLES, NUM_ACTIONS)
    t = torch.rand(BATCH_SIZE, NUM_MC_SAMPLES, 1)

    loss = policy.compute_flow_loss(obs, action, eps, t)
    total_loss = loss.mean()
    total_loss.backward()

    has_grad = False
    for p in policy.velocity_field.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, f"No gradients computed in mode '{mode}'"


def test_cfm_loss_sum_vs_mean(obs_dim_dict, module_config, history_length):
    """Sum reduction produces action_dim times larger loss than mean reduction."""
    policy = _make_flow_policy(obs_dim_dict, module_config, history_length, "velocity")
    obs = torch.randn(BATCH_SIZE, 48)
    action = torch.randn(BATCH_SIZE, NUM_ACTIONS)
    eps = torch.randn(BATCH_SIZE, NUM_MC_SAMPLES, NUM_ACTIONS)
    t = torch.rand(BATCH_SIZE, NUM_MC_SAMPLES, 1)

    loss_sum = policy.compute_flow_loss(obs, action, eps, t, reduction="sum")
    loss_mean = policy.compute_flow_loss(obs, action, eps, t, reduction="mean")

    assert loss_sum.shape == loss_mean.shape == (BATCH_SIZE, NUM_MC_SAMPLES, 1)
    torch.testing.assert_close(loss_sum, loss_mean * NUM_ACTIONS, rtol=1e-4, atol=1e-6)

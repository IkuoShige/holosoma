"""Phase 2 unit tests for the vendored FlashSAC code path.

These run on CPU without launching IsaacSim, so they catch namespace-rewrite
bugs, Hydra global-state issues, dataclass field drift, and basic
agent/buffer plumbing before we burn time on the GPU smoke run.

Run with::

    pytest tests/unit/test_flash_rl_vendor.py -v

(Requires the ``hssim`` conda env activated.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, fields

import numpy as np
import pytest
import torch
from gymnasium import spaces


def test_imports_pure_modules() -> None:
    """Vendored FlashSAC modules import without launching IsaacSim."""
    import holosoma._vendored.flash_rl  # noqa: F401
    from holosoma._vendored.flash_rl.agents import create_agent  # noqa: F401
    from holosoma._vendored.flash_rl.agents.flashSAC.agent import (  # noqa: F401
        FlashSACAgent,
        FlashSACConfig,
    )
    from holosoma._vendored.flash_rl.agents.flashSAC.layer import UnitLinear  # noqa: F401
    from holosoma._vendored.flash_rl.agents.flashSAC.network import (  # noqa: F401
        FlashSACActor,
        FlashSACDoubleCritic,
    )
    from holosoma._vendored.flash_rl.agents.flashSAC.update import (  # noqa: F401
        update_actor,
        update_critic,
        update_temperature,
    )
    from holosoma._vendored.flash_rl.agents.utils.network import Network  # noqa: F401
    from holosoma._vendored.flash_rl.agents.utils.reward_normalization import (  # noqa: F401
        RewardNormalizer,
    )
    from holosoma._vendored.flash_rl.agents.utils.scheduler import (  # noqa: F401
        warmup_cosine_decay_scheduler,
    )
    from holosoma._vendored.flash_rl.buffers.torch_buffer import TorchUniformBuffer  # noqa: F401
    from holosoma._vendored.flash_rl.common import create_logger  # noqa: F401
    from holosoma._vendored.flash_rl.common.logger import (  # noqa: F401
        TensorboardTrainerLogger,
    )
    from holosoma._vendored.flash_rl.envs import create_envs  # noqa: F401
    from holosoma._vendored.flash_rl.evaluation import evaluate, record_video  # noqa: F401


def test_resolve_compile_mode_for_torch_2_7() -> None:
    """``_resolve_compile_mode('auto')`` resolves to ``reduce-overhead`` for torch < 2.9."""
    from holosoma._vendored.flash_rl.agents.flashSAC.agent import _resolve_compile_mode

    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    expected = "max-autotune" if (major, minor) >= (2, 9) else "reduce-overhead"
    assert _resolve_compile_mode("auto") == expected
    # explicit modes pass through unchanged
    assert _resolve_compile_mode("default") == "default"
    assert _resolve_compile_mode("reduce-overhead") == "reduce-overhead"


def test_flashsac_config_dataclass_construction() -> None:
    """``FlashSACConfig`` accepts the same field set the YAML defines."""
    from holosoma._vendored.flash_rl.agents.flashSAC.agent import FlashSACConfig

    cfg = FlashSACConfig(
        seed=0,
        normalize_reward=False,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type="cpu",
        buffer_max_length=128,
        buffer_min_length=8,
        buffer_device_type="cpu",
        sample_batch_size=8,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=10,
        actor_num_blocks=1,
        actor_hidden_dim=16,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=8,
        actor_update_period=1,
        critic_num_blocks=1,
        critic_hidden_dim=16,
        critic_num_bins=11,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,
        gamma=0.99,
        n_step=1,
        use_compile=False,
        compile_mode="reduce-overhead",
        use_amp=False,
        load_optimizer=True,
        load_reward_normalizer=True,
    )
    assert cfg.actor_hidden_dim == 16
    assert cfg.use_compile is False
    # All field names look reasonable; if upstream renames a field this dies fast.
    field_names = {f.name for f in fields(FlashSACConfig)}
    assert "asymmetric_observation" in field_names
    assert "critic_num_bins" in field_names


def test_flashsac_agent_cpu_init_and_one_update() -> None:
    """Construct ``FlashSACAgent`` on CPU and run a single update with synthetic data.

    Smoke-tests the entire FlashSAC update path (network init, weight
    normalization, replay buffer, categorical critic update, target EMA) on
    CPU so failures here cannot blame IsaacSim.
    """
    from holosoma._vendored.flash_rl.agents.flashSAC.agent import (
        FlashSACAgent,
        FlashSACConfig,
    )

    obs_dim = 6
    act_dim = 2
    num_envs = 4
    cfg = FlashSACConfig(
        seed=0,
        normalize_reward=False,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type="cpu",
        buffer_max_length=64,
        buffer_min_length=4,
        buffer_device_type="cpu",
        sample_batch_size=4,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=10,
        actor_num_blocks=1,
        actor_hidden_dim=16,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=4,
        actor_update_period=1,
        critic_num_blocks=1,
        critic_hidden_dim=16,
        critic_num_bins=11,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,
        gamma=0.99,
        n_step=1,
        use_compile=False,
        compile_mode="reduce-overhead",
        use_amp=False,
        load_optimizer=True,
        load_reward_normalizer=True,
    )

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    env_info = {"actor_observation_size": (obs_dim,), "asymmetric_obs": False}

    agent = FlashSACAgent(
        observation_space=obs_space,
        action_space=act_space,
        env_info=env_info,
        cfg=cfg,
    )
    assert not agent.can_start_training()  # buffer empty

    # Push enough transitions to satisfy buffer_min_length=4 (with num_envs=4 → 1 step)
    rng = np.random.default_rng(0)
    for step in range(2):
        obs = rng.standard_normal((num_envs, obs_dim)).astype(np.float32)
        next_obs = rng.standard_normal((num_envs, obs_dim)).astype(np.float32)
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
        rewards = rng.standard_normal((num_envs,)).astype(np.float32)
        terminated = np.zeros((num_envs,), dtype=bool)
        truncated = np.zeros((num_envs,), dtype=bool)
        agent.process_transition(
            {
                "observation": obs,
                "action": actions,
                "reward": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": next_obs,
            }
        )

    assert agent.can_start_training()
    update_info = agent.update()
    assert isinstance(update_info, dict)
    assert len(update_info) > 0
    # at least one finite numeric metric (e.g., a critic/actor loss scalar)
    finite_metrics = [
        v for v in update_info.values()
        if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    assert len(finite_metrics) > 0, f"no finite metrics in update_info: {update_info}"


def test_hydra_compose_idempotent() -> None:
    """``build_cfg`` can be called repeatedly within the same process."""
    from holosoma._vendored.flash_rl.train import build_cfg

    cfg1 = build_cfg(
        overrides=[
            "env=isaaclab",
            "env.env_name=Isaac-Velocity-Flat-G1-v0",
            "num_env_steps=320",
            "num_train_envs=64",
        ]
    )
    cfg2 = build_cfg(overrides=["env=isaaclab", "agent.use_compile=false"])
    cfg3 = build_cfg(overrides=["env=mujoco"])
    assert cfg1.env.env_name == "Isaac-Velocity-Flat-G1-v0"
    assert cfg2.agent.use_compile is False
    assert cfg3.env.env_type == "mujoco"


def test_yaml_compose_resolves_eval() -> None:
    """The ``${eval: ...}`` resolver actually evaluates expressions."""
    from holosoma._vendored.flash_rl.train import build_cfg

    cfg = build_cfg(
        overrides=[
            "env=isaaclab",
            "num_env_steps=320",
            "num_train_envs=64",
        ]
    )
    # num_interaction_steps = num_env_steps / num_train_envs = 320 / 64 = 5
    assert int(cfg.num_interaction_steps) == 5


def test_torch_buffer_can_sample_off_by_one() -> None:
    """Vectorized add of N transitions makes ``can_sample()`` true at exactly N.

    Codex flagged that the original plan's ``buffer_min_length=64`` claim said
    update() runs at iteration 2; in fact ``TorchUniformBuffer.can_sample`` uses
    ``>=`` so update() runs at iteration 1.
    """
    from holosoma._vendored.flash_rl.buffers.torch_buffer import TorchUniformBuffer

    obs_dim = 4
    act_dim = 2
    num_envs = 8
    buf = TorchUniformBuffer(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        max_length=64,
        min_length=8,
        sample_batch_size=4,
        n_step=1,
        gamma=0.99,
        device_type="cpu",
    )
    assert not buf.can_sample(), "freshly constructed buffer cannot sample"
    # one vectorized add of 8 transitions
    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    nxt = np.zeros((num_envs, obs_dim), dtype=np.float32)
    act = np.zeros((num_envs, act_dim), dtype=np.float32)
    rew = np.zeros((num_envs,), dtype=np.float32)
    done = np.zeros((num_envs,), dtype=bool)
    buf.add(
        {
            "observation": obs,
            "action": act,
            "reward": rew,
            "terminated": done,
            "truncated": done,
            "next_observation": nxt,
        }
    )
    # After exactly one vectorized add of `num_envs` transitions, len(buffer) == num_envs
    # which is >= min_length=8, so can_sample() must be True.
    assert len(buf) == num_envs
    assert buf.can_sample(), "buffer with num_envs == min_length should be sampleable"

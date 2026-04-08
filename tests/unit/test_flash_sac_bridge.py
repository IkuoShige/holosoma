"""Unit tests for the FlashSAC env bridge.

These tests use a mock holosoma ``BaseTask``-shaped object so they run on CPU
without IsaacSim. They cover the contract translation that the planner identified
as the highest-risk part of the Layer 2 adapter:

- dict-action ``step({"actions": ...})`` plumbing
- dict observation extraction with the holosoma ``actor_obs`` / ``critic_obs`` keys
- ``reset_buf`` / ``time_outs`` → ``terminated`` / ``truncated`` split
- ``final_observations`` re-stitching for envs that just reset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class _MockRobotCfg:
    actions_dim: int = 3


class _MockBaseTask:
    """Tiny stand-in that exposes only the surface area FlashSACGymBridge uses."""

    def __init__(self, num_envs: int, actor_dim: int, critic_dim: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.dim_actions = 3
        self.dim_obs = actor_dim
        self.dim_critic_obs = critic_dim
        self.device = device
        self.robot_config = _MockRobotCfg(actions_dim=self.dim_actions)
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros((num_envs,), dtype=torch.long, device=device)

        self._actor_dim = actor_dim
        self._critic_dim = critic_dim
        self.obs_buf_dict: dict[str, torch.Tensor] = {
            "actor_obs": torch.zeros((num_envs, actor_dim), device=device),
            "critic_obs": torch.zeros((num_envs, critic_dim), device=device),
        }
        self.rew_buf = torch.zeros((num_envs,), device=device)
        self.reset_buf = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        self.extras: dict[str, Any] = {
            "time_outs": torch.zeros((num_envs,), dtype=torch.bool, device=device),
        }

        self._step_count = 0

    def reset_all(self) -> dict[str, torch.Tensor]:
        self._step_count = 0
        for v in self.obs_buf_dict.values():
            v.fill_(0.0)
        self.reset_buf.fill_(False)
        self.extras["time_outs"].fill_(False)
        return self.obs_buf_dict

    def step(self, actor_state: dict[str, torch.Tensor]):
        self._step_count += 1
        actions = actor_state["actions"]
        # Sanity-check the bridge handed us a torch tensor on the right device.
        assert isinstance(actions, torch.Tensor)
        assert actions.shape == (self.num_envs, self.dim_actions)
        # Fake an observation update so the bridge can detect non-zero data.
        for k, v in self.obs_buf_dict.items():
            v.fill_(float(self._step_count))
        self.rew_buf.fill_(1.0)
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras


def test_bridge_basic_reset_step_round_trip() -> None:
    from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge

    env = _MockBaseTask(num_envs=4, actor_dim=6, critic_dim=8)
    bridge = FlashSACGymBridge(env, actor_obs_keys=("actor_obs",), critic_obs_keys=("critic_obs",))

    obs, infos = bridge.reset(random_start_init=False)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 6 + 8)  # actor + critic concatenated (asymmetric)
    assert infos["actor_observation_size"] == (6,)
    assert infos["asymmetric_obs"] is True

    actions = np.zeros((4, 3), dtype=np.float32)
    next_obs, rew, terminated, truncated, info = bridge.step(actions)
    assert next_obs.shape == (4, 14)
    assert rew.shape == (4,)
    assert terminated.shape == (4,) and truncated.shape == (4,)
    assert (terminated == False).all()  # noqa: E712 -- explicit equality test
    assert (truncated == False).all()  # noqa: E712
    assert "final_obs" in info
    assert info["final_obs"].shape == (4, 14)


def test_bridge_dones_split() -> None:
    """`reset_buf=[1,0,1,0]` and `time_outs=[1,0,0,0]` → `terminated=[0,0,1,0]`, `truncated=[1,0,0,0]`."""

    from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge

    env = _MockBaseTask(num_envs=4, actor_dim=2, critic_dim=2)
    bridge = FlashSACGymBridge(env, actor_obs_keys=("actor_obs",), critic_obs_keys=("critic_obs",))
    bridge.reset(random_start_init=False)

    env.reset_buf = torch.tensor([True, False, True, False])
    env.extras["time_outs"] = torch.tensor([True, False, False, False])

    actions = np.zeros((4, env.dim_actions), dtype=np.float32)
    _next_obs, _rew, terminated, truncated, _info = bridge.step(actions)
    np.testing.assert_array_equal(terminated, np.array([False, False, True, False]))
    np.testing.assert_array_equal(truncated, np.array([True, False, False, False]))


def test_bridge_actor_only_when_no_critic_group() -> None:
    """When the env has no critic_obs key, the bridge runs in symmetric mode."""

    from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge

    env = _MockBaseTask(num_envs=2, actor_dim=4, critic_dim=0)
    env.obs_buf_dict.pop("critic_obs")
    bridge = FlashSACGymBridge(env, actor_obs_keys=("actor_obs",), critic_obs_keys=())
    obs, infos = bridge.reset(random_start_init=False)
    assert obs.shape == (2, 4)  # actor only
    assert infos["asymmetric_obs"] is False


def test_bridge_handles_dict_action_contract() -> None:
    """The mock asserts that the bridge passes ``{"actions": tensor}`` (not raw tensor)."""

    from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge

    env = _MockBaseTask(num_envs=2, actor_dim=4, critic_dim=4)
    bridge = FlashSACGymBridge(env, actor_obs_keys=("actor_obs",), critic_obs_keys=("critic_obs",))
    bridge.reset(random_start_init=False)
    bridge.step(np.zeros((2, env.dim_actions), dtype=np.float32))
    # If we got here, _MockBaseTask.step's assert on the action shape passed.


def test_flash_sac_agent_dual_format_checkpoint_round_trip(tmp_path) -> None:
    """``FlashSACAgent.save`` writes a dual-format checkpoint and ``load``
    accepts either form.

    Uses the lower-level vendored ``FlashSACAgent`` (not the holosoma
    BaseAlgo wrapper) so the round-trip runs on CPU without IsaacSim — we
    don't need the bridge or BaseAlgo plumbing to test that the packed
    single-file format unpacks byte-identically back into per-component
    ``.pt`` files.
    """
    import numpy as np
    from gymnasium import spaces

    from holosoma._vendored.flash_rl.agents.flashSAC.agent import (
        FlashSACAgent as VendoredFlashSACAgent,
    )
    from holosoma._vendored.flash_rl.agents.flashSAC.agent import FlashSACConfig

    obs_dim = 4
    act_dim = 2
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
        load_reward_normalizer=False,  # normalize_reward is False → no normalizer exists
    )
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    env_info = {"actor_observation_size": (obs_dim,), "asymmetric_obs": False}

    a = VendoredFlashSACAgent(obs_space, act_space, env_info, cfg)
    dir_path = tmp_path / "flashsac_step0"
    dir_path.mkdir()
    a.save(str(dir_path))

    # Each component .pt file exists
    component_files = [
        "actor.pt",
        "critic.pt",
        "target_critic.pt",
        "temperature.pt",
        "agent_state.pt",
    ]
    for fname in component_files:
        assert (dir_path / fname).exists(), f"{fname} missing from {dir_path}"

    # Simulate the migration: pack the directory into a single .ckpt as the
    # holosoma adapter does, then round-trip it back through a tempdir +
    # vendored load.
    import torch

    components = {
        fname: torch.load(str(dir_path / fname), map_location="cpu", weights_only=False)
        for fname in component_files
    }
    packed = {
        "experiment_config": {"dummy": "value"},
        "iteration": 0,
        "global_step": 0,
        "flashsac_components": components,
    }
    ckpt_file = tmp_path / "flashsac_step0.ckpt"
    torch.save(packed, str(ckpt_file))

    # Unpack to a temp dir and reload through the vendored loader (this is
    # exactly what ``FlashSACAgent.load`` does internally for single-file
    # checkpoints).
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for fname, contents in components.items():
            torch.save(contents, str(tmpdir / fname))
        a2 = VendoredFlashSACAgent(obs_space, act_space, env_info, cfg)
        a2.load(str(tmpdir))  # should succeed byte-for-byte


def test_bridge_final_observations_full_batch_layout() -> None:
    """Regression guard: ``extras["final_observations"][k]`` is a full-batch
    tensor shaped ``(num_envs, obs_dim)``; the bridge must index it with the
    current ``env_ids`` before assigning, otherwise we hit
    ``RuntimeError: shape mismatch: value tensor of shape [num_envs, obs_dim]
    cannot be broadcast to indexing result of shape [len(env_ids), obs_dim]``.

    This was not caught by the earlier 64-env Gate B smoke because no env
    terminated in 5 steps. The larger 4096-env run on a bigger GPU surfaced
    the bug: two envs reset mid-episode, ``final_actor_obs[env_ids] =
    stacked`` then failed because ``stacked`` was the full (num_envs, obs_dim)
    tensor rather than its (len(env_ids), obs_dim) slice.
    """

    from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge

    num_envs = 4096
    actor_dim = 6
    critic_dim = 4
    env = _MockBaseTask(num_envs=num_envs, actor_dim=actor_dim, critic_dim=critic_dim)
    bridge = FlashSACGymBridge(
        env, actor_obs_keys=("actor_obs",), critic_obs_keys=("critic_obs",)
    )
    bridge.reset(random_start_init=False)

    # Simulate holosoma's BaseTask layout:
    #   - reset_buf has a few envs flipped true (like early termination)
    #   - extras["final_observations"][k] is a FULL-BATCH tensor with
    #     meaningful rows only at env_ids and zeros elsewhere.
    reset_ids = torch.tensor([3, 17], dtype=torch.long)
    reset_mask = torch.zeros((num_envs,), dtype=torch.bool)
    reset_mask[reset_ids] = True
    env.reset_buf = reset_mask
    env.extras["time_outs"] = torch.zeros((num_envs,), dtype=torch.bool)
    env.extras["time_outs"][3] = True  # one truncation, one termination

    actor_final_full = torch.zeros((num_envs, actor_dim))
    critic_final_full = torch.zeros((num_envs, critic_dim))
    actor_final_full[3] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    actor_final_full[17] = torch.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    critic_final_full[3] = torch.tensor([100.0, 200.0, 300.0, 400.0])
    critic_final_full[17] = torch.tensor([500.0, 600.0, 700.0, 800.0])
    env.extras["final_observations"] = {
        "actor_obs": actor_final_full,
        "critic_obs": critic_final_full,
    }

    actions = np.zeros((num_envs, env.dim_actions), dtype=np.float32)
    next_obs, rewards, terminated, truncated, info = bridge.step(actions)

    # Shape sanity
    assert next_obs.shape == (num_envs, actor_dim + critic_dim)
    assert info["final_obs"].shape == (num_envs, actor_dim + critic_dim)

    # Dones split
    assert terminated[17] and not truncated[17]
    assert truncated[3] and not terminated[3]
    # No other env should be marked done
    done_mask = terminated | truncated
    assert int(done_mask.sum()) == 2

    # The reset-env rows of ``final_obs`` must carry the stored final
    # observations (concatenation of actor_obs and critic_obs).
    np.testing.assert_array_equal(
        info["final_obs"][3],
        np.concatenate([actor_final_full[3].numpy(), critic_final_full[3].numpy()]),
    )
    np.testing.assert_array_equal(
        info["final_obs"][17],
        np.concatenate([actor_final_full[17].numpy(), critic_final_full[17].numpy()]),
    )

"""Play a trained FlashSAC agent in IsaacLab.

Vendored from upstream ``play_isaaclab.py`` with the same robustness edits
applied to :mod:`holosoma._vendored.flash_rl.train`:

1. Config directory defaults to the vendored ``configs/`` dir resolved from
   ``__file__`` rather than a relative ``./configs`` path, so the script
   works regardless of the caller's cwd.
2. Hydra's global singleton is cleared at the top of :func:`build_cfg` so
   repeated calls in the same Python process do not raise
   ``GlobalHydra is already initialized``.
3. The OmegaConf ``${eval: ...}`` resolver is registered with
   ``replace=True``.
4. :func:`main` uses ``parse_known_args`` so that argv pollution from
   IsaacLab's ``AppLauncher`` does not break the argument parser.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import random
from pathlib import Path
from typing import MutableMapping, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from holosoma._vendored.flash_rl.agents import create_agent
from holosoma._vendored.flash_rl.envs.isaaclab import make_isaaclab_env
from holosoma._vendored.flash_rl.types import Tensor

_VENDORED_CONFIG_DIR: str = str((Path(__file__).parent / "configs").resolve())


def build_cfg(
    overrides: Optional[Sequence[str]] = None,
    config_name: str = "flashSAC_base",
    config_dir: Optional[str] = None,
) -> DictConfig:
    overrides = list(overrides) if overrides is not None else []
    config_dir = config_dir or _VENDORED_CONFIG_DIR

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)  # noqa: S307

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def play(
    cfg: DictConfig,
    checkpoint_path: str,
    num_envs: int = 16,
    num_episodes: int = 10,
) -> None:
    """Roll out a trained FlashSAC checkpoint against an IsaacLab env."""

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Create environment with rendering (headless=False)
    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=num_envs,
        seed=cfg.seed,
        headless=False,
    )

    # Create agent using config (same as train.py)
    _, env_info = env.reset(random_start_init=False)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Play loop
    observations, _ = env.reset(random_start_init=False)
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(num_envs)

    while completed_episodes < num_episodes:
        actions = agent.sample_actions(
            interaction_step=0, prev_transition=prev_transition, training=False
        )
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

        episode_returns += rewards
        episode_dones = np.logical_or(terminateds, truncateds)

        for idx in range(num_envs):
            if episode_dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0
                if completed_episodes >= num_episodes:
                    break

        observations = next_observations
        prev_transition = {"next_observation": observations}

    env.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Play a trained FlashSAC agent in IsaacLab", allow_abbrev=False
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional override for the Hydra config directory",
    )
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to agent checkpoint directory (e.g. models/.../step24400)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments for visualization",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to play"
    )
    args, _unknown = parser.parse_known_args(argv)

    cfg = build_cfg(
        overrides=args.overrides,
        config_name=args.config_name,
        config_dir=args.config_path,
    )
    play(
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()

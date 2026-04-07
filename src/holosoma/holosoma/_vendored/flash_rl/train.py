"""Vendored FlashSAC training entry-point.

Refactored from upstream ``train.py`` (https://github.com/joonleesky/FlashSAC)
into three composable functions so it can be both:

- invoked as a CLI subprocess via ``python -m holosoma._vendored.flash_rl.train``
- imported in-process by holosoma's adapter layer (``holosoma.agents.flash_sac``)

The main internal training loop in :func:`run` is bit-equivalent to upstream
``train.py`` lines 50-204; only the surrounding boilerplate (Hydra
initialization, argument parsing) was restructured.

Key holosoma vendoring deviations from upstream:

1. ``build_cfg`` clears the global Hydra singleton before composing so the
   function is safe to call repeatedly inside the same Python process
   (e.g. from pytest).
2. ``build_cfg`` uses :func:`hydra.initialize_config_dir` with an absolute path
   resolved from ``__file__``, so the call works regardless of the caller's
   current working directory.
3. The OmegaConf ``${eval: ...}`` resolver is registered idempotently in
   :mod:`holosoma._vendored.flash_rl.__init__`, so importing the package once
   is enough.
4. :func:`main` uses ``parse_known_args`` so that argv pollution from
   IsaacLab's ``AppLauncher`` (which adds ``--headless``, ``--device`` etc. via
   side effect when imported) does not break our argument parser.
5. The initial unconditional ``evaluate`` / ``record_video`` calls are
   short-circuited when ``num_eval_episodes == 0`` / ``num_record_episodes == 0``
   so that smoke runs do not produce ``np.mean([])`` warnings or attempt to
   call ``env.render()`` on environments that do not implement it.
6. The save-path base is rooted at ``Path(__file__).parent`` rather than
   ``sys.argv[0]`` so checkpoints land next to the vendored package regardless
   of how it is launched.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import torch
import tqdm
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from holosoma._vendored.flash_rl.agents import create_agent
from holosoma._vendored.flash_rl.common import create_logger
from holosoma._vendored.flash_rl.envs import create_envs
from holosoma._vendored.flash_rl.evaluation import evaluate, record_video
from holosoma._vendored.flash_rl.types import Tensor

# Absolute path to the vendored Hydra config tree, resolved from this file's
# location so that it works regardless of the caller's cwd or whether
# the package is installed editable vs site-packages.
_VENDORED_CONFIG_DIR: str = str((Path(__file__).parent / "configs").resolve())


def build_cfg(
    overrides: Optional[Sequence[str]] = None,
    config_name: str = "flashSAC_base",
    config_dir: Optional[str] = None,
) -> DictConfig:
    """Build a resolved FlashSAC config object.

    Wraps Hydra's ``initialize_config_dir`` + ``compose`` so that:

    - the call is idempotent (clears the Hydra global singleton first);
    - the OmegaConf ``${eval: ...}`` resolver is registered with
      ``replace=True`` so a second call does not raise;
    - the config tree default points at the vendored
      ``holosoma/_vendored/flash_rl/configs/`` directory.
    """

    overrides = list(overrides) if overrides is not None else []
    config_dir = config_dir or _VENDORED_CONFIG_DIR

    # Idempotency for repeated calls (pytest, in-process adapter).
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # The eval resolver is registered eagerly in
    # ``holosoma._vendored.flash_rl.__init__``; ``replace=True`` here is a
    # belt-and-suspenders fallback in case some other code path cleared it.
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)  # noqa: S307

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def run(cfg: DictConfig) -> None:
    """Execute the FlashSAC training loop against a resolved config.

    The body is byte-equivalent to upstream ``train.py`` lines 50-204 except
    that:

    - the initial unconditional ``evaluate`` / ``record_video`` calls are
      gated on ``num_eval_episodes`` / ``num_record_episodes`` to avoid noise
      on smoke runs;
    - ``save_path_base`` is rooted at this file's directory rather than at
      ``sys.argv[0]`` so it is launcher-independent.
    """

    ###############################
    # seeding / configuration
    ###############################

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    #############################
    # envs
    #############################
    train_env, eval_env, record_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # agent
    #############################

    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    #############################
    # train
    #############################

    logger = create_logger(cfg)

    # Output paths are rooted at the vendored package directory so that
    # checkpoints land somewhere stable regardless of how the script was
    # launched (CLI subprocess vs in-process import).
    script_dir = str(Path(__file__).parent.resolve())
    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = os.path.join(script_dir, save_path_resolved)

    if cfg.agent_load_path is not None:
        load_path = os.path.join(script_dir, cfg.agent_load_path)
        agent.load(load_path)
    if cfg.buffer_load_path is not None:
        load_path = os.path.join(script_dir, cfg.buffer_load_path)
        agent.load_replay_buffer(load_path)

    # initial evaluation (skipped on smoke runs where num_eval_episodes==0)
    if cfg.num_eval_episodes:
        eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
        logger.update_metric(**eval_info)
    if cfg.num_record_episodes:
        video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
        logger.update_metric(**video_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    observations, env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter: float = 0.0
    update_info: dict[str, Any] = {}
    env_step = 0

    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)),
        smoothing=0.1,
        mininterval=0.5,
    ):
        # using env steps simplifies the comparison with the performance reported in the paper.
        env_step = interaction_step * cfg.num_train_envs

        # collect data. use random actions until agent.can_start_training()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = train_env.action_space.sample()

        assert actions is not None
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])

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
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1

            # evaluation
            if (
                cfg.evaluation_per_interaction_step
                and interaction_step % cfg.evaluation_per_interaction_step == 0
                and cfg.num_eval_episodes
            ):
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
                logger.update_metric(**eval_info)

            # metrics
            if cfg.metrics_per_interaction_step and interaction_step % cfg.metrics_per_interaction_step == 0:
                metrics_info = agent.get_metrics()
                logger.update_metric(**metrics_info)

            # video recording
            if (
                cfg.recording_per_interaction_step
                and interaction_step % cfg.recording_per_interaction_step == 0
                and cfg.num_record_episodes
            ):
                video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
                logger.update_metric(**video_info)

            # logging
            if cfg.logging_per_interaction_step and interaction_step % cfg.logging_per_interaction_step == 0:
                logger.log_metric(step=env_step)
                logger.reset()

            # checkpointing
            if (
                cfg.save_checkpoint_per_interaction_step
                and interaction_step % cfg.save_checkpoint_per_interaction_step == 0
            ):
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save(save_path)

            # save buffer
            if cfg.save_buffer_per_interaction_step and interaction_step % cfg.save_buffer_per_interaction_step == 0:
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save_replay_buffer(save_path)

    # final evaluation (also gated on smoke flags)
    if cfg.num_eval_episodes:
        eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
        logger.update_metric(**eval_info)
    if cfg.num_record_episodes:
        video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
        logger.update_metric(**video_info)
    logger.log_metric(step=env_step)
    logger.reset()

    train_env.close()
    eval_env.close()
    record_env.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point.

    Uses ``parse_known_args`` to tolerate argv pollution from IsaacLab's
    ``AppLauncher``, which injects flags via side effect when imported.
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override for the Hydra config directory")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    args, _unknown = parser.parse_known_args(argv)
    cfg = build_cfg(
        overrides=args.overrides,
        config_name=args.config_name,
        config_dir=args.config_path,
    )
    run(cfg)


if __name__ == "__main__":
    main()

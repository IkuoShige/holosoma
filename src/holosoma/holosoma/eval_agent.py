from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.logger import WandbLoggerConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _is_media_only_mode() -> bool:
    # Default to media-only uploads for eval runs unless explicitly disabled.
    return os.getenv("HOLOSOMA_EVAL_WANDB_MEDIA_ONLY", "1").lower() not in {"0", "false", "no"}


def _init_eval_wandb(tyro_config: ExperimentConfig, eval_log_dir: Path) -> bool:
    if tyro_config.logger.type != "wandb":
        return False

    import wandb

    logger_cfg = tyro_config.logger
    assert isinstance(logger_cfg, WandbLoggerConfig), "Logger config must be WandbLoggerConfig when type is wandb"

    default_project = tyro_config.training.project or logger_cfg.project or "default_project"
    default_run_name = (
        f"{get_timestamp()}_{tyro_config.training.name or 'run'}_eval_{tyro_config.robot.asset.robot_type}"
    )

    wandb_dir = Path(logger_cfg.dir or (eval_log_dir / ".wandb"))
    wandb_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving wandb eval logs to {wandb_dir}")

    wandb_kwargs: dict[str, object] = {
        "project": logger_cfg.project or default_project,
        "name": logger_cfg.name or default_run_name,
        "config": dataclasses.asdict(tyro_config),
        "dir": str(wandb_dir),
        "mode": logger_cfg.mode,
    }
    if _is_media_only_mode():
        wandb_kwargs["settings"] = wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            save_code=False,
            _disable_stats=True,
            _disable_meta=True,
            _disable_machine_info=True,
        )
    if logger_cfg.entity:
        wandb_kwargs["entity"] = logger_cfg.entity
    if logger_cfg.group:
        wandb_kwargs["group"] = logger_cfg.group
    if logger_cfg.id:
        wandb_kwargs["id"] = logger_cfg.id
    if logger_cfg.tags:
        wandb_kwargs["tags"] = list(logger_cfg.tags)
    if logger_cfg.resume is not None:
        wandb_kwargs["resume"] = logger_cfg.resume

    wandb.init(**wandb_kwargs)
    if wandb.run is None:
        logger.warning("W&B logger requested but run initialization failed. Video uploads will be skipped.")
        return False

    logger.info(f"W&B run initialized: {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
    return True


def _prepare_eval_video_config(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Normalize eval video behavior for bounded replays.

    Evaluation often runs a single long episode and exits by `max_eval_steps`,
    so `video.interval>1` can result in zero captured videos. Force interval=1
    when video recording is enabled to ensure at least one rollout is recorded.
    """
    logger_cfg = tyro_config.logger
    if not logger_cfg.video.enabled and not logger_cfg.headless_recording:
        return tyro_config

    if logger_cfg.video.upload_to_wandb and logger_cfg.type != "wandb":
        logger.warning(
            "Video upload to W&B requested with non-wandb logger. "
            "Auto-promoting eval logger to `wandb`."
        )
        promoted_logger = WandbLoggerConfig(
            mode=os.getenv("WANDB_MODE", "online") if os.getenv("WANDB_MODE") in {"online", "offline"} else "online",
            entity=getattr(logger_cfg, "entity", None),
            project=getattr(logger_cfg, "project", None),
            name=getattr(logger_cfg, "name", None),
            group=getattr(logger_cfg, "group", None),
            id=getattr(logger_cfg, "id", None),
            tags=tuple(getattr(logger_cfg, "tags", ()) or ()),
            dir=getattr(logger_cfg, "dir", None),
            resume=getattr(logger_cfg, "resume", None),
            video=logger_cfg.video,
            headless_recording=logger_cfg.headless_recording,
            base_dir=logger_cfg.base_dir,
        )
        tyro_config = dataclasses.replace(tyro_config, logger=promoted_logger)
        logger_cfg = tyro_config.logger

    if logger_cfg.video.interval == 1:
        patched_cfg = tyro_config
    else:
        logger.info(
            "For evaluation, forcing `logger.video.interval=1` to guarantee video capture "
            "for short/step-limited runs."
        )
        patched_video = dataclasses.replace(logger_cfg.video, interval=1)
        patched_logger = dataclasses.replace(logger_cfg, video=patched_video)
        patched_cfg = dataclasses.replace(tyro_config, logger=patched_logger)
        logger_cfg = patched_cfg.logger

    if _is_media_only_mode() and patched_cfg.training.export_onnx:
        logger.info("Media-only mode enabled: forcing `training.export_onnx=False` to avoid non-media uploads.")
        patched_training = dataclasses.replace(patched_cfg.training, export_onnx=False)
        patched_cfg = dataclasses.replace(patched_cfg, training=patched_training)

    return patched_cfg


def _cleanup_video_recording(env) -> None:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return
    video_recorder = getattr(simulator, "video_recorder", None)
    if video_recorder is None:
        return

    logger.info("Flushing evaluation video recorder before shutdown.")
    video_recorder.cleanup()


def _teardown_wandb_if_needed(wandb_enabled: bool) -> None:
    if not wandb_enabled:
        return

    import wandb

    logger.info("Shutting down wandb...")
    wandb.teardown()


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    tyro_config = _prepare_eval_video_config(tyro_config)

    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    wandb_enabled = _init_eval_wandb(tyro_config, eval_log_dir)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    eval_config_path = eval_log_dir / CONFIG_NAME
    tyro_config.save_config(str(eval_config_path))
    if wandb_enabled and not _is_media_only_mode():
        import wandb

        wandb.save(str(eval_config_path), base_path=str(eval_log_dir))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    # Keep eval-side artifacts inside eval log dir so wandb.save(base_path=log_dir)
    # can upload them without crossing path boundaries.
    exported_policy_dir_path = os.path.join(str(eval_log_dir), "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        try:
            algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
            logger.info(f"Exported policy as onnx to: {exported_onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to export ONNX during evaluation (continuing eval): {e}")

    try:
        algo.evaluate_policy(
            max_eval_steps=tyro_config.training.max_eval_steps,
        )
    finally:
        # Ensure headless recordings are finalized even when evaluation ends early.
        try:
            _cleanup_video_recording(env)
        except Exception as e:
            logger.warning(f"Failed to flush video recorder during eval shutdown: {e}")

        try:
            _teardown_wandb_if_needed(wandb_enabled)
        finally:
            if simulation_app:
                close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()

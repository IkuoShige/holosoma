"""Evaluate velocity tracking performance with practical commands.

Usage:
    source scripts/source_isaacsim_setup.sh
    DISPLAY=:99 python scripts/eval_tracking.py \
        --checkpoint logs/hv-k1-manager/<run_dir>/model_0100000.pt

    # With video recording + wandb upload:
    DISPLAY=:99 python scripts/eval_tracking.py \
        --checkpoint logs/hv-k1-manager/<run_dir>/model_0100000.pt --record

Reports actual/commanded velocity ratio for each test scenario.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.logger import WandbLoggerConfig
from holosoma.config_types.video import VideoConfig
from holosoma.managers.observation.terms.locomotion import get_base_ang_vel, get_base_lin_vel
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint, load_saved_experiment_config
from holosoma.utils.experiment_paths import get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG
from loguru import logger


@dataclass
class TestScenario:
    name: str
    vx: float
    vy: float
    yaw: float
    steps: int = 400


SCENARIOS = [
    TestScenario("停止 (stand)", 0.0, 0.0, 0.0, steps=400),
    TestScenario("前進 (forward)", 1.0, 0.0, 0.0),
    TestScenario("後退 (backward)", -0.8, 0.0, 0.0),
    TestScenario("左移動 (strafe L)", 0.0, 0.5, 0.0),
    TestScenario("右移動 (strafe R)", 0.0, -0.5, 0.0),
    TestScenario("直進+旋回 (fwd+turn)", 1.0, 0.0, 0.8),
    TestScenario("高速前進 (fast fwd)", 1.5, 0.0, 0.0),
    TestScenario("斜め移動 (diagonal)", 1.0, 0.6, 0.0),
    TestScenario("横+旋回 (strafe+turn)", 0.0, 0.8, 1.0),
]

WARMUP_STEPS = 100


def _enable_video_recording(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Patch config to enable video recording with wandb upload."""
    video_cfg = VideoConfig(
        enabled=True,
        interval=1,
        upload_to_wandb=True,
        show_command_overlay=True,
        width=640,
        height=360,
        output_format="h264",
        playback_rate=1.0,
    )
    logger_cfg = tyro_config.logger
    if isinstance(logger_cfg, WandbLoggerConfig):
        patched_logger = dataclasses.replace(logger_cfg, video=video_cfg, headless_recording=True)
    else:
        patched_logger = WandbLoggerConfig(
            mode=os.getenv("WANDB_MODE", "online"),
            project=getattr(logger_cfg, "project", None) or tyro_config.training.project,
            video=video_cfg,
            headless_recording=True,
            base_dir=logger_cfg.base_dir,
        )
    return dataclasses.replace(tyro_config, logger=patched_logger)


def _init_wandb(tyro_config: ExperimentConfig, log_dir: Path) -> bool:
    """Initialize wandb for eval tracking."""
    import wandb  # noqa: PLC0415

    logger_cfg = tyro_config.logger
    if not isinstance(logger_cfg, WandbLoggerConfig):
        return False

    project = logger_cfg.project or tyro_config.training.project or "hv-eval"
    run_name = f"{get_timestamp()}_eval_tracking_{tyro_config.robot.asset.robot_type}"

    wandb_dir = Path(logger_cfg.dir or (log_dir / ".wandb"))
    wandb_dir.mkdir(exist_ok=True, parents=True)

    wandb.init(
        project=project,
        name=run_name,
        config=dataclasses.asdict(tyro_config),
        dir=str(wandb_dir),
        mode=logger_cfg.mode,
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            save_code=False,
            _disable_stats=True,
            _disable_meta=True,
            _disable_machine_info=True,
        ),
        tags=["eval_tracking"],
    )
    if wandb.run is None:
        logger.warning("W&B init failed. Recording without upload.")
        return False
    logger.info(f"W&B run: {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
    return True


def _log_results_to_wandb(results: list[dict]) -> None:
    """Log results table and summary metrics to wandb."""
    import wandb  # noqa: PLC0415

    if wandb.run is None:
        return

    columns = [
        "scenario",
        "cmd_vx",
        "cmd_vy",
        "cmd_yaw",
        "actual_vx",
        "actual_vy",
        "actual_yaw",
        "vx_ratio",
        "vy_ratio",
        "yaw_ratio",
        "vx_std",
        "vy_std",
        "yaw_std",
        "fallen",
    ]
    table = wandb.Table(columns=columns)

    for r in results:

        def ratio(actual, cmd):
            return actual / cmd if abs(cmd) > 0.01 else None

        table.add_data(
            r["name"],
            r["cmd_vx"],
            r["cmd_vy"],
            r["cmd_yaw"],
            r["actual_vx"],
            r["actual_vy"],
            r["actual_yaw"],
            ratio(r["actual_vx"], r["cmd_vx"]),
            ratio(r["actual_vy"], r["cmd_vy"]),
            ratio(r["actual_yaw"], r["cmd_yaw"]),
            r["vx_std"],
            r["vy_std"],
            r["yaw_std"],
            r["fallen"],
        )

    wandb.log({"eval_tracking/results": table})

    # Log key summary metrics
    stand = results[0]
    wandb.log(
        {
            "eval_tracking/stand_vx_std": stand["vx_std"],
            "eval_tracking/stand_vy_std": stand["vy_std"],
            "eval_tracking/stand_yaw_std": stand["yaw_std"],
        }
    )
    for r in results:
        if abs(r["cmd_vx"]) > 0.01:
            tag = r["name"].split("(")[1].rstrip(")").replace(" ", "_") if "(" in r["name"] else r["name"]
            wandb.log({f"eval_tracking/{tag}_vx_ratio": r["actual_vx"] / r["cmd_vx"]})


def _is_ppo(algo) -> bool:
    return hasattr(algo, "actor_obs_keys")


def _set_commands(env, scenario: TestScenario):
    commands = env.command_manager.commands
    commands[:, 0] = scenario.vx
    commands[:, 1] = scenario.vy
    commands[:, 2] = scenario.yaw


def run_scenario(env, algo, scenario: TestScenario) -> dict:
    """Run one test scenario and return tracking metrics.

    Supports both PPO (obs_dict-based) and FastSAC (tensor-based) agents.
    """
    # Access the raw env (unwrapped from FastSACEnv if applicable)
    raw_env = getattr(algo, "unwrapped_env", None) or env
    is_ppo = _is_ppo(algo)

    if is_ppo:
        obs_dict = env.reset_all()
        inference_fn = algo.get_inference_policy()
    else:
        obs = env.reset()

    _set_commands(raw_env, scenario)

    vx_samples, vy_samples, yaw_samples = [], [], []
    fallen = False

    total_steps = WARMUP_STEPS + scenario.steps
    for step in range(total_steps):
        _set_commands(raw_env, scenario)

        with torch.no_grad():
            if is_ppo:
                actor_obs = torch.cat([obs_dict[k] for k in algo.actor_obs_keys], dim=1)
                actions = inference_fn({"actor_obs": actor_obs})
                obs_dict, _, dones, _ = env.step({"actions": actions})
            else:
                if algo.obs_normalization:
                    normalized_obs = algo.obs_normalizer(obs, update=False)
                else:
                    normalized_obs = obs
                actions = algo.actor(normalized_obs)[0]
                obs, _, dones, _ = algo.env.step(actions)

        if dones.any():
            fallen = True
            if is_ppo:
                obs_dict = env.reset_all()
            else:
                obs = algo.env.reset()
            _set_commands(raw_env, scenario)

        # Collect after warmup
        if step >= WARMUP_STEPS:
            lin_vel = get_base_lin_vel(raw_env)
            ang_vel = get_base_ang_vel(raw_env)
            vx_samples.append(lin_vel[0, 0].item())
            vy_samples.append(lin_vel[0, 1].item())
            yaw_samples.append(ang_vel[0, 2].item())

    n = len(vx_samples)
    vx_mean = sum(vx_samples) / n
    vy_mean = sum(vy_samples) / n
    yaw_mean = sum(yaw_samples) / n
    vx_std = (sum((v - vx_mean) ** 2 for v in vx_samples) / n) ** 0.5
    vy_std = (sum((v - vy_mean) ** 2 for v in vy_samples) / n) ** 0.5
    yaw_std = (sum((v - yaw_mean) ** 2 for v in yaw_samples) / n) ** 0.5

    return {
        "name": scenario.name,
        "cmd_vx": scenario.vx,
        "cmd_vy": scenario.vy,
        "cmd_yaw": scenario.yaw,
        "actual_vx": vx_mean,
        "actual_vy": vy_mean,
        "actual_yaw": yaw_mean,
        "vx_std": vx_std,
        "vy_std": vy_std,
        "yaw_std": yaw_std,
        "fallen": fallen,
    }


def fmt_ratio(actual: float, cmd: float) -> str:
    if abs(cmd) < 0.01:
        return f"{actual:+.3f}"
    return f"{actual / cmd * 100:.0f}%"


def main():
    init_eval_logging()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--record", action="store_true", help="Enable video recording and wandb upload")
    known, remaining = parser.parse_known_args()

    checkpoint_cfg = CheckpointConfig(checkpoint=known.checkpoint)
    saved_cfg, _saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining,
        description="Eval tracking",
        config=TYRO_CONIFG,
    )

    # Enable video recording if requested
    wandb_enabled = False
    if known.record:
        tyro_config = _enable_video_recording(tyro_config)

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    log_dir = Path("logs/eval_tracking_tmp")
    log_dir.mkdir(parents=True, exist_ok=True)

    if known.record:
        wandb_enabled = _init_wandb(tyro_config, log_dir)

    checkpoint = load_checkpoint(known.checkpoint, str(log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.load(checkpoint_path)

    # Set eval mode
    if hasattr(algo, "_eval_mode"):
        algo._eval_mode()
    env.set_is_evaluating()

    results = []
    for sc in SCENARIOS:
        logger.info(f"Testing: {sc.name} (vx={sc.vx}, vy={sc.vy}, yaw={sc.yaw})")
        result = run_scenario(env, algo, sc)
        results.append(result)

    # Flush the last scenario's video
    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        video_recorder = getattr(simulator, "video_recorder", None)
        if video_recorder is not None:
            logger.info("Flushing video recorder.")
            video_recorder.cleanup()

    # Print table
    print("\n" + "=" * 95)
    print(" 速度追従評価 (Velocity Tracking Evaluation)")
    print("=" * 95)
    hdr = f"{'シナリオ':<25} {'指令':>10} {'実測':>10} {'追従率':>8} {'振動std':>8} {'転倒':>4}"
    print(hdr)
    print("-" * 95)

    for r in results:
        # Print vx row
        vx_ratio = fmt_ratio(r["actual_vx"], r["cmd_vx"])
        fallen_mark = "Y" if r["fallen"] else "N"
        print(
            f"{r['name']:<25} vx={r['cmd_vx']:>+5.1f}   vx={r['actual_vx']:>+6.3f}"
            f"   {vx_ratio:>6}   {r['vx_std']:.4f}   {fallen_mark:>2}"
        )
        # Print vy row if nonzero
        if abs(r["cmd_vy"]) > 0.01 or abs(r["actual_vy"]) > 0.05:
            vy_ratio = fmt_ratio(r["actual_vy"], r["cmd_vy"])
            print(f"{'':25} vy={r['cmd_vy']:>+5.1f}   vy={r['actual_vy']:>+6.3f}   {vy_ratio:>6}   {r['vy_std']:.4f}")
        # Print yaw row if nonzero
        if abs(r["cmd_yaw"]) > 0.01 or abs(r["actual_yaw"]) > 0.05:
            yaw_ratio = fmt_ratio(r["actual_yaw"], r["cmd_yaw"])
            print(
                f"{'':25} yw={r['cmd_yaw']:>+5.1f}   yw={r['actual_yaw']:>+6.3f}   {yaw_ratio:>6}   {r['yaw_std']:.4f}"
            )

    print("-" * 95)

    # Zero-velocity summary
    s = results[0]
    print("\n■ 停止時の振動:")
    print(f"  vx std = {s['vx_std']:.4f} m/s,  vy std = {s['vy_std']:.4f} m/s,  yaw std = {s['yaw_std']:.4f} rad/s")
    thr = 0.05
    if s["vx_std"] < thr and s["vy_std"] < thr and s["yaw_std"] < thr:
        print(f"  → 発振なし (all std < {thr})")
    else:
        print(f"  → 発振あり (threshold={thr})")
    print()

    # Upload results to wandb
    if wandb_enabled:
        _log_results_to_wandb(results)
        import wandb  # noqa: PLC0415

        logger.info("Shutting down wandb...")
        wandb.finish()

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()

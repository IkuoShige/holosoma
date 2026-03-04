"""Evaluate velocity tracking performance with practical commands.

Usage:
    source scripts/source_isaacsim_setup.sh
    DISPLAY=:99 python scripts/eval_tracking.py \
        --checkpoint logs/hv-k1-manager/<run_dir>/model_0100000.pt \
        --eval-overrides.headless True --logger.headless-recording True

Reports actual/commanded velocity ratio for each test scenario.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import tyro
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.managers.observation.terms.locomotion import get_base_ang_vel, get_base_lin_vel
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


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
    TestScenario("高速前進 (fast fwd)", 2.0, 0.0, 0.0),
]

WARMUP_STEPS = 100


def run_scenario(env, algo, scenario: TestScenario) -> dict:
    """Run one test scenario and return tracking metrics."""
    # Access the raw env (unwrapped from FastSACEnv)
    raw_env = algo.unwrapped_env
    commands = raw_env.command_manager.commands

    # Reset via the wrapped env (FastSACEnv.reset)
    obs = algo.env.reset()
    commands[:, 0] = scenario.vx
    commands[:, 1] = scenario.vy
    commands[:, 2] = scenario.yaw

    vx_samples, vy_samples, yaw_samples = [], [], []
    fallen = False

    total_steps = WARMUP_STEPS + scenario.steps
    for step in range(total_steps):
        # Force commands every step (eval mode prevents resampling)
        commands[:, 0] = scenario.vx
        commands[:, 1] = scenario.vy
        commands[:, 2] = scenario.yaw

        with torch.no_grad():
            if algo.obs_normalization:
                normalized_obs = algo.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            actions = algo.actor(normalized_obs)[0]

        obs, _, dones, _ = algo.env.step(actions)

        if dones.any():
            fallen = True
            obs = algo.env.reset()
            commands[:, 0] = scenario.vx
            commands[:, 1] = scenario.vy
            commands[:, 2] = scenario.yaw

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
        "cmd_vx": scenario.vx, "cmd_vy": scenario.vy, "cmd_yaw": scenario.yaw,
        "actual_vx": vx_mean, "actual_vy": vy_mean, "actual_yaw": yaw_mean,
        "vx_std": vx_std, "vy_std": vy_std, "yaw_std": yaw_std,
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
    known, remaining = parser.parse_known_args()

    checkpoint_cfg = CheckpointConfig(checkpoint=known.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    tyro_config = tyro.cli(
        ExperimentConfig, default=eval_cfg, args=remaining,
        description="Eval tracking", config=TYRO_CONIFG,
    )

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    checkpoint = load_checkpoint(known.checkpoint, "logs/eval_tracking_tmp")
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo = algo_class(
        device=device, env=env, config=tyro_config.algo.config,
        log_dir="logs/eval_tracking_tmp", multi_gpu_cfg=None,
    )
    algo.setup()
    algo.load(checkpoint_path)

    # Set eval mode (zeros commands initially)
    env.set_is_evaluating()

    results = []
    for sc in SCENARIOS:
        logger.info(f"Testing: {sc.name} (vx={sc.vx}, vy={sc.vy}, yaw={sc.yaw})")
        result = run_scenario(env, algo, sc)
        results.append(result)

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
        print(f"{r['name']:<25} vx={r['cmd_vx']:>+5.1f}   vx={r['actual_vx']:>+6.3f}   {vx_ratio:>6}   {r['vx_std']:.4f}   {'Y' if r['fallen'] else 'N':>2}")
        # Print vy row if nonzero
        if abs(r["cmd_vy"]) > 0.01 or abs(r["actual_vy"]) > 0.05:
            vy_ratio = fmt_ratio(r["actual_vy"], r["cmd_vy"])
            print(f"{'':25} vy={r['cmd_vy']:>+5.1f}   vy={r['actual_vy']:>+6.3f}   {vy_ratio:>6}   {r['vy_std']:.4f}")
        # Print yaw row if nonzero
        if abs(r["cmd_yaw"]) > 0.01 or abs(r["actual_yaw"]) > 0.05:
            yaw_ratio = fmt_ratio(r["actual_yaw"], r["cmd_yaw"])
            print(f"{'':25} yw={r['cmd_yaw']:>+5.1f}   yw={r['actual_yaw']:>+6.3f}   {yaw_ratio:>6}   {r['yaw_std']:.4f}")

    print("-" * 95)

    # Zero-velocity summary
    s = results[0]
    print(f"\n■ 停止時の振動:")
    print(f"  vx std = {s['vx_std']:.4f} m/s,  vy std = {s['vy_std']:.4f} m/s,  yaw std = {s['yaw_std']:.4f} rad/s")
    thr = 0.05
    if s["vx_std"] < thr and s["vy_std"] < thr and s["yaw_std"] < thr:
        print(f"  → 発振なし (all std < {thr})")
    else:
        print(f"  → 発振あり (threshold={thr})")
    print()

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()

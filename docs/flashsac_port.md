# FlashSAC Port

This document describes the holosoma port of [FlashSAC](https://github.com/joonleesky/FlashSAC) — the off-policy SAC variant published in *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control* (Kim et al., arXiv 2026).

> **Honest status (snapshot after Option A config change):** every upstream file is vendored, every test gate is green, the IsaacSim Gate B path trains a G1 policy that walks (~0.28 m/s forward on the dedicated stripped preset). Recent findings: the port itself is byte-identical to upstream on all audited paths — a line-cited code audit of the 4 items Codex flagged cleared them all. The previous `temp_target_sigma=0.30` widening (`20260408_181344`) did NOT improve gait quality empirically and was reverted. Latest configuration: upstream algo defaults verbatim (sigma=0.15, asymmetric_observation=false, n_step=3) + holosoma's PPO-default reward+observation (`g1_29dof_loco` + `g1_29dof_loco_single_wolinvel`). Whether this "Option A" recipe converges to clean walking is the single most important open question; the previous stripped preset remains registered as a verified fallback (walks at 20260408_142832 / 20260408_154733). See *Open work #3* for the Option A rationale and *Current state and limitations* for caveats.

## Three layers

```
src/holosoma/holosoma/
├── _vendored/
│   └── flash_rl/                   # Layer 1: verbatim mirror of upstream flash_rl/
│       ├── agents/flashSAC/
│       ├── buffers/
│       ├── envs/isaaclab.py        #   + import isaaclab_tasks (registration safety)
│       ├── common/logger.py
│       ├── configs/                #   Hydra YAMLs, resolved via Path(__file__).parent
│       ├── train.py                #   refactored into build_cfg / run / main(argv)
│       └── ...
└── agents/
    └── flash_sac/                  # Layer 2: holosoma-native BaseAlgo adapter
        ├── flash_sac_agent.py      #   FlashSACAgent(BaseAlgo) + dual-format save/load + evaluate_policy
        └── flash_sac_env_bridge.py #   FlashSACGymBridge: BaseTask -> gym VectorEnv

scripts/
├── run_flashsac_isaaclab_smoke.sh   # Layer 3: Gate A (vendored train.py path)
├── run_flashsac_holosoma_smoke.sh   # Layer 3: Gate B (holosoma BaseAlgo path)
├── run_flashsac_mjwarp_smoke.sh     # Layer 3: Gate B-mjwarp
├── flashsac_migrate_checkpoint.py   # convert legacy directory checkpoints to single-file .ckpt
└── _vendored/flash_rl/
    └── run_*.sh                     # All upstream batch-experiment scripts
```

### Layer 1 — Verbatim vendoring

Every file under `/workspace/FlashSAC/flash_rl/` is copied to
`src/holosoma/holosoma/_vendored/flash_rl/` with these mechanical edits only:

| Edit | Where | Reason |
|---|---|---|
| `from flash_rl…` → `from holosoma._vendored.flash_rl…` | every `.py` | required for the new namespace |
| `import isaaclab_tasks  # noqa: F401` | `_vendored/flash_rl/envs/isaaclab.py` | guarantees gymnasium task IDs are registered before `parse_env_cfg` |
| `jax.numpy` import made optional | `_vendored/flash_rl/types.py` | jax is not installed in `hssim`; the FlashSAC torch path does not need it |
| `train.py` refactored into `build_cfg / run / main(argv=None)` | `_vendored/flash_rl/train.py` | needed so Layer 2 can build configs in-process and so pytest can call `build_cfg` repeatedly without `GlobalHydra is already initialized` failures |
| `play_isaaclab.py` refactored similarly | `_vendored/flash_rl/play_isaaclab.py` | same rationale (cwd-independent + idempotent + library form) |
| `OmegaConf.register_new_resolver("eval", …)` made idempotent | `_vendored/flash_rl/__init__.py`, `_vendored/flash_rl/train.py` | second `register_new_resolver` call would otherwise raise |
| `pyproject.toml` ruff per-file-ignores adds `_vendored/**/*.py = ["ALL"]` | repo root | vendored upstream code is exempt from holosoma's lint |

`flash_rl/configs/` is mirrored under `_vendored/flash_rl/configs/`. The vendored `train.py`'s `build_cfg` resolves the directory via `Path(__file__).parent / "configs"` so it works regardless of cwd.

`flash_rl/agents/flashSAC/` (camelCase) is **kept verbatim** rather than renamed to `flash_sac` so that future re-syncs from upstream produce a clean git diff.

### Layer 2 — Holosoma-native adapter

`src/holosoma/holosoma/agents/flash_sac/` wraps the vendored algorithm so it can be run through holosoma's standard training entry point (`train_agent.py exp:g1-29dof-flash-sac`) against `LeggedRobotLocomotionManager`.

#### `FlashSACGymBridge`

Translates between the holosoma `BaseTask` contract and the gymnasium `VectorEnv` contract that the vendored FlashSAC algorithm expects:

| Concept | Holosoma `BaseTask` | FlashSAC `IsaacLabVectorEnv` |
|---|---|---|
| Step input | `step({"actions": tensor})` | `step(numpy_actions)` |
| Step return | `(obs_buf_dict, rew_buf, reset_buf, extras)` | `(obs, rew, terminated, truncated, infos)` |
| Observations | `obs_buf_dict[group_key]` (torch dict) | flat numpy array |
| Termination | `reset_buf = terminated \| time_outs` (single OR'd channel) | separate `terminated` / `truncated` |
| Time-outs | `extras["time_outs"]` | `infos["time_outs"]` |
| Final obs (pre-reset) | `extras["final_observations"][group_key]` | `infos["final_obs"]` |
| Action dim | `env.dim_actions` / `env.robot_config.actions_dim` | `single_action_space.shape` |
| Actor obs dim | `env.dim_obs` | `infos["actor_observation_size"]` |
| Action range | `tanh ∈ [-1, 1]` × bridge multiplier × env `action_scale` | `tanh ∈ [-1, 1]` × IsaacLab stock `scale=0.5` |

The bridge handles the dones split:

```python
truncated = time_outs.bool()
terminated = reset_buf.bool() & ~truncated
```

and re-stitches `final_obs` from `extras["final_observations"]` (a persistent full-batch tensor) by indexing only the `env_ids` rows of the current step:

```python
env_ids = reset_bool.nonzero(as_tuple=False).flatten()
final_actor_obs[env_ids] = stacked[env_ids]  # NOT `stacked` — see commit 0986f36
```

It also applies a **uniform per-axis action multiplier** so the actor's `tanh ∈ [-1, 1]` reaches the same effective ±0.5 rad joint target range that IsaacLab stock G1 uses (`JointPositionActionCfg(scale=0.5)`):

```python
multiplier = target_action_scale_rad / env.robot_config.control.action_scale
            = 0.5 / 0.25
            = 2.0  # uniform scalar, NOT per-joint
```

This is **not** FastSAC-style per-joint scaling (`max_range / env_action_scale`, which yields 8-13× factors and pushes FlashSAC's narrow deterministic policy into extreme joint configurations). It is the uniform scale FlashSAC's hyperparameters were tuned against. Pass `target_action_scale_rad=None` to disable when running against an env that already scales internally.

#### `FlashSACAgent`

Subclass of `holosoma.agents.base_algo.BaseAlgo`. Constructor wraps the holosoma env in `FlashSACGymBridge`, then `setup()` instantiates the vendored `FlashSACAgent` directly via `FlashSACConfig(**asdict(holosoma_dataclass))` (no OmegaConf round-trip; Hydra stays quarantined inside Layer 1).

Key behaviors:

- **`learn()`** mirrors upstream `train.py` lines 113-200, but accumulates `update_info` into a `window_update_info` dict that is flushed at every `logging_interval`. This is necessary because the vendored `_update_networks` only returns `actor/*` and `temperature/*` keys on even `update_step` calls (`_update_step % actor_update_period == 0`); naïvely overwriting the dict each call drops actor metrics whenever the last update of a logging window happens to be an odd step. With `actor_update_period=2` and `updates_per_interaction_step=2.0` and `logging_interval=100`, the parity is always odd → actor metrics would otherwise never appear in TensorBoard. See commit `d37b991`.
- **`evaluate_policy(max_eval_steps)`** runs deterministic rollouts through the bridge using `sample_actions(training=False)`. Lets `holosoma.eval_agent.py` drive the FlashSAC adapter.
- **Dual-format `save(path, name="model.ckpt")`** writes BOTH the vendored per-component directory (`actor.pt`, `critic.pt`, `target_critic.pt`, `temperature.pt`, `reward_normalizer.pt`, `agent_state.pt` — for upstream `play_isaaclab.py` compatibility) AND a single `.ckpt` file at `<path>/<name>` containing the same component contents inline plus the holosoma `_checkpoint_metadata` (experiment_config, wandb_run_path, iteration). The `.ckpt` is what `eval_agent.py` ingests.
- **`load(ckpt_path)`** auto-detects file vs directory. A directory defers to the vendored loader; a `.ckpt` unpacks `flashsac_components` into a tempdir and re-uses the vendored loader so the on-disk format stays the single source of truth.
- The adapter forces `load_reward_normalizer=False` whenever `normalize_reward=False` so the upstream `assert reward_normalizer is not None` footgun (when normalize_reward was off during training but load_reward_normalizer defaulted to True at eval) cannot fire.
- `learning_rate_warmup_step` and `learning_rate_decay_step` are computed at `setup()` time from `learning_rate_*_rate * num_learning_iterations * updates_per_interaction_step` (upstream Hydra computes these via `${eval: …}`).

#### Checkpoint migration script

`scripts/flashsac_migrate_checkpoint.py` converts a legacy directory-only checkpoint
(saved by `FlashSACAgent.save` *before* the dual-format change) into the single-file
`.ckpt` that `eval_agent.py` expects:

```bash
python scripts/flashsac_migrate_checkpoint.py \
    logs/hv-g1-manager/<timestamp>-...-locomotion/flashsac_step50000
```

It reads each component `.pt` from the directory plus the sibling `holosoma_config.yaml`,
packs them into `flashsac_step50000.ckpt` next to the directory, and exits 0.
After migration, point `--checkpoint` at the `.ckpt` file.

### Layer 3 — Smoke gates

| Gate | Script | Test | Proves |
|---|---|---|---|
| **A** | `scripts/run_flashsac_isaaclab_smoke.sh` | `tests/e2e/test_flashsac_isaaclab_smoke.py` | Vendored FlashSAC runs IsaacLab's stock `Isaac-Velocity-Flat-G1-v0` for 5 interaction steps end-to-end (config + env + agent + buffer + update + logger). |
| **B** | `scripts/run_flashsac_holosoma_smoke.sh` | `tests/e2e/test_flashsac_holosoma_smoke.py` | Holosoma adapter runs the vendored algorithm against `LeggedRobotLocomotionManager` on the IsaacSim backend via `train_agent.py exp:g1-29dof-flash-sac` for 5 outer iterations. |
| **B-mjwarp** | `scripts/run_flashsac_mjwarp_smoke.sh` | `tests/e2e/test_flashsac_mjwarp_smoke.py` | Same Layer 2 adapter, but with `simulator=simulator.mjwarp` (`g1_29dof_flash_sac_mjwarp`). Runs in the `hsmujoco` conda env. |

The IsaacSim e2e tests are marked `@pytest.mark.isaacsim @pytest.mark.slow` and the mjwarp e2e test is marked `@pytest.mark.mujoco @pytest.mark.slow`. All three shell out to bash via `subprocess.run` so each invocation runs in a fresh Python process — required because IsaacSim is single-process per `AppLauncher` and to keep the two conda envs (`hssim` vs `hsmujoco`) cleanly separated.

Test counts as of this snapshot: **14 unit tests + 3 e2e smoke tests = 17 total**, all green on RTX 5090 / hssim.

## Reward and observation: holosoma is incompatible with FlashSAC's hyperparameters

This is the single most important finding from the porting sprint and the
biggest blocker on calling FlashSAC "fully usable in holosoma".

### What FlashSAC's hyperparameters assume

FlashSAC's `configs/agent/flashSAC.yaml` defaults (and the `scripts/run_isaaclab.sh`
preset) were tuned exclusively against IsaacLab's stock
`Isaac-Velocity-Flat-G1-v0` task in
`isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`. That
task's reward and observation shape is:

```python
# IsaacLab stock G1 flat
JointPositionActionCfg(scale=0.5)            # uniform action scale, ±0.5 rad

# rewards
track_lin_vel_xy_exp     weight=1.0  std=sqrt(0.25)
track_ang_vel_z_exp      weight=0.5  std=sqrt(0.25)
lin_vel_z_l2             weight=-2.0
ang_vel_xy_l2            weight=-0.05
flat_orientation_l2      weight=0.0   # disabled in flat task
dof_torques_l2           weight=-1.0e-5
dof_acc_l2               weight=-2.5e-7
feet_air_time            weight=0.125
# NO alive, NO feet_phase, NO pose, NO close_feet_xy, NO feet_ori
# NO sin_phase / cos_phase observation terms
```

Crucially, FlashSAC's actor collapses to a near-deterministic policy very
early (entropy hits the heuristic target ≈ -14 for 29-D actions within ~10 k
gradient steps, temperature drops to ~3 × 10⁻⁴). Once narrow, the policy
exploits whatever local optimum is closest to its initial distribution and
cannot break out by noise alone. This is fine when the reward landscape's
dominant gradient points toward walking — which it does for IsaacLab stock,
because the only positive signals are tracking + air-time.

### What goes wrong with holosoma's standard reward

Holosoma's `g1_29dof_loco` (PPO default) and `g1_29dof_loco_fast_sac`
(FastSAC preset) both add several shaping terms that are absent from
IsaacLab stock:

| Term | Holosoma weight | IsaacLab stock | Why FlashSAC trips on it |
|---|---|---|---|
| `feet_phase` | +5.0 (σ=0.008) | not present | Pure foot-height tracking against a clock-phase observation. **No coupling to forward velocity, COM progression, or stance impulse.** A near-deterministic policy harvests it by lifting and putting down feet in place to match the clock — without ever moving forward. In our second-to-last training run it dominated the per-term reward decomposition (+36 cumulative ep_sum vs +14 for `tracking_lin_vel`). |
| `alive` (FastSAC preset) | +10.0 | not present | Constant +10 per step for not falling. Together with the modest tracking max (~+3.5), best-case standing reward (~+10) competes with best-case walking reward (~+13.5). FastSAC's wide action_std explores past it; FlashSAC's narrow policy locks onto the +10 attractor. |
| `alive` (PPO preset) | +1.0 | not present | Even at weight 1.0, with all the *other* shaping terms removed it still pulls the policy toward standing. |
| `pose` | -0.5 × per-joint | not present | Per-joint default-pose deviation penalty with **50.0 weight on every upper-body joint** (17 joints × 50 = -850 max upper-body penalty). Strongly discourages the torso sway any walking gait produces. |
| `penalty_close_feet_xy` | -10.0 (binary, threshold 0.15) | not present | Hard threshold penalty. |
| `penalty_feet_ori` | -5.0 | not present | Foot orientation penalty. |
| `penalty_orientation` | -10.0 | flat task: weight 0.0 | Strong torso-tilt penalty. |
| `penalty_action_rate` | -2.0 | -0.005 (via dof_torques/dof_acc) | 400× stronger than IsaacLab. Penalizes any large step-to-step action change, which is exactly what walking gait requires. |

Combined: every holosoma-specific term either creates a "fake walking"
attractor (feet_phase, alive) or punishes the very motions a walking gait
needs (pose on upper body, action_rate at -2.0, orientation at -10.0). The
narrow FlashSAC policy lands in the locally-best-rewarded basin — which is
"stand still" or "twitch feet to match the clock without going anywhere" —
and stays there.

### The dedicated FlashSAC reward / observation preset

To work around this we added two new presets used **only** by the FlashSAC
experiments. They mirror IsaacLab stock's reward shape and are deliberately
NOT shared with PPO/FastSAC/FPO.

`config_values/loco/g1/observation.py` — `g1_29dof_loco_single_flashsac`:

```
actor_obs (7 terms, no phase clock):
  base_ang_vel, projected_gravity, command_lin_vel, command_ang_vel,
  dof_pos, dof_vel, actions

critic_obs (8 terms = same + base_lin_vel):
  base_lin_vel, base_ang_vel, projected_gravity, command_lin_vel,
  command_ang_vel, dof_pos, dof_vel, actions
```

`sin_phase` / `cos_phase` are removed because the phase clock is only
meaningful when paired with `feet_phase`. Stripping it removes the input
the policy could otherwise use to time the in-place clock-matching exploit.

`config_values/loco/g1/reward.py` — `g1_29dof_loco_flashsac`:

```
tracking_lin_vel    weight=2.0   sigma=0.25
tracking_ang_vel    weight=1.5   sigma=0.25
penalty_ang_vel_xy  weight=-0.05  (was -1.0; matches IsaacLab)
penalty_orientation weight=-1.0   (was -10.0)
penalty_action_rate weight=-0.005 (was -2.0; was killing walking)
```

Removed entirely: `feet_phase`, `alive`, `pose`, `penalty_feet_ori`,
`penalty_close_feet_xy`. PPO/FastSAC presets are unchanged.

`config_values/loco/g1/experiment.py` wires both `g1_29dof_flash_sac` and
`g1_29dof_flash_sac_mjwarp` to use these presets.

### Trial-and-error history (so future contributors do not repeat it)

Eight training attempts on G1 walking, each diagnosing one layer:

| # | Run dir suffix | Fix applied | Empirical result | Root cause unblocked |
|---|---|---|---|---|
| 1 | `20260407_170834` | (initial port) | Stands still, dist 0.20 m / 10 s | Action range `±0.25 rad` was 8-11% of hip joint range |
| 2 | `20260408_065029` | re-train with no changes (was actually re-running attempt 1's recipe) | Same as #1 | – |
| 3 | `20260408_083642` | per-joint scaling 13× (FastSAC-style) | Robot thrashes, root z=0.24, reward -8.9 | Per-joint scaling too aggressive for FlashSAC's narrow policy |
| 4 | `20260408_094513` | uniform scaling 2× (= IsaacLab stock 0.5 rad) | Robot upright, dist 0.27 m / 10 s | Action range fixed; reward shape still wrong |
| 5 | `20260408_124759` | switched reward `_fast_sac` (alive=10) → `g1_29dof_loco` (alive=1) | **dist 0.049 m / 500 steps** (≈ 0.005 m/s). `feet_phase` ep_sum +36 dominated the per-term decomposition; the policy learned to match the foot-phase clock *in place*. | alive bonus reduced; feet_phase exploit revealed |
| 6 | `20260408_142832` | dedicated `g1_29dof_loco_flashsac` reward + `g1_29dof_loco_single_flashsac` obs (this preset) | **Walks. dist 2.85 m / 10 s, avg speed 0.28 m/s**, stays upright, tracks commanded velocity + yaw through curving world-frame trajectories. Not benchmark-clean but unambiguously walking. | Reward shape stripped to IsaacLab-stock minimum |
| 7 | `20260408_154733` | reproducibility re-run of #6 with a different training seed | **Walks. dist 2.80 m / 10 s, avg speed 0.28 m/s**. Final TB metrics (critic_loss 2.10, actor_loss -2.71, entropy -13.81) match #6 within ±0.02 — the convergence basin is stable across seeds. | – |
| 8 | `20260408_181344` | widen `temp_target_sigma` 0.15 → 0.30 (stripped reward held fixed) | Entropy converged to theoretical target +6.23 (mechanically correct), but gait quality did NOT improve over #6/#7: visibly walks with bent posture. Policy gait plateau is not an entropy-budget issue. | Identified that the bottleneck is reward/env-side, not action noise |
| **9** | **(Option A — running)** | σ reverted to 0.15 + reward swapped to `g1_29dof_loco` (PPO default) + obs to `g1_29dof_loco_single_wolinvel` | Pending | Will answer whether FlashSAC is usable against holosoma's default reward zoo |

Codex was consulted at attempt #5 → #6 transition and converged on the same
diagnosis (degenerate `feet_phase` attractor + restrictive pose penalty +
narrow FlashSAC policy = "step in place and torso-locked" local optimum).
After #8 a second dual-audit pass (Claude's python-reviewer + Codex rescue)
was run against the port's 4 Codex-flagged correctness items; both cleared
every item as byte-identical-to-upstream. The bent-posture symptom is
therefore a reward/env issue, not a port-side correctness bug, which is
why #9 (Option A) swaps back to holosoma's PPO-default reward+observation
as the next empirical probe.

> **Historical note**: an earlier version of this doc reported that
> attempt #6 "does not really walk (dist 0.049 m / 10 s)" based on a probe
> whose output was mis-attributed to the wrong run directory. A careful
> re-probe of both checkpoints confirmed that #5 (alive=1 with
> feet_phase still present) is the "twitch in place" run and #6 (stripped
> reward) is the walking run.

## Running on MuJoCo Warp (mjwarp)

The same `FlashSACAgent(BaseAlgo)` adapter also works with the GPU-accelerated
MuJoCo Warp backend, because the bridge talks to `BaseTask` (simulator-agnostic)
rather than to the IsaacSim API directly. A sister experiment
`g1_29dof_flash_sac_mjwarp` is wired with `simulator=simulator.mjwarp` and lives
alongside the IsaacSim variant. Both use the holosoma PPO-default
`g1_29dof_loco` reward and `g1_29dof_loco_single_wolinvel` observation
(see Option A / Open work #3); the older dedicated `g1_29dof_loco_flashsac`
/ `g1_29dof_loco_single_flashsac` presets remain registered as a verified
fallback that walks at ~0.28 m/s (runs `20260408_142832` / `20260408_154733`).

Required setup (one-time):

```bash
source scripts/source_mujoco_setup.sh    # activates the hsmujoco conda env
pip install hydra-core gymnasium         # the only deps missing in hsmujoco
```

Smoke run:

```bash
bash scripts/run_flashsac_mjwarp_smoke.sh
# tunables (env vars): FLASHSAC_NUM_ENVS, FLASHSAC_SEED, FLASHSAC_NUM_ITERS
```

Full run (note: untested at scale — only 5-step smoke has been run):

```bash
source scripts/source_mujoco_setup.sh

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-flash-sac-mjwarp \
  --training.num-envs=1024 \
  --algo.config.num-learning-iterations=48829 \
  --training.headless=True
```

Pytest gate (separate marker so it does not run alongside the IsaacSim suite):

```bash
/root/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python -m pytest \
  -m "mujoco and slow" tests/e2e/test_flashsac_mjwarp_smoke.py -v
```

Notes:

- The hsmujoco env runs Python 3.10 / torch 2.10. `_resolve_compile_mode("auto")`
  returns `"max-autotune"` here (vs `"reduce-overhead"` in hssim), so the smoke
  script disables compile and AMP for safety. Re-enable both once the smoke is
  green.
- `scripts/source_mujoco_setup.sh` references `LD_LIBRARY_PATH` unconditionally,
  so the smoke script pre-initializes it before sourcing under `set -u`.
- The two backends share artifact paths; checkpoints land at
  `logs/hv-g1-manager/<timestamp>-g1_29dof_flash_sac_mjwarp_manager-locomotion/flashsac_step{N}/`.

## Training and play recipes

### Recommended training (IsaacSim, paper-matched ratio)

```bash
git pull
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-flash-sac \
  --training.num-envs=1024 \
  --algo.config.num-learning-iterations=48829 \
  --training.headless=True
```

`num_envs=1024` and `num_learning_iterations=48829` matches FlashSAC's paper
preset exactly (`scripts/run_isaaclab.sh`: `num_train_envs=1024`,
`num_env_steps=50_000_896` ≈ 48 829 interaction steps × 1 024 envs).
Larger `num_envs` (e.g. 4 096) requires `updates_per_interaction_step` to be
scaled up by the same factor (so 8.0 instead of 2.0) to keep the
update-to-sample ratio at the paper's ~1.95 × 10⁻³. Using `num_envs=4 096`
with the default `updates_per_interaction_step=2.0` gives a 4× lower update
ratio and produces an under-trained policy.

Useful tunables to override on the CLI:

```
--algo.config.use-compile=False   # disable torch.compile if hitting cache issues
--algo.config.use-amp=False       # disable AMP if hitting fp16 instability
--algo.config.normalize-reward=False
--algo.config.temp-target-sigma=0.30   # widen target entropy (default 0.15)
                                       # — tested at 20260408_181344 with
                                       # negative result; see Open work #1
```

**Fallback to the dedicated stripped preset** (if Option A / Open work #3
collapses on the PPO-default reward zoo):

```
# The stripped preset stays registered but is no longer the default.
# If Option A fails, override on the CLI:
--reward:g1-29dof-loco-flashsac
--observation:g1-29dof-loco-single-flashsac
```

Runs `20260408_142832` / `20260408_154733` trained this fallback combination
to a stable walking basin (~0.28 m/s forward, bent-posture quality).

### Watch metrics

A successful walking training should look like:

| metric | early | mid | late |
|---|---|---|---|
| `flashsac/critic/loss` | ~5 | ~1 | < 1.5 (still trending down or stable) |
| `flashsac/actor/loss` | ~+0.1 | ~-0.5 | continuing to decrease |
| `flashsac/actor/entropy` | ~0 | -10 | -14 (target) |
| `flashsac/temperature/value` | 0.01 | < 0.001 | collapsed (this is normal) |
| `flashsac/actor/mean_action` | ~0 | grows | non-trivial spread per joint |

Red flags from past failed runs:

- `actor/loss ≈ 0` for the entire run → policy stuck at a flat-Q local optimum
- `critic/loss` rises after midpoint → actor/critic divergence (policy exploiting
  a region the critic cannot value)
- `actor/loss` decreases but `mean_action ≈ 0` → policy collapsed to outputting nothing

### Play / evaluation

Step 1 — migrate any *legacy* (directory-only) checkpoint to the dual format
that `eval_agent.py` understands:

```bash
python scripts/flashsac_migrate_checkpoint.py \
    logs/hv-g1-manager/<timestamp>-...-locomotion/flashsac_step48829
```

Checkpoints saved by the *current* `FlashSACAgent.save` already produce both
formats automatically, so this step is only needed for runs from before the
dual-format change.

Step 2 — run `eval_agent.py` against the `.ckpt` file:

```bash
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint logs/hv-g1-manager/<timestamp>-...-locomotion/flashsac_step48829.ckpt \
    --training.num-envs=16 \
    --training.headless=False \
    --training.max-eval-steps=2000 \
    --training.export-onnx=False              # required: ONNX export not implemented
    --algo.config.load-reward-normalizer=False  # optional; auto-handled when normalize_reward=False
```

Notes:

- Do **not** pass `exp:g1-29dof-flash-sac` here. `eval_agent.py` reads the base
  config from the `.ckpt` directly; CLI args layer overrides on top.
- `--training.export-onnx=False` is required because
  `FlashSACAgent.actor_onnx_wrapper` raises `NotImplementedError`.
- `--training.headless=False` opens the viser viewer.

## Smoke override cheat sheet

The Gate A bash script encodes the minimum overrides needed to run vendored FlashSAC against G1 for 5 interaction steps without GPU compile / AMP / checkpointing / video / eval, while still actually exercising `agent.update()`:

```bash
python -m holosoma._vendored.flash_rl.train \
  --config_name flashSAC_base \
  --overrides env=isaaclab \
  --overrides env.env_name=Isaac-Velocity-Flat-G1-v0 \
  --overrides num_env_steps=320 \
  --overrides num_train_envs=64 \
  --overrides num_eval_envs=null --overrides num_record_envs=null \
  --overrides num_eval_episodes=0 --overrides num_record_episodes=0 \
  --overrides evaluation_per_interaction_step=0 \
  --overrides recording_per_interaction_step=0 \
  --overrides metrics_per_interaction_step=0 \
  --overrides logging_per_interaction_step=1 \
  --overrides save_checkpoint_per_interaction_step=0 \
  --overrides save_buffer_per_interaction_step=null \
  --overrides agent.use_compile=false --overrides agent.use_amp=false \
  --overrides agent.buffer_min_length=64 \
  --overrides agent.sample_batch_size=64 \
  --overrides agent.buffer_max_length=4096 \
  --overrides agent.normalize_reward=false \
  --overrides logger_type=tensorboard
```

With 64 envs × 5 steps = 320 transitions and `buffer_min_length=64`, the replay buffer becomes sampleable on iteration 1 and 5 update() calls run before exit.

## Re-syncing from upstream

To pull new changes from `/workspace/FlashSAC` (or whichever clone of upstream FlashSAC), run:

```bash
SRC=/path/to/FlashSAC
DST=src/holosoma/holosoma/_vendored/flash_rl
# 1. Diff first to know what changed
diff -r "$SRC/flash_rl" "$DST" | grep -v '^---\|^+++\|holosoma._vendored' | head
# 2. Mirror selectively
cd "$SRC/flash_rl" && find . -name '*.py' -newer "$DST/__init__.py" | while read f; do
    cp "$f" "$DST/$f"
done
# 3. Re-run namespace rewrite
find "$DST" -name '*.py' | xargs -I {} perl -i -pe '
    s/(\bfrom\s+)flash_rl(\.|\s)/$1holosoma._vendored.flash_rl$2/g;
    s/(\bimport\s+)flash_rl(\.|\s|$)/$1holosoma._vendored.flash_rl$2/g;
' {}
# 4. Re-apply the train.py / play_isaaclab.py refactor patches if upstream touched them
# 5. Re-apply the bridge action-scaling change if upstream changed sample_actions
# 6. Re-run unit tests
pytest tests/unit/test_flash_rl_vendor.py tests/unit/test_flash_sac_bridge.py -v
# 7. Re-run Gate A smoke
bash scripts/run_flashsac_isaaclab_smoke.sh
```

## Verification status

| Phase | What | Status |
|---|---|---|
| 0 | Pre-flight (deps, G1 task registration, Hydra reinit) | ✅ |
| 1 | Verbatim vendoring (44 .py + 13 yaml + train.py + play_isaaclab.py + 8 scripts) | ✅ |
| 1.5 | `train.py` library refactor (`build_cfg`/`run`/`main`) | ✅ |
| 2 | Unit tests (CPU) — 7 vendor + 7 bridge & adapter | ✅ 14/14 |
| 3 | Gate A manual run | ✅ |
| 4 | Gate A pytest e2e | ✅ |
| 5 | Layer 2 adapter (bridge, agent, config, experiment, subcommand) | ✅ |
| 6 | Gate B manual run (IsaacSim) | ✅ |
| 6b | Gate B pytest e2e (IsaacSim) | ✅ |
| 6c | Gate B-mjwarp manual run | ✅ (smoke only) |
| 6d | Gate B-mjwarp pytest e2e | ✅ |
| 7 | Vendored shell scripts (8 under `scripts/_vendored/flash_rl/`) | ✅ |
| 8 | Docs (this file) | ✅ |
| 9 | `play_isaaclab.py` library refactor | ✅ |
| 10 | `eval_agent.py` integration via dual-format `.ckpt` | ✅ |
| 11 | Checkpoint migration script for legacy directories | ✅ |
| 12 | **Walking on G1 IsaacSim with the FlashSAC-tuned reward preset** | ⚠️ "kind of walking" (not benchmark-clean) |
| 13 | Walking on G1 with `reward.g1_29dof_loco` (PPO default) | ⏳ Option A empirical run pending (earlier "stand-still" result was at attempt #5 / `20260408_124759`, before the action-scale fix. Re-opened as Open work #3 under the now-current uniform `scale=0.5` action bridge + upstream sigma=0.15 algo defaults.) |
| 14 | Full mjwarp training (not just smoke) | ❌ untested |
| 15 | ONNX export | ❌ raises NotImplementedError |
| 16 | Multi-GPU validation | ❌ untested |
| 17 | T1 / K1 / WBT compatibility | ❌ untested |

## Current state and limitations

The port is **functionally complete** as a holosoma-installable RL algorithm:
the bridge exists, the adapter wires through `train_agent.py`, the eval flow
works through `eval_agent.py`, all 17 tests are green on hssim, and the
checkpoint format is compatible with holosoma's standard inspection tools.

The port is **not production-quality** in the following ways:

### 1. Whether FlashSAC needs a dedicated reward/observation preset is now being re-tested

The previous conclusion ("FlashSAC cannot use `reward.g1_29dof_loco` and requires the stripped preset") was drawn from attempts #1-#5 which were **also** confounded by the action-scale bug (attempts #1, #2, #3, #4 all had the wrong action range) and by the FastSAC alive-bonus carry-over (attempt #5). After those fixes landed, the stripped-preset run at attempt #6 walked, but the PPO-default preset was never re-tested on top of the fully-fixed action bridge.

**Open work #3 (Option A)** re-runs exactly that control. As of the current HEAD `g1_29dof_flash_sac` / `g1_29dof_flash_sac_mjwarp` use `reward=g1_29dof_loco` + `observation=g1_29dof_loco_single_wolinvel` + upstream algo defaults (sigma=0.15, asymmetric_observation=false, n_step=3). Whether this converges to walking is pending empirical verification.

If it converges, the dedicated stripped preset is no longer required and the port becomes drop-in for holosoma's reward zoo. If it collapses, the stripped preset remains registered as a verified fallback (20260408_142832 / 20260408_154733 baseline).

### 2. Walking quality is "it walks" not "benchmark-clean"

The latest training run (`20260408_142832` / reproduced by `20260408_154733`)
produces a policy that empirically walks at avg 0.28 m/s over 10 sim seconds,
stays upright, and tracks commanded yaw through curving trajectories. The
gait is visibly not polished (the user's assessment was 「綺麗ではないが歩いて
はいた」 — "it's walking but not pretty"). FlashSAC's paper benchmarks on
`Isaac-Velocity-Flat-G1-v0` show much higher fidelity. The gap is most likely
due to:

- holosoma's reward shape, even after stripping, differs from IsaacLab stock in
  small ways (tracking sigma, term names, command sampling distribution,
  termination conditions)
- holosoma's action manager applies a fixed `0.25 × tanh × bridge_multiplier`
  instead of IsaacLab's direct `0.5 × tanh` — there is a small extra layer of
  PD control inside holosoma's `JointPositionActionTerm`
- Hyperparameters (especially `temp_target_sigma`, `actor_noise_zeta_*`) have
  not been tuned for holosoma's specific reward landscape

### 3. Hyperparameters are stock paper defaults, not holosoma-tuned

Every hyperparameter in `algo.flash_sac` is the upstream `configs/agent/flashSAC.yaml` default. No tuning was done for holosoma's reward, action layer, or simulator backend.

### 4. Several auxiliary features are unimplemented

- `actor_onnx_wrapper` raises `NotImplementedError`
- Multi-GPU is not validated (the adapter accepts `multi_gpu_cfg` but the
  vendored agent is single-process under the hood)
- Resume-from-checkpoint is supported by the dual-format `.ckpt` but never
  tested in continuous training
- `g1_29dof_flash_sac_mjwarp` has only been smoke-tested (5 steps); never
  trained to convergence
- T1, K1, WBT robots have no FlashSAC experiment configs

### 5. Six legacy checkpoints are mutually incompatible

Each major fix changed the env transition the policy learns against
(action-scale regime, reward shape), so no two of the six training runs
share a compatible policy. Only `20260408_142832` is currently meaningful.

### 6. Some port-side details have not been deeply audited

Codex's review at the end of the porting sprint flagged a checklist of
port-side things to verify carefully if walking did not converge. After the
reward fix made walking emerge, those items were not re-checked at code
level. Specifically:

- `_sample_flashsac_actions` — confirm stochastic sample is used during
  data collection and `tanh(mean)` only at eval; confirm tanh-squash log-prob
  correction is present in `update_actor`
- Target entropy sign and magnitude for 29-D actions — heuristic gives ≈ -14,
  worth confirming the sign convention matches the actor's entropy
  computation
- Bridge `terminated` vs `truncated` handling alignment with the n-step
  buffer's reset semantics
- Reward and done arrays alignment with the same transition row in the
  replay buffer

These are subtle but important. If walking quality remains a concern after
hyperparameter tuning, these are the next things to inspect.

## Open work

Suggested order, in increasing scope:

1. ~~**Re-run with `temp_target_sigma=0.30` (was 0.15)** against the current
   `g1_29dof_loco_flashsac` preset.~~ ❌ **DONE and NEGATIVE.** Training run
   `20260408_181344` converged the actor entropy to the theoretical target
   (+6.23 matches `0.5 * 29 * log(2πe * 0.09)`), so the sigma widening was
   mechanically correct, but the resulting gait was not visibly better than
   the σ=0.15 baseline (`20260408_142832` / `20260408_154733`) — the policy
   still walked with visibly bent posture. Conclusion: action-noise budget is
   NOT the bottleneck. Override reverted to upstream 0.15 alongside the
   Option A change below.
2. ~~**Audit port-side items Codex flagged.**~~ ✅ **DONE.** Line-cited audit
   of `_vendored/flash_rl/agents/flashSAC/agent.py`, `layer.py`, `update.py`,
   `buffers/torch_buffer.py`, and the holosoma adapter/bridge against
   `/workspace/FlashSAC` found:
   - **Item 1 (stochastic sample vs deterministic eval):** OK. `training=True`
     uses `tanh(mu + sigma*eps*temperature=1.0)`; `training=False` uses
     `tanh(mu)` directly (temperature=0.0 short-circuit in
     `_sample_flashsac_actions`). The flag plumbs correctly from
     `FlashSACAgent.evaluate_policy` → inner `sample_actions`.
   - **Item 2 (tanh-squash log-prob correction):** OK.
     `safe_tanh_log_det_jacobian(x) = 2*(log(2) - x - softplus(-2x))` is
     applied in `NormalTanhPolicy.forward` and threaded through both
     `update_actor` and `update_temperature` via `info["log_prob"]`. Byte-
     identical to upstream.
   - **Item 3 (target entropy sign / magnitude):** OK. Formula at
     `agent.py:355` matches the unbounded-Gaussian heuristic; temperature
     loss `alpha * (entropy - target_entropy)` has the correct sign (raises
     alpha when entropy < target, lowers when > target). Numerical check:
     σ=0.15 → -13.86, σ=0.30 → +6.24.
   - **Item 4 (n-step buffer + terminated/truncated alignment):** OK.
     Bridge splits `truncated = time_outs`, `terminated = reset & ~time_outs`;
     buffer n-step mask uses `done = terminated | truncated` so the n-step
     return truncates at either; but `batch["terminated"]` (the zero-bootstrap
     flag used by the critic target) preserves only the true terminations.
     Final-obs alignment between bridge and buffer was separately verified.
   - **False alarm on "privileged observation leakage":** the initial audit
     flagged that `asymmetric_observation=False` (holosoma default) makes the
     actor consume the full `[actor_obs || critic_obs]` concat. This is the
     **intended upstream behavior**: `/workspace/FlashSAC/scripts/run_isaaclab.sh`
     *explicitly* sets `--overrides agent.asymmetric_observation=false` for
     `Isaac-Velocity-Flat-G1-v0`. Our bridge matches upstream exactly.
   - **Dormant bug in `get_inference_policy`:** the closure at
     `flash_sac_agent.py:385-401` only feeds `obs[actor_obs_keys[0]]` to the
     actor, but the actor was trained with `input_dim = total_obs_dim`.
     Shape-mismatch at call time. However, the closure is currently only
     reachable via `actor_onnx_wrapper`, which raises `NotImplementedError`,
     so it is dormant. The `eval_agent.py` → `FlashSACAgent.evaluate_policy` →
     bridge path returns the full concat and is unaffected. Must be fixed
     before Open work #4 (ONNX export).
3. **Option A — pair FlashSAC with holosoma's PPO-default reward+observation
   AND revert `temp_target_sigma` to upstream 0.15.** ⏳ **Config applied,
   empirical run pending.** `g1_29dof_flash_sac` / `g1_29dof_flash_sac_mjwarp`
   now use `reward=g1_29dof_loco` + `observation=g1_29dof_loco_single_wolinvel`
   + upstream algo defaults (`temp_target_sigma=0.15`, `asymmetric_observation
   =false`, `n_step=3`, `updates_per_interaction_step=2`, `use_amp=true`).
   This is **one training run** that answers two questions at once:
   - Is holosoma's PPO-default reward zoo compatible with FlashSAC? (If yes,
     the dedicated stripped preset becomes optional.)
   - Was the σ=0.30 widening actively harmful relative to the paper default?
   If this run walks cleanly, the port is usable across holosoma's reward
   zoo. If it collapses into the previously-observed "twitch in place" /
   "stand still and collect alive" attractors, fall back to the stripped
   preset (`g1_29dof_loco_flashsac` + `g1_29dof_loco_single_flashsac` are
   still registered and verified to walk at ~0.28 m/s on `20260408_142832`
   and `20260408_154733`) and move on to Open work #4+.
4. **Implement `actor_onnx_wrapper`** (and fix the `get_inference_policy`
   shape-mismatch bug uncovered in #2). Mirror `FastSACAgent.actor_onnx_wrapper`
   but with the FlashSAC actor's `get_mean_and_std` head, feeding the full
   `[actor_obs || critic_obs]` concat that matches what `evaluate_policy`
   already gives the actor. ~1-2 hours.
5. **Full mjwarp training.** Run the full ~48 829-iteration recipe in
   `hsmujoco` and confirm walking quality is comparable to the IsaacSim run.
6. **Multi-GPU validation.** Run `torchrun --nproc_per_node=2
   train_agent.py exp:g1-29dof-flash-sac …` and confirm gradients converge.
7. **T1 / K1 experiment configs.** Add `t1_29dof_flash_sac` and
   `k1_22dof_flash_sac`. Whether they need dedicated FlashSAC-tuned presets
   depends on the outcome of Option A (#3).
8. **Re-train all FlashSAC G1 runs** with the converged hyperparameters and
   the reward preset that ends up working, and replace the legacy
   checkpoints with a single canonical "good" checkpoint and a recorded run.
9. **WBT (whole-body tracking) compatibility** — much harder, since WBT has
   a different action and observation schema and FlashSAC's collapse
   dynamics may be even harsher there.

Task #3 (Option A) is by far the highest leverage remaining — one 45-minute
training run answers the usability question for the whole port.

## Commit history (porting sprint)

The full sequence of fixes, in commit order:

| hash | what it fixed |
|---|---|
| `f3857eb` | Initial verbatim vendoring of every flash_rl file + configs + scripts |
| `a6879be` | Layer 2 adapter (`FlashSACAgent` + `FlashSACGymBridge`) + experiment configs |
| `2be11be` | Unit tests, e2e smoke tests, conftest markers |
| `0b6ad39` | First version of this docs file |
| `0986f36` | Bridge `final_obs` indexing fix (was assigning full tensor instead of `env_ids` slice) |
| `35166cb` | `play_isaaclab.py` library refactor for cwd-independence |
| `72899c1` | Dual-format checkpoint save/load + `evaluate_policy` + migration script |
| `d37b991` | Logging fix: accumulate `update_info` across window so actor metrics actually appear |
| `f3165c7` | Per-joint action scaling (FastSAC-style) — *superseded* |
| `d2faae4` | **Uniform action scaling matching IsaacLab stock G1 (`scale=0.5`)** |
| `bc57ff3` | Switch reward to `g1_29dof_loco` (alive=1.0) — *superseded by next* |
| `a05648b` | **Dedicated `g1_29dof_loco_flashsac` reward + `g1_29dof_loco_single_flashsac` observation preset** |
| (this commit) | Updated docs reflecting actual state at end of porting sprint |

`f3165c7` and `bc57ff3` are kept in history rather than rebased away because
the trial-and-error process is part of how to interpret the port: each
commit unblocked one diagnostic layer and reading them in order is the
fastest way to understand why the current `FlashSACGymBridge` and
`g1_29dof_loco_flashsac` look the way they do.

## File index

```
src/holosoma/holosoma/_vendored/flash_rl/    (44 .py files)
src/holosoma/holosoma/_vendored/flash_rl/configs/  (13 .yaml files)
scripts/_vendored/flash_rl/                  (8 .sh files)
src/holosoma/holosoma/agents/flash_sac/      (3 .py files)
src/holosoma/holosoma/config_types/algo.py            (FlashSACVendorConfig + FlashSACVendorAlgoConfig added)
src/holosoma/holosoma/config_values/algo.py           (flash_sac default added)
src/holosoma/holosoma/config_values/loco/g1/observation.py (g1_29dof_loco_single_flashsac added)
src/holosoma/holosoma/config_values/loco/g1/reward.py     (g1_29dof_loco_flashsac added)
src/holosoma/holosoma/config_values/loco/g1/experiment.py (g1_29dof_flash_sac, g1_29dof_flash_sac_mjwarp added)
src/holosoma/holosoma/config_values/observation.py    (g1_29dof_loco_single_flashsac registered)
src/holosoma/holosoma/config_values/reward.py         (g1_29dof_loco_flashsac registered)
src/holosoma/holosoma/config_values/experiment.py     (subcommands registered)
tests/unit/test_flash_rl_vendor.py
tests/unit/test_flash_sac_bridge.py
tests/e2e/test_flashsac_isaaclab_smoke.py
tests/e2e/test_flashsac_holosoma_smoke.py
tests/e2e/test_flashsac_mjwarp_smoke.py
scripts/run_flashsac_isaaclab_smoke.sh
scripts/run_flashsac_holosoma_smoke.sh
scripts/run_flashsac_mjwarp_smoke.sh
scripts/flashsac_migrate_checkpoint.py
docs/flashsac_port.md
```

# FlashSAC Port

This document describes the holosoma port of [FlashSAC](https://github.com/joonleesky/FlashSAC) — the off-policy SAC variant published in *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control* (Kim et al., arXiv 2026).

The port is **complete**: every Python module, every Hydra YAML, and every shell script under upstream `flash_rl/`, `configs/`, and `scripts/` is mirrored under holosoma. The only allowed deviations are mechanical (namespace rewrites) or quality-of-life (Hydra reinit safety, `parse_known_args`, library-form `train.py`). The original training loop body is byte-equivalent to upstream.

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
        ├── flash_sac_agent.py      #   FlashSACAgent(BaseAlgo)
        └── flash_sac_env_bridge.py #   FlashSACGymBridge: BaseTask -> gym VectorEnv

scripts/
├── run_flashsac_isaaclab_smoke.sh  # Layer 3: Gate A (vendored train.py path)
├── run_flashsac_holosoma_smoke.sh  # Layer 3: Gate B (holosoma BaseAlgo path)
└── _vendored/flash_rl/
    └── run_*.sh                    # All upstream batch-experiment scripts
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

The bridge handles the dones split:

```python
truncated = time_outs.bool()
terminated = reset_buf.bool() & ~truncated
```

and re-stitches `final_obs` from `extras["final_observations"]` for envs that reset on the current step.

#### `FlashSACAgent`

Subclass of `holosoma.agents.base_algo.BaseAlgo`. Constructor wraps the holosoma env in `FlashSACGymBridge`, then `setup()` instantiates the vendored `FlashSACAgent` directly via `FlashSACConfig(**asdict(holosoma_dataclass))`. The `learn()` loop is in-line (mirroring upstream `train.py` lines 113-200) so it can plug into holosoma's logger / checkpoint conventions.

`learning_rate_warmup_step` and `learning_rate_decay_step` are computed at `setup()` time from `learning_rate_*_rate * num_learning_iterations * updates_per_interaction_step` (upstream Hydra computes these via `${eval: …}`).

### Layer 3 — Smoke gates

| Gate | Script | Test | Proves |
|---|---|---|---|
| **A** | `scripts/run_flashsac_isaaclab_smoke.sh` | `tests/e2e/test_flashsac_isaaclab_smoke.py` | Vendored FlashSAC runs IsaacLab's stock `Isaac-Velocity-Flat-G1-v0` for 5 interaction steps end-to-end (config + env + agent + buffer + update + logger). |
| **B** | `scripts/run_flashsac_holosoma_smoke.sh` | `tests/e2e/test_flashsac_holosoma_smoke.py` | Holosoma adapter runs the vendored algorithm against `LeggedRobotLocomotionManager` on the IsaacSim backend via `train_agent.py exp:g1-29dof-flash-sac` for 5 outer iterations. |
| **B-mjwarp** | `scripts/run_flashsac_mjwarp_smoke.sh` | `tests/e2e/test_flashsac_mjwarp_smoke.py` | Same Layer 2 adapter, but with `simulator=simulator.mjwarp` (`g1_29dof_flash_sac_mjwarp`). Runs in the `hsmujoco` conda env. |

The IsaacSim e2e tests are marked `@pytest.mark.isaacsim @pytest.mark.slow` and the mjwarp e2e test is marked `@pytest.mark.mujoco @pytest.mark.slow`. All three shell out to bash via `subprocess.run` so each invocation runs in a fresh Python process — required because IsaacSim is single-process per `AppLauncher` and to keep the two conda envs (`hssim` vs `hsmujoco`) cleanly separated.

## Running on MuJoCo Warp (mjwarp)

The same `FlashSACAgent(BaseAlgo)` adapter also works with the GPU-accelerated
MuJoCo Warp backend, because the bridge talks to `BaseTask` (simulator-agnostic)
rather than to the IsaacSim API directly. A sister experiment
`g1_29dof_flash_sac_mjwarp` is wired with `simulator=simulator.mjwarp` and lives
alongside the IsaacSim variant.

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

Full run:

```bash
source scripts/source_mujoco_setup.sh

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-flash-sac-mjwarp \
  --training.num-envs=1024 \
  --algo.config.num-learning-iterations=50000 \
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
# 4. Re-apply the train.py refactor patches if upstream touched train.py
# 5. Re-run unit tests
pytest tests/unit/test_flash_rl_vendor.py tests/unit/test_flash_sac_bridge.py -v
# 6. Re-run Gate A smoke
bash scripts/run_flashsac_isaaclab_smoke.sh
```

## Verification status

| Phase | What | Status | Wall-clock |
|---|---|---|---|
| 0 | Pre-flight | ✅ all deps present, G1 task registered, Hydra reinit OK | <30 s |
| 1 | Verbatim vendoring | ✅ 44 .py + 13 yaml + train.py + play_isaaclab.py + 8 scripts | – |
| 1.5 | `train.py` library refactor | ✅ build_cfg/run/main, idempotent compose | – |
| 2 | Unit tests (CPU) | ✅ 7/7 pytest pass | ~3 s |
| 3 | Gate A manual run | ✅ 5/5 tqdm, TensorBoard event with 5 critic updates + 3 actor updates | ~13 s |
| 4 | Gate A pytest e2e | ✅ pass | ~12 s |
| 5 | Layer 2 adapter | ✅ bridge, agent, config, experiment, subcommand registered | – |
| 5b | Bridge unit tests | ✅ 4/4 pytest pass (mock BaseTask) | ~1 s |
| 6 | Gate B manual run (IsaacSim) | ✅ 5/5 tqdm, holosoma checkpoint dir + TensorBoard event | ~14 s |
| 6b | Gate B pytest e2e (IsaacSim) | ✅ pass | ~19 s |
| 6c | Gate B-mjwarp manual run | ✅ 5/5 tqdm, mjwarp checkpoint dir + TensorBoard event | ~13 s |
| 6d | Gate B-mjwarp pytest e2e | ✅ pass | ~20 s |
| 7 | Vendored shell scripts | ✅ 8 scripts under `scripts/_vendored/flash_rl/` | – |
| 8 | Docs | ✅ this file | – |

## Known limitations / non-goals

- **ONNX export**: `FlashSACAgent.actor_onnx_wrapper` raises `NotImplementedError`. Implementing ONNX export was out of scope for the smoke proof.
- **Multi-GPU**: The adapter accepts `multi_gpu_cfg` but the underlying vendored agent is single-process. Multi-GPU support is a follow-up.
- **`torch.compile` / AMP**: Both default to `True` in the holosoma config but are disabled in the smoke scripts. The compile path uses `reduce-overhead` mode on torch 2.7.0 (verified by `_resolve_compile_mode('auto')`).
- **Optional simulator wrappers** (`mujoco_playground`, `genesis`, `dmc`, etc.): vendored verbatim but never imported in the IsaacSim hot path; they are deferred imports in `flash_rl/envs/__init__.py`.
- **`asymmetric_observation` mode** is supported by the bridge but not exercised by the G1 smoke (the stock `Isaac-Velocity-Flat-G1-v0` task uses a single `policy` observation group).

## File index

A complete map of every vendored file is generated at port time and lives at:

```
src/holosoma/holosoma/_vendored/flash_rl/    (44 .py files)
src/holosoma/holosoma/_vendored/flash_rl/configs/  (13 .yaml files)
scripts/_vendored/flash_rl/                  (8 .sh files)
src/holosoma/holosoma/agents/flash_sac/      (3 .py files)
tests/unit/test_flash_rl_vendor.py
tests/unit/test_flash_sac_bridge.py
tests/e2e/test_flashsac_isaaclab_smoke.py
tests/e2e/test_flashsac_holosoma_smoke.py
scripts/run_flashsac_isaaclab_smoke.sh
scripts/run_flashsac_holosoma_smoke.sh
docs/flashsac_port.md
```

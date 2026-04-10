# FlashSAC Reward Translation Guide

How to convert any holosoma PPO locomotion reward preset into a FlashSAC-compatible preset, and how to expand FlashSAC support to new robots.

## Why FlashSAC cannot use PPO-default rewards directly

FlashSAC's policy is **narrow by design**: `temp_target_sigma=0.15` produces a target entropy of `0.5 * d * log(2πe * 0.15²)` which is strongly negative for typical humanoid action dims (≈ -13.86 for G1 29-DoF, ≈ -10.51 for K1 22-DoF). The policy commits to near-deterministic actions early in training and cannot broadly explore.

Holosoma's PPO-default reward contains terms that create **zero-action attractors** — local optima where the easiest reward-maximising strategy for a narrow policy is "don't move":

| Term | PPO weight | Why it kills FlashSAC |
|---|---|---|
| `alive` | +1.0 | Constant +1/step for surviving. Narrow policy: stand still = free reward. |
| `feet_phase` | 5.0 | Phase clock match with micro-oscillation gives +36 ep_sum without forward progress. |
| `pose` | -0.5 (ub×50) | Upper body weight=50 locks torso to default. Policy freezes. |
| `penalty_action_rate` | -2.0 | Punishes any movement. Narrow policy: zero action = zero penalty. |
| `penalty_orientation` | -10.0 | Very strong tilt penalty discourages any weight transfer. |

PPO and FastSAC handle these because their policies start wide (PPO: on-policy exploration; FastSAC: `target_entropy = -action_dim`, broader). FlashSAC's `temp_target_sigma=0.15` sabotages exactly the exploration phase that discovers walking.

**Empirical confirmation** (run `20260409_053822`, Option A): pairing FlashSAC with holosoma's full `g1_29dof_loco` reward at upstream sigma=0.15 collapsed deterministically — `actor/loss → -0.01`, `mean_action → -0.0015`. Robot holds default pose and falls.

## The canonical v5 translation recipe

Validated on G1 across 16 training runs. Best result: **1.44× composite PPO-divergence** (run `20260409_153439`).

### Rule: keep all terms except `alive`, re-weight to FlashSAC-safe levels

| Term | PPO weight | FlashSAC weight | Ratio | Rationale |
|---|---|---|---|---|
| `tracking_lin_vel` | 2.0 | **2.0** | 100% | Task-defining. Keep at full weight. |
| `tracking_ang_vel` | 1.5 | **1.5** | 100% | Task-defining. Keep at full weight. |
| `feet_phase` | 5.0 | **4.0** | 80% | Forces alternating gait. 0.5 was too weak for backward walking with stiff upper body; 4.0 works. |
| `pose` (outer weight) | -0.5 | **-0.2** | 40% | -0.05 left upper body 3-11× too loose; -0.5 collapsed FlashSAC. |
| `pose` (upper body joint weights) | 50 | **150** | 300% | Compensates for 2.5× weaker outer weight. Effective: -0.2×150 = -30 (PPO: -25). |
| `pose` (leg joint weights) | per-joint | **same as PPO** | 100% | Leg weights are already small (0.01-5.0); no change needed. |
| `penalty_orientation` | -10.0 | **-1.0** | 10% | -3.0 worsened stability (2.48×). -1.0 is the sweet spot. |
| `penalty_ang_vel_xy` | -1.0 | **-0.05** | 5% | -0.3 improved roll but worsened pitch (1.63×). -0.05 is the sweet spot. |
| `penalty_action_rate` | -2.0 | **-0.005** | 0.25% | 400× reduction. PPO's -2.0 instantly kills exploration. |
| `penalty_feet_ori` | -5.0 | **-0.5** | 10% | Gentle foot orientation guidance. |
| `penalty_close_feet_xy` | -10.0 | **-1.0** | 10% | Gentle anti-crossing. |
| ~~`alive`~~ | 1.0 | **DROPPED** | — | Creates lazy-attractor even at reduced weight. |

### Observation: use PPO-default unchanged

The v5 recipe retains `feet_phase` in the reward, so the `sin_phase`/`cos_phase` clock terms must remain in the observation. Use the same `{robot}_loco_single_wolinvel` preset as PPO.

### Algorithm config: upstream FlashSAC defaults verbatim

```
temp_target_sigma=0.15     # upstream default (0.30 tested and negative)
asymmetric_observation=False  # upstream default (actor sees [actor_obs||critic_obs] concat)
n_step=3
updates_per_interaction_step=2
use_amp=True
normalize_reward=True
```

### Curriculum: use `{robot}_curriculum_fast_sac`

The fast_sac curriculum starts penalties at 50% and ramps to 100%, which gives FlashSAC a gentler warm-up than PPO's curriculum (starts at 10%).

## Tuning history (G1, 16 runs)

| # | Run | Change | Composite | Result |
|---|---|---|---|---|
| 6 | `20260408_142832` | Initial stripped preset (5 terms only) | — | Walks 0.28 m/s, bent posture |
| 7 | `20260408_154733` | Reproduce #6 | — | Walks 0.28 m/s (seed-stable) |
| 8 | `20260408_181344` | σ=0.30 on stripped | — | Entropy converged but gait unchanged. σ is NOT the bottleneck. |
| 9 | `20260409_053822` | Option A: full PPO reward + σ=0.15 | — | **Collapsed**. mean_action→0. Fell. |
| 10 | `20260409_071046` | Add pose=-0.05 only | 1.87× | Forward improved, backward two-foot jumping |
| 11 | `20260409_075929` | Add all shaping at 10× weaker, feet_phase=0.5 | — | Backward still jumping |
| 12 | `20260409_090517` | feet_phase 0.5→2.5 | — | Backward fixed. Visually good. |
| 13 | `20260409_120728` | pose -0.05→-0.2 | 1.87× | Upper body sway reduced |
| 14 | `20260409_125729` | orientation -1→-3 | 2.48× | **Worse**. Reverted. |
| 15 | `20260409_142000` | pose ub weights 50→150 | 1.72× | Upper body improved, backward hopping returned |
| 16 | `20260409_153439` | feet_phase 2.5→4.0 | **1.44×** | **Best.** Backward fixed, upper body tight. |
| 17 | `20260410_023100` | ang_vel_xy -0.05→-0.3 | 1.63× | **Worse**. Reverted. |

### Key findings

1. **`alive` is the one term that must be fully removed.** Even at reduced weight it creates a standing-still attractor.
2. **`feet_phase` needs ≥80% of PPO weight** for backward gait. Lower values allow two-foot hopping.
3. **`pose` outer weight must be ≤40% of PPO**, compensated by 3× upper-body joint weights.
4. **Penalty terms need dramatic weakening** (5-400×). Increasing them toward PPO values consistently worsened the composite score.
5. **`penalty_orientation` and `penalty_ang_vel_xy` are sensitive** — modest increases (3× and 6×) both backfired by making the policy too stiff.

### Remaining gap (1.44×)

| Metric | PPO | FlashSAC | Ratio |
|---|---|---|---|
| root pitch std | 0.92° | 1.04° | 1.1× |
| height std | 0.014 m | 0.015 m | 1.1× |
| leg joints | — | — | 1.2× |
| upper body joints | — | — | 1.6× |
| **root roll std** | **1.16°** | **2.80°** | **2.4×** |

Root roll (left-right sway) is the primary remaining gap. Attempts to close it via `penalty_ang_vel_xy` or `penalty_orientation` both backfired. This may be an inherent characteristic of FlashSAC's narrow policy — the near-deterministic gait has less margin to absorb lateral perturbations compared to PPO's wider policy.

## Using the transform helper

```python
from holosoma.config_values.loco.flashsac_transform import (
    make_flashsac_reward,
    make_flashsac_observation,
)
from holosoma.config_values.loco.k1.reward import k1_22dof_loco
from holosoma.config_values.loco.k1.observation import k1_22dof_loco_single_wolinvel

# Generate FlashSAC-compatible reward (drops alive, re-weights everything else)
k1_flashsac_reward = make_flashsac_reward(k1_22dof_loco)

# Observation: PPO-default passthrough (no changes needed for v5 recipe)
k1_flashsac_obs = make_flashsac_observation(k1_22dof_loco_single_wolinvel)
```

The helper is validated by unit tests in `tests/unit/test_flashsac_transform.py` that verify:
- G1 transform output matches the hand-tuned `g1_29dof_loco_flashsac` preset exactly
- Only `alive` is dropped
- K1 transform produces 9 terms with expected weights
- Upper body pose_weights are set to 150
- Transform is idempotent

## K1 expansion roadmap

### Prerequisites (done)

- [x] FlashSAC port complete (vendored + adapter + bridge)
- [x] G1 canonical recipe established (v5, 1.44× composite)
- [x] Translation helper validated on G1 ground truth
- [x] Translation helper produces correct K1 output (unit test green)

### Steps to enable K1

1. **Create `k1_22dof_loco_flashsac` reward preset**
   ```python
   # In config_values/loco/k1/reward.py
   from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
   k1_22dof_loco_flashsac = make_flashsac_reward(k1_22dof_loco)
   ```
   Register in `config_values/reward.py` DEFAULTS dict.

2. **Create `k1_22dof_flash_sac` experiment**
   ```python
   # In config_values/loco/k1/experiment.py
   k1_22dof_flash_sac = ExperimentConfig(
       env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
       training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_manager"),
       algo=algo.flash_sac,
       simulator=simulator.isaacsim,
       robot=robot.k1_22dof,
       terrain=terrain.terrain_locomotion_mix,
       observation=observation.k1_22dof_loco_single_wolinvel,  # PPO-default (has phase clock)
       action=action.k1_22dof_joint_pos,
       termination=termination.k1_22dof_termination,
       randomization=randomization.k1_22dof_randomization,
       command=command.k1_22dof_command,
       curriculum=curriculum.k1_22dof_curriculum_fast_sac,
       reward=reward.k1_22dof_loco_flashsac,
   )
   ```
   Register in `config_values/experiment.py` DEFAULTS dict.

3. **Unit smoke test** — verify the experiment config resolves without error and the observation/reward dimensions are self-consistent.

4. **5-step e2e smoke** — run for 5 interaction steps to catch shape mismatches:
   ```bash
   python src/holosoma/holosoma/train_agent.py exp:k1-22dof-flash-sac \
     --training.num-envs=64 \
     --algo.config.num-learning-iterations=5 \
     --training.headless=True
   ```

5. **Full training** (~45 min on RTX 5090):
   ```bash
   python src/holosoma/holosoma/train_agent.py exp:k1-22dof-flash-sac \
     --training.num-envs=1024 \
     --algo.config.num-learning-iterations=48829 \
     --training.headless=True
   ```

6. **Quantitative eval** — record trajectory NPZ and compare against K1 PPO baseline using the same `compare_eval.py` methodology used for G1. Target: composite ≤1.5×.

7. **If K1 composite > 2.0×** — the translation may need K1-specific weight tuning. Start from the G1 v5 recipe and adjust the term that shows the largest per-metric gap (same iterative approach as G1 runs #10-#17).

### Expected K1 differences from G1

- **22-DoF** (vs 29): target_entropy = -10.51 (vs -13.86). Narrower action space → faster convergence but also faster collapse if reward is wrong.
- **Different joint layout**: K1 has 10 upper body DoFs + 12 leg DoFs. G1 has 17 upper body + 12 leg. Upper body constraint may need different `upper_body_start_idx` — verify the K1 pose_weights ordering in `k1_22dof_loco`.
- **Different dynamics**: K1 is a different robot — gait frequency, COM height, foot clearance all differ. The reward WEIGHTS should transfer (same transform), but the reward PARAMETERS (e.g., `swing_height=0.09` in feet_phase, `close_feet_threshold=0.15`) may need K1-specific values — check what `k1_22dof_loco` already uses and preserve those.

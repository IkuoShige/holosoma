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
    K1_UPPER_BODY_POSE_INDICES,
)
from holosoma.config_values.loco.k1.reward import k1_22dof_loco
from holosoma.config_values.loco.k1.observation import k1_22dof_loco_single_wolinvel

# Generate FlashSAC-compatible reward
# IMPORTANT: K1 has upper body FIRST (indices 0-9), unlike G1 (12-28).
# You MUST pass K1_UPPER_BODY_POSE_INDICES explicitly.
k1_flashsac_reward = make_flashsac_reward(
    k1_22dof_loco,
    upper_body_pose_indices=K1_UPPER_BODY_POSE_INDICES,
)

# Observation: PPO-default passthrough (no changes needed for v5 recipe)
k1_flashsac_obs = make_flashsac_observation(k1_22dof_loco_single_wolinvel)
```

### Joint order caveat

The transform helper's `upper_body_pose_indices` parameter replaces the old `upper_body_start_idx` (which assumed legs always come first). Robot joint orders differ:

| Robot | Joint order | Upper body indices |
|---|---|---|
| **G1 (29-DoF)** | legs (0-11) → upper body (12-28) | `G1_UPPER_BODY_POSE_INDICES = range(12, 29)` |
| **K1 (22-DoF)** | upper body (0-9) → legs (10-21) | `K1_UPPER_BODY_POSE_INDICES = range(0, 10)` |

Using the wrong indices will boost leg joints to 150 instead of upper body — the robot won't walk.

The helper is validated by unit tests in `tests/unit/test_flashsac_transform.py` that verify:
- G1 transform output matches the hand-tuned `g1_29dof_loco_flashsac` preset exactly
- Only `alive` is dropped
- K1 transform produces 9 terms with expected weights
- K1 upper body (indices 0-9) boosted to 150, legs (10-21) unchanged
- K1 preset has K1-specific tuning (feet_phase=12.0, swing_height=0.04, etc.)
- Transform is idempotent

## K1 expansion (completed 2026-04-10)

### Implementation

- [x] FlashSAC port complete (vendored + adapter + bridge)
- [x] G1 canonical recipe established (v5, 1.44× composite)
- [x] Fixed `flashsac_transform.py` API: `upper_body_start_idx` → `upper_body_pose_indices` to support K1's upper-body-first joint order
- [x] Fixed false-positive K1 unit test (was checking leg indices 12+ instead of upper body 0-9)
- [x] Updated K1 robot model from `booster_assets`
- [x] Created `k1_22dof_loco_flashsac` reward preset with K1-specific tuning
- [x] Created `k1_22dof_flash_sac` (IsaacSim) and `k1_22dof_flash_sac_mjwarp` (MJWarp) experiment configs
- [x] Registered in base `experiment.py` and `reward.py`
- [x] Smoke test: 5-step training completes (10.62 it/s)
- [x] Full training runs completed, iterative tuning in progress

### K1 training commands

```bash
# Full training (IsaacSim)
python src/holosoma/holosoma/train_agent.py exp:k1-22dof-flash-sac

# Full training (MJWarp)
python src/holosoma/holosoma/train_agent.py exp:k1-22dof-flash-sac-mjwarp

# Smoke test (5 steps)
bash scripts/run_flashsac_k1_holosoma_smoke.sh
```

### K1 tuning history

| # | Run | Change | Observation |
|---|---|---|---|
| 1 | `20260410_043117` | G1 v5 defaults (feet_phase=4.0, swing=0.09, action_rate=-0.005, tracking=2.0) | **Shuffle gait** — small rapid steps, no foot lifting. actor/loss=-2.24, temp→0.0004. |
| 2 | `20260410_054053` | feet_phase 4→7, swing 0.09→0.065, action_rate→-0.001 | Marginal improvement, still shuffling. actor/loss=-3.04. |
| 3 | `20260410_065007` | feet_phase 7→12, swing 0.065→0.04, tracking 2.0→**1.0** | **Worse** — marching in place. Halved tracking killed forward drive. |
| 4 | `20260410_072839` | Restore tracking→2.0, feet_phase 12→10, keep swing 0.04 | **Still no walk.** Temperature=0.0004 in ALL v1-v4 runs. |
| 5 | `20260410_080903` | **Algorithm fix:** sigma 0.15→0.25, reward=v2 | Back to shuffle. sigma fix helped entropy (+0.49) but shuffle persists. |
| 6 | `20260410_084759` | Upstream-aligned minimal reward (5 terms), sigma=0.25 | Still shuffle. entropy=+0.49, temp=0.0011. Reward is NOT the problem. |
| 7 | (pending) | **Physics fix:** leg Kp 200→80, Kd 5→2.5 + upstream reward + sigma=0.25 | Root cause: Kp=200 saturates effort_limit=45Nm at ±0.225rad. Robot physically cannot stride. |

### Why K1 shuffles (FlashSAC-specific failure mode)

FlashSAC's `temp_target_sigma=0.15` causes **early temperature collapse** (α → 0.0004 within ~3k steps). The policy locks into the first local optimum it discovers.

For K1, the easiest early optimum is **shuffle**: rapid small foot vibrations track `tracking_lin_vel` somewhat without risking termination. PPO avoids this because its high-entropy policy (`init_noise_std=0.8`) keeps exploring and eventually discovers proper gait. FlashSAC's narrow policy cannot.

K1 is especially vulnerable because it lacks waist DOFs (0 vs G1's 3). G1 can use torso rotation for stride length; K1 must rely entirely on leg swing, making proper gait harder to discover.

**Strategy**: Make foot-lifting the dominant reward from step 1, so the first local optimum the policy finds IS proper gait:

| Parameter | G1 v5 | K1 v5 | Rationale |
|---|---|---|---|
| `temp_target_sigma` | 0.15 | **0.25** | K1 needs wider exploration (no waist DOFs). Prevents temperature collapse. |
| `feet_phase` weight | 4.0 | **7.0** | Stronger gait enforcement for K1. |
| `swing_height` | 0.09 | **0.065** | K1 shorter legs. |
| `penalty_action_rate` | -0.005 | **-0.001** | 5× weaker. Allow large joint movements. |
| `tracking_lin_vel` | 2.0 | **2.0** | Keep full weight (halving killed forward drive in v3). |

### K1 vs G1 morphology differences

| Aspect | G1 (29-DoF) | K1 (22-DoF) | Impact on FlashSAC |
|---|---|---|---|
| Waist DOFs | 3 (yaw, roll, pitch) | **0** | K1 can't use torso rotation → shuffle is easier than stepping |
| Head DOFs | 0 | 2 (yaw, pitch) | Head pose_weights boosted to 150 (keep head stable) |
| Arm DOFs per arm | 7 (incl. wrist) | 4 (no wrist) | Less upper body inertia for K1 |
| Joint order | legs first (0-11) | **upper body first (0-9)** | Must use `K1_UPPER_BODY_POSE_INDICES` |
| action_scale | 0.25 | 0.25 | Same → FlashSACGymBridge multiplier=2.0 identical |
| target_entropy | -13.86 | -10.51 | K1 collapses faster (fewer DOFs) |

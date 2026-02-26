# K1 Agile Gait Improvement Work Log

This document records the K1 agile gait improvement work completed on February 24, 2026,
with follow-up updates through February 26, 2026.
It summarizes the observed problem, root-cause analysis, code changes, and recommended
training/deployment flow.

## Scope

- Robot: Booster K1 (`k1_22dof`)
- Task: locomotion policy (`exp:k1-22dof-agile`)
- Goal:
  - reach higher stable walking speed in real robot deployment
  - improve robustness under high joystick commands
  - improve behavior under simultaneous translation + yaw commands (soccer locomotion use case)

## Observed Symptoms

- In deployment, full left-stick input did not produce expected high speed.
- Full stick input caused unstable gait and heading drift in real robot tests.
- Around half-stick input, sim-to-real behavior was significantly more stable.

## Root Cause Summary

1. Inference command-limit mismatch:
   - Agile training command ranges were much higher than default joystick limits used in booster inference.
   - Result: command distribution mismatch between training and deployment.
2. High-speed reward tradeoff:
   - Strong stability penalties and narrow tracking sigma made high-speed tracking harder to optimize.
3. Gait cadence limitation:
   - Fixed cadence caused limited pitch growth at high command speeds.
4. Limited emphasis on coupled high-speed translation + turning:
   - Existing command sampling covered independent axes, but did not explicitly bias hard coupled cases.
5. One-sided foot-width constraint:
   - Existing reward penalized only feet-too-close cases, so wider lateral stance was not discouraged.
6. Disturbance reward leakage:
   - If an external push happened in the same direction as command, tracking velocity reward could rise
     even when the motion was disturbance-driven.

## Implemented Changes

## A) Inference: align joystick limits with policy metadata

- Added optional task config overrides:
  - `joystick_max_vx`
  - `joystick_max_vy`
  - `joystick_max_vyaw`
  - `joystick_deadzone`
- Added ONNX metadata parsing for `command_ranges` in policy loader.
- Added runtime joystick limit configuration in `InterfaceWrapper`.
- Automatically apply joystick limits from:
  1. CLI override (if provided)
  2. otherwise ONNX `command_ranges` metadata

Main files:

- `src/holosoma_inference/holosoma_inference/config/config_types/task.py`
- `src/holosoma_inference/holosoma_inference/policies/base.py`
- `src/holosoma_inference/holosoma_inference/sdk/interface_wrapper.py`

## B) Training command generation: improve high-speed coupled maneuvers

- Extended `LocomotionCommand` to support optional combined-motion sampling parameters:
  - `combined_motion_prob`
  - `combined_lin_vel_x_range`
  - `combined_lin_vel_y_range`
  - `combined_ang_vel_yaw_abs_range`
- Added K1 agile preset values to bias samples toward fast forward + turning commands.

Main files:

- `src/holosoma/holosoma/managers/command/terms/locomotion.py`
- `src/holosoma/holosoma/config_values/loco/k1/command.py`

## C) High-constant gait pitch

- Added gait-frequency controls in `LocomotionGait` and configured agile cadence to be high and mostly constant.
- Current agile target is around `2.17 Hz` gait cycles (`gait_period: 0.46`), with narrow randomization only.
- Speed-coupled frequency gains are set to `0.0` to avoid pitch changes tied to command magnitude.

Main file:

- `src/holosoma/holosoma/config_values/loco/k1/command.py`

## D) Reward shaping for stride-dominant speed gain

- Increased base `swing_height` and disabled speed-coupled swing-height scaling to keep clearance margin
  under high fixed pitch.
- Added a new stride reward:
  - `stride_pitch_coupling` now uses phase-locked foot-x targets in base frame.
  - This enforces actual leg cycling (front foot must come back, rear foot must come forward), while
    desired stride magnitude still increases with command speed.
- Added `penalty_far_feet_xy`:
  - discourages excessive lateral leg spread while preserving existing close-feet protection.
- Added optional push-velocity compensation in `tracking_lin_vel`:
  - subtracts short-horizon decayed push contribution before computing velocity-tracking error.
  - avoids accidental positive reward from same-direction pushes.
- Added `penalty_head_ang_vel_xy`:
  - penalizes head-link pitch/roll angular speed (with deadzone) to reduce camera blur while walking.
- FastSAC anti-stall adjustments:
  - increased `tracking_lin_vel` weight
  - reduced `alive` and `penalty_action_rate` dominance
  - added `penalty_stall_when_commanded`
  - gated `stride_pitch_coupling` by actual translational speed ratio (prevents in-place stepping from scoring)
- Curriculum settings were left unchanged.

Main files:

- `src/holosoma/holosoma/config_values/loco/k1/reward.py`
- `src/holosoma/holosoma/managers/reward/terms/locomotion.py`

## Training Command

PPO agile retraining:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:k1-22dof-agile \
  simulator:isaacgym \
  logger:wandb \
  --training.seed 1
```

FastSAC agile retraining:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:k1-22dof-agile-fast-sac \
  simulator:isaacgym \
  logger:wandb \
  --training.seed 1
```

## Deployment Notes

- For inference, if ONNX metadata contains `command_ranges`, joystick limits are now auto-aligned.
- You can still explicitly override limits from CLI if needed:

```bash
--task.joystick-max-vx 2.5 \
--task.joystick-max-vy 1.0 \
--task.joystick-max-vyaw 1.5 \
--task.joystick-deadzone 0.10
```

## Validation Checklist

1. Confirm ONNX metadata includes `command_ranges`.
2. Confirm startup log prints updated joystick limits.
3. Run staged real-robot tests:
   - low speed straight walk
   - high speed straight walk
   - coupled high speed forward + yaw commands
4. Log and compare:
   - commanded velocity vs measured base velocity
   - heading error under coupled commands
   - stumble/fall events

## Known Risks

- If adaptive gait-frequency gains are too large, cadence can rise faster than actuator capability.
- If stride coupling weight is too high, the policy can over-prioritize step geometry over disturbance recovery.
- Real hardware limits (actuator thermal, friction variation) can still bound top stable speed.

## Follow-up Update (2026-02-26)

### 1) Asset sync for K1 reproducibility

- Imported the latest K1 model assets directly into holosoma:
  - `src/holosoma/holosoma/data/robots/k1/k1_22dof.urdf`
  - `src/holosoma/holosoma/data/robots/k1/k1_22dof.xml`
  - latest `meshes/*`
- Purpose: keep train/eval/deploy behavior consistent without depending on external asset repos at runtime.

### 2) Eval video workflow for gait checks

- Updated eval workflow so headless gait replay reliably records videos in short step-limited runs:
  - force eval video interval to `1`
  - explicit recorder flush on eval shutdown
- Added W&B media-oriented eval logging mode to reduce non-essential uploads while keeping rollout videos.

### 3) Current status summary

- Agile reward/command shaping updates are in place (stride target, fixed higher pitch, anti-stall terms, push-compensated tracking).
- FastSAC can still fall into undesirable local minima depending on run seed/settings.
- Recommended operational flow:
  1. validate gait shape in sim with short eval videos (headless)
  2. select stable checkpoint by commanded-vs-actual speed and heading stability
  3. then run staged real-robot speed ramp tests

# Ball Play (Kick & Dribble) Design Spec

## Overview

Two independent RL environments for the K1 humanoid robot that extend the existing locomotion policy to ball interaction tasks:

1. **BallKickTask** - approach ball, align to kick direction, kick toward target
2. **BallDribbleTask** - track and follow ball, push it in a commanded direction

Both build on `LeggedRobotLocomotionManager` and run on Isaac Sim with rigid body ball physics.

## Motivation

Based on the "RL Ball Playing" report analysis: ball play is not a standalone kick motion but a continuous task combining locomotion, positioning, alignment, and contact. The key insight is to extend a command-conditioned locomotion policy to a task-conditioned one, not to train from scratch.

---

## Architecture

### Class Hierarchy

```
BaseTask
  └── LeggedRobotLocomotionManager
        ├── BallKickTask
        └── BallDribbleTask
```

Both inherit locomotion fundamentals (gait phase, joint control, terrain handling) and add ball actor management, ball-specific buffers, and reset logic.

### Directory Structure

```
src/holosoma/holosoma/
├── envs/ball_play/
│   ├── __init__.py
│   ├── ball_kick_task.py
│   └── ball_dribble_task.py
├── managers/
│   ├── reward/terms/ball_play.py
│   ├── observation/terms/ball_play.py
│   ├── command/terms/ball_play.py
│   └── termination/terms/ball_play.py
├── config_values/ball_play/k1/
│   ├── reward_kick.py
│   ├── reward_dribble.py
│   ├── observation_kick.py
│   ├── observation_dribble.py
│   ├── command_kick.py
│   ├── command_dribble.py
│   ├── termination_kick.py
│   └── termination_dribble.py
└── data/robots/ball/
    └── ball.urdf
```

---

## Robot & Ball Assets

### Robot

K1 22-DOF from booster_assets (`/workspace/booster_assets/robots/K1/K1_22dof.urdf`). Full body: head (2), arms (8), legs (12).

### Ball

Copied from htwk-gym (`/workspace/htwk-gym/resources/T1/ball.urdf`):
- Radius: 0.075 m
- Mass: 0.2 kg
- Inertia: 0.00045 (all axes)
- Simple sphere collision

### Ball Spawning (Isaac Sim)

Uses existing `RigidObject` mechanism in `isaacsim.py` `_setup_scene` (lines 398-433). Set `robot_config.object.object_urdf_path` to the ball URDF path. Ball spawns at `/World/envs/env_.*/Object`.

Ball physics properties (friction, restitution) are set via `sim_utils.RigidBodyMaterialCfg` or shape property overrides.

Ball state access via unified API: `simulator.get_actor_states(["object"], env_ids)` returns `[num_envs, 13]` (pos[3] + quat[4] + lin_vel[3] + ang_vel[3]).

---

## Environment Classes

### BallKickTask

```python
class BallKickTask(LeggedRobotLocomotionManager):
    """Locomotion + ball approach + directional kick."""

    def _get_task_name(self) -> str:
        return "ball_kick"

    def _init_buffers(self):
        super()._init_buffers()
        self.ball_pos = torch.zeros(num_envs, 3)          # world frame
        self.ball_vel = torch.zeros(num_envs, 3)           # world frame
        self.ball_pos_relative = torch.zeros(num_envs, 3)  # robot frame
        self.last_ball_vel = torch.zeros(num_envs, 3)
        self.kick_target_dir = torch.zeros(num_envs, 2)    # (cos, sin)
        self.time_ball_still = torch.zeros(num_envs)
        self.time_ball_moving = torch.zeros(num_envs)

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        # Fetch ball state from simulator
        # Transform ball position to robot local frame
        # Update ball activity timers

    def _reset_robot_states_callback(self, env_ids):
        super()._reset_robot_states_callback(env_ids)
        # Reset ball in front of robot with randomized offset
        # Resample kick target direction

    def _reset_buffers_callback(self, env_ids):
        super()._reset_buffers_callback(env_ids)
        # Reset ball timers
```

### BallDribbleTask

```python
class BallDribbleTask(LeggedRobotLocomotionManager):
    """Locomotion + ball tracking + directional dribbling."""

    def _get_task_name(self) -> str:
        return "ball_dribble"

    def _init_buffers(self):
        super()._init_buffers()
        self.ball_pos = torch.zeros(num_envs, 3)
        self.ball_vel = torch.zeros(num_envs, 3)
        self.ball_pos_relative = torch.zeros(num_envs, 3)
        self.perceived_ball_pos_relative = torch.zeros(num_envs, 2)  # delayed detection
        self.last_perceived_ball_pos_relative = torch.zeros(num_envs, 2)
        self.ball_detection_timer = torch.zeros(num_envs)
        self.dribble_target_dir = torch.zeros(num_envs, 2)  # world frame

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        # Simulate ~30 FPS ball detection with jitter
        # Reset ball only (not robot) if ball too far

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        # Fetch ball state, transform to robot frame
```

### Ball Reset Logic (shared pattern)

```python
def _reset_ball_at_robot_front(self, env_ids):
    robot_pos = self.simulator.robot_root_states[env_ids, :3]
    robot_quat = self.simulator.robot_root_states[env_ids, 3:7]

    # Compute forward direction from quaternion
    forward = quat_rotate(robot_quat, forward_vec)

    # Random offset in front of robot
    offset_x = torch_rand_float(x_min, x_max, ...)
    offset_y = torch_rand_float(y_min, y_max, ...)

    ball_pos = robot_pos[:, :2] + forward[:, :2] * offset
    ball_z = ball_radius  # on ground

    # Write via unified API
    new_states[:, :3] = [ball_x, ball_y, ball_z]
    new_states[:, 3:7] = identity_quat
    new_states[:, 7:13] = 0  # zero velocity
    simulator.set_actor_states(["object"], env_ids, new_states)
```

---

## Observations

### Kick Task Actor Observations (78 dims)

| Term | Dims | Source |
|------|------|--------|
| projected_gravity | 3 | locomotion.py |
| base_ang_vel | 3 | locomotion.py |
| ball_pos_relative_xy | 2 | ball_play.py (new) |
| kick_target_dir | 2 | ball_play.py (new) |
| sin_phase | 1 | locomotion.py |
| cos_phase | 1 | locomotion.py |
| dof_pos | 22 | locomotion.py |
| dof_vel | 22 | locomotion.py |
| actions | 22 | locomotion.py |

### Dribble Task Actor Observations (80 dims)

| Term | Dims | Source |
|------|------|--------|
| projected_gravity | 3 | locomotion.py |
| base_ang_vel | 3 | locomotion.py |
| dribble_target_dir_local | 2 | ball_play.py (new) |
| ball_pos_relative_perceived_xy | 2 | ball_play.py (new) |
| last_ball_pos_relative_perceived_xy | 2 | ball_play.py (new) |
| sin_phase | 1 | locomotion.py |
| cos_phase | 1 | locomotion.py |
| dof_pos | 22 | locomotion.py |
| dof_vel | 22 | locomotion.py |
| actions | 22 | locomotion.py |

### Critic Observations (additional to actor)

| Term | Dims | Description |
|------|------|-------------|
| base_lin_vel | 3 | Privileged: true base velocity |
| ball_vel_world_xy | 2 | Privileged: ball velocity |
| base_height | 1 | Privileged: terrain-relative height |

### New Observation Functions (`managers/observation/terms/ball_play.py`)

```python
def ball_pos_relative(env) -> Tensor:       # [num_envs, 2]
def ball_pos_relative_perceived(env) -> Tensor:  # [num_envs, 2]
def last_ball_pos_relative_perceived(env) -> Tensor:  # [num_envs, 2]
def kick_target_direction(env) -> Tensor:   # [num_envs, 2]
def dribble_target_direction_local(env) -> Tensor:  # [num_envs, 2]
def ball_vel_world(env) -> Tensor:          # [num_envs, 2]
```

All ball positions are transformed to robot local frame via `quat_rotate_inverse(base_quat, ball_pos_world - robot_pos_world)`.

---

## Rewards

### Kick Task Rewards (`managers/reward/terms/ball_play.py`)

**Task rewards:**

| Name | Weight | Shape | Description |
|------|--------|-------|-------------|
| ball_velocity_target_direction | +10.0 | linear * exp_decay | Ball velocity projected onto target direction, with time decay after kick |
| ball_acceleration_toward_target | +0.25 | tanh | Ball acceleration in target direction |
| kicking_foot_approach_ball | +10.0 | exp(-d/sigma) | Foot-ball proximity, active only when ball is stationary (speed < 0.1 m/s) |
| body_alignment_for_kick | +1.0 | exp | Alignment: robot_forward dot (ball_to_target_normalized) |
| waiting_penalty | -1.0 | quadratic | (t / t_max)^2 penalty to encourage early kicks |

**Locomotion rewards (inherited/configured):**

| Name | Weight | Description |
|------|--------|-------------|
| survival | +0.25 | Alive bonus |
| base_height | -200.0 | Height deviation penalty |
| penalty_orientation | -20.0 | Non-flat orientation penalty |
| penalty_action_rate | -1.5 | Action smoothness |
| penalty_ang_vel_xy | -0.1 | Roll/pitch rate penalty |
| feet_phase | +1.0 | Gait phase tracking |

### Dribble Task Rewards

**Task rewards:**

| Name | Weight | Shape | Description |
|------|--------|-------|-------------|
| ball_velocity_tracking | +2.0 | cos_sim * exp(-speed_err/sigma) | Direction alignment (cosine similarity) times speed matching |
| ball_distance_penalty | -0.25 | exp(d/sigma) - 1 | Distance from robot to optimal dribble position (behind ball) |
| look_at_ball | +0.5 | exp(-angle_err^2/sigma) | Yaw alignment toward ball |

**Locomotion rewards (inherited/configured):**

| Name | Weight | Description |
|------|--------|-------------|
| survival | +0.25 | Alive bonus |
| base_height | -20.0 | Height deviation penalty |
| penalty_action_rate | -1.5 | Action smoothness |
| feet_phase | +2.0 | Gait phase tracking (stronger for stable dribble walking) |
| penalty_feet_slip | -0.1 | Foot slip penalty |

### Reward Function Signatures

```python
def ball_velocity_target_direction(env, decay_time: float = 0.1,
                                    max_reward: float = 10.0) -> Tensor

def ball_acceleration_toward_target(env, scale: float = 80.0,
                                     max_reward: float = 80.0) -> Tensor

def kicking_foot_approach_ball(env, proximity_sigma: float = 0.1,
                                stationary_threshold: float = 0.1,
                                max_reward: float = 50.0) -> Tensor

def body_alignment_for_kick(env, sigma: float = 0.5,
                             max_reward: float = 1.0) -> Tensor

def waiting_penalty(env, max_still_time: float = 2.0) -> Tensor

def ball_velocity_tracking(env, sigma: float = 1.0,
                            min_speed: float = 0.1) -> Tensor

def ball_distance_penalty(env, sigma: float = 1.0,
                           max_dist: float = 3.0) -> Tensor

def look_at_ball(env, sigma: float = 0.5) -> Tensor
```

---

## Commands

### Kick Task: KickTargetCommand

No locomotion velocity commands. The kick target direction is sampled at reset and held for the episode.

```python
class KickTargetCommand(CommandTermBase):
    def setup(self):
        self.kick_target_dir = torch.zeros(num_envs, 2)  # (cos, sin)

    def reset(self, env_ids):
        angles = torch_rand_float(-pi, pi, (len(env_ids), 1))
        self.kick_target_dir[env_ids, 0] = torch.cos(angles.squeeze())
        self.kick_target_dir[env_ids, 1] = torch.sin(angles.squeeze())

    def step(self):
        pass  # No resampling within episode
```

### Dribble Task: DribbleCommand

Target dribble velocity vector in world frame. Resampled every 3-8 seconds.

```python
class DribbleCommand(CommandTermBase):
    def setup(self):
        self.commands = torch.zeros(num_envs, 2)  # (vx, vy) world frame

    def reset(self, env_ids):
        angles = torch_rand_float(-pi, pi, (len(env_ids), 1))
        speeds = torch_rand_float(0.1, 3.0, (len(env_ids), 1))
        self.commands[env_ids, 0] = (torch.cos(angles) * speeds).squeeze()
        self.commands[env_ids, 1] = (torch.sin(angles) * speeds).squeeze()

    def step(self):
        # Resample at random intervals (3-8 seconds)
```

---

## Termination Conditions

### Kick Task

| Condition | Type | Threshold |
|-----------|------|-----------|
| Contact force exceeded | terminate | Trunk/hip/shank contacts > 1.0 N |
| Root velocity exceeded | terminate | velocity^2 sum > 50.0 |
| Base height too low | terminate | height < 0.45 m |
| Episode timeout | timeout | 7.0 seconds |
| Ball still too long | terminate | 2.0 seconds |
| Ball moving too long | terminate | 5.0 seconds |

### Dribble Task

| Condition | Type | Threshold |
|-----------|------|-----------|
| Contact force exceeded | terminate | Same as kick |
| Root velocity exceeded | terminate | velocity^2 sum > 70.0 |
| Base height too low | terminate | height < 0.35 m |
| Episode timeout | timeout | 30.0 seconds |
| Ball too far | ball_reset_only | distance > 3.0 m (reset ball, not robot) |

New termination functions in `managers/termination/terms/ball_play.py`:

```python
def ball_still_too_long(env, max_still_time: float = 2.0) -> Tensor
def ball_moving_too_long(env, max_moving_time: float = 5.0) -> Tensor
def ball_too_far(env, max_distance: float = 3.0) -> Tensor  # returns mask for ball-only reset
```

---

## Curriculum

Progressive difficulty via existing `CurriculumManager`:

### Kick Task Curriculum

| Stage | Ball Distance | Ball Init Velocity | Trigger |
|-------|--------------|-------------------|---------|
| 1 (initial) | 0.25-0.4 m | 0 m/s | Start |
| 2 (medium) | 0.3-1.5 m | 0 m/s | avg_episode > 3.0s |
| 3 (hard) | 0.5-3.0 m | 0-2.0 m/s | avg_episode > 4.5s |

### Dribble Task Curriculum

| Stage | Ball Distance | Ball Init Velocity | Trigger |
|-------|--------------|-------------------|---------|
| 1 (initial) | 0.3-1.0 m | 0 m/s | Start |
| 2 (medium) | 0.5-2.0 m | 0-0.5 m/s | avg_episode > 10s |
| 3 (hard) | 1.0-3.0 m | 0-1.0 m/s | avg_episode > 15s |

Implementation: A new `BallCurriculum` term that adjusts ball spawn distance range and initial velocity range based on average episode length thresholds.

---

## Config Values

All config presets use holosoma's `RewardManagerCfg`, `ObservationManagerCfg`, etc. pattern:

```python
# Example: config_values/ball_play/k1/reward_kick.py
k1_22dof_ball_kick = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "ball_velocity_target_direction": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:ball_velocity_target_direction",
            weight=10.0,
            params={"decay_time": 0.1, "max_reward": 10.0},
        ),
        "kicking_foot_approach_ball": RewardTermCfg(
            func="holosoma.managers.reward.terms.ball_play:kicking_foot_approach_ball",
            weight=10.0,
            params={"proximity_sigma": 0.1, "stationary_threshold": 0.1},
        ),
        # ... locomotion rewards ...
    }
)
```

---

## Ball Physics Properties

### Kick Task (from htwk-gym Kicking_K1.yaml)

| Property | Value |
|----------|-------|
| Restitution | 0.0 |
| Friction | 1.0 |
| Rolling friction | 1.0 |
| Density | 0.001 |

### Dribble Task (from htwk-gym Dribble_K1.yaml)

| Property | Value |
|----------|-------|
| Restitution | 0.1 |
| Friction | 1.0 |
| Rolling friction | 0.3 |
| Torsion friction | 0.1 |
| Density | 200 |

These are set via Isaac Sim's `RigidBodyMaterialCfg` and rigid body property overrides in the environment setup.

---

## Noise & Normalization

### Observation Noise (both tasks)

| Term | Range | Distribution |
|------|-------|-------------|
| gravity | [0, 0.01] | gaussian |
| ang_vel | [0, 0.1] | gaussian |
| dof_pos | [0, 0.01] | gaussian |
| dof_vel | [0, 0.1] | gaussian |
| ball_pos (kick) | [-0.01, 0.01] | gaussian |
| ball_pos (dribble) | [-0.03, 0.03] | gaussian |

### Normalization

| Term | Kick | Dribble |
|------|------|---------|
| ball_pos | 1.0 | 0.333 |
| dof_vel | 0.1 | 0.1 |
| ball_vel | - | 1.0 |

---

## Files to Create

| File | Type | Description |
|------|------|-------------|
| `data/robots/ball/ball.urdf` | new | Ball asset (from htwk-gym) |
| `envs/ball_play/__init__.py` | new | Package init |
| `envs/ball_play/ball_kick_task.py` | new | Kick environment class |
| `envs/ball_play/ball_dribble_task.py` | new | Dribble environment class |
| `managers/reward/terms/ball_play.py` | new | 8 reward functions |
| `managers/observation/terms/ball_play.py` | new | 6 observation functions |
| `managers/command/terms/ball_play.py` | new | 2 command classes |
| `managers/termination/terms/ball_play.py` | new | 3 termination functions |
| `config_values/ball_play/__init__.py` | new | Package init |
| `config_values/ball_play/k1/__init__.py` | new | Package init |
| `config_values/ball_play/k1/reward_kick.py` | new | Kick reward config |
| `config_values/ball_play/k1/reward_dribble.py` | new | Dribble reward config |
| `config_values/ball_play/k1/observation_kick.py` | new | Kick observation config |
| `config_values/ball_play/k1/observation_dribble.py` | new | Dribble observation config |
| `config_values/ball_play/k1/command_kick.py` | new | Kick command config |
| `config_values/ball_play/k1/command_dribble.py` | new | Dribble command config |
| `config_values/ball_play/k1/termination_kick.py` | new | Kick termination config |
| `config_values/ball_play/k1/termination_dribble.py` | new | Dribble termination config |

## Files to Modify (minimal)

No core framework modifications required. Ball spawning uses existing `robot_config.object.object_urdf_path` mechanism in `isaacsim.py`.

# FlashSAC Sigma / Exploration / G1 Comparison Report

Updated: 2026-04-10

## Scope

This report answers two concrete questions against the current local codebase:

1. How `temp_target_sigma` propagates through FlashSAC as
   `sigma -> target_entropy -> alpha -> learned std`.
2. Why upstream FlashSAC can still learn locomotion even though it tends to run
   a relatively low-entropy policy, and how that upstream regime differs from
   holosoma's G1 setup in action scale and reward shaping.

Primary sources used in this report:

- Upstream FlashSAC repo at `/workspace/FlashSAC`
- Holosoma repo at `/workspace/holosoma`
- Installed IsaacLab task sources in the active `hssim` conda env

## Executive Summary

- The claim "FlashSAC explores only a little" is directionally true but easy to overstate.
  Upstream FlashSAC does not rely on a large long-lived entropy bonus like PPO-style wide policies.
  However, it still explores through four channels: random replay warmup, learned actor std,
  temporally correlated noise repetition, and massive parallel data collection.
- `temp_target_sigma` is not the direct rollout noise multiplier.
  It is converted into a target entropy, which trains the temperature parameter `alpha`.
  `alpha` then influences how much the actor is rewarded for staying stochastic.
- In upstream FlashSAC locomotion, even a fairly small target sigma can work because the task is dense,
  the reward landscape is simple, the stock IsaacLab action scale is generous (`scale=0.5`),
  and training runs at 1024 parallel environments with a 100k-sample random warmup.
- In holosoma, FlashSAC itself is basically upstream. The problem is the environment regime.
  Holosoma's native locomotion reward and robot action scale were not originally tuned for
  FlashSAC's narrower policy. Current holosoma code compensates by:
  - restoring IsaacLab-equivalent action authority through a uniform bridge multiplier
  - using a dedicated FlashSAC reward preset with weaker shaping terms
- There is documentation drift in the repo.
  The code is currently more trustworthy than some comments and older docs snapshots.

## Current Codebase Status

### Core FlashSAC in holosoma is still upstream FlashSAC

The vendored agent/config under `src/holosoma/holosoma/_vendored/flash_rl/` matches upstream behaviorally.
The local diff audit showed only namespace import rewrites in the vendored agent file, and the vendored
`configs/agent/flashSAC.yaml` is byte-identical to upstream.

Holosoma's active default FlashSAC preset explicitly says it mirrors upstream FlashSAC defaults,
including `temp_target_sigma=0.15`, `actor_noise_zeta_mu=2.0`, `actor_noise_zeta_max=16`,
`n_step=3`, and `updates_per_interaction_step=2.0`.
See [config_values/algo.py](/workspace/holosoma/src/holosoma/holosoma/config_values/algo.py#L246).

### There is doc drift inside holosoma

- `docs/flashsac_port.md` still says the "latest configuration" is the old Option A setup using the
  PPO-default reward (`g1_29dof_loco`). See [flashsac_port.md](/workspace/holosoma/docs/flashsac_port.md#L5).
- The current experiment config does **not** do that. It uses `reward.g1_29dof_loco_flashsac`.
  See [g1/experiment.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/experiment.py#L60).
- The comment in `g1/experiment.py` also says the FlashSAC reward preset strips out terms like
  `feet_phase` and `pose`, but the actual reward preset currently keeps them at reduced weights.
  Compare [g1/experiment.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/experiment.py#L74)
  with [g1/reward.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/reward.py#L241).

For the rest of this report, the code is treated as the source of truth.

## Part 1: Sigma -> Target Entropy -> Alpha -> Learned Std

### 1. Upstream definition

FlashSAC's docs define a unified entropy target through a fixed target action std `sigma_tgt`.
See [docs/index.html](/workspace/FlashSAC/docs/index.html#L375).

The implementation computes:

```text
target_entropy = 0.5 * action_dim * log(2 * pi * e * temp_target_sigma^2)
```

See [agent.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/agent.py#L355).

The default upstream config is:

```yaml
temp_initial_value: 0.01
temp_target_sigma: 0.15
```

See [flashSAC.yaml](/workspace/FlashSAC/configs/agent/flashSAC.yaml#L44).

### 2. What `alpha` is

FlashSAC parameterizes temperature as `alpha = exp(log_temp)`.
See [network.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/network.py#L108).

So `temp_target_sigma` does **not** directly set the actor's runtime action noise.
It only sets the entropy target that trains `alpha`.

### 3. Where `alpha` enters learning

#### Actor loss

The actor update uses:

```text
actor_loss = E[ alpha * log_prob(a|s) - Q(s,a) ]
```

See [update.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/update.py#L125).

Because `log_prob` is typically negative, a larger `alpha` puts more weight on keeping the policy stochastic.
A smaller `alpha` lets the Q term dominate, which usually collapses the actor toward a narrower policy.

#### Temperature update

The temperature loss is:

```text
temperature_loss = alpha * (entropy - target_entropy)
```

See [update.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/update.py#L303).

This means:

- If current entropy is above target entropy, gradient descent pushes `alpha` down.
- If current entropy is below target entropy, gradient descent pushes `alpha` up.

For humanoid action dimensions, `sigma=0.15` implies a strongly negative target entropy,
so the optimization typically drives `alpha` downward quickly.

#### Critic target

`alpha` also enters the critic target through the entropy bonus on next actions:

```text
next_actor_entropy = alpha * log_prob(next_action | next_state)
```

See [update.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/update.py#L211).

So the effect chain is:

```text
temp_target_sigma
  -> target_entropy
  -> learned alpha
  -> actor entropy pressure and critic entropy bonus
  -> learned policy std
```

### 4. `sigma` is not the same as rollout-time `std * noise`

At training time, FlashSAC samples rollout actions as:

```text
action = tanh(mean + std * noise * temperature)
```

See [agent.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/agent.py#L223).

But the rollout helper hard-codes:

- `temperature = 1.0` during training
- `temperature = 0.0` during evaluation

See [agent.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/agent.py#L424).

So:

- the training-time action perturbation comes from the actor's own learned `std`
- `temp_target_sigma` only affects that `std` indirectly through the entropy objective
- `alpha` is not used as a direct multiplier on rollout noise in `sample_actions()`

### 5. Numerical examples

Derived from the upstream formula above:

| Action dim | sigma=0.05 | sigma=0.15 | sigma=0.25 |
|---|---:|---:|---:|
| 12 | -18.92 | -5.74 | 0.39 |
| 22 | -34.69 | -10.52 | 0.72 |
| 29 | -45.73 | -13.87 | 0.95 |

Interpretation:

- `sigma=0.15` for G1 29-DoF gives a target entropy of about `-13.87`.
- That is low enough that FlashSAC tends to become narrow early.
- Raising sigma to `0.25` fundamentally changes the regime because the target entropy becomes near zero or positive.

This matches holosoma's own K1 notes, which already describe `0.15` as causing temperature collapse and `0.25`
as a meaningful widening move.

### 6. Noise repetition matters more than the name suggests

Upstream docs explicitly present noise repetition as one of the two exploration mechanisms.
See [docs/index.html](/workspace/FlashSAC/docs/index.html#L378).

The default settings are:

- `actor_noise_zeta_mu = 2.0`
- `actor_noise_zeta_max = 16`

See [flashSAC.yaml](/workspace/FlashSAC/configs/agent/flashSAC.yaml#L30).

Derived from those defaults, the expected noise repeat length is about `2.13` steps.
The distribution is front-loaded but not trivial:

- `P(k=1) ~= 0.631`
- `P(k=2) ~= 0.158`
- `P(k=3) ~= 0.070`
- tail extends to `k=16`

So FlashSAC's exploration is not just per-step Gaussian dithering.
It deliberately creates short temporally correlated action bursts.

## Part 2: Why Upstream FlashSAC Still Learns Locomotion

### 1. Upstream training scale is large

The upstream IsaacLab training script runs locomotion with:

- `num_env_steps = 50,000,896`
- `num_train_envs = 1024`
- `updates_per_interaction_step = 2`
- `buffer_min_length = 100,000`

See [run_isaaclab.sh](/workspace/FlashSAC/scripts/run_isaaclab.sh#L25) and
[flashSAC.yaml](/workspace/FlashSAC/configs/agent/flashSAC.yaml#L10).

This has two important consequences.

First, there is a non-trivial pure-random warmup.
FlashSAC uses random actions until the replay buffer can sample.
See [train.py](/workspace/FlashSAC/train.py#L123) and [agent.py](/workspace/FlashSAC/flash_rl/agents/flashSAC/agent.py#L472).

With `buffer_min_length=100000` and `1024` train envs, that is about:

```text
100000 / 1024 ~= 97.7
```

or roughly the first 98 interaction steps worth of fully random exploration.

Second, the dataset is broad even when per-env exploration is modest.
1024 environments with randomized resets, commands, and dynamics can generate a lot of coverage.

### 2. The stock IsaacLab locomotion task is exploration-friendly

The base IsaacLab locomotion reward is simple and dense:

- tracking linear velocity
- tracking angular velocity
- moderate penalties
- feet-air-time bonus

See [velocity_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L230).

The G1 stock flat config keeps that structure and does not add terms like `alive`, `pose`, or holosoma's strong clock-based shaping.
See [flat_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py#L27).

This matters because a low-entropy policy can still learn if the main reward gradient already points toward walking.
The policy does not need extremely broad exploration to discover a useful direction.

### 3. The stock action scale is generous

Upstream IsaacLab uses:

```python
JointPositionActionCfg(..., scale=0.5, ...)
```

See [velocity_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L108).

That gives the policy enough joint authority to express a gait even with a relatively narrow action distribution.

### 4. The ablations show low sigma is not fatal

The exploration ablation CSVs in upstream FlashSAC show that locomotion still learns even with smaller target sigmas.

#### G1 rough, final avg_return at 50,000,896 env steps

| Setting | Final avg_return |
|---|---:|
| `target_std_0.05` | 37.88 |
| `no_noise_repeat` | 34.33 |
| `target_std_0.25` | 42.41 |

Sources:

- [target_std_0.05.csv](/workspace/FlashSAC/results/ablation_exploration/target_std_0.05.csv#L12)
- [no_noise_repeat.csv](/workspace/FlashSAC/results/ablation_exploration/no_noise_repeat.csv#L12)
- [target_std_0.25.csv](/workspace/FlashSAC/results/ablation_exploration/target_std_0.25.csv#L12)

#### G1 flat, final avg_return at 50,000,896 env steps

| Setting | Final avg_return |
|---|---:|
| `target_std_0.05` | 35.67 |
| `no_noise_repeat` | 32.82 |
| `target_std_0.25` | 38.36 |

Sources:

- [target_std_0.05.csv](/workspace/FlashSAC/results/ablation_exploration/target_std_0.05.csv#L34)
- [no_noise_repeat.csv](/workspace/FlashSAC/results/ablation_exploration/no_noise_repeat.csv#L34)
- [target_std_0.25.csv](/workspace/FlashSAC/results/ablation_exploration/target_std_0.25.csv#L34)

Interpretation:

- Low target sigma does reduce exploration pressure.
- But upstream locomotion still learns in that regime.
- Higher sigma helps, but the algorithm is not depending on very wide entropy alone.
- Noise repetition is useful, but removing it degrades performance rather than fully breaking locomotion.

## Part 3: Upstream IsaacLab G1 vs Holosoma G1

## 3.1 Action scale comparison

| Regime | Effective joint-position action scale | Source |
|---|---:|---|
| Upstream IsaacLab stock G1 | `0.5` | [velocity_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L112) |
| Holosoma G1 robot default | `0.25` | [robot.py](/workspace/holosoma/src/holosoma/holosoma/config_values/robot.py#L524) |
| Holosoma FlashSAC bridge target | `0.5` via uniform multiplier `0.5 / 0.25 = 2.0` | [flash_sac_env_bridge.py](/workspace/holosoma/src/holosoma/holosoma/agents/flash_sac/flash_sac_env_bridge.py#L116) |

This is a critical porting point.

Without the bridge multiplier, holosoma's base G1 locomotion setup gives FlashSAC only half the joint authority that upstream IsaacLab tuning assumed.
That alone can make a narrow policy look "under-exploratory" when the real issue is insufficient action authority.

## 3.2 Reward comparison

### Upstream IsaacLab stock G1 locomotion reward

Shared base reward:

| Term | Base weight |
|---|---:|
| `track_lin_vel_xy_exp` | 1.0 |
| `track_ang_vel_z_exp` | 0.5 |
| `lin_vel_z_l2` | -2.0 |
| `ang_vel_xy_l2` | -0.05 |
| `dof_torques_l2` | -1.0e-5 |
| `dof_acc_l2` | -2.5e-7 |
| `action_rate_l2` | -0.01 |
| `feet_air_time` | 0.125 |
| `undesired_contacts` | -1.0 |
| `flat_orientation_l2` | 0.0 |

See [velocity_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L235).

G1 flat tweaks:

| Term | Weight |
|---|---:|
| `track_ang_vel_z_exp` | 1.0 |
| `lin_vel_z_l2` | -0.2 |
| `action_rate_l2` | -0.005 |
| `dof_acc_l2` | -1.0e-7 |
| `feet_air_time` | 0.75 |

See [flat_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py#L27).

G1 rough tweaks:

| Term | Weight |
|---|---:|
| `termination_penalty` | -200.0 |
| `feet_air_time` | 0.25 |
| `flat_orientation_l2` | -1.0 |
| `action_rate_l2` | -0.005 |
| `dof_acc_l2` | -1.25e-7 |
| `dof_torques_l2` | -1.5e-7 |
| `undesired_contacts` | removed |

See [rough_env_cfg.py](/root/.holosoma_deps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py#L103).

### Holosoma PPO-default G1 reward

Current holosoma PPO-default G1 reward includes strong extra shaping terms absent from upstream stock:

| Term | Weight |
|---|---:|
| `tracking_lin_vel` | 2.0 |
| `tracking_ang_vel` | 1.5 |
| `penalty_ang_vel_xy` | -1.0 |
| `penalty_orientation` | -10.0 |
| `penalty_action_rate` | -2.0 |
| `feet_phase` | 5.0 |
| `pose` | -0.5 |
| `penalty_close_feet_xy` | -10.0 |
| `penalty_feet_ori` | -5.0 |
| `alive` | 1.0 |

See [g1/reward.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/reward.py#L5).

This is exactly the kind of reward surface that a narrow policy struggles with.
Holosoma's own code comments say as much:

- `feet_phase` can be harvested in place by a near-deterministic policy
- `alive` creates a lazy attractor
- `pose` can freeze the torso

See [g1/reward.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/reward.py#L220).

### Holosoma current FlashSAC G1 reward

Current code does **not** use the PPO-default reward for FlashSAC.
It uses a dedicated translated preset:

| Term | Weight | Relation to PPO-default |
|---|---:|---|
| `tracking_lin_vel` | 2.0 | unchanged |
| `tracking_ang_vel` | 1.5 | unchanged |
| `penalty_ang_vel_xy` | -0.05 | 20x weaker |
| `penalty_orientation` | -1.0 | 10x weaker |
| `penalty_action_rate` | -0.005 | 400x weaker |
| `pose` | -0.2 | weaker outer weight, stronger upper-body per-joint weight |
| `feet_phase` | 4.0 | slightly weaker |
| `penalty_feet_ori` | -0.5 | 10x weaker |
| `penalty_close_feet_xy` | -1.0 | 10x weaker |
| `alive` | omitted | removed |

See [g1/reward.py](/workspace/holosoma/src/holosoma/holosoma/config_values/loco/g1/reward.py#L241).

So the current holosoma strategy is not "copy upstream stock exactly".
It is closer to:

1. keep upstream FlashSAC algorithm defaults
2. restore upstream action authority
3. translate holosoma's locomotion reward into a FlashSAC-safe regime

## Part 4: Why the Upstream Regime Works Better for FlashSAC

The simplest explanation is:

- upstream stock G1 reward has fewer local optima unrelated to walking
- upstream stock action scale gives the actor enough authority to realize a gait
- upstream data collection is very broad even if the policy entropy is not high for long

In contrast, holosoma PPO-default locomotion originally had both of the main failure ingredients:

- **less action authority** (`0.25` vs upstream `0.5`)
- **more non-task shaping** (`alive`, `feet_phase`, strong `pose`, strong `action_rate`, strong `orientation`)

That combination can make FlashSAC look like it has "too little exploration", but the failure is actually a system-level mismatch:

```text
narrower policy
  + weak action authority
  + reward landscape with easy standing/in-place local optima
  = no gait discovery
```

## Part 5: Practical Conclusions

### A. Is it true that FlashSAC explores only a little?

Yes, in the sense that `temp_target_sigma=0.15` creates a low target entropy for humanoid action dimensions,
which tends to drive `alpha` down and the actor toward a narrow policy.

No, in the sense that the algorithm still uses:

- replay-buffer random warmup
- learned actor std
- temporally correlated noise repetition
- large-scale parallel environment collection

So the accurate statement is:

> FlashSAC is a relatively low-entropy off-policy method, not a no-exploration method.

### B. Why can upstream FlashSAC still get locomotion?

Because upstream locomotion is trained in a regime where low-to-moderate exploration is still sufficient:

- dense velocity-tracking rewards
- stock IsaacLab action scale `0.5`
- simple reward surface
- large parallel data collection
- random warmup
- noise repetition

### C. What should holosoma preserve if it wants FlashSAC to work reliably?

Priority order:

1. Match upstream effective action scale first.
2. Keep reward local optima shallow; remove or weaken standing/in-place attractors.
3. Only then tune `temp_target_sigma`.

In other words, `sigma` is important, but it is not the first thing to fix when the action authority and reward landscape are mismatched.

## Final Bottom Line

The current codebase supports the following interpretation:

- FlashSAC's `temp_target_sigma=0.15` does make the policy comparatively narrow.
- That narrowness is real and matters.
- But upstream FlashSAC still learns locomotion because the upstream IsaacLab G1 setup is specifically friendly to a narrow policy:
  broad data collection, stock action scale `0.5`, simple dense reward, and no strong standing/in-place attractors.
- Holosoma's FlashSAC port succeeds only when it recreates those conditions as closely as possible.

For holosoma, the most important lesson is not "always raise sigma".
It is:

> First align action scale and reward landscape with the upstream regime.
> Then decide whether sigma widening is actually needed.

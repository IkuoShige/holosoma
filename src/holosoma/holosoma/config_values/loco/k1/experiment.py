from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

k1_22dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.75, "inf"]},
    ),
)

k1_22dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_fast_sac_manager"),
    algo=replace(
        algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=100000, use_symmetry=True)
    ),
    simulator=simulator.isaacgym,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum_fast_sac,
    reward=reward.k1_22dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.65, "inf"], "Episode/rew_tracking_lin_vel": [0.9, "inf"]},
    ),
)

# v36: PPO-compatible FlashSAC reward + exploration-tuned.
#
# v35 proved: PPO's raw penalties (-2.0, -10.0) kill FlashSAC even
# with exploration tuning (100% termination, temperature collapse by 40M).
# v28-v34 proved: FlashSAC-specific rewards (stride_progress, feet_air_time)
# can walk but diverge from PPO quality.
#
# v36 middle ground: PPO reward STRUCTURE (same terms, same positive
# weights) but penalties reduced to 1/10 of PPO. No FlashSAC-specific
# reward terms. This is "PPO-compatible" — same objective, different
# penalty tolerance for bounded exploration.
#
# Exploration: keep v35's working settings (sigma=0.25, zeta_mu=1.2,
# temp_initial=0.03) but restore standard buffer (10M). Short buffer
# (262k in v35) caused reward oscillation and instability.
k1_22dof_flash_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_manager", num_envs=1024),
    # v39: v34 base + moderate exploration tuning.
    # v34 walked with stock exploration (sigma=0.15, zeta=2, buffer 10M).
    # v38 failed: sigma=0.25 + zeta_mu=1.2 was too aggressive → legs split.
    # v39: halfway between v34 and v35 exploration. Standard buffer.
    algo=replace(algo.flash_sac, config=replace(
        algo.flash_sac.config,
        asymmetric_observation=True,
        gamma=0.97,
        n_step=1,
        # v43: revert to uniform scaling (v42 per-joint caused thrashing).
        target_action_scale_rad=1.0,
        # v39 exploration (walked stably)
        temp_initial_value=0.02,
        temp_target_sigma=0.20,
        actor_noise_zeta_mu=1.5,
        actor_noise_zeta_max=20,
        buffer_max_length=10_000_000,
        buffer_min_length=100_000,
        updates_per_interaction_step=2.0,
        sample_batch_size=2048,
        num_learning_iterations=100_000,
    )),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    # v43: v39 forward-only command + stand_prob 0.1→0.3.
    # v39's zero-velocity oscillation is from insufficient standing
    # practice. Fix by increasing stand_prob WITHOUT broadening
    # command range (v41 showed all-at-once broadening degrades gait).
    command=replace(
        command.k1_22dof_command,
        setup_terms={
            **command.k1_22dof_command.setup_terms,
            "locomotion_command": replace(
                command.k1_22dof_command.setup_terms["locomotion_command"],
                params={
                    # v48: full range (PPO default).
                    "command_ranges": {
                        "lin_vel_x": [-1.0, 1.0],
                        "lin_vel_y": [-1.0, 1.0],
                        "ang_vel_yaw": [-1.0, 1.0],
                        "heading": [-3.14, 3.14],
                    },
                    "stand_prob": 0.3,
                },
            ),
        },
    ),
    # PPO curriculum (initial_scale=0.1) — start with weak penalties
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco_flashsac,
)

# v50 / Plan A: G1 attempt #6 recipe ported to K1.
#
# Every prior K1 FlashSAC experiment (v1–v49) keeps the full PPO reward
# structure with K1-specific additions (feet_air_time, stride_progress,
# penalty_stand_still) and iterates on the weights. The one G1 recipe
# that produced clean walking — attempt #6 (``20260408_142832``) — is a
# 5-term IsaacLab-stock mirror with NO phase clock in the observation and
# NO shaping terms at all. That recipe has never been run on K1 under
# the post-v13 ``target_action_scale_rad=1.0`` action-scale regime.
#
# This preset is the minimal port of that recipe:
#   - reward = ``k1_22dof_loco_flashsac_stripped`` (5 terms, no feet_phase,
#     no pose, no alive, no close_feet, no feet_ori)
#   - observation = ``k1_22dof_loco_single_flashsac`` (no sin/cos phase)
#   - algo = stock ``algo.flash_sac`` (sigma=0.15, asymmetric_observation
#     =False, n_step=3, updates_per_interaction_step=2.0) — upstream paper
#     defaults, unchanged
#   - target_action_scale_rad=1.0 — K1-specific; v13 proved this recovers
#     hip swing amplitude from 0.14→0.30 rad (FlashSAC is tanh-bounded,
#     unlike PPO). This is the only K1-specific algo override.
#   - num_envs=1024 + num_learning_iterations=100_000 — matching paper
#     preset (num_envs * iter ≈ 100M interaction steps).
#
# If this fails (policy collapses to stand-still, as Option A on G1 did at
# ``20260409_053822``), the stripped preset is definitionally not enough
# for K1 and the next step is plan B (PPO warm-start) or accepting the
# v49 12-term recipe as the best achievable.
k1_22dof_flash_sac_stripped = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-k1-manager",
        name="k1_22dof_flash_sac_stripped_manager",
        num_envs=1024,
    ),
    algo=replace(
        algo.flash_sac,
        config=replace(
            algo.flash_sac.config,
            target_action_scale_rad=1.0,
            num_learning_iterations=100_000,
        ),
    ),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_flashsac,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco_flashsac_stripped,
)

# v51 / Plan A+: G1 v5 recipe (9 terms) applied literally to K1.
#
# Plan A (``k1_22dof_flash_sac_stripped``, run 20260418_124413) was smooth
# but the 5-term reward lets a narrow FlashSAC policy settle into a
# static-splits local optimum (one foot forward, one back, body low). The
# G1 v5 recipe adds feet_phase (forces alternating foot lift),
# pose (weak leg constraint + strong upper body), and
# penalty_close_feet_xy — the minimal shaping that prevents the splits
# attractor — without the K1-specific feet_air_time / stride_progress /
# stand_still terms that v28–v49 never converged with. Observation goes
# back to ``k1_22dof_loco_single_wolinvel`` so the policy can consume
# sin_phase/cos_phase required for feet_phase tracking.
k1_22dof_flash_sac_v5 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-k1-manager",
        name="k1_22dof_flash_sac_v5_manager",
        num_envs=1024,
    ),
    algo=replace(
        algo.flash_sac,
        config=replace(
            algo.flash_sac.config,
            target_action_scale_rad=1.0,
            num_learning_iterations=100_000,
        ),
    ),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco_flashsac_v5,
)

# v53: v5 recipe + 10x stronger penalty_action_rate (K1 v10 finding).
#
# v51 (``k1_22dof_flash_sac_v5``, run 20260418_153529) TB analysis:
#   rew_feet_phase = 3.12/4.0 = 78% (clock IS being followed)
#   rew_tracking_lin_vel = 1.79/2.0 = 90% (forward tracking works)
#   mean_action = +0.027 (non-zero but small)
# Yet visually: tap-tap cadence, grazing feet, small stride.
#
# Diagnosis: feet_phase kernel exp(-err^2 / sigma) with sigma=0.008 has
# effective tolerance ~= 9cm. Narrow policy gained reward by high-
# frequency small-amplitude foot oscillation whose average matched the
# Bezier target, without actual sinusoidal swing. penalty_action_rate
# at -0.005 (G1 v5 default) is too weak to punish this. K1 v10
# (``20260410_125717``) empirically proved -0.05 gives smooth gait.
#
# v53 changes ONE weight only: penalty_action_rate -0.005 -> -0.05.
# No swing_height, tracking_sigma, algo, or command changes from v51.
# num_learning_iterations=50_000 per FlashSAC paper preset (v51/v52
# converged in the first half of 100k; 50k halves the feedback loop).
k1_22dof_flash_sac_v5_smooth = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-k1-manager",
        name="k1_22dof_flash_sac_v5_smooth_manager",
        num_envs=1024,
    ),
    algo=replace(
        algo.flash_sac,
        config=replace(
            algo.flash_sac.config,
            target_action_scale_rad=1.0,
            num_learning_iterations=50_000,
        ),
    ),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco_flashsac_v5_smooth,
)


# v52: v5 recipe + K1 short-leg stride fix.
#
# v51 (``k1_22dof_flash_sac_v5``, run 20260418_153529) walked but feet
# grazed the ground, cadence was rapid staccato, and standstill posture
# was arched hip-forward. Diagnosis: the feet_phase kernel
# (``swing_height=0.09m``, ``tracking_sigma=0.008m``) is unreachable and
# too sharp for K1 kinematics, so the narrow policy abandons feet_phase
# reward. Without that gradient, stride stays tiny and clock lock is
# lost — the observed cadence is the policy's natural oscillation, not
# the ``gait_period=1.0s`` clock.
#
# v52 relaxes feet_phase parameters ONLY, nothing else changes from v51:
#   swing_height    0.09 → 0.065  (K1 v11–v15 short-leg finding)
#   tracking_sigma  0.008 → 0.015 (partial reward restores gradient)
#
# If stride increases and clock lock returns, cadence should also
# normalize without touching gait_period. penalty_stand_still and
# gait_period=1.2s stay in reserve for v53 if standstill posture and
# timing still need direct intervention.
k1_22dof_flash_sac_v5_stride = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="hv-k1-manager",
        name="k1_22dof_flash_sac_v5_stride_manager",
        num_envs=1024,
    ),
    algo=replace(
        algo.flash_sac,
        config=replace(
            algo.flash_sac.config,
            target_action_scale_rad=1.0,
            num_learning_iterations=100_000,
        ),
    ),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco_flashsac_v5_stride,
)

k1_22dof_flash_sac_mjwarp = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_mjwarp_manager"),
    algo=algo.flash_sac,
    simulator=simulator.mjwarp,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum_fast_sac,
    reward=reward.k1_22dof_loco_flashsac,
)

k1_22dof_fpo = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_fpo_manager"),
    algo=replace(algo.fpo, config=replace(algo.fpo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command.k1_22dof_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward.k1_22dof_loco,
)

__all__ = [
    "k1_22dof",
    "k1_22dof_fast_sac",
    "k1_22dof_flash_sac",
    "k1_22dof_flash_sac_stripped",
    "k1_22dof_flash_sac_v5",
    "k1_22dof_flash_sac_v5_smooth",
    "k1_22dof_flash_sac_v5_stride",
    "k1_22dof_flash_sac_mjwarp",
    "k1_22dof_fpo",
]

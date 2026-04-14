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

__all__ = ["k1_22dof", "k1_22dof_fast_sac", "k1_22dof_flash_sac", "k1_22dof_flash_sac_mjwarp", "k1_22dof_fpo"]

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

# v35: PPO original reward + FlashSAC exploration-tuned + short buffer.
#
# Thesis: "FlashSAC の探索設計を task に合わせれば、PPO の reward が
# そのまま動くはず。FlashSAC 専用の reward shaping は workaround。"
#
# Phase 1 (this config): on-policy-like FlashSAC.
#   - PPO の ORIGINAL reward (k1_22dof_loco) をそのまま使う
#   - PPO の curriculum (initial_scale=0.1, penalties 段階的)
#   - Short buffer (262k) = 最新データのみ保持 (on-policy 的)
#   - temp_target_sigma=0.25 (collapse 防止, entropy target +0.72)
#   - temp_initial_value=0.03 (初期探索 3x 強化)
#   - actor_noise_zeta_mu=1.2 (低い = 長い noise repetition)
#   - UTD=1.0 (on-policy 寄り)
#   - Forward-biased command + stand_prob=0.1
#
# Abort criteria: 3k iter で temperature < 0.002 かつ forward progress 低下
# Transition to Phase 2: 10-15k iter で歩行確認後, buffer/UTD を off-policy に
k1_22dof_flash_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_manager", num_envs=1024),
    algo=replace(algo.flash_sac, config=replace(
        algo.flash_sac.config,
        # T1-parity critic settings
        asymmetric_observation=True,
        gamma=0.97,
        n_step=1,
        # Action scale for PPO-level hip amplitude
        target_action_scale_rad=1.0,
        # Exploration: on-policy-like
        temp_initial_value=0.03,
        temp_target_sigma=0.25,
        actor_noise_zeta_mu=1.2,
        actor_noise_zeta_max=25,
        # Short buffer: ~256 vector steps. Forces learning from recent data.
        buffer_max_length=262_144,
        buffer_min_length=32_768,
        updates_per_interaction_step=1.0,
        sample_batch_size=2048,
        # Phase 1: 20k iterations. Codex: "abort if no forward gait by 20k"
        num_learning_iterations=20_000,
    )),
    simulator=simulator.isaacsim,
    robot=robot.k1_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=randomization.k1_22dof_randomization,
    # Forward-biased command for Phase 1 gait discovery.
    command=replace(
        command.k1_22dof_command,
        setup_terms={
            **command.k1_22dof_command.setup_terms,
            "locomotion_command": replace(
                command.k1_22dof_command.setup_terms["locomotion_command"],
                params={
                    "command_ranges": {
                        "lin_vel_x": [0.4, 0.8],
                        "lin_vel_y": [-0.1, 0.1],
                        "ang_vel_yaw": [-0.2, 0.2],
                        "heading": [-3.14, 3.14],
                    },
                    "stand_prob": 0.1,
                },
            ),
        },
    ),
    # PPO's original curriculum (initial_scale=0.1, not FastSAC's 0.5)
    curriculum=curriculum.k1_22dof_curriculum,
    # PPO's ORIGINAL reward — no stride_progress, no feet_air_time
    reward=reward.k1_22dof_loco,
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

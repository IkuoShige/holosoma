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

# FlashSAC-specific K1 control: match upstream mujoco_playground T1 gains
# (Booster-family robot with same hip_pitch=-0.2, knee_pitch=0.4 default pose).
#
# Root-cause analysis (v1-v18 forensic):
#   K1 effort limits: hip_pitch/knee 45 Nm, hip_roll/yaw 30 Nm, ankle 20 Nm.
#   At Kp=200 (stock): max static offset = 45/200 = 0.225 rad. Way too little.
#   At Kp=80 (v7-v18): hip_pitch 45/80 = 0.5625 rad, hip_roll 30/80 = 0.375 rad,
#                       ankle 20/40 = 0.5 rad. ALL below the 1.0 rad tanh range.
#   Policy outputs ∈ [-1, 1] rad target, but anything > 0.56 collapses into
#   the same clipped torque — a saturation plateau where FlashSAC's narrow
#   tanh policy cannot get a gradient signal.
# At T1's Kp=30 (hip/knee), Kp=10 (ankle): hip_pitch 45/30 = 1.5 rad, hip_roll
#   30/30 = 1.0 rad, ankle 20/10 = 2.0 rad. ALL >= 1.0 rad tanh range.
#   Full action space is physically reachable → policy gradient is clean.
# v19 (this config) ports this gain schedule to K1.
_k1_flashsac_robot = replace(
    robot.k1_22dof,
    control=replace(
        robot.k1_22dof.control,
        stiffness={
            "Head_yaw": 5.0, "Head_pitch": 5.0,
            "Hip_Yaw": 30.0, "Hip_Roll": 30.0, "Hip_Pitch": 30.0,
            "Knee": 30.0, "Ankle_Pitch": 10.0, "Ankle_Roll": 10.0,
            "Shoulder_Pitch": 20.0, "Shoulder_Roll": 20.0,
            "Elbow_Pitch": 20.0, "Elbow_Yaw": 20.0,
        },
        damping={
            "Head_yaw": 0.5, "Head_pitch": 0.5,
            "Hip_Yaw": 3.0, "Hip_Roll": 3.0, "Hip_Pitch": 3.0,
            "Knee": 3.0, "Ankle_Pitch": 3.0, "Ankle_Roll": 3.0,
            "Shoulder_Pitch": 0.5, "Shoulder_Roll": 0.5,
            "Elbow_Pitch": 0.5, "Elbow_Yaw": 0.5,
        },
    ),
)

k1_22dof_flash_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_manager"),
    # K1 needs wider exploration than G1: no waist DOFs makes gait harder
    # to discover. sigma 0.15→0.25 raises target_entropy from -10.53 to
    # +0.74, preventing the early temperature collapse that locks into
    # shuffle/march-in-place (confirmed across v1-v4 runs).
    # PPO actions are unbounded (Normal dist, no tanh); FlashSAC is tanh-bounded.
    # PPO hip_pitch output ≈3.4 → 3.4×0.25=0.85 rad. FlashSAC max is
    # tanh=1.0 → multiplier×0.25.
    # target_action_scale_rad=1.0 is the v13/v15/v19 baseline.
    # v16 (2.0) caused splits, v17 (1.5) was neutral — the tanh-gradient
    # hypothesis was wrong. v19 fixes the real issue (Kp saturation plateau)
    # via PD gain reduction in _k1_flashsac_robot above.
    algo=replace(algo.flash_sac, config=replace(
        algo.flash_sac.config, temp_target_sigma=0.25, target_action_scale_rad=1.0,
    )),
    simulator=simulator.isaacsim,
    robot=_k1_flashsac_robot,
    terrain=terrain.terrain_locomotion_mix,
    # PPO-default observation (with sin/cos phase clock): canonical v5
    # retains feet_phase in the reward, so the phase clock must remain.
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    # FlashSAC-specific: softer friction range [0.5, 1.25] (G1-matched).
    # Stock K1 uses [0.1, 1.0] which biases toward conservative shuffle.
    randomization=replace(
        randomization.k1_22dof_randomization,
        setup_terms={
            **randomization.k1_22dof_randomization.setup_terms,
            "randomize_friction_startup": replace(
                randomization.k1_22dof_randomization.setup_terms["randomize_friction_startup"],
                params={"friction_range": [0.5, 1.25], "enabled": True},
            ),
        },
    ),
    # v19: revert v18 command tweaks (gait_period, lin_vel_x, stand_prob,
    # tracking_sigma) — they were an attempt to compensate for Kp saturation
    # which is now fixed at the physics level via Kp=30. Only keep the
    # K1-specific gait_period=1.2 with G1-matched randomization width 0.2.
    command=replace(
        command.k1_22dof_command,
        setup_terms={
            **command.k1_22dof_command.setup_terms,
            "locomotion_gait": replace(
                command.k1_22dof_command.setup_terms["locomotion_gait"],
                params={
                    "gait_period": 1.2,
                    "gait_period_randomization_width": 0.2,
                },
            ),
        },
    ),
    curriculum=curriculum.k1_22dof_curriculum_fast_sac,
    reward=reward.k1_22dof_loco_flashsac,
)

k1_22dof_flash_sac_mjwarp = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_22dof_flash_sac_mjwarp_manager"),
    algo=replace(algo.flash_sac, config=replace(
        algo.flash_sac.config, temp_target_sigma=0.25, target_action_scale_rad=1.0,
    )),
    simulator=simulator.mjwarp,
    robot=_k1_flashsac_robot,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_22dof_loco_single_wolinvel,
    action=action.k1_22dof_joint_pos,
    termination=termination.k1_22dof_termination,
    randomization=replace(
        randomization.k1_22dof_randomization,
        setup_terms={
            **randomization.k1_22dof_randomization.setup_terms,
            "randomize_friction_startup": replace(
                randomization.k1_22dof_randomization.setup_terms["randomize_friction_startup"],
                params={"friction_range": [0.5, 1.25], "enabled": True},
            ),
        },
    ),
    command=replace(
        command.k1_22dof_command,
        setup_terms={
            **command.k1_22dof_command.setup_terms,
            "locomotion_gait": replace(
                command.k1_22dof_command.setup_terms["locomotion_gait"],
                params={
                    "gait_period": 1.2,
                    "gait_period_randomization_width": 0.2,
                },
            ),
        },
    ),
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

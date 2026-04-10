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

# FlashSAC-specific K1 control: reduce leg stiffness so the PD controller
# does not saturate the effort limit. Stock K1 uses Kp=200 on hips/knees
# with effort_limit=45 Nm, which means the joint can only move
# 45/200=0.225 rad per timestep — far too little for a walking gait.
# Reducing to G1-comparable stiffness (~80 Nm/rad) gives
# 45/80=0.5625 rad, enough for real steps. Damping halved to match.
# PPO uses the stock gains and compensates via high-entropy exploration;
# FlashSAC's narrow policy cannot, so we must fix the physics.
# PD gains: Kp=80 (reduced from stock 200) is better for FlashSAC.
# v13(Kp=80): hip amp 0.30rad. v14(Kp=200): hip amp 0.14rad (worse).
# Kp=80 is more compliant → policy can output larger actions safely.
# Kp=200 makes PD too stiff → policy learns cautious small actions.
_k1_flashsac_robot = replace(
    robot.k1_22dof,
    control=replace(
        robot.k1_22dof.control,
        stiffness={
            "Head_yaw": 5.0, "Head_pitch": 5.0,
            "Hip_Yaw": 80.0, "Hip_Roll": 80.0, "Hip_Pitch": 80.0,
            "Knee": 80.0, "Ankle_Pitch": 40.0, "Ankle_Roll": 40.0,
            "Shoulder_Pitch": 20.0, "Shoulder_Roll": 20.0,
            "Elbow_Pitch": 20.0, "Elbow_Yaw": 20.0,
        },
        damping={
            "Head_yaw": 0.5, "Head_pitch": 0.5,
            "Hip_Yaw": 2.5, "Hip_Roll": 2.5, "Hip_Pitch": 2.5,
            "Knee": 2.5, "Ankle_Pitch": 1.5, "Ankle_Roll": 1.5,
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
    # tanh=1.0 → multiplier×0.25. target_action_scale_rad=1.0 gives ±1.0 rad,
    # covering PPO's observed 0.85-0.93 rad peaks.
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
    # Slower gait clock for K1's shorter legs (0.49m vs G1 0.76m).
    # gait_period 1.0→1.2s with randomization (G1-matched).
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

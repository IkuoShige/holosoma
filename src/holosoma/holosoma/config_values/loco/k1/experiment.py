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

# FlashSAC-specific K1 control: Kp=80 (down from stock 200), Kd=2.5/1.5.
#
# History: v7-v18 used Kp=80 (best behavior at v13/v15). v19 ported T1's
# Kp=30 schedule, but K1's body mass requires the PD to develop ~30 Nm
# of static support torque per leg. At Kp=30, that needs a 1.0 rad sag
# from the default pose — the robot literally collapses into a squat
# before it can generate enough torque to stand. T1's lighter body
# tolerates Kp=30; K1 does not.
#
# v20 reverts to Kp=80 (the v13/v15 baseline that was the local best).
# The real bottleneck (per Codex post-mortem) was likely the contact
# termination at force_threshold=1.0 N: if the robot sags or wobbles
# enough that any "Hip"/"Trunk"/"Arm"/"Head" body touches the ground
# even mildly, the episode ends. v20 also raises the threshold to
# 50 N (see the termination override below).
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


# v20: relax the contact termination threshold from 1.0 N (stock) to 50 N.
# The 1.0 N threshold ends episodes on the slightest "Hip/Trunk/Arm/Head"
# body contact with the ground. With self-collisions disabled, this is
# essentially a fall proxy — but at 1.0 N it triggers far too easily.
# 50 N still detects clear ground impacts but tolerates exploratory wobble.
# This is the most likely root cause of the persistent v13-v19 metric
# plateau: episodes were ending too quickly to learn long-stride gaits.
_k1_flashsac_termination = replace(
    termination.k1_22dof_termination,
    terms={
        **termination.k1_22dof_termination.terms,
        "contact": replace(
            termination.k1_22dof_termination.terms["contact"],
            params={
                "force_threshold": 50.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
    },
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
    termination=_k1_flashsac_termination,
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
    termination=_k1_flashsac_termination,
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

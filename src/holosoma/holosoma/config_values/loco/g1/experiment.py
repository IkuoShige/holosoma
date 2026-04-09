from dataclasses import replace

from holosoma.config_types.algo import FPOModuleDictConfig, LayerConfig, ModuleConfig
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

g1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_flash_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_flash_sac_manager"),
    # Upstream FlashSAC algo defaults verbatim: temp_target_sigma=0.15,
    # asymmetric_observation=False, n_step=3, updates_per_interaction_step=2,
    # use_amp=True, etc. — matching configs/agent/flashSAC.yaml and
    # scripts/run_isaaclab.sh in /workspace/FlashSAC.
    algo=algo.flash_sac,
    # FlashSAC port targets the IsaacSim backend (Gate B). The vendored
    # algorithm itself is simulator-agnostic; only the underlying physics
    # backend differs from the existing g1_29dof_fast_sac (which uses
    # IsaacGym).
    simulator=simulator.isaacsim,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    # Dedicated FlashSAC-compatible reward and observation presets
    # (g1_29dof_loco_flashsac / g1_29dof_loco_single_flashsac). These strip
    # the exploration-killing terms {feet_phase, alive, pose, penalty_feet_ori,
    # penalty_close_feet_xy} and the sin/cos phase clock, leaving only the
    # task-defining tracking_lin_vel / tracking_ang_vel plus minimal
    # stability penalties. Mirrors IsaacLab stock ``Isaac-Velocity-Flat-G1-v0``
    # (the reference task FlashSAC's hyperparameters were tuned against).
    #
    # Option A (Open work #3, commit d422082) tested pairing FlashSAC with
    # the full holosoma PPO-default ``g1_29dof_loco`` reward on 20260409_053822
    # and confirmed empirically that the stripped preset is required: actor
    # collapsed to mean_action ≈ 0, fell at deploy. The earlier negative
    # result at 20260408_124759 (also on g1_29dof_loco) was NOT masking the
    # action-scale bug after all. Reverted to the stripped preset.
    # See docs/flashsac_port.md "Open work #3" for the post-mortem.
    observation=observation.g1_29dof_loco_single_flashsac,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_flashsac,
)

g1_29dof_flash_sac_mjwarp = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_flash_sac_mjwarp_manager"),
    # Mirror ``g1_29dof_flash_sac``: upstream FlashSAC algo defaults.
    algo=algo.flash_sac,
    # Same FlashSAC algorithm + same manager-based env, but driven by the
    # GPU-accelerated MuJoCo Warp (mjwarp) backend instead of IsaacSim.
    # Requires the ``hsmujoco`` conda env (Python 3.10, torch 2.10, mujoco_warp).
    simulator=simulator.mjwarp,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    # Mirror ``g1_29dof_flash_sac``: dedicated FlashSAC-compatible presets.
    # See ``g1_29dof_flash_sac`` above for the empirical history and
    # ``docs/flashsac_port.md`` Open work #3 for the Option A post-mortem.
    observation=observation.g1_29dof_loco_single_flashsac,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_flashsac,
)

g1_29dof_fpo = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_manager"),
    algo=replace(
        algo.fpo,
        config=replace(algo.fpo.config, num_learning_iterations=25000, use_symmetry=True, clip_param=0.01, lam=0.5),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

g1_29dof_fpo_refdiag = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_refdiag"),
    algo=replace(
        algo.fpo,
        config=replace(
            algo.fpo.config,
            num_learning_iterations=25000,
            use_symmetry=True,
            clip_param=0.01,
            lam=0.2,
            ratio_mode="legacy_avg",
            ratio_log_clip=1.0,
            num_mc_samples=16,
            mc_chunk_size=16,
            trust_region_mode="ppo_clip",
            cfm_reg_coef=0.0,
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

g1_29dof_fpo_data = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_data"),
    algo=replace(
        algo.fpo,
        config=replace(
            algo.fpo.config,
            num_learning_iterations=25000,
            use_symmetry=True,
            clip_param=0.01,
            lam=0.5,
            flow_param_mode="data",
            cfm_reg_coef=0.05,
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

g1_29dof_fpo_pp_repro = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_pp_repro"),
    algo=replace(
        algo.fpo,
        config=replace(
            algo.fpo.config,
            num_learning_iterations=25000,
            use_symmetry=True,
            num_learning_epochs=8,
            num_steps_per_env=24,
            num_mini_batches=4,
            lam=0.95,
            clip_param=0.05,
            cfm_reg_coef=0.0,
            num_flow_steps=64,
            num_mc_samples=16,
            ratio_mode="per_sample",
            ratio_log_clip=1.0,
            trust_region_mode="aspo",
            action_bound=1.5,
            action_bound_warmup_iters=0,
            max_grad_norm=0.5,
            flow_param_mode="velocity",
            cfm_loss_reduction="mean",
            cfm_loss_clip=10000.0,
            obs_normalization=True,
            divergence_guard_enabled=True,
            module_dict=FPOModuleDictConfig(
                actor=ModuleConfig(
                    type="MLP",
                    input_dim=["actor_obs"],
                    output_dim=["robot_action_dim"],
                    layer_config=LayerConfig(hidden_dims=[256, 256, 256], activation="ELU"),
                ),
                critic=ModuleConfig(
                    type="MLP",
                    input_dim=["critic_obs"],
                    output_dim=[1],
                    layer_config=LayerConfig(hidden_dims=[768, 768, 768], activation="ELU"),
                ),
            ),
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

g1_29dof_fpo_pp_warmstart_probe = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_pp_warmstart_probe"),
    algo=replace(
        algo.fpo,
        config=replace(
            algo.fpo.config,
            num_learning_iterations=25000,
            use_symmetry=True,
            num_learning_epochs=8,
            num_steps_per_env=24,
            num_mini_batches=4,
            lam=0.95,
            clip_param=0.05,
            cfm_reg_coef=0.0,
            num_flow_steps=64,
            num_mc_samples=16,
            ratio_mode="per_sample",
            ratio_log_clip=1.0,
            trust_region_mode="aspo",
            action_bound=1.5,
            action_bound_warmup_iters=0,
            max_grad_norm=0.5,
            flow_param_mode="velocity",
            cfm_loss_reduction="mean",
            cfm_loss_clip=10000.0,
            obs_normalization=True,
            divergence_guard_enabled=True,
            # BC warm-start from PPO teacher
            warm_start_mode="bc_teacher",
            warm_start_checkpoint=None,  # Set via CLI: --algo.config.warm_start_checkpoint /path/to/ppo.pt
            warm_start_bc_steps=200,
            warm_start_load_critic=False,
            warm_start_teacher_module=ModuleConfig(
                type="MLP",
                input_dim=["actor_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(hidden_dims=[512, 256, 128], activation="ELU"),
            ),
            module_dict=FPOModuleDictConfig(
                actor=ModuleConfig(
                    type="MLP",
                    input_dim=["actor_obs"],
                    output_dim=["robot_action_dim"],
                    layer_config=LayerConfig(hidden_dims=[256, 256, 256], activation="ELU"),
                ),
                critic=ModuleConfig(
                    type="MLP",
                    input_dim=["critic_obs"],
                    output_dim=[1],
                    layer_config=LayerConfig(hidden_dims=[768, 768, 768], activation="ELU"),
                ),
            ),
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

g1_29dof_fpo_pp_paper_default = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_pp_paper_default"),
    algo=replace(
        algo.fpo,
        config=replace(
            algo.fpo.config,
            num_learning_iterations=2000,
            use_symmetry=True,
            # Paper Table A.1 base; epochs reduced 32→8 to fix old_loss staleness
            # 128 gradient updates/iter → 32: keeps old_loss relevant across training
            num_learning_epochs=8,
            num_steps_per_env=24,
            num_mini_batches=4,
            lam=0.95,
            clip_param=0.05,
            cfm_reg_coef=0.0,
            num_flow_steps=64,
            num_mc_samples=16,
            ratio_mode="per_sample",
            ratio_log_clip=1.0,  # 0.5→1.0: rho_max=e^1=2.72, more signal with less staleness
            trust_region_mode="aspo",
            action_bound=3.0,
            action_bound_warmup_iters=0,
            max_grad_norm=0.5,
            flow_param_mode="velocity",
            cfm_loss_reduction="mean",  # /action_dim で ratio スケールを O(1) に正規化
            cfm_loss_clip=10.0,  # Stage 1: initial δ (overridden by adaptive)
            cfm_loss_clip_adaptive=True,  # Auto-adjust δ to maintain stage1_rate 10-30%
            cfm_loss_clip_quantile=0.8,
            cfm_loss_clip_ema_alpha=0.1,
            cfm_loss_clip_min=4.0,
            cfm_loss_clip_max=30.0,
            cfm_loss_clip_update_interval=10,
            cfm_loss_dim_clip=None,
            obs_normalization=True,
            divergence_guard_enabled=False,  # Don't auto-stop; observe the full divergence
            module_dict=FPOModuleDictConfig(
                actor=ModuleConfig(
                    type="MLP",
                    input_dim=["actor_obs"],
                    output_dim=["robot_action_dim"],
                    layer_config=LayerConfig(hidden_dims=[256, 256, 256], activation="ELU"),
                ),
                critic=ModuleConfig(
                    type="MLP",
                    input_dim=["critic_obs"],
                    output_dim=[1],
                    layer_config=LayerConfig(hidden_dims=[768, 768, 768], activation="ELU"),
                ),
            ),
        ),
    ),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fpo,
    reward=reward.g1_29dof_loco_fpo,
)

g1_29dof_fpo_pp = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fpo_pp"),
    algo=replace(
        algo.fpo_pp,
        config=replace(
            algo.fpo_pp.config,
            num_learning_iterations=2000,
            num_learning_epochs=32,  # humanoid: 32 epochs (official Table A.1)
            num_mc_samples=32,  # humanoid: 32 samples (official)
            use_symmetry=True,
            save_interval=50,
        ),
    ),
    simulator=simulator.isaacgym,
    # FPO++ requires tight action clipping (official: clip_actions=2.0 in env wrapper).
    # Without it, unbounded flow outputs cause action_rate penalty explosion.
    robot=replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_clip_value=2.0),
    ),
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
)

__all__ = [
    "g1_29dof",
    "g1_29dof_fast_sac",
    "g1_29dof_flash_sac",
    "g1_29dof_flash_sac_mjwarp",
    "g1_29dof_fpo",
    "g1_29dof_fpo_data",
    "g1_29dof_fpo_pp",
    "g1_29dof_fpo_pp_paper_default",
    "g1_29dof_fpo_pp_repro",
    "g1_29dof_fpo_pp_warmstart_probe",
    "g1_29dof_fpo_refdiag",
]

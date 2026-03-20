"""Ball play experiment presets for the K1 robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.robot import ObjectConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    randomization,
    robot,
    simulator,
    terrain,
)
from holosoma.config_values.ball_play.k1 import (
    command_dribble,
    command_kick,
    observation_dribble,
    observation_kick,
    reward_dribble,
    reward_kick,
    termination_dribble,
    termination_kick,
)

# Robot config with ball object
_k1_22dof_with_ball = replace(
    robot.k1_22dof,
    object=ObjectConfig(
        object_urdf_path="holosoma/data/robots/ball/ball.urdf",
    ),
)

# PPO config with symmetry disabled and min_noise_std set
_ball_play_algo = replace(
    algo.ppo,
    config=replace(
        algo.ppo.config,
        num_learning_iterations=25000,
        use_symmetry=False,
        module_dict=replace(
            algo.ppo.config.module_dict,
            actor=replace(algo.ppo.config.module_dict.actor, min_noise_std=0.01),
        ),
    ),
)

k1_22dof_ball_kick = ExperimentConfig(
    env_class="holosoma.envs.ball_play.ball_kick_task.BallKickTask",
    training=TrainingConfig(project="hv-k1-ball-play", name="k1_22dof_ball_kick"),
    algo=_ball_play_algo,
    simulator=simulator.isaacsim,
    robot=_k1_22dof_with_ball,
    terrain=terrain.terrain_locomotion_plane,
    observation=observation_kick.k1_22dof_ball_kick,
    action=action.k1_22dof_joint_pos,
    termination=termination_kick.k1_22dof_ball_kick_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command_kick.k1_22dof_ball_kick_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward_kick.k1_22dof_ball_kick,
)

k1_22dof_ball_dribble = ExperimentConfig(
    env_class="holosoma.envs.ball_play.ball_dribble_task.BallDribbleTask",
    training=TrainingConfig(project="hv-k1-ball-play", name="k1_22dof_ball_dribble"),
    algo=_ball_play_algo,
    simulator=simulator.isaacsim,
    robot=_k1_22dof_with_ball,
    terrain=terrain.terrain_locomotion_plane,
    observation=observation_dribble.k1_22dof_ball_dribble,
    action=action.k1_22dof_joint_pos,
    termination=termination_dribble.k1_22dof_ball_dribble_termination,
    randomization=randomization.k1_22dof_randomization,
    command=command_dribble.k1_22dof_ball_dribble_command,
    curriculum=curriculum.k1_22dof_curriculum,
    reward=reward_dribble.k1_22dof_ball_dribble,
)

# MuJoCo Warp variants
k1_22dof_ball_kick_mjwarp = replace(
    k1_22dof_ball_kick,
    simulator=simulator.mjwarp,
)

k1_22dof_ball_dribble_mjwarp = replace(
    k1_22dof_ball_dribble,
    simulator=simulator.mjwarp,
)

__all__ = [
    "k1_22dof_ball_kick",
    "k1_22dof_ball_dribble",
    "k1_22dof_ball_kick_mjwarp",
    "k1_22dof_ball_dribble_mjwarp",
]

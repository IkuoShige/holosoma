"""Dribble observation presets for the K1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

k1_22dof_ball_dribble = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.01,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                    noise=0.1,
                ),
                "dribble_target_dir_local": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:dribble_target_direction_local",
                    scale=1.0,
                    noise=0.0,
                ),
                "ball_pos_perceived": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:ball_pos_relative_perceived",
                    scale=0.333,
                    noise=0.03,
                ),
                "last_ball_pos_perceived": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:last_ball_pos_relative_perceived",
                    scale=0.333,
                    noise=0.03,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=1.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                ),
                "dribble_target_dir_local": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:dribble_target_direction_local",
                    scale=1.0,
                ),
                "ball_pos_perceived": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:ball_pos_relative_perceived",
                    scale=0.333,
                ),
                "ball_vel_world": ObsTermCfg(
                    func="holosoma.managers.observation.terms.ball_play:ball_vel_world",
                    scale=1.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                ),
            },
        ),
    },
)

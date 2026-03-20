import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.k1.experiment import (
    k1_22dof,
    k1_22dof_agile,
    k1_22dof_agile_fast_sac,
    k1_22dof_fast_sac,
)
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
)
from holosoma.config_values.ball_play.k1.experiment import (
    k1_22dof_ball_kick,
    k1_22dof_ball_dribble,
    k1_22dof_ball_kick_mjwarp,
    k1_22dof_ball_dribble_mjwarp,
)
from typing_extensions import Annotated

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "k1_22dof": k1_22dof,
    "k1_22dof_fast_sac": k1_22dof_fast_sac,
    "k1_22dof_agile": k1_22dof_agile,
    "k1_22dof_agile_fast_sac": k1_22dof_agile_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "k1_22dof_ball_kick": k1_22dof_ball_kick,
    "k1_22dof_ball_dribble": k1_22dof_ball_dribble,
    "k1_22dof_ball_kick_mjwarp": k1_22dof_ball_kick_mjwarp,
    "k1_22dof_ball_dribble_mjwarp": k1_22dof_ball_dribble_mjwarp,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]

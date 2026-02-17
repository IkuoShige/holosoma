"""Default curriculum manager configurations."""

from holosoma.config_values.loco.g1.curriculum import g1_29dof_curriculum, g1_29dof_curriculum_fast_sac
from holosoma.config_values.loco.k1.curriculum import (
    k1_22dof_agile_curriculum,
    k1_22dof_agile_curriculum_fast_sac,
    k1_22dof_curriculum,
    k1_22dof_curriculum_fast_sac,
)
from holosoma.config_values.loco.t1.curriculum import t1_29dof_curriculum, t1_29dof_curriculum_fast_sac
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_curriculum,
    "k1_22dof": k1_22dof_curriculum,
    "k1_22dof_agile": k1_22dof_agile_curriculum,
    "g1_29dof": g1_29dof_curriculum,
    "t1_29dof_fast_sac": t1_29dof_curriculum_fast_sac,
    "k1_22dof_fast_sac": k1_22dof_curriculum_fast_sac,
    "k1_22dof_agile_fast_sac": k1_22dof_agile_curriculum_fast_sac,
    "g1_29dof_fast_sac": g1_29dof_curriculum_fast_sac,
    "g1_29dof_wbt_curriculum": g1_29dof_wbt_curriculum,
}

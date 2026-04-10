"""Transform holosoma PPO/FastSAC reward+observation presets into FlashSAC-compatible form.

FlashSAC uses a narrow deterministic policy (``temp_target_sigma=0.15``, target
entropy ≈ ``0.5 * d * log(2πe * 0.15²)`` which is strongly negative for typical
humanoid action dims). This means the policy commits early and cannot escape
local optima created by strong shaping terms at full PPO weight.

The canonical translation (validated on G1 across 16 training runs to a
composite PPO-divergence score of 1.44×) keeps **all PPO reward terms except
``alive``** but re-weights them to stay below FlashSAC's collapse threshold.
The observation uses the **PPO-default** preset unchanged (including
``sin_phase`` / ``cos_phase`` which couple to the retained ``feet_phase``).

See ``docs/flashsac_reward_translation.md`` for the full empirical rationale,
per-term tuning history, and K1 expansion roadmap.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

# -------------------------------------------------------------------
# Canonical v5 recipe — validated on G1 run 20260409_153439
# (composite 1.44× vs PPO, pitch 1.1×, height 1.1×, upper body 1.6×).
# -------------------------------------------------------------------

#: The only term to DROP entirely. ``alive`` creates a lazy-attractor
#: (constant +1/step for standing still) that FlashSAC exploits even at
#: reduced weight. Empirically confirmed at run #9 (Option A).
DEFAULT_DROP_TERMS: frozenset[str] = frozenset({"alive"})

#: Per-term weight overrides. Each term is kept but at a weight tuned to
#: stay below FlashSAC's zero-action collapse threshold while providing
#: enough gradient to shape a PPO-like gait.
#:
#: Tuning history (G1, 16 runs):
#:   penalty_ang_vel_xy:  -0.3 worsened pitch (1.63×); -0.05 is the sweet spot.
#:   penalty_orientation: -3.0 worsened composite (2.48×); -1.0 is the sweet spot.
#:   penalty_action_rate: PPO -2.0 is 400× too strong; -0.005 works.
#:   pose:                -0.5 (full PPO) collapses; -0.2 with ub_weights=150 works.
#:   feet_phase:          0.5 too weak for backward gait with stiff upper body;
#:                        4.0 (80% PPO) is the sweet spot.
#:   penalty_feet_ori:    -0.5 (10× weaker) works.
#:   penalty_close_feet_xy: -1.0 (10× weaker) works.
DEFAULT_WEIGHT_OVERRIDES: dict[str, float] = {
    "penalty_ang_vel_xy": -0.05,
    "penalty_orientation": -1.0,
    "penalty_action_rate": -0.005,
    "pose": -0.2,
    "feet_phase": 4.0,
    "penalty_feet_ori": -0.5,
    "penalty_close_feet_xy": -1.0,
}

#: Upper-body pose_weights multiplier. PPO uses 50 for upper body joints;
#: FlashSAC needs 150 (3×) to compensate for the 2.5× weaker outer pose weight.
#: Effective penalty: -0.2 × 150 = -30 (PPO: -0.5 × 50 = -25, ~120%).
DEFAULT_UPPER_BODY_POSE_WEIGHT: float = 150.0

#: Index at which upper-body joints start in the pose_weights list.
#: For both G1 (29-DoF) and K1 (22-DoF): left leg (6) + right leg (6) = 12.
UPPER_BODY_START_IDX: int = 12


def make_flashsac_reward(
    source: RewardManagerCfg,
    *,
    drop_terms: frozenset[str] = DEFAULT_DROP_TERMS,
    weight_overrides: dict[str, float] = DEFAULT_WEIGHT_OVERRIDES,
    upper_body_pose_weight: float = DEFAULT_UPPER_BODY_POSE_WEIGHT,
    upper_body_start_idx: int = UPPER_BODY_START_IDX,
) -> RewardManagerCfg:
    """Derive a FlashSAC-compatible reward from a PPO/FastSAC locomotion preset.

    Parameters
    ----------
    source:
        The PPO-default ``RewardManagerCfg`` (e.g. ``g1_29dof_loco``).
    drop_terms:
        Term names to remove entirely (default: only ``alive``).
    weight_overrides:
        Term name → new weight. Terms not listed here keep their PPO weight.
    upper_body_pose_weight:
        Replacement pose_weight value for upper-body joints (index >=
        ``upper_body_start_idx``). Set to 0 to skip pose_weights patching.
    upper_body_start_idx:
        First index in ``pose_weights`` that is an upper-body joint.
    """
    new_terms: dict[str, RewardTermCfg] = {}
    for name, term in source.terms.items():
        if name in drop_terms:
            continue
        if name in weight_overrides:
            term = replace(term, weight=weight_overrides[name])
        # Patch upper-body pose_weights if this is the ``pose`` term.
        if name == "pose" and upper_body_pose_weight > 0:
            pw = list(term.params.get("pose_weights", []))
            if pw:
                for i in range(upper_body_start_idx, len(pw)):
                    pw[i] = upper_body_pose_weight
                term = replace(term, params={**term.params, "pose_weights": pw})
        new_terms[name] = term
    return replace(source, terms=new_terms)


def make_flashsac_observation(
    source: ObservationManagerCfg,
) -> ObservationManagerCfg:
    """Return the PPO-default observation unchanged.

    The canonical v5 recipe retains ``feet_phase`` in the reward, so the
    ``sin_phase`` / ``cos_phase`` clock terms must remain in the observation.
    No observation transform is needed — this function exists as a no-op
    placeholder so the interface stays symmetric with ``make_flashsac_reward``.
    """
    return source

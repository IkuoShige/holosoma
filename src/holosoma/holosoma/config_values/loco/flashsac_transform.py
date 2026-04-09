"""Transform holosoma PPO/FastSAC reward+observation presets into FlashSAC-compatible form.

FlashSAC uses a narrow deterministic policy (``temp_target_sigma=0.15``, target
entropy ≈ ``0.5 * d * log(2πe * 0.15²)`` which is strongly negative for typical
humanoid action dims). This means the policy commits early and cannot escape
local optima created by strong shaping terms.

Holosoma's PPO-default locomotion reward contains several terms that create
**zero-action attractors** for FlashSAC's narrow policy:

- ``alive`` (+1.0/step): constant reward for surviving → standing still is safe
- ``feet_phase`` (high weight, tight sigma): narrow policy matches the phase
  clock with micro-oscillation, exploiting +36 ep_sum without forward progress
- ``pose`` (upper-body weight 50): strongly penalises deviations from default
  joint angles → the policy locks the torso
- ``penalty_action_rate`` (-2.0): punishes movement harder than exploration
  can overcome
- ``penalty_feet_ori``, ``penalty_close_feet_xy``: additional penalties that
  reinforce "don't move"

This module provides ``make_flashsac_reward`` and ``make_flashsac_observation``
which mechanically apply the empirically-validated transformation that was
hand-tuned on G1 (runs ``20260408_142832`` / ``20260408_154733``, walking at
~0.28 m/s forward). The transforms are designed to be **robot-agnostic** so
they can be applied to K1, T1, or any future holosoma humanoid without
per-robot trial-and-error.

Empirical evidence (9 G1 training runs):
- Runs #6/#7 (stripped preset): walks at 0.28 m/s, σ=0.15 ✓
- Run #8 (σ=0.30 on stripped): no gait improvement → σ is NOT the bottleneck
- Run #9 (Option A: PPO-default reward + σ=0.15): collapsed to mean_action≈0
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

# -------------------------------------------------------------------
# Default configuration — empirically validated on G1 runs #6/#7.
# -------------------------------------------------------------------

#: Reward terms that create zero-action attractors for FlashSAC.
DEFAULT_DROP_TERMS: frozenset[str] = frozenset({
    "alive",
    "feet_phase",
    "pose",
    "penalty_feet_ori",
    "penalty_close_feet_xy",
})

#: Penalty weights to weaken so exploration is not punished.
#: Mapping from term name → new weight.
DEFAULT_WEAKEN_WEIGHTS: dict[str, float] = {
    "penalty_ang_vel_xy": -0.05,     # PPO default: -1.0   (20× weaker)
    "penalty_orientation": -1.0,      # PPO default: -10.0  (10× weaker)
    "penalty_action_rate": -0.005,    # PPO default: -2.0   (400× weaker)
}

#: Observation terms to drop (clock signals that only couple to feet_phase).
DEFAULT_DROP_OBS_TERMS: frozenset[str] = frozenset({
    "sin_phase",
    "cos_phase",
})


def make_flashsac_reward(
    source: RewardManagerCfg,
    *,
    drop_terms: frozenset[str] = DEFAULT_DROP_TERMS,
    weaken_weights: dict[str, float] = DEFAULT_WEAKEN_WEIGHTS,
    strip_curriculum_tags: bool = True,
) -> RewardManagerCfg:
    """Derive a FlashSAC-compatible reward from a PPO/FastSAC locomotion preset.

    Parameters
    ----------
    source:
        The PPO-default ``RewardManagerCfg`` (e.g. ``g1_29dof_loco``).
    drop_terms:
        Term names to remove entirely.
    weaken_weights:
        Term names to keep but with reduced weight.
    strip_curriculum_tags:
        If True, remove ``tags`` from surviving penalty terms. FlashSAC's
        curriculum interaction is through the ``fast_sac`` curriculum which
        starts at 0.5, so curriculum tags on already-weakened penalties are
        unnecessary and may cause subtle scaling issues.
    """
    new_terms: dict[str, RewardTermCfg] = {}
    for name, term in source.terms.items():
        if name in drop_terms:
            continue
        if name in weaken_weights:
            term = replace(term, weight=weaken_weights[name])
            if strip_curriculum_tags:
                term = replace(term, tags=[])
        new_terms[name] = term
    return replace(source, terms=new_terms)


def make_flashsac_observation(
    source: ObservationManagerCfg,
    *,
    drop_obs_terms: frozenset[str] = DEFAULT_DROP_OBS_TERMS,
) -> ObservationManagerCfg:
    """Derive a FlashSAC-compatible observation from a PPO/FastSAC observation preset.

    Removes the phase-clock terms (``sin_phase``, ``cos_phase``) from both
    ``actor_obs`` and ``critic_obs`` groups. These are only meaningful when
    coupled to ``feet_phase`` in the reward; without that coupling they add
    noise and a degenerate "match clock in place" attractor.

    All other terms (including ``base_lin_vel`` in the critic group) are
    preserved unchanged.
    """
    new_groups: dict[str, ObsGroupCfg] = {}
    for group_name, group in source.groups.items():
        new_terms = {
            name: term
            for name, term in group.terms.items()
            if name not in drop_obs_terms
        }
        new_groups[group_name] = replace(group, terms=new_terms)
    return replace(source, groups=new_groups)

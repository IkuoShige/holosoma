"""Validate that make_flashsac_reward/observation reproduces the G1 ground truth.

The hand-crafted ``g1_29dof_loco_flashsac`` and ``g1_29dof_loco_single_flashsac``
presets were empirically validated on training runs 20260408_142832 / 20260408_154733
(walks at ~0.28 m/s forward with σ=0.15). These tests verify that the transform
helper produces byte-identical output from the PPO-default presets.
"""

from __future__ import annotations

import pytest


def test_g1_reward_transform_matches_handcrafted() -> None:
    """make_flashsac_reward(g1_29dof_loco) == g1_29dof_loco_flashsac."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_flashsac

    generated = make_flashsac_reward(g1_29dof_loco)

    # Same term names.
    assert set(generated.terms.keys()) == set(g1_29dof_loco_flashsac.terms.keys()), (
        f"Term mismatch: generated={set(generated.terms.keys())}, "
        f"handcrafted={set(g1_29dof_loco_flashsac.terms.keys())}"
    )

    # Same weights and params per term.
    for name in g1_29dof_loco_flashsac.terms:
        gen = generated.terms[name]
        ref = g1_29dof_loco_flashsac.terms[name]
        assert gen.weight == ref.weight, (
            f"Weight mismatch on '{name}': generated={gen.weight}, handcrafted={ref.weight}"
        )
        assert gen.func == ref.func, (
            f"Func mismatch on '{name}': generated={gen.func}, handcrafted={ref.func}"
        )
        assert gen.params == ref.params, (
            f"Params mismatch on '{name}': generated={gen.params}, handcrafted={ref.params}"
        )
        # Handcrafted stripped preset has no tags on penalty terms.
        assert gen.tags == ref.tags, (
            f"Tags mismatch on '{name}': generated={gen.tags}, handcrafted={ref.tags}"
        )


def test_g1_observation_transform_matches_handcrafted() -> None:
    """make_flashsac_observation(g1_29dof_loco_single_wolinvel) == g1_29dof_loco_single_flashsac."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_observation
    from holosoma.config_values.loco.g1.observation import (
        g1_29dof_loco_single_flashsac,
        g1_29dof_loco_single_wolinvel,
    )

    generated = make_flashsac_observation(g1_29dof_loco_single_wolinvel)

    for group_name in ("actor_obs", "critic_obs"):
        gen_group = generated.groups[group_name]
        ref_group = g1_29dof_loco_single_flashsac.groups[group_name]

        assert set(gen_group.terms.keys()) == set(ref_group.terms.keys()), (
            f"Terms mismatch in '{group_name}': generated={set(gen_group.terms.keys())}, "
            f"handcrafted={set(ref_group.terms.keys())}"
        )

        for term_name in ref_group.terms:
            gen = gen_group.terms[term_name]
            ref = ref_group.terms[term_name]
            assert gen.func == ref.func, f"{group_name}/{term_name}: func mismatch"
            assert gen.scale == ref.scale, f"{group_name}/{term_name}: scale mismatch"
            assert gen.noise == ref.noise, f"{group_name}/{term_name}: noise mismatch"


def test_transform_drops_correct_terms() -> None:
    """Verify the default drop list removes exactly the known killers."""
    from holosoma.config_values.loco.flashsac_transform import (
        DEFAULT_DROP_TERMS,
        make_flashsac_reward,
    )
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco

    generated = make_flashsac_reward(g1_29dof_loco)
    dropped = set(g1_29dof_loco.terms.keys()) - set(generated.terms.keys())
    assert dropped == DEFAULT_DROP_TERMS, (
        f"Expected to drop {DEFAULT_DROP_TERMS}, actually dropped {dropped}"
    )


def test_transform_is_idempotent() -> None:
    """Applying the transform twice produces the same result."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco

    once = make_flashsac_reward(g1_29dof_loco)
    twice = make_flashsac_reward(once)

    assert set(once.terms.keys()) == set(twice.terms.keys())
    for name in once.terms:
        assert once.terms[name].weight == twice.terms[name].weight


def test_k1_reward_transform_produces_expected_terms() -> None:
    """k1_22dof_loco transforms to 5 terms with expected weights."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.k1.reward import k1_22dof_loco

    generated = make_flashsac_reward(k1_22dof_loco)

    expected_terms = {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.5,
        "penalty_ang_vel_xy": -0.05,
        "penalty_orientation": -1.0,
        "penalty_action_rate": -0.005,
    }

    assert set(generated.terms.keys()) == set(expected_terms.keys()), (
        f"K1 FlashSAC transform term mismatch: got {set(generated.terms.keys())}"
    )
    for name, expected_weight in expected_terms.items():
        assert generated.terms[name].weight == expected_weight, (
            f"K1 '{name}' weight: got {generated.terms[name].weight}, expected {expected_weight}"
        )

"""Validate that make_flashsac_reward reproduces the G1 v5 canonical recipe.

The v5 recipe (run 20260409_153439) achieved a composite PPO-divergence
score of 1.44× — pitch 1.1×, height 1.1×, upper body 1.6×, leg 1.2×.
These tests verify that the transform helper produces output matching
the hand-tuned ``g1_29dof_loco_flashsac`` preset.
"""

from __future__ import annotations


def test_g1_reward_transform_matches_v5_preset() -> None:
    """make_flashsac_reward(g1_29dof_loco) matches g1_29dof_loco_flashsac."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_flashsac

    generated = make_flashsac_reward(g1_29dof_loco)

    assert set(generated.terms.keys()) == set(g1_29dof_loco_flashsac.terms.keys()), (
        f"Term mismatch: generated={set(generated.terms.keys())}, "
        f"handcrafted={set(g1_29dof_loco_flashsac.terms.keys())}"
    )

    for name in g1_29dof_loco_flashsac.terms:
        gen = generated.terms[name]
        ref = g1_29dof_loco_flashsac.terms[name]
        assert gen.weight == ref.weight, (
            f"Weight mismatch on '{name}': generated={gen.weight}, expected={ref.weight}"
        )
        assert gen.func == ref.func, f"Func mismatch on '{name}'"
        # Check pose_weights for the pose term
        if name == "pose":
            gen_pw = gen.params.get("pose_weights", [])
            ref_pw = ref.params.get("pose_weights", [])
            assert len(gen_pw) == len(ref_pw), f"pose_weights length mismatch"
            for i, (g, r) in enumerate(zip(gen_pw, ref_pw)):
                assert g == r, f"pose_weights[{i}] mismatch: generated={g}, expected={r}"


def test_only_alive_is_dropped() -> None:
    """The v5 recipe drops only ``alive``, not feet_phase/pose/etc."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco

    generated = make_flashsac_reward(g1_29dof_loco)
    dropped = set(g1_29dof_loco.terms.keys()) - set(generated.terms.keys())
    assert dropped == {"alive"}, f"Expected to drop only 'alive', dropped {dropped}"


def test_transform_is_idempotent() -> None:
    """Applying the transform twice produces the same result."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_reward
    from holosoma.config_values.loco.g1.reward import g1_29dof_loco

    once = make_flashsac_reward(g1_29dof_loco)
    twice = make_flashsac_reward(once)

    assert set(once.terms.keys()) == set(twice.terms.keys())
    for name in once.terms:
        assert once.terms[name].weight == twice.terms[name].weight


def test_observation_passthrough() -> None:
    """make_flashsac_observation returns PPO observation unchanged."""
    from holosoma.config_values.loco.flashsac_transform import make_flashsac_observation
    from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel

    result = make_flashsac_observation(g1_29dof_loco_single_wolinvel)
    assert result is g1_29dof_loco_single_wolinvel


def test_k1_reward_transform_produces_expected_terms() -> None:
    """k1_22dof_loco transforms to 9 terms (all except alive) with correct weights."""
    from holosoma.config_values.loco.flashsac_transform import (
        K1_UPPER_BODY_POSE_INDICES,
        make_flashsac_reward,
    )
    from holosoma.config_values.loco.k1.reward import k1_22dof_loco

    generated = make_flashsac_reward(
        k1_22dof_loco, upper_body_pose_indices=K1_UPPER_BODY_POSE_INDICES
    )

    assert "alive" not in generated.terms, "alive should be dropped"
    assert len(generated.terms) == 9, f"Expected 9 terms, got {len(generated.terms)}"

    expected_weights = {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.5,
        "penalty_ang_vel_xy": -0.05,
        "penalty_orientation": -1.0,
        "penalty_action_rate": -0.005,
        "pose": -0.2,
        "feet_phase": 4.0,
        "penalty_feet_ori": -0.5,
        "penalty_close_feet_xy": -1.0,
    }
    for name, expected in expected_weights.items():
        assert generated.terms[name].weight == expected, (
            f"K1 '{name}': got {generated.terms[name].weight}, expected {expected}"
        )

    # K1 joint order: upper body FIRST (indices 0-9), then legs (10-21).
    pw = generated.terms["pose"].params["pose_weights"]
    # Upper body (indices 0-9) should be boosted to 150
    for i in range(0, 10):
        assert pw[i] == 150.0, f"K1 pose_weights[{i}] (upper body) should be 150, got {pw[i]}"
    # Leg joints (indices 10-21) should keep their PPO values
    assert pw[10] == 0.01, f"K1 pose_weights[10] (Left_Hip_Pitch) should be 0.01, got {pw[10]}"
    assert pw[13] == 0.01, f"K1 pose_weights[13] (Left_Knee_Pitch) should be 0.01, got {pw[13]}"
    assert pw[12] == 5.0, f"K1 pose_weights[12] (Left_Hip_Yaw) should be 5.0, got {pw[12]}"


def test_k1_flashsac_preset_matches_transform() -> None:
    """The hand-registered k1_22dof_loco_flashsac matches make_flashsac_reward output."""
    from holosoma.config_values.loco.flashsac_transform import (
        K1_UPPER_BODY_POSE_INDICES,
        make_flashsac_reward,
    )
    from holosoma.config_values.loco.k1.reward import k1_22dof_loco, k1_22dof_loco_flashsac

    generated = make_flashsac_reward(
        k1_22dof_loco, upper_body_pose_indices=K1_UPPER_BODY_POSE_INDICES
    )

    assert set(generated.terms.keys()) == set(k1_22dof_loco_flashsac.terms.keys())
    for name in k1_22dof_loco_flashsac.terms:
        gen = generated.terms[name]
        ref = k1_22dof_loco_flashsac.terms[name]
        assert gen.weight == ref.weight, f"Weight mismatch on '{name}'"
        if name == "pose":
            gen_pw = gen.params.get("pose_weights", [])
            ref_pw = ref.params.get("pose_weights", [])
            assert gen_pw == ref_pw, f"pose_weights mismatch"

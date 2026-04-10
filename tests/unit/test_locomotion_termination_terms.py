from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.managers.termination.terms import locomotion as terms


def _make_env(*, num_envs: int = 2) -> SimpleNamespace:
    env = SimpleNamespace()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.simulator = SimpleNamespace()
    return env


def test_gravity_tilt_exceeded_does_not_require_env_config(monkeypatch) -> None:
    env = _make_env()
    projected_gravity = torch.tensor(
        [
            [0.8, 0.0, -0.6],
            [0.2, 0.1, -0.97],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(terms, "get_projected_gravity", lambda _env: projected_gravity)

    actual = terms.gravity_tilt_exceeded(env, threshold_x=0.7, threshold_y=0.7)

    expected = torch.tensor([True, False], dtype=torch.bool)
    torch.testing.assert_close(actual, expected)


def test_base_height_below_threshold_does_not_require_env_config() -> None:
    env = _make_env()
    env.simulator.robot_root_states = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.35],
        ],
        dtype=torch.float32,
    )

    actual = terms.base_height_below_threshold(env, min_height=0.3)

    expected = torch.tensor([True, False], dtype=torch.bool)
    torch.testing.assert_close(actual, expected)


def test_dof_velocity_limit_exceeded_uses_term_parameter_threshold_scale() -> None:
    env = _make_env()
    env.dof_vel_limits = torch.tensor([2.0, 2.0], dtype=torch.float32)
    env.simulator.dof_vel = torch.tensor(
        [
            [1.8, 0.2],
            [0.9, 0.5],
        ],
        dtype=torch.float32,
    )

    actual = terms.dof_velocity_limit_exceeded(env, threshold_scale=0.8)

    expected = torch.tensor([True, False], dtype=torch.bool)
    torch.testing.assert_close(actual, expected)


def test_torque_limit_exceeded_uses_term_parameter_threshold_scale() -> None:
    env = _make_env()
    env.torque_limits = torch.tensor([10.0, 10.0], dtype=torch.float32)
    env.action_manager = SimpleNamespace(
        get_term=lambda _name: SimpleNamespace(
            torques=torch.tensor(
                [
                    [9.5, 0.0],
                    [4.0, 4.0],
                ],
                dtype=torch.float32,
            )
        )
    )

    actual = terms.torque_limit_exceeded(env, threshold_scale=0.9)

    expected = torch.tensor([True, False], dtype=torch.bool)
    torch.testing.assert_close(actual, expected)

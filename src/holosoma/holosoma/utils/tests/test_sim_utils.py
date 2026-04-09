from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.sim_utils import is_effective_headless


def test_is_effective_headless_uses_training_flag() -> None:
    cfg = ExperimentConfig()

    assert is_effective_headless(cfg) is True


def test_is_effective_headless_treats_viser_as_headless() -> None:
    base_cfg = ExperimentConfig()
    cfg = ExperimentConfig(
        training=replace(base_cfg.training, headless=False),
        simulator=replace(
            base_cfg.simulator,
            config=replace(base_cfg.simulator.config, viser=replace(base_cfg.simulator.config.viser, enabled=True)),
        ),
    )

    assert is_effective_headless(cfg) is True


def test_is_effective_headless_supports_run_sim_config() -> None:
    cfg = RunSimConfig()

    assert is_effective_headless(cfg) is False

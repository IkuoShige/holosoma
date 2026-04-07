"""Holosoma-native adapter for the vendored FlashSAC algorithm.

This package wraps :mod:`holosoma._vendored.flash_rl` so it can be invoked
through ``holosoma.train_agent`` against ``LeggedRobotLocomotionManager`` and
holosoma's standard logging / checkpointing surface.
"""

from holosoma.agents.flash_sac.flash_sac_agent import FlashSACAgent  # noqa: F401
from holosoma.agents.flash_sac.flash_sac_env_bridge import FlashSACGymBridge  # noqa: F401

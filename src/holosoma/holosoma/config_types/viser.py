"""Configuration for ViserBridge web-based 3D visualization."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ViserBridgeConfig:
    """Configuration for viser web viewer bridge.

    When enabled, streams robot body transforms from the simulator
    to a browser-accessible 3D viewer via viser.
    """

    enabled: bool = False
    """Master toggle. False by default so training is unaffected."""

    host: str = "0.0.0.0"
    """Bind address for the viser web server."""

    port: int = 8080
    """Web server port."""

    show_grid: bool = True
    """Show ground plane grid."""

    grid_width: float = 8.0
    """Ground grid width in meters."""

    grid_height: float = 8.0
    """Ground grid height in meters."""

    update_freq: int = 4
    """Update viser every N physics steps (decimation). Higher = less overhead."""

    max_envs: int = 1
    """Number of environments to visualize (1 = env 0 only)."""

    env_spacing: float = 3.0
    """Spacing between environments in multi-env mode."""

    fps_limit: int = 10
    """Cap viser updates to this frame rate. Lower = less training overhead."""

    show_terrain: bool = False
    """Show terrain mesh in viser viewer. Can be heavy for large terrains."""

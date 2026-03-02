"""Bridge holosoma TerrainManager state to mjlab TerrainEntityCfg."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from holosoma.managers.terrain.manager import TerrainManager


def build_terrain_entity_cfg(terrain_manager: TerrainManager) -> object | None:
    """Convert holosoma terrain state to an mjlab TerrainEntityCfg.

    Parameters
    ----------
    terrain_manager : TerrainManager
        Holosoma terrain manager containing terrain configuration.

    Returns
    -------
    TerrainEntityCfg | None
        mjlab terrain config, or None for flat/none terrain.
    """
    from mjlab.terrains import TerrainEntityCfg  # noqa: PLC0415

    terrain_state = terrain_manager.get_state("locomotion_terrain")

    if terrain_state.mesh_type in ["none", "fake"]:
        logger.info("MjLab: No terrain (mesh_type='none'/'fake')")
        return None

    if terrain_state.mesh_type == "plane":
        logger.info("MjLab: Using default ground plane terrain")
        return TerrainEntityCfg(terrain_type="plane")

    # For trimesh/heightfield terrains, use the plane fallback.
    # Full terrain mesh bridging would require converting holosoma's terrain mesh
    # data into mjlab's TerrainGeneratorCfg, which is out of scope for initial
    # integration. Users should configure terrain via mjlab's native terrain system.
    logger.warning(
        f"MjLab: Terrain mesh_type='{terrain_state.mesh_type}' not yet bridged. "
        "Falling back to ground plane. Configure terrain via mjlab's native TerrainEntityCfg."
    )
    return TerrainEntityCfg(terrain_type="plane")

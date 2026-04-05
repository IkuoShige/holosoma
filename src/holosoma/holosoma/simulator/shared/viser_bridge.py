"""ViserBridge: simulator-agnostic bridge to viser web-based 3D viewer.

Streams per-body transforms from any holosoma simulator backend to a
browser-accessible 3D viewer. Loads robot URDF via ``ViserUrdf`` and
updates joint configurations each frame.

Requires optional dependencies: ``pip install viser yourdfpy``
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from holosoma.config_types.viser import ViserBridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class ViserBridge:
    """Bridge between holosoma simulator and viser web viewer.

    Parameters
    ----------
    simulator : BaseSimulator
        The simulator instance (provides robot config, dof_pos, body transforms).
    config : ViserBridgeConfig
        Viser bridge configuration.
    """

    def __init__(self, simulator: BaseSimulator, config: ViserBridgeConfig) -> None:
        import viser  # type: ignore[import-not-found]
        import yourdfpy  # type: ignore[import-untyped]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]

        self._simulator = simulator
        self._config = config
        self._step_count = 0
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / max(config.fps_limit, 1)

        # Resolve URDF path
        urdf_path = self._resolve_urdf_path()
        logger.info(f"ViserBridge: loading URDF from {urdf_path}")

        # Create viser server
        self._server = viser.ViserServer(host=config.host, port=config.port)

        # Load URDF and create robot visualization
        robot_root = self._server.scene.add_frame("/robot", show_axes=False)
        urdf_yourdfpy = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
        self._viser_robot = ViserUrdf(self._server, urdf_or_path=urdf_yourdfpy, root_node_name="/robot")
        self._robot_root = robot_root

        # Ground grid
        if config.show_grid:
            self._server.scene.add_grid(
                "/grid",
                width=config.grid_width,
                height=config.grid_height,
                position=(0.0, 0.0, 0.0),
            )

        # Build joint name mapping (simulator DOF index → ViserUrdf joint index)
        self._joint_mapping = self._build_joint_mapping()

        # ViserUrdf joint count
        joint_limits = self._viser_robot.get_actuated_joint_limits()
        self._viser_dof = len(joint_limits)

        logger.info(
            f"ViserBridge: ready at http://{config.host}:{config.port} "
            f"(mapped {len(self._joint_mapping)}/{simulator.num_dofs} DOFs)"
        )

    def _resolve_urdf_path(self) -> str:
        """Resolve robot URDF path from simulator's robot config."""
        from holosoma.utils.module_utils import get_holosoma_root

        asset_root = self._simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        urdf_file = self._simulator.robot_config.asset.urdf_file
        return os.path.join(asset_root, urdf_file)

    def _build_joint_mapping(self) -> dict[int, int]:
        """Map simulator DOF indices to ViserUrdf joint indices.

        Handles prefix differences (e.g., 'robot_left_hip_roll' → 'left_hip_roll').
        """
        joint_limits = self._viser_robot.get_actuated_joint_limits()
        viser_joint_names = [name for name, _ in joint_limits]

        sim_to_viser: dict[int, int] = {}
        unmapped = []

        for sim_idx, sim_name in enumerate(self._simulator.dof_names):
            # Try exact match
            if sim_name in viser_joint_names:
                sim_to_viser[sim_idx] = viser_joint_names.index(sim_name)
                continue

            # Try stripping common prefixes
            matched = False
            for prefix in ["robot_", "Robot_"]:
                stripped = sim_name.removeprefix(prefix)
                if stripped != sim_name and stripped in viser_joint_names:
                    sim_to_viser[sim_idx] = viser_joint_names.index(stripped)
                    matched = True
                    break

            if not matched:
                unmapped.append(sim_name)

        if unmapped:
            logger.warning(f"ViserBridge: {len(unmapped)} unmapped DOFs: {unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")

        return sim_to_viser

    def update(self) -> None:
        """Push current simulator state to viser viewer.

        Called each control step. Respects update_freq decimation and fps_limit.
        """
        self._step_count += 1
        if self._step_count % self._config.update_freq != 0:
            return

        now = time.monotonic()
        if (now - self._last_update_time) < self._min_update_interval:
            return
        self._last_update_time = now

        sim = self._simulator
        import numpy as np

        # Read state for env 0 (primary visualization target)
        # dof_pos: [num_envs, num_dof] → [num_dof]
        dof_pos = sim.dof_pos[0].detach().cpu().numpy()

        # Root body transform: _rigid_body_pos [num_envs, num_bodies, 3]
        #                      _rigid_body_rot [num_envs, num_bodies, 4] (xyzw)
        root_pos = sim._rigid_body_pos[0, 0].detach().cpu().numpy()  # [3]
        root_rot_xyzw = sim._rigid_body_rot[0, 0].detach().cpu().numpy()  # [4] xyzw

        # Convert xyzw → wxyz for viser
        root_rot_wxyz = np.array([
            root_rot_xyzw[3], root_rot_xyzw[0], root_rot_xyzw[1], root_rot_xyzw[2]
        ])

        # Build ViserUrdf joint config array
        joint_cfg = np.zeros(self._viser_dof)
        for sim_idx, viser_idx in self._joint_mapping.items():
            joint_cfg[viser_idx] = dof_pos[sim_idx]

        # Apply updates atomically to avoid flickering
        with self._server.atomic():
            self._robot_root.position = root_pos
            self._robot_root.wxyz = root_rot_wxyz
            self._viser_robot.update_cfg(joint_cfg)

    def cleanup(self) -> None:
        """Shut down the viser server."""
        try:
            self._server.close()
            logger.info("ViserBridge: server closed")
        except Exception as e:
            logger.warning(f"ViserBridge: error during cleanup: {e}")

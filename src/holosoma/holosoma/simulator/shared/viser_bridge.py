"""ViserBridge: simulator-agnostic bridge to mjviser web-based 3D viewer.

Uses mjviser's ``ViserMujocoScene`` for all backends:
  - **MuJoCo**: Direct passthrough of ``root_model``/``root_data``.
  - **IsaacSim / IsaacGym**: Shadow MuJoCo model loaded from robot MJCF.
    Each frame, ``robot_root_states`` and ``dof_pos`` are copied into
    ``mj_data.qpos``, ``mj_forward`` computes body transforms, and
    mjviser renders them.

Requires: ``pip install mjviser``
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import mujoco

    from holosoma.config_types.viser import ViserBridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class ViserBridge:
    """Unified viser bridge using mjviser for all simulator backends.

    Parameters
    ----------
    simulator : BaseSimulator
        The simulator instance.
    config : ViserBridgeConfig
        Viser bridge configuration.
    mj_model : mujoco.MjModel | None
        If provided (MuJoCo backend), use directly. Otherwise, load shadow model from MJCF.
    mj_data : mujoco.MjData | None
        If provided (MuJoCo backend), use directly. Otherwise, create from shadow model.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        config: ViserBridgeConfig,
        mj_model: mujoco.MjModel | None = None,
        mj_data: mujoco.MjData | None = None,
    ) -> None:
        import logging

        import mujoco as mj
        import viser  # type: ignore[import-not-found]
        from mjviser import ViserMujocoScene  # type: ignore[import-not-found]

        # Suppress noisy DEBUG logs
        for noisy in ("websockets", "websockets.server", "trimesh", "trimesh.util"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._simulator = simulator
        self._config = config
        self._step_count = 0
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / max(config.fps_limit, 1)

        # Determine mode: direct (MuJoCo) or shadow (IsaacSim/IsaacGym)
        if mj_model is not None and mj_data is not None:
            self._mode = "direct"
            self._mj_model = mj_model
            self._mj_data = mj_data
            self._shadow_qpos_root_addr = None
            self._shadow_dof_addrs: list[int] = []
            logger.info("ViserBridge: direct mode (MuJoCo backend)")
        else:
            self._mode = "shadow"
            self._mj_model, self._mj_data = self._create_shadow_model()
            self._resolve_shadow_addressing()
            logger.info(
                f"ViserBridge: shadow mode (FK-only, {self._mj_model.nq} qpos, "
                f"{len(self._shadow_dof_addrs)} mapped DOFs)"
            )

        # Create viser server and mjviser scene
        self._server = viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        # Full GUI: Scene (camera tracking, env selector) + Overlay + Groups
        self._scene.create_visualization_gui()

        logger.info(f"ViserBridge: ready at http://{config.host}:{config.port}")

    # ================================================================
    # Shadow model (for non-MuJoCo backends)
    # ================================================================

    def _create_shadow_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load robot MJCF into a standalone MuJoCo model for FK-only rendering."""
        import mujoco as mj

        mjcf_path = self._resolve_mjcf_path()
        logger.info(f"ViserBridge: loading shadow model from {mjcf_path}")
        model = mj.MjModel.from_xml_path(mjcf_path)
        data = mj.MjData(model)
        return model, data

    def _resolve_mjcf_path(self) -> str:
        """Resolve robot MJCF/XML path from simulator's robot config."""
        from holosoma.utils.module_utils import get_holosoma_root

        asset_root = self._simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return os.path.join(asset_root, self._simulator.robot_config.asset.xml_file)

    def _resolve_shadow_addressing(self) -> None:
        """Build qpos address mapping between simulator DOFs and shadow model joints.

        Shadow model qpos layout (freejoint robot):
        [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
                                    ^-- joint qpos starts at index 7
        """
        import mujoco as mj

        model = self._mj_model

        # Find root freejoint qpos address
        # The first joint of the robot body is typically a freejoint
        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            self._shadow_qpos_root_addr = model.jnt_qposadr[0]
        else:
            self._shadow_qpos_root_addr = None
            logger.warning("ViserBridge: shadow model has no freejoint, root pose won't update")

        # Map simulator DOF names → shadow model joint qpos addresses
        sim_dof_names = self._simulator.dof_names

        # Build shadow model joint name → qpos address mapping
        shadow_joint_map: dict[str, int] = {}
        for jnt_id in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name and model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_FREE:
                shadow_joint_map[jnt_name] = model.jnt_qposadr[jnt_id]

        self._shadow_dof_addrs = []
        for sim_name in sim_dof_names:
            # Try exact match
            if sim_name in shadow_joint_map:
                self._shadow_dof_addrs.append(shadow_joint_map[sim_name])
                continue
            # Try stripping common prefixes (MuJoCo scene_manager adds "robot_")
            matched = False
            for prefix in ["robot_", "Robot_"]:
                stripped = sim_name.removeprefix(prefix)
                if stripped != sim_name and stripped in shadow_joint_map:
                    self._shadow_dof_addrs.append(shadow_joint_map[stripped])
                    matched = True
                    break
            if not matched:
                # Unmapped DOF — use -1 as sentinel
                self._shadow_dof_addrs.append(-1)
                logger.debug(f"ViserBridge: unmapped shadow DOF: {sim_name}")

    def _sync_shadow_state(self) -> None:
        """Copy simulator state into shadow mj_data and run forward kinematics."""
        import mujoco as mj

        sim = self._simulator
        data = self._mj_data

        # Root state: [num_envs, 13] = [x,y,z, qx,qy,qz,qw, ...]
        root_state = sim.robot_root_states[0].detach().cpu().numpy()

        if self._shadow_qpos_root_addr is not None:
            addr = self._shadow_qpos_root_addr
            # Position
            data.qpos[addr : addr + 3] = root_state[:3]
            # Quaternion: holosoma xyzw → MuJoCo qpos wxyz
            qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]
            data.qpos[addr + 3] = qw
            data.qpos[addr + 4] = qx
            data.qpos[addr + 5] = qy
            data.qpos[addr + 6] = qz

        # Joint positions
        dof_pos = sim.dof_pos[0].detach().cpu().numpy()
        for sim_idx, qpos_addr in enumerate(self._shadow_dof_addrs):
            if qpos_addr >= 0:
                data.qpos[qpos_addr] = dof_pos[sim_idx]

        # Forward kinematics only (no physics)
        mj.mj_forward(self._mj_model, data)

    # ================================================================
    # Update loop
    # ================================================================

    def update(self) -> None:
        """Push current simulator state to mjviser.

        Fast path (early return) is integer ops only.
        """
        self._step_count += 1
        if self._step_count % self._config.update_freq != 0:
            return

        if self._scene.paused:
            return

        now = time.monotonic()
        if (now - self._last_update_time) < self._min_update_interval:
            return
        self._last_update_time = now

        self._push_state()

    def _push_state(self) -> None:
        """Transfer simulator state to mjviser."""
        if self._mode == "shadow":
            self._sync_shadow_state()

        self._scene.update_from_mjdata(self._mj_data)

    # ================================================================
    # Lifecycle
    # ================================================================

    def cleanup(self) -> None:
        """Shut down the viser server."""
        try:
            self._server.close()
            logger.info("ViserBridge: server closed")
        except Exception as e:
            logger.warning(f"ViserBridge: error during cleanup: {e}")

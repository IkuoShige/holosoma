"""ViserBridge: simulator-agnostic bridge to viser web-based 3D viewer.

Streams per-body transforms from any holosoma simulator backend to a
browser-accessible 3D viewer. Loads robot URDF via ``ViserUrdf`` and
updates joint configurations each frame.

Features:
  - Camera auto-tracking (mjviser-style scene offset)
  - Velocity command arrows
  - GUI controls (camera tracking, velocity display, play/pause)

Requires optional dependencies: ``pip install viser yourdfpy``
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from holosoma.config_types.viser import ViserBridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


def _quat_xyzw_to_rotation_matrix_2d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Extract 2x2 rotation matrix (XY plane) from xyzw quaternion."""
    # R_00 = 1 - 2*(qy^2 + qz^2), R_01 = 2*(qx*qy - qz*qw)
    # R_10 = 2*(qx*qy + qz*qw), R_11 = 1 - 2*(qx^2 + qz^2)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz)],
    ])


class ViserBridge:
    """Bridge between holosoma simulator and viser web viewer."""

    def __init__(self, simulator: BaseSimulator, config: ViserBridgeConfig) -> None:
        import logging

        import viser  # type: ignore[import-not-found]
        import yourdfpy  # type: ignore[import-untyped]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]

        # Suppress noisy DEBUG logs from viser dependencies
        for noisy_logger in ("websockets", "websockets.server", "trimesh", "trimesh.util"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        self._simulator = simulator
        self._config = config
        self._step_count = 0
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / max(config.fps_limit, 1)

        # State
        self._camera_tracking = True
        self._scene_offset = np.zeros(3)
        self._is_playing = True

        # Resolve URDF path
        urdf_path = self._resolve_urdf_path()
        logger.info(f"ViserBridge: loading URDF from {urdf_path}")

        # Create viser server
        self._server = viser.ViserServer(host=config.host, port=config.port)

        # Scene root frame (shifted for camera tracking)
        self._scene_root = self._server.scene.add_frame("/scene", show_axes=False)

        # Load URDF and create robot visualization under scene root
        self._robot_root = self._server.scene.add_frame("/scene/robot", show_axes=False)
        urdf_yourdfpy = yourdfpy.URDF.load(urdf_path, load_meshes=True, build_scene_graph=True)
        self._viser_robot = ViserUrdf(
            self._server, urdf_or_path=urdf_yourdfpy, root_node_name="/scene/robot"
        )

        # Ground grid (under scene root so it tracks with camera)
        self._grid_handle = None
        if config.show_grid:
            self._grid_handle = self._server.scene.add_grid(
                "/scene/grid",
                width=config.grid_width,
                height=config.grid_height,
                position=(0.0, 0.0, 0.0),
            )

        # Velocity command arrow
        self._velocity_handle = self._server.scene.add_line_segments(
            "/scene/velocity_arrow",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0.2, 0.4, 1.0], [0.2, 0.4, 1.0]]], dtype=np.float32),
            line_width=4.0,
        )
        self._velocity_scale = 1.0
        self._show_velocity = True

        # ViserUrdf actuated joint names (dict: name → (lower, upper))
        joint_limits = self._viser_robot.get_actuated_joint_limits()
        self._viser_joint_names = list(joint_limits.keys())
        self._viser_dof = len(self._viser_joint_names)

        # Build joint name mapping (simulator DOF index → ViserUrdf joint index)
        self._joint_mapping = self._build_joint_mapping()

        # Setup GUI controls
        self._setup_gui()

        # Set initial camera position (behind and above)
        az, el, dist = math.radians(180), math.radians(25), 3.0
        cam_offset = np.array([
            -math.cos(el) * math.cos(az),
            -math.cos(el) * math.sin(az),
            math.sin(el),
        ]) * dist
        for client in self._server.get_clients().values():
            client.camera.position = cam_offset
            client.camera.look_at = np.zeros(3)

        # Also set for future connecting clients
        @self._server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = cam_offset
            client.camera.look_at = np.zeros(3)

        logger.info(
            f"ViserBridge: ready at http://{config.host}:{config.port} "
            f"(mapped {len(self._joint_mapping)}/{simulator.num_dof} DOFs)"
        )

    # ================================================================
    # Setup
    # ================================================================

    def _resolve_urdf_path(self) -> str:
        from holosoma.utils.module_utils import get_holosoma_root

        asset_root = self._simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return os.path.join(asset_root, self._simulator.robot_config.asset.urdf_file)

    def _build_joint_mapping(self) -> dict[int, int]:
        """Map simulator DOF indices to ViserUrdf joint indices."""
        viser_joint_names = self._viser_joint_names
        sim_to_viser: dict[int, int] = {}
        unmapped = []

        for sim_idx, sim_name in enumerate(self._simulator.dof_names):
            if sim_name in viser_joint_names:
                sim_to_viser[sim_idx] = viser_joint_names.index(sim_name)
                continue
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
            logger.warning(
                f"ViserBridge: {len(unmapped)} unmapped DOFs: "
                f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}"
            )
        return sim_to_viser

    def _setup_gui(self) -> None:
        """Create GUI controls in the viser sidebar."""
        import viser  # type: ignore[import-not-found]

        server = self._server

        # --- Camera ---
        with server.gui.add_folder("Camera"):
            cb_track = server.gui.add_checkbox("Track robot", initial_value=True)

            @cb_track.on_update
            def _(_: viser.GuiEvent) -> None:
                self._camera_tracking = cb_track.value
                if cb_track.value:
                    # Reset camera for all clients
                    for client in server.get_clients().values():
                        client.camera.look_at = np.zeros(3)

        # --- Velocity ---
        with server.gui.add_folder("Velocity Command"):
            cb_vel = server.gui.add_checkbox("Show arrows", initial_value=True)
            sl_scale = server.gui.add_slider(
                "Arrow scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
            )

            @cb_vel.on_update
            def _(_: viser.GuiEvent) -> None:
                self._show_velocity = cb_vel.value
                self._velocity_handle.visible = cb_vel.value

            @sl_scale.on_update
            def _(_: viser.GuiEvent) -> None:
                self._velocity_scale = sl_scale.value

        # --- Playback ---
        with server.gui.add_folder("Playback"):
            btn_pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
            btn_play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY, visible=False)

            @btn_pause.on_click
            def _(_: viser.GuiEvent) -> None:
                self._is_playing = False
                btn_pause.visible = False
                btn_play.visible = True

            @btn_play.on_click
            def _(_: viser.GuiEvent) -> None:
                self._is_playing = True
                btn_play.visible = False
                btn_pause.visible = True

    # ================================================================
    # Update loop
    # ================================================================

    def update(self) -> None:
        """Push current simulator state to viser viewer.

        Fast path (early return) is just integer ops — no syscalls, no tensor ops.
        """
        self._step_count += 1
        if self._step_count % self._config.update_freq != 0:
            return

        if not self._is_playing:
            return

        now = time.monotonic()
        if (now - self._last_update_time) < self._min_update_interval:
            return
        self._last_update_time = now

        self._push_state()

    def _push_state(self) -> None:
        """Transfer simulator state to viser (GPU→CPU sync happens here)."""
        sim = self._simulator

        # dof_pos: [num_envs, num_dof] → [num_dof]
        dof_pos = sim.dof_pos[0].detach().cpu().numpy()

        # Robot root state: [num_envs, 13] = [x,y,z, qx,qy,qz,qw, ...]
        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        root_pos = root_state[:3].copy()
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]

        # Camera tracking: offset scene so robot stays near origin
        if self._camera_tracking:
            # Track XY only; keep Z grounded so camera doesn't bob vertically
            self._scene_offset = np.array([-root_pos[0], -root_pos[1], 0.0])

        # Compute robot position in scene-offset space
        scene_root_pos = root_pos + self._scene_offset

        # Build ViserUrdf joint config array
        joint_cfg = np.zeros(self._viser_dof)
        for sim_idx, viser_idx in self._joint_mapping.items():
            joint_cfg[viser_idx] = dof_pos[sim_idx]

        # Velocity command arrow
        vel_points = self._compute_velocity_arrow(root_state, scene_root_pos)

        # Apply updates atomically to avoid flickering
        with self._server.atomic():
            # Move grid to scene offset (ground tracks robot)
            if self._grid_handle is not None:
                self._grid_handle.position = self._scene_offset

            # Robot pose
            self._robot_root.position = scene_root_pos
            self._robot_root.wxyz = (qw, qx, qy, qz)
            self._viser_robot.update_cfg(joint_cfg)

            # Velocity arrow
            if self._show_velocity and vel_points is not None:
                self._velocity_handle.points = vel_points

    def _compute_velocity_arrow(
        self, root_state: np.ndarray, scene_root_pos: np.ndarray
    ) -> np.ndarray | None:
        """Compute velocity command arrow in world frame."""
        sim = self._simulator

        # Try to read velocity command from simulator's command tensor
        # command layout: [vx, vy, yaw_rate, ...]
        if not hasattr(sim, "commands") or sim.commands is None:
            return None

        try:
            cmd = sim.commands[0].detach().cpu().numpy()  # [cmd_dim]
        except (IndexError, AttributeError):
            return None

        if len(cmd) < 2:
            return None

        vx_local, vy_local = cmd[0], cmd[1]

        # Skip tiny commands
        if abs(vx_local) < 0.01 and abs(vy_local) < 0.01:
            return np.zeros((1, 2, 3), dtype=np.float32)

        # Transform local velocity to world frame using root orientation
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]
        rot2d = _quat_xyzw_to_rotation_matrix_2d(qx, qy, qz, qw)
        vel_world = rot2d @ np.array([vx_local, vy_local]) * self._velocity_scale

        # Arrow: from robot base (slightly above ground) to velocity endpoint
        z_offset = 0.15
        start = np.array([scene_root_pos[0], scene_root_pos[1], z_offset])
        end = start + np.array([vel_world[0], vel_world[1], 0.0])

        return np.array([[start, end]], dtype=np.float32)

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

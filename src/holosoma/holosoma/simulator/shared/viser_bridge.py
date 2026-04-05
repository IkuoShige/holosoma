"""ViserBridge: simulator-agnostic bridge to mjviser web-based 3D viewer.

Always uses a **shadow MuJoCo model** (robot-only, no terrain) so the
visualization works identically for all backends (MuJoCo, IsaacSim,
IsaacGym).  Each frame the bridge copies ``robot_root_states`` and
``dof_pos`` into the shadow ``mj_data.qpos``, runs ``mj_forward`` for
FK, and lets mjviser render the result.

Requires: ``pip install mjviser``
"""

from __future__ import annotations

import math
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
    """Unified viser bridge using mjviser for all simulator backends."""

    def __init__(
        self,
        simulator: BaseSimulator,
        config: ViserBridgeConfig,
        **_kwargs,
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
        self._paused = False

        # Always use shadow model (robot-only MJCF, no terrain)
        self._mj_model, self._mj_data = self._create_shadow_model()
        self._resolve_shadow_addressing()
        logger.info(
            f"ViserBridge: shadow model loaded ({self._mj_model.nq} qpos, "
            f"{len([a for a in self._shadow_dof_addrs if a >= 0])} mapped DOFs)"
        )

        # Create viser server + mjviser scene
        self._server = viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        # mjviser GUI (Scene + Overlay + Groups in tabs)
        tab_group = self._scene.create_visualization_gui()

        # Add holosoma-specific controls in a new tab
        self._setup_holosoma_gui(tab_group)

        # Velocity command arrow
        self._velocity_handle = self._server.scene.add_line_segments(
            "/velocity_cmd",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0.1, 0.4, 1.0], [0.1, 0.4, 1.0]]], dtype=np.float32),
            line_width=5.0,
        )
        self._velocity_actual_handle = self._server.scene.add_line_segments(
            "/velocity_actual",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0.0, 0.8, 0.4], [0.0, 0.8, 0.4]]], dtype=np.float32),
            line_width=4.0,
        )
        self._show_velocity = True
        self._velocity_scale = 1.0

        # Set initial camera for connecting clients
        self._camera_offset = self._compute_camera_offset(
            azimuth=150.0, elevation=25.0, distance=3.0
        )

        @self._server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = self._camera_offset
            client.camera.look_at = np.zeros(3)
            client.camera.fov = 60.0

        logger.info(f"ViserBridge: ready at http://{config.host}:{config.port}")

    # ================================================================
    # Shadow model
    # ================================================================

    def _create_shadow_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load robot MJCF into a standalone MuJoCo model for FK-only rendering.

        Strips ``<contact>`` section from XML before loading, since it
        references external geoms (e.g. 'floor') absent in the shadow model.
        """
        import mujoco as mj
        from xml.etree import ElementTree as ET

        mjcf_path = self._resolve_mjcf_path()
        logger.info(f"ViserBridge: loading shadow model from {mjcf_path}")

        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Remove <contact> section (references floor/ground geoms)
        for contact_elem in root.findall("contact"):
            root.remove(contact_elem)

        xml_string = ET.tostring(root, encoding="unicode")

        # MuJoCo needs the asset path for mesh files
        import os
        assets_dir = os.path.dirname(mjcf_path)
        model = mj.MjModel.from_xml_string(xml_string, assets=self._load_mesh_assets(assets_dir, root))
        data = mj.MjData(model)
        return model, data

    @staticmethod
    def _load_mesh_assets(assets_dir: str, xml_root) -> dict[str, bytes]:
        """Load mesh files referenced in MJCF as a dict for from_xml_string."""
        import os
        assets: dict[str, bytes] = {}

        # Find mesh dir from <compiler meshdir="..."/>
        mesh_dir = assets_dir
        for compiler in xml_root.findall("compiler"):
            meshdir = compiler.get("meshdir")
            if meshdir:
                mesh_dir = os.path.join(assets_dir, meshdir)
                break

        # Load all referenced mesh files
        for mesh in xml_root.iter("mesh"):
            mesh_file = mesh.get("file")
            if mesh_file:
                full_path = os.path.join(mesh_dir, mesh_file)
                if os.path.isfile(full_path):
                    assets[mesh_file] = open(full_path, "rb").read()

        return assets

    def _resolve_mjcf_path(self) -> str:
        from holosoma.utils.module_utils import get_holosoma_root

        asset_root = self._simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return os.path.join(asset_root, self._simulator.robot_config.asset.xml_file)

    def _resolve_shadow_addressing(self) -> None:
        """Build qpos address mapping between simulator DOFs and shadow model joints."""
        import mujoco as mj

        model = self._mj_model

        # Root freejoint address
        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            self._shadow_qpos_root_addr: int | None = model.jnt_qposadr[0]
        else:
            self._shadow_qpos_root_addr = None
            logger.warning("ViserBridge: shadow model has no freejoint")

        # Build shadow joint name → qpos address mapping
        shadow_joint_map: dict[str, int] = {}
        for jnt_id in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name and model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_FREE:
                shadow_joint_map[jnt_name] = model.jnt_qposadr[jnt_id]

        # Map simulator DOFs → shadow qpos addresses
        self._shadow_dof_addrs: list[int] = []
        unmapped = []
        for sim_name in self._simulator.dof_names:
            addr = shadow_joint_map.get(sim_name)
            if addr is None:
                for prefix in ["robot_", "Robot_"]:
                    stripped = sim_name.removeprefix(prefix)
                    if stripped != sim_name:
                        addr = shadow_joint_map.get(stripped)
                        if addr is not None:
                            break
            if addr is not None:
                self._shadow_dof_addrs.append(addr)
            else:
                self._shadow_dof_addrs.append(-1)
                unmapped.append(sim_name)

        if unmapped:
            logger.warning(f"ViserBridge: {len(unmapped)} unmapped DOFs: {unmapped[:5]}")

    def _sync_shadow_state(self) -> None:
        """Copy simulator state into shadow mj_data and run forward kinematics."""
        import mujoco as mj

        sim = self._simulator
        data = self._mj_data

        # Root state: [x,y,z, qx,qy,qz,qw, ...]
        root_state = sim.robot_root_states[0].detach().cpu().numpy()

        if self._shadow_qpos_root_addr is not None:
            a = self._shadow_qpos_root_addr
            data.qpos[a : a + 3] = root_state[:3]  # xyz
            # holosoma xyzw → MuJoCo qpos wxyz
            data.qpos[a + 3] = root_state[6]  # qw
            data.qpos[a + 4] = root_state[3]  # qx
            data.qpos[a + 5] = root_state[4]  # qy
            data.qpos[a + 6] = root_state[5]  # qz

        # Joint positions
        dof_pos = sim.dof_pos[0].detach().cpu().numpy()
        for sim_idx, qpos_addr in enumerate(self._shadow_dof_addrs):
            if qpos_addr >= 0:
                data.qpos[qpos_addr] = dof_pos[sim_idx]

        # FK only (no physics step)
        mj.mj_forward(self._mj_model, data)

    # ================================================================
    # GUI
    # ================================================================

    def _setup_holosoma_gui(self, tab_group) -> None:
        """Add holosoma-specific controls as an additional tab."""
        import viser  # type: ignore[import-not-found]

        server = self._server

        with tab_group.add_tab("Controls"):
            # --- Playback ---
            with server.gui.add_folder("Playback", expand_by_default=True):
                btn_pause = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
                btn_play = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY, visible=False)

                @btn_pause.on_click
                def _(_: viser.GuiEvent) -> None:
                    self._paused = True
                    btn_pause.visible = False
                    btn_play.visible = True

                @btn_play.on_click
                def _(_: viser.GuiEvent) -> None:
                    self._paused = False
                    btn_play.visible = False
                    btn_pause.visible = True

            # --- Velocity Command ---
            with server.gui.add_folder("Velocity Command", expand_by_default=True):
                cb_vel = server.gui.add_checkbox("Show arrows", initial_value=True)
                sl_scale = server.gui.add_slider(
                    "Arrow scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
                )

                @cb_vel.on_update
                def _(_: viser.GuiEvent) -> None:
                    self._show_velocity = cb_vel.value
                    self._velocity_handle.visible = cb_vel.value
                    self._velocity_actual_handle.visible = cb_vel.value

                @sl_scale.on_update
                def _(_: viser.GuiEvent) -> None:
                    self._velocity_scale = sl_scale.value

            # --- Info ---
            with server.gui.add_folder("Info"):
                sim_name = type(self._simulator).__name__
                num_dof = self._simulator.num_dof
                mapped = len([a for a in self._shadow_dof_addrs if a >= 0])
                server.gui.add_markdown(
                    f"**Simulator:** {sim_name}\n\n"
                    f"**DOFs:** {mapped}/{num_dof} mapped\n\n"
                    f"**Mode:** shadow FK"
                )

    @staticmethod
    def _compute_camera_offset(azimuth: float, elevation: float, distance: float) -> np.ndarray:
        az = math.radians(azimuth)
        el = math.radians(elevation)
        return np.array([
            -math.cos(el) * math.cos(az),
            -math.cos(el) * math.sin(az),
            math.sin(el),
        ]) * distance

    # ================================================================
    # Update loop
    # ================================================================

    def update(self) -> None:
        """Push current simulator state to mjviser. Fast path is integer ops only."""
        self._step_count += 1
        if self._step_count % self._config.update_freq != 0:
            return
        if self._paused:
            return
        now = time.monotonic()
        if (now - self._last_update_time) < self._min_update_interval:
            return
        self._last_update_time = now

        self._sync_shadow_state()
        self._scene.update_from_mjdata(self._mj_data)
        self._update_velocity_arrows()

    def _update_velocity_arrows(self) -> None:
        """Draw velocity command and actual velocity arrows from robot base."""
        if not self._show_velocity:
            return

        sim = self._simulator
        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        root_pos = root_state[:3]
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]

        # 2D rotation matrix from quaternion (XY plane)
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz)],
        ])

        # Scene offset from camera tracking
        scene_offset = np.zeros(3)
        if self._scene.camera_tracking_enabled:
            tracked = self._mj_data.xpos[self._scene._tracked_body_id] if self._scene._tracked_body_id is not None else root_pos
            scene_offset = -tracked.copy()
            scene_offset[2] = 0.0

        base = root_pos + scene_offset
        z_offset = 0.15
        start = np.array([base[0], base[1], z_offset])

        # --- Command velocity arrow (blue) ---
        cmd_points = np.zeros((1, 2, 3), dtype=np.float32)
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                cmd = sim.commands[0].detach().cpu().numpy()
                if len(cmd) >= 2:
                    vx, vy = cmd[0], cmd[1]
                    vel_world = R @ np.array([vx, vy]) * self._velocity_scale
                    cmd_points[0, 0] = start
                    cmd_points[0, 1] = start + np.array([vel_world[0], vel_world[1], 0.0])
            except (IndexError, AttributeError):
                pass
        self._velocity_handle.points = cmd_points

        # --- Actual velocity arrow (green) ---
        actual_points = np.zeros((1, 2, 3), dtype=np.float32)
        vx_actual = root_state[7]  # linear vel x (world frame)
        vy_actual = root_state[8]  # linear vel y (world frame)
        if abs(vx_actual) > 0.01 or abs(vy_actual) > 0.01:
            actual_points[0, 0] = start + np.array([0, 0, 0.05])
            actual_points[0, 1] = actual_points[0, 0] + np.array([
                vx_actual * self._velocity_scale,
                vy_actual * self._velocity_scale,
                0.0,
            ])
        self._velocity_actual_handle.points = actual_points

    # ================================================================
    # Lifecycle
    # ================================================================

    def cleanup(self) -> None:
        try:
            self._server.close()
            logger.info("ViserBridge: server closed")
        except Exception as e:
            logger.warning(f"ViserBridge: error during cleanup: {e}")

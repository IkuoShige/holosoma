"""ViserBridge: simulator-agnostic bridge to mjviser web-based 3D viewer.

Always uses a **shadow MuJoCo model** (robot-only, no terrain) so the
visualization works identically for all backends (MuJoCo, IsaacSim,
IsaacGym).  Each frame the bridge copies ``robot_root_states`` and
``dof_pos`` into the shadow ``mj_data.qpos``, runs ``mj_forward`` for
FK, and lets mjviser render the result.

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

        # Shadow model (robot-only MJCF, no terrain)
        self._mj_model, self._mj_data = self._create_shadow_model()
        self._resolve_shadow_addressing()

        mapped = len([a for a in self._shadow_dof_addrs if a >= 0])
        logger.info(f"ViserBridge: shadow model ({mapped}/{simulator.num_dof} DOFs mapped)")

        # Create viser server + mjviser scene
        self._server = viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        # mjviser full GUI — handles camera, overlays, groups, FOV, tracking
        # This sets up on_client_connect, camera position, FOV, etc.
        tab_group = self._scene.create_visualization_gui(
            camera_distance=3.0,
            camera_azimuth=150.0,
            camera_elevation=25.0,
        )

        # Add holosoma-specific controls tab
        self._show_velocity = True
        self._velocity_scale = 1.0
        self._setup_controls_tab(tab_group)

        # Velocity command arrows
        self._velocity_cmd_handle = self._server.scene.add_line_segments(
            "/velocity_cmd",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0.2, 0.3, 1.0], [0.2, 0.3, 1.0]]], dtype=np.float32),
            line_width=5.0,
        )
        self._velocity_actual_handle = self._server.scene.add_line_segments(
            "/velocity_actual",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0.0, 0.8, 0.3], [0.0, 0.8, 0.3]]], dtype=np.float32),
            line_width=4.0,
        )

        logger.info(f"ViserBridge: ready at http://{config.host}:{config.port}")

    # ================================================================
    # Shadow model
    # ================================================================

    def _create_shadow_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load robot MJCF with dummy floor so contact pairs compile."""
        import mujoco as mj
        from xml.etree import ElementTree as ET

        mjcf_path = self._resolve_mjcf_path()
        logger.info(f"ViserBridge: loading shadow model from {mjcf_path}")

        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Inject invisible dummy floor geom so contact pairs compile
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "geom", {
            "name": "floor",
            "type": "plane",
            "size": "10 10 0.01",
            "rgba": "0 0 0 0",
            "contype": "1",
            "conaffinity": "1",
        })

        # Write temp file in same dir so MuJoCo resolves mesh paths
        import tempfile
        mjcf_dir = os.path.dirname(mjcf_path)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=mjcf_dir, delete=False
        ) as tmp:
            tree.write(tmp, xml_declaration=True, encoding="utf-8")
            tmp_path = tmp.name

        try:
            model = mj.MjModel.from_xml_path(tmp_path)
        finally:
            os.unlink(tmp_path)

        data = mj.MjData(model)
        return model, data

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

        # Shadow joint name → qpos address
        shadow_joint_map: dict[str, int] = {}
        for jnt_id in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name and model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_FREE:
                shadow_joint_map[jnt_name] = model.jnt_qposadr[jnt_id]

        # Map simulator DOFs → shadow qpos addresses
        self._shadow_dof_addrs: list[int] = []
        for sim_name in self._simulator.dof_names:
            addr = shadow_joint_map.get(sim_name)
            if addr is None:
                for prefix in ["robot_", "Robot_"]:
                    stripped = sim_name.removeprefix(prefix)
                    if stripped != sim_name:
                        addr = shadow_joint_map.get(stripped)
                        if addr is not None:
                            break
            self._shadow_dof_addrs.append(addr if addr is not None else -1)

    def _sync_shadow_state(self) -> None:
        """Copy simulator state into shadow mj_data and run mj_forward."""
        import mujoco as mj

        sim = self._simulator
        data = self._mj_data

        root_state = sim.robot_root_states[0].detach().cpu().numpy()

        if self._shadow_qpos_root_addr is not None:
            a = self._shadow_qpos_root_addr
            data.qpos[a : a + 3] = root_state[:3]
            # holosoma xyzw → MuJoCo qpos wxyz
            data.qpos[a + 3] = root_state[6]  # qw
            data.qpos[a + 4] = root_state[3]  # qx
            data.qpos[a + 5] = root_state[4]  # qy
            data.qpos[a + 6] = root_state[5]  # qz

        dof_pos = sim.dof_pos[0].detach().cpu().numpy()
        for sim_idx, qpos_addr in enumerate(self._shadow_dof_addrs):
            if qpos_addr >= 0:
                data.qpos[qpos_addr] = dof_pos[sim_idx]

        mj.mj_forward(self._mj_model, data)

    # ================================================================
    # GUI (holosoma-specific controls)
    # ================================================================

    def _setup_controls_tab(self, tab_group) -> None:
        import viser  # type: ignore[import-not-found]

        server = self._server

        with tab_group.add_tab("Controls"):
            # Velocity arrows
            with server.gui.add_folder("Velocity Command", expand_by_default=True):
                cb_vel = server.gui.add_checkbox("Show arrows", initial_value=True)
                sl_scale = server.gui.add_slider(
                    "Arrow scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
                )

                @cb_vel.on_update
                def _(_: viser.GuiEvent) -> None:
                    self._show_velocity = cb_vel.value
                    self._velocity_cmd_handle.visible = cb_vel.value
                    self._velocity_actual_handle.visible = cb_vel.value

                @sl_scale.on_update
                def _(_: viser.GuiEvent) -> None:
                    self._velocity_scale = sl_scale.value

            # Info
            with server.gui.add_folder("Info"):
                sim_name = type(self._simulator).__name__
                mapped = len([a for a in self._shadow_dof_addrs if a >= 0])
                server.gui.add_markdown(
                    f"**Simulator:** {sim_name}\n\n"
                    f"**DOFs:** {mapped}/{self._simulator.num_dof}\n\n"
                    f"**FPS limit:** {self._config.fps_limit}"
                )

    # ================================================================
    # Update loop
    # ================================================================

    def update(self) -> None:
        self._step_count += 1
        if self._step_count % self._config.update_freq != 0:
            return
        if self._scene.paused:
            return
        now = time.monotonic()
        if (now - self._last_update_time) < self._min_update_interval:
            return
        self._last_update_time = now

        self._sync_shadow_state()

        with self._server.atomic():
            self._scene.update_from_mjdata(self._mj_data)
            self._update_velocity_arrows()

    def _update_velocity_arrows(self) -> None:
        if not self._show_velocity:
            return

        sim = self._simulator
        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        root_pos = root_state[:3]
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]

        # 2D rotation (XY plane) from quaternion
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz)],
        ])

        # Scene offset from mjviser camera tracking
        scene_offset = self._scene._scene_offset
        base = root_pos + scene_offset
        z_off = 0.15
        start = np.array([base[0], base[1], z_off])

        # Command velocity arrow (blue)
        cmd_pts = np.zeros((1, 2, 3), dtype=np.float32)
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                cmd = sim.commands[0].detach().cpu().numpy()
                if len(cmd) >= 2 and (abs(cmd[0]) > 0.01 or abs(cmd[1]) > 0.01):
                    vel_w = R @ np.array([cmd[0], cmd[1]]) * self._velocity_scale
                    cmd_pts[0, 0] = start
                    cmd_pts[0, 1] = start + np.array([vel_w[0], vel_w[1], 0.0])
            except (IndexError, AttributeError):
                pass
        self._velocity_cmd_handle.points = cmd_pts

        # Actual velocity arrow (green)
        act_pts = np.zeros((1, 2, 3), dtype=np.float32)
        vx, vy = root_state[7], root_state[8]
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            act_start = start + np.array([0, 0, 0.05])
            act_pts[0, 0] = act_start
            act_pts[0, 1] = act_start + np.array([
                vx * self._velocity_scale, vy * self._velocity_scale, 0.0
            ])
        self._velocity_actual_handle.points = act_pts

    # ================================================================
    # Lifecycle
    # ================================================================

    def cleanup(self) -> None:
        try:
            self._server.close()
            logger.info("ViserBridge: server closed")
        except Exception as e:
            logger.warning(f"ViserBridge: error during cleanup: {e}")

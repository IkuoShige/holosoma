"""ViserBridge: simulator-agnostic bridge to mjviser web-based 3D viewer.

Always uses a **shadow MuJoCo model** (robot-only) so the visualization
works identically for all backends.  Each frame copies simulator state
into shadow ``mj_data.qpos``, runs ``mj_forward``, and lets mjviser render.

Features:
  - Full mjviser GUI (Scene / Overlay / Groups tabs)
  - mjlab-style 3D velocity arrows (shaft + cone head) above robot head
  - Controls tab: play/pause, speed, velocity joystick, info

Requires: ``pip install mjviser``
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from loguru import logger

if TYPE_CHECKING:
    import mujoco
    import viser

    from holosoma.config_types.viser import ViserBridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# Arrow geometry constants (mjlab style)
_ARROW_SHAFT_RATIO = 0.8  # shaft = 80% of total length
_ARROW_HEAD_RATIO = 0.2  # head = 20%
_ARROW_WIDTH = 0.015
_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) rotating *src* onto *dst* (both unit vectors)."""
    c = np.dot(src, dst)
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-8:
        # 180-degree rotation around any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(src, dst)
    w = 1.0 + c
    q = np.array([w, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def _make_shaft_mesh() -> trimesh.Trimesh:
    """Unit cylinder with base at origin, extending +Z."""
    m = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
    m.apply_translation([0, 0, 0.5])
    return m


def _make_head_mesh() -> trimesh.Trimesh:
    """Unit cone with base at origin, pointing +Z."""
    m = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
    return m


class _Arrow3D:
    """A single 3D arrow (shaft cylinder + cone head) in the viser scene."""

    def __init__(self, server: viser.ViserServer, name: str, color: tuple[int, int, int]):
        shaft = _make_shaft_mesh()
        shaft.visual.face_colors = [(*color, 200)] * len(shaft.faces)
        head = _make_head_mesh()
        head.visual.face_colors = [(*color, 200)] * len(head.faces)
        self._shaft = server.scene.add_mesh_trimesh(f"{name}/shaft", shaft)
        self._head = server.scene.add_mesh_trimesh(f"{name}/head", head)
        self.visible = True

    def update(self, start: np.ndarray, end: np.ndarray, scene_offset: np.ndarray) -> None:
        s = start + scene_offset
        e = end + scene_offset
        d = e - s
        length = float(np.linalg.norm(d))

        if length < 1e-4:
            self._shaft.visible = False
            self._head.visible = False
            return

        self._shaft.visible = self.visible
        self._head.visible = self.visible
        if not self.visible:
            return

        direction = d / length
        q = _rotation_between(_Z_AXIS, direction)

        shaft_len = _ARROW_SHAFT_RATIO * length
        head_len = _ARROW_HEAD_RATIO * length
        w = _ARROW_WIDTH

        self._shaft.position = s
        self._shaft.wxyz = q
        self._shaft.scale = (w, w, shaft_len)

        self._head.position = s + direction * shaft_len
        self._head.wxyz = q
        self._head.scale = (w, w, head_len)

    def set_visible(self, v: bool) -> None:
        self.visible = v
        self._shaft.visible = v
        self._head.visible = v


class ViserBridge:
    """Unified viser bridge using mjviser for all simulator backends."""

    def __init__(self, simulator: BaseSimulator, config: ViserBridgeConfig, **_kw) -> None:
        import logging

        import mujoco as mj
        import viser as _viser
        from mjviser import ViserMujocoScene

        for noisy in ("websockets", "websockets.server", "trimesh", "trimesh.util"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._simulator = simulator
        self._config = config
        self._step_count = 0
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / max(config.fps_limit, 1)

        # Shadow model
        self._mj_model, self._mj_data = self._create_shadow_model()
        self._resolve_shadow_addressing()
        mapped = len([a for a in self._shadow_dof_addrs if a >= 0])
        logger.info(f"ViserBridge: shadow model ({mapped}/{simulator.num_dof} DOFs)")

        # Viser server + mjviser scene
        self._server = _viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        # mjviser GUI tabs
        tab_group = self._scene.create_visualization_gui(
            camera_distance=3.0, camera_azimuth=150.0, camera_elevation=25.0,
        )

        # 3D arrows (mjlab style: shaft + cone head)
        self._arrow_cmd_lin = _Arrow3D(self._server, "/arrows/cmd_lin", (50, 70, 230))
        self._arrow_cmd_ang = _Arrow3D(self._server, "/arrows/cmd_ang", (50, 150, 50))
        self._arrow_actual_lin = _Arrow3D(self._server, "/arrows/actual_lin", (0, 150, 255))
        self._arrow_actual_ang = _Arrow3D(self._server, "/arrows/actual_ang", (0, 230, 100))
        self._velocity_scale = 0.5
        self._show_velocity = True
        self._vel_z_offset = 0.2

        # Holosoma controls tab
        self._vel_joystick_enabled = False
        self._vel_joystick_vx = 0.0
        self._vel_joystick_vy = 0.0
        self._vel_joystick_yaw = 0.0
        self._setup_controls_tab(tab_group)

        logger.info(f"ViserBridge: http://{config.host}:{config.port}")

    # ================================================================
    # Shadow model
    # ================================================================

    def _create_shadow_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        import mujoco as mj
        from xml.etree import ElementTree as ET
        import tempfile

        mjcf_path = self._resolve_mjcf_path()
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "geom", {
            "name": "floor", "type": "plane", "size": "10 10 0.01",
            "rgba": "0 0 0 0", "contype": "1", "conaffinity": "1",
        })

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

        return model, mj.MjData(model)

    def _resolve_mjcf_path(self) -> str:
        from holosoma.utils.module_utils import get_holosoma_root
        asset_root = self._simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return os.path.join(asset_root, self._simulator.robot_config.asset.xml_file)

    def _resolve_shadow_addressing(self) -> None:
        import mujoco as mj
        model = self._mj_model

        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            self._shadow_qpos_root_addr: int | None = model.jnt_qposadr[0]
        else:
            self._shadow_qpos_root_addr = None

        shadow_joint_map: dict[str, int] = {}
        for jnt_id in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name and model.jnt_type[jnt_id] != mj.mjtJoint.mjJNT_FREE:
                shadow_joint_map[jnt_name] = model.jnt_qposadr[jnt_id]

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
        import mujoco as mj
        sim = self._simulator
        data = self._mj_data

        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        if self._shadow_qpos_root_addr is not None:
            a = self._shadow_qpos_root_addr
            data.qpos[a:a + 3] = root_state[:3]
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
    # GUI
    # ================================================================

    def _setup_controls_tab(self, tab_group) -> None:
        import viser as _viser

        server = self._server

        with tab_group.add_tab("Controls", viser_icon=_viser.Icon.SETTINGS):
            # --- Velocity arrows ---
            with server.gui.add_folder("Velocity Arrows", expand_by_default=True):
                cb_show = server.gui.add_checkbox("Show", initial_value=True)
                sl_scale = server.gui.add_slider(
                    "Scale", min=0.1, max=3.0, step=0.1, initial_value=0.5,
                )
                sl_z = server.gui.add_slider(
                    "Height offset", min=0.0, max=1.0, step=0.05, initial_value=0.2,
                )

                @cb_show.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._show_velocity = cb_show.value
                    for arr in (self._arrow_cmd_lin, self._arrow_cmd_ang,
                                self._arrow_actual_lin, self._arrow_actual_ang):
                        arr.set_visible(cb_show.value)

                @sl_scale.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._velocity_scale = sl_scale.value

                @sl_z.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_z_offset = sl_z.value

                # Legend
                server.gui.add_markdown(
                    "**Blue** = cmd linear &nbsp; **Green** = cmd angular\n\n"
                    "**Cyan** = actual linear &nbsp; **Lime** = actual angular"
                )

            # --- Velocity joystick ---
            with server.gui.add_folder("Velocity Joystick", expand_by_default=False):
                cb_joy = server.gui.add_checkbox("Enable override", initial_value=False)
                sl_vx = server.gui.add_slider("lin_vel_x", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                sl_vy = server.gui.add_slider("lin_vel_y", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                sl_yaw = server.gui.add_slider("ang_vel_z", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
                btn_zero = server.gui.add_button("Zero", icon=_viser.Icon.SQUARE_X)

                @cb_joy.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_joystick_enabled = cb_joy.value

                @sl_vx.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_joystick_vx = sl_vx.value

                @sl_vy.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_joystick_vy = sl_vy.value

                @sl_yaw.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_joystick_yaw = sl_yaw.value

                @btn_zero.on_click
                def _(_: _viser.GuiEvent) -> None:
                    sl_vx.value = 0.0
                    sl_vy.value = 0.0
                    sl_yaw.value = 0.0
                    self._vel_joystick_vx = 0.0
                    self._vel_joystick_vy = 0.0
                    self._vel_joystick_yaw = 0.0

            # --- Info ---
            with server.gui.add_folder("Info"):
                server.gui.add_markdown(
                    f"**Simulator:** {type(self._simulator).__name__}\n\n"
                    f"**DOFs:** {len([a for a in self._shadow_dof_addrs if a >= 0])}"
                    f"/{self._simulator.num_dof}\n\n"
                    f"**FPS limit:** {self._config.fps_limit}\n\n"
                    f"**Update freq:** every {self._config.update_freq} steps"
                )

    # ================================================================
    # Update
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

        # Apply velocity joystick override
        if self._vel_joystick_enabled:
            self._apply_joystick_override()

        self._sync_shadow_state()

        with self._server.atomic():
            self._scene.update_from_mjdata(self._mj_data)
            if self._show_velocity:
                self._update_velocity_arrows()

    def _apply_joystick_override(self) -> None:
        """Override simulator velocity commands with joystick values."""
        sim = self._simulator
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                sim.commands[0, 0] = self._vel_joystick_vx
                sim.commands[0, 1] = self._vel_joystick_vy
                if sim.commands.shape[1] > 2:
                    sim.commands[0, 2] = self._vel_joystick_yaw
            except (IndexError, AttributeError):
                pass

    def _update_velocity_arrows(self) -> None:
        """Draw 4 mjlab-style velocity arrows above robot head."""
        sim = self._simulator
        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        base_pos = root_state[:3]
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]

        # 3x3 rotation matrix from quaternion (xyzw)
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])

        scene_offset = self._scene._scene_offset
        scale = self._velocity_scale
        z_off = self._vel_z_offset

        def local_to_world(v: np.ndarray) -> np.ndarray:
            return base_pos + R @ (v * scale)

        origin = local_to_world(np.array([0, 0, z_off]))

        # Command velocity
        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, 0.0
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                cmd = sim.commands[0].detach().cpu().numpy()
                cmd_vx = cmd[0] if len(cmd) > 0 else 0.0
                cmd_vy = cmd[1] if len(cmd) > 1 else 0.0
                cmd_yaw = cmd[2] if len(cmd) > 2 else 0.0
            except (IndexError, AttributeError):
                pass

        # Actual velocity (world frame → body frame)
        lin_vel_w = root_state[7:10]
        ang_vel_w = root_state[10:13]
        lin_vel_b = R.T @ lin_vel_w
        ang_vel_b = R.T @ ang_vel_w

        # 4 arrows (mjlab convention)
        self._arrow_cmd_lin.update(
            origin,
            local_to_world(np.array([0, 0, z_off]) + np.array([cmd_vx, cmd_vy, 0])),
            scene_offset,
        )
        self._arrow_cmd_ang.update(
            origin,
            local_to_world(np.array([0, 0, z_off]) + np.array([0, 0, cmd_yaw])),
            scene_offset,
        )
        self._arrow_actual_lin.update(
            origin,
            local_to_world(np.array([0, 0, z_off]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])),
            scene_offset,
        )
        self._arrow_actual_ang.update(
            origin,
            local_to_world(np.array([0, 0, z_off]) + np.array([0, 0, ang_vel_b[2]])),
            scene_offset,
        )

    # ================================================================
    # Lifecycle
    # ================================================================

    @property
    def vel_joystick_enabled(self) -> bool:
        return self._vel_joystick_enabled

    def cleanup(self) -> None:
        try:
            self._server.close()
            logger.info("ViserBridge: server closed")
        except Exception as e:
            logger.warning(f"ViserBridge cleanup: {e}")

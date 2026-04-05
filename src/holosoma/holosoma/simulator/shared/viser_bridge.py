"""ViserBridge: simulator-agnostic bridge to mjviser web-based 3D viewer.

Uses a shadow MuJoCo model (robot-only) + mjviser for all backends.

GUI tabs (mjlab-style):
  - Scene / Visualization / Groups — from mjviser
  - Controls — play/pause, step info, speed, velocity arrows, joystick
  - Rewards — live reward time-series plots

Requires: ``pip install mjviser``
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from loguru import logger

if TYPE_CHECKING:
    import mujoco
    import viser

    from holosoma.config_types.viser import ViserBridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

_ARROW_SHAFT_RATIO = 0.8
_ARROW_HEAD_RATIO = 0.2
_ARROW_WIDTH = 0.015
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_REWARD_HISTORY_LEN = 200


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) rotating *src* onto *dst*."""
    c = np.dot(src, dst)
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-8:
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(src, dst)
    w = 1.0 + c
    q = np.array([w, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def _colored_mesh(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.face_colors = [rgba] * len(mesh.faces)
    return mesh


class _Arrow3D:
    """3D arrow (cylinder shaft + cone head) in the viser scene."""

    def __init__(self, server: viser.ViserServer, name: str, rgba: tuple[int, int, int, int]):
        shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=12)
        shaft.apply_translation([0, 0, 0.5])
        head = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
        self._shaft = server.scene.add_mesh_trimesh(f"{name}/shaft", _colored_mesh(shaft, rgba))
        self._head = server.scene.add_mesh_trimesh(f"{name}/head", _colored_mesh(head, rgba))
        self.visible = True

    def update(self, start: np.ndarray, end: np.ndarray, offset: np.ndarray) -> None:
        s, e = start + offset, end + offset
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
        w = _ARROW_WIDTH
        self._shaft.position = s
        self._shaft.wxyz = q
        self._shaft.scale = (w, w, _ARROW_SHAFT_RATIO * length)
        self._head.position = s + direction * _ARROW_SHAFT_RATIO * length
        self._head.wxyz = q
        self._head.scale = (w, w, _ARROW_HEAD_RATIO * length)

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
        self._total_steps = 0
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / max(config.fps_limit, 1)
        self._fps_counter_time = time.monotonic()
        self._fps_counter_frames = 0
        self._current_fps = 0.0

        # Shadow model
        self._mj_model, self._mj_data = self._create_shadow_model()
        self._resolve_shadow_addressing()
        self._mapped_dofs = len([a for a in self._shadow_dof_addrs if a >= 0])

        # Viser server + mjviser scene
        self._server = _viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, self._mj_model, num_envs=1)

        # Terrain mesh
        self._terrain_handle = None
        self._add_terrain()

        # 3D arrows
        self._arrow_cmd_lin = _Arrow3D(self._server, "/arrows/cmd_lin", (50, 70, 230, 200))
        self._arrow_cmd_ang = _Arrow3D(self._server, "/arrows/cmd_ang", (50, 150, 50, 200))
        self._arrow_actual_lin = _Arrow3D(self._server, "/arrows/actual_lin", (0, 150, 255, 180))
        self._arrow_actual_ang = _Arrow3D(self._server, "/arrows/actual_ang", (0, 230, 100, 180))
        self._velocity_scale = 0.5
        self._show_velocity = True
        self._vel_z_offset = 0.2

        # Velocity joystick state
        self._vel_joystick_enabled = False
        self._vel_joystick_vx = 0.0
        self._vel_joystick_vy = 0.0
        self._vel_joystick_yaw = 0.0

        # Speed control
        self._speed_multiplier = 1.0

        # Reward tracking
        self._reward_history: deque[float] = deque(maxlen=_REWARD_HISTORY_LEN)
        self._reward_timesteps: deque[float] = deque(maxlen=_REWARD_HISTORY_LEN)
        self._reward_plot_handle = None

        # GUI — Controls tab FIRST (mjlab convention), then mjviser tabs
        self._info_handle = None
        tab_group = self._server.gui.add_tab_group()
        self._setup_controls_tab(tab_group)
        self._setup_rewards_tab(tab_group)
        # mjviser Scene / Visualization / Groups tabs
        with tab_group.add_tab("Scene"):
            self._scene.create_scene_gui(
                camera_distance=3.0, camera_azimuth=150.0, camera_elevation=25.0,
            )
        with tab_group.add_tab("Visualization"):
            self._scene.create_overlay_gui()
        with tab_group.add_tab("Groups"):
            self._scene.create_groups_gui()
        self._setup_checkpoints_tab(tab_group)

        logger.info(f"ViserBridge: http://{config.host}:{config.port}")

    # ================================================================
    # Shadow model
    # ================================================================

    def _create_shadow_model(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        import mujoco as mj
        import tempfile
        from xml.etree import ElementTree as ET

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
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", dir=mjcf_dir, delete=False) as tmp:
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

    def _add_terrain(self) -> None:
        """Add terrain mesh from terrain_manager if available."""
        sim = self._simulator
        tm = getattr(sim, "terrain_manager", None)
        if tm is None:
            return

        mesh = getattr(tm, "_mesh", None)
        if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return

        # Downsample for large terrains (>100k faces) to keep viser responsive
        terrain_mesh = mesh.copy()
        if len(terrain_mesh.faces) > 100_000:
            terrain_mesh = terrain_mesh.simplify_quadric_decimation(100_000)

        # Color terrain green-brown
        terrain_mesh.visual.face_colors = [(140, 170, 110, 180)] * len(terrain_mesh.faces)

        self._terrain_handle = self._server.scene.add_mesh_trimesh(
            "/terrain", terrain_mesh
        )
        logger.info(
            f"ViserBridge: terrain added ({len(terrain_mesh.vertices)} verts, "
            f"{len(terrain_mesh.faces)} faces)"
        )

    def _resolve_shadow_addressing(self) -> None:
        import mujoco as mj
        model = self._mj_model
        self._shadow_qpos_root_addr: int | None = (
            model.jnt_qposadr[0] if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE else None
        )
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
        sim, data = self._simulator, self._mj_data
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
    # GUI — Controls tab
    # ================================================================

    def _setup_controls_tab(self, tab_group) -> None:
        import viser as _viser

        server = self._server

        with tab_group.add_tab("Controls"):
            # --- Info panel ---
            with server.gui.add_folder("Info", expand_by_default=True):
                self._info_handle = server.gui.add_markdown(self._build_info_text())

            # --- Simulation ---
            with server.gui.add_folder("Simulation", expand_by_default=True):
                btn_pause = server.gui.add_button("Pause", icon=_viser.Icon.PLAYER_PAUSE)
                btn_play = server.gui.add_button("Play", icon=_viser.Icon.PLAYER_PLAY, visible=False)

                @btn_pause.on_click
                def _(_: _viser.GuiEvent) -> None:
                    self._scene.paused = True
                    btn_pause.visible = False
                    btn_play.visible = True

                @btn_play.on_click
                def _(_: _viser.GuiEvent) -> None:
                    self._scene.paused = False
                    btn_play.visible = False
                    btn_pause.visible = True

                # Speed controls
                speed_group = server.gui.add_button_group("Speed", ("Slower", "1x", "Faster"))

                @speed_group.on_click
                def _(_: _viser.GuiEvent) -> None:
                    if speed_group.value == "Slower":
                        self._speed_multiplier = max(0.125, self._speed_multiplier / 2.0)
                    elif speed_group.value == "Faster":
                        self._speed_multiplier = min(8.0, self._speed_multiplier * 2.0)
                    else:
                        self._speed_multiplier = 1.0

            # --- Terrain ---
            if self._terrain_handle is not None:
                with server.gui.add_folder("Terrain", expand_by_default=False):
                    cb_terrain = server.gui.add_checkbox("Show terrain", initial_value=True)

                    @cb_terrain.on_update
                    def _(_: _viser.GuiEvent) -> None:
                        if self._terrain_handle is not None:
                            self._terrain_handle.visible = cb_terrain.value

            # --- Velocity arrows ---
            with server.gui.add_folder("Velocity Arrows", expand_by_default=True):
                cb_show = server.gui.add_checkbox("Show", initial_value=True)
                sl_scale = server.gui.add_slider("Scale", min=0.1, max=3.0, step=0.1, initial_value=0.5)
                sl_z = server.gui.add_slider("Height", min=0.0, max=1.0, step=0.05, initial_value=0.2)

                @cb_show.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._show_velocity = cb_show.value
                    for a in (self._arrow_cmd_lin, self._arrow_cmd_ang,
                              self._arrow_actual_lin, self._arrow_actual_ang):
                        a.set_visible(cb_show.value)

                @sl_scale.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._velocity_scale = sl_scale.value

                @sl_z.on_update
                def _(_: _viser.GuiEvent) -> None:
                    self._vel_z_offset = sl_z.value

                server.gui.add_markdown(
                    "🔵 cmd lin &nbsp; 🟢 cmd ang\n\n🔷 actual lin &nbsp; 🟩 actual ang"
                )

            # --- Velocity joystick ---
            with server.gui.add_folder("Commands", expand_by_default=True):
                cb_joy = server.gui.add_checkbox("Enable joystick", initial_value=False)
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
                    sl_vx.value = sl_vy.value = sl_yaw.value = 0.0
                    self._vel_joystick_vx = self._vel_joystick_vy = self._vel_joystick_yaw = 0.0

    def _build_info_text(self) -> str:
        status = "⏸ Paused" if self._scene.paused else "▶ Running"
        return (
            f"**Status:** {status}\n\n"
            f"**Step:** {self._total_steps}\n\n"
            f"**Speed:** {self._speed_multiplier}x\n\n"
            f"**FPS:** {self._current_fps:.0f}\n\n"
            f"**Simulator:** {type(self._simulator).__name__}\n\n"
            f"**DOFs:** {self._mapped_dofs}/{self._simulator.num_dof}"
        )

    # ================================================================
    # GUI — Rewards tab
    # ================================================================

    def _setup_rewards_tab(self, tab_group) -> None:
        import colorsys

        import viser as _viser

        try:
            with tab_group.add_tab("Rewards"):
                # Total reward plot
                self._server.gui.add_markdown("### Total Reward")
                series = (
                    _viser.uplot.Series(label="step"),
                    _viser.uplot.Series(label="Total", stroke="rgb(0, 150, 255)", width=2),
                )
                self._reward_plot_handle = self._server.gui.add_uplot(
                    data=(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)),
                    series=series,
                    scales={
                        "x": _viser.uplot.Scale(time=False, auto=True),
                        "y": _viser.uplot.Scale(auto=True),
                    },
                    legend=_viser.uplot.Legend(show=False),
                    aspect=1.5,
                )

                # Per-term reward plots (created dynamically on first push)
                self._term_plots_folder = self._server.gui.add_folder(
                    "Per-Term Rewards", expand_by_default=True
                )
                self._term_plot_handles: dict[str, object] = {}
                self._term_histories: dict[str, deque] = {}
                self._term_colors: dict[str, str] = {}
                self._term_plots_initialized = False
        except Exception as e:
            logger.debug(f"ViserBridge: could not create reward plot: {e}")
            self._reward_plot_handle = None

    def _init_term_plots(self, term_names: list[str]) -> None:
        """Create per-term reward plots on first reward push."""
        import colorsys

        import viser as _viser

        with self._term_plots_folder:
            for i, name in enumerate(term_names):
                hue = i / max(1, len(term_names))
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                self._term_colors[name] = color
                self._term_histories[name] = deque(maxlen=_REWARD_HISTORY_LEN)

                self._server.gui.add_markdown(
                    f"<span style='color:{color}'>**{name}**</span>"
                )
                series = (
                    _viser.uplot.Series(label="step"),
                    _viser.uplot.Series(label=name, stroke=color, width=2),
                )
                self._term_plot_handles[name] = self._server.gui.add_uplot(
                    data=(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)),
                    series=series,
                    scales={
                        "x": _viser.uplot.Scale(time=False, auto=True),
                        "y": _viser.uplot.Scale(auto=True),
                    },
                    legend=_viser.uplot.Legend(show=False),
                    aspect=1.0,
                )
        self._term_plots_initialized = True

    # ================================================================
    # GUI — Checkpoints tab
    # ================================================================

    def _setup_checkpoints_tab(self, tab_group) -> None:
        """Checkpoint selector for hot-swapping during eval."""
        import viser as _viser

        self._checkpoint_load_callback = None
        self._checkpoint_dropdown = None

        try:
            with tab_group.add_tab("Checkpoints"):
                self._server.gui.add_markdown("### Load Checkpoint")
                self._checkpoint_dropdown = self._server.gui.add_dropdown(
                    "Checkpoint", options=["(scanning...)"],
                )
                btn_refresh = self._server.gui.add_button("Refresh", icon=_viser.Icon.REFRESH)
                btn_load = self._server.gui.add_button("Load", icon=_viser.Icon.DOWNLOAD)
                self._checkpoint_status = self._server.gui.add_markdown("*Select a checkpoint*")

                @btn_refresh.on_click
                def _(_: _viser.GuiEvent) -> None:
                    self._scan_checkpoints()

                @btn_load.on_click
                def _(_: _viser.GuiEvent) -> None:
                    if self._checkpoint_load_callback and self._checkpoint_dropdown:
                        path = self._checkpoint_dropdown.value
                        if path and path != "(no checkpoints)":
                            self._checkpoint_status.content = f"Loading **{os.path.basename(path)}**..."
                            try:
                                self._checkpoint_load_callback(path)
                                self._checkpoint_status.content = f"Loaded **{os.path.basename(path)}**"
                            except Exception as e:
                                self._checkpoint_status.content = f"**Error:** {e}"

                # Initial scan
                self._scan_checkpoints()
        except Exception as e:
            logger.debug(f"ViserBridge: could not create checkpoints tab: {e}")

    def _scan_checkpoints(self) -> None:
        """Scan log directory for .pt checkpoint files."""
        if self._checkpoint_dropdown is None:
            return
        # Try to find log directory from simulator's training config
        log_dir = getattr(self._simulator, "training_config", None)
        if log_dir is not None:
            log_dir = getattr(log_dir, "log_dir", None)

        # Also check common patterns
        search_dirs = []
        if log_dir and os.path.isdir(str(log_dir)):
            search_dirs.append(str(log_dir))
        # Search in current working dir / logs
        for candidate in ["logs", "."]:
            if os.path.isdir(candidate):
                search_dirs.append(candidate)

        checkpoints = []
        for d in search_dirs:
            for root, _dirs, files in os.walk(d):
                for f in sorted(files):
                    if f.endswith(".pt"):
                        checkpoints.append(os.path.join(root, f))

        if checkpoints:
            # Show most recent first
            checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)[:20]
            self._checkpoint_dropdown.options = checkpoints
        else:
            self._checkpoint_dropdown.options = ["(no checkpoints)"]

    def set_checkpoint_loader(self, callback) -> None:
        """Register a callback to load checkpoints: callback(path: str) -> None."""
        self._checkpoint_load_callback = callback

    # ================================================================
    # Public API — push data from eval loop
    # ================================================================

    def push_rewards(self, rewards: np.ndarray | float, term_rewards: dict[str, float] | None = None) -> None:
        """Push per-step rewards into history for plotting."""
        r = float(rewards[0]) if hasattr(rewards, "__getitem__") else float(rewards)
        self._reward_history.append(r)
        self._reward_timesteps.append(float(self._total_steps))

        # Per-term rewards
        if term_rewards and hasattr(self, "_term_plots_initialized"):
            if not self._term_plots_initialized:
                self._init_term_plots(list(term_rewards.keys()))
            for name, val in term_rewards.items():
                if name in self._term_histories:
                    self._term_histories[name].append(float(val))

    # ================================================================
    # Update
    # ================================================================

    @property
    def speed_multiplier(self) -> float:
        """Current playback speed multiplier (for eval loop pacing)."""
        return self._speed_multiplier

    def update(self) -> None:
        self._step_count += 1
        self._total_steps += 1
        if self._step_count % self._config.update_freq != 0:
            return
        if self._scene.paused:
            return
        now = time.monotonic()
        # Adjust update interval by speed: faster speed → shorter interval
        interval = self._min_update_interval / max(self._speed_multiplier, 0.1)
        if (now - self._last_update_time) < interval:
            return
        self._last_update_time = now

        # FPS counter
        self._fps_counter_frames += 1
        elapsed = now - self._fps_counter_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter_frames / elapsed
            self._fps_counter_frames = 0
            self._fps_counter_time = now

        # Apply joystick
        if self._vel_joystick_enabled:
            self._apply_joystick_override()

        self._sync_shadow_state()

        with self._server.atomic():
            self._scene.update_from_mjdata(self._mj_data)
            self._update_terrain_offset()
            if self._show_velocity:
                self._update_velocity_arrows()
            self._update_info_panel()
            self._update_reward_plot()

    def _apply_joystick_override(self) -> None:
        sim = self._simulator
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                sim.commands[0, 0] = self._vel_joystick_vx
                sim.commands[0, 1] = self._vel_joystick_vy
                if sim.commands.shape[1] > 2:
                    sim.commands[0, 2] = self._vel_joystick_yaw
            except (IndexError, AttributeError):
                pass

    def _update_terrain_offset(self) -> None:
        """Move terrain mesh to follow camera tracking offset."""
        if self._terrain_handle is not None:
            offset = self._compute_scene_offset()
            self._terrain_handle.position = offset

    def _update_info_panel(self) -> None:
        if self._info_handle is not None and self._total_steps % 50 == 0:
            self._info_handle.content = self._build_info_text()

    def _update_reward_plot(self) -> None:
        if len(self._reward_timesteps) < 2:
            return
        t = np.array(self._reward_timesteps, dtype=np.float32)

        # Total reward
        if self._reward_plot_handle is not None:
            r = np.array(self._reward_history, dtype=np.float32)
            self._reward_plot_handle.data = (t, r)

        # Per-term rewards
        if hasattr(self, "_term_plot_handles"):
            for name, handle in self._term_plot_handles.items():
                if name in self._term_histories and len(self._term_histories[name]) >= 2:
                    tr = np.array(self._term_histories[name], dtype=np.float32)
                    # Align lengths (term history may be shorter than timesteps)
                    n = min(len(t), len(tr))
                    handle.data = (t[-n:], tr[-n:])

    def _compute_scene_offset(self) -> np.ndarray:
        """Compute scene offset from shadow mj_data (matches mjviser's internal tracking)."""
        offset = np.zeros(3)
        if self._scene.camera_tracking_enabled:
            tracked_id = getattr(self._scene, "_tracked_body_id", None)
            if tracked_id is not None and tracked_id < self._mj_model.nbody:
                tracked_pos = self._mj_data.xpos[tracked_id].copy()
                offset = -tracked_pos
                offset[2] = 0.0  # keep Z grounded
        return offset

    def _update_velocity_arrows(self) -> None:
        sim = self._simulator
        root_state = sim.robot_root_states[0].detach().cpu().numpy()
        base_pos = root_state[:3]
        qx, qy, qz, qw = root_state[3], root_state[4], root_state[5], root_state[6]
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])

        # Compute scene offset directly from shadow mj_data (reliable)
        offset = self._compute_scene_offset()
        scale = self._velocity_scale
        z_off = self._vel_z_offset

        def l2w(v):
            """Local body-frame vector → world position."""
            return base_pos + R @ (v * scale)

        origin = l2w(np.array([0, 0, z_off]))

        # Command velocity
        cmd_vx = cmd_vy = cmd_yaw = 0.0
        if hasattr(sim, "commands") and sim.commands is not None:
            try:
                cmd = sim.commands[0].detach().cpu().numpy()
                cmd_vx = float(cmd[0]) if len(cmd) > 0 else 0.0
                cmd_vy = float(cmd[1]) if len(cmd) > 1 else 0.0
                cmd_yaw = float(cmd[2]) if len(cmd) > 2 else 0.0
            except (IndexError, AttributeError):
                pass

        # Actual velocity (world → body frame)
        lin_vel_b = R.T @ root_state[7:10]
        ang_vel_b = R.T @ root_state[10:13]

        # 4 arrows (mjlab convention: all originate from z_offset above base)
        base_offset = np.array([0, 0, z_off])
        self._arrow_cmd_lin.update(
            origin, l2w(base_offset + np.array([cmd_vx, cmd_vy, 0])), offset)
        self._arrow_cmd_ang.update(
            origin, l2w(base_offset + np.array([0, 0, cmd_yaw])), offset)
        self._arrow_actual_lin.update(
            origin, l2w(base_offset + np.array([lin_vel_b[0], lin_vel_b[1], 0])), offset)
        self._arrow_actual_ang.update(
            origin, l2w(base_offset + np.array([0, 0, ang_vel_b[2]])), offset)

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

"""MjLab simulator implementation.

Bridges mjlab's GPU-accelerated MuJoCo Warp framework to holosoma's
BaseSimulator interface, providing GPU-batched physics via mjlab's
Simulation, Scene, Entity, and EntityData classes.
"""

from __future__ import annotations

import os

import mujoco
import torch
from loguru import logger

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.mjlab.terrain_bridge import build_terrain_entity_cfg
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.shared.virtual_gantry import create_virtual_gantry
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.module_utils import get_holosoma_root


# ---------------------------------------------------------------------------
# Warp < 1.12 compatibility: mjlab assumes model/data attributes return
# torch.Tensor, but older warp returns wp.array.  This proxy wraps warp
# objects so attribute access transparently returns torch tensors (zero-copy).
# Once warp >= 1.12 is released this shim can be removed.
# ---------------------------------------------------------------------------
def _install_warp_compat_patches() -> None:
    """Monkey-patch mjlab Entity/EntityData for warp < 1.12 compatibility."""
    import warp as wp  # noqa: PLC0415

    # Quick check: if warp already returns torch tensors, skip patching
    try:
        _test_model_cls = wp.array  # just checking warp is importable
    except Exception:
        return

    from mjlab.entity.entity import Entity  # noqa: PLC0415

    if getattr(Entity, "_holosoma_warp_patched", False):
        return  # already patched

    class _WarpTorchProxy:
        """Proxy that converts warp array attributes to torch tensors (zero-copy)."""

        __slots__ = ("_warp_obj",)

        def __init__(self, warp_obj: object) -> None:
            object.__setattr__(self, "_warp_obj", warp_obj)

        def __getattr__(self, name: str) -> object:
            val = getattr(object.__getattribute__(self, "_warp_obj"), name)
            if isinstance(val, wp.array):
                return wp.to_torch(val)
            return val

        def __setattr__(self, name: str, value: object) -> None:
            setattr(object.__getattribute__(self, "_warp_obj"), name, value)

    _orig_entity_init = Entity.initialize

    def _patched_entity_initialize(self, mj_model, model, data, device):  # type: ignore[no-untyped-def]
        # Wrap warp model/data so mjlab code sees torch tensors
        wrapped_model = _WarpTorchProxy(model) if not isinstance(model, _WarpTorchProxy) else model
        wrapped_data = _WarpTorchProxy(data) if not isinstance(data, _WarpTorchProxy) else data
        return _orig_entity_init(self, mj_model, wrapped_model, wrapped_data, device)

    Entity.initialize = _patched_entity_initialize
    Entity._holosoma_warp_patched = True  # type: ignore[attr-defined]


class MjLabScene:
    """MjLab Scene implementation following holosoma's SceneInterface protocol."""

    def __init__(self, env_origins: torch.Tensor, device: str) -> None:
        self._env_origins = env_origins
        self._device = device

    @property
    def env_origins(self) -> torch.Tensor:
        return self._env_origins


class _MjLabWpModelProxy:
    """Proxy around wp_model that forwards _expanded_fields from Simulation."""

    def __init__(self, wp_model, sim) -> None:
        object.__setattr__(self, "_wp_model", wp_model)
        object.__setattr__(self, "_sim", sim)

    @property
    def _expanded_fields(self) -> set[str]:
        return self._sim.expanded_fields

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_wp_model"), name)


class _MjLabBackendAdapter:
    """Adapter exposing mjlab internals in the format expected by randomize_field().

    Provides ``model`` (CPU MuJoCo model), ``warp_model_bridge`` (torch-wrapped
    GPU model), and ``mjw_model`` (raw warp model) — matching the interface that
    ``holosoma.simulator.mujoco.backends.warp_randomization.randomize_field``
    accesses via ``simulator.backend.*``.
    """

    def __init__(self, mjlab_sim) -> None:
        self._sim = mjlab_sim

    @property
    def model(self) -> mujoco.MjModel:
        return self._sim.mj_model

    @property
    def warp_model_bridge(self):
        return self._sim.model  # WarpBridge around wp_model

    @property
    def mjw_model(self):
        return _MjLabWpModelProxy(self._sim.wp_model, self._sim)


class MjLab(BaseSimulator):
    """MjLab physics simulator backend.

    Uses mjlab's GPU-accelerated MuJoCo Warp framework with its
    manager-based Scene/Entity API, bridged to holosoma's BaseSimulator
    interface for seamless CLI selection via ``simulator:mjlab``.
    """

    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str) -> None:
        logger.info("=== MjLab Simulator Initialization Started ===")
        logger.info(f"Device: {device}")

        super().__init__(tyro_config, terrain_manager, device)

        self.tyro_config = tyro_config
        self.device = device
        self.robot_config = tyro_config.robot
        self.num_envs = self.training_config.num_envs

        # mjlab objects (initialized during load_assets)
        self.mjlab_sim = None  # mjlab.Simulation
        self.mjlab_scene = None  # mjlab.Scene
        self.robot_entity = None  # mjlab.Entity

        # MuJoCo model/data (from mjlab, for viewer/video compatibility)
        self.root_model: mujoco.MjModel | None = None
        self.root_data: mujoco.MjData | None = None

        # Robot metadata
        self.num_dof: int = 0
        self.num_bodies: int = 0
        self.dof_names: list[str] = []
        self.body_names: list[str] = []
        self._body_list: list[str] = []

        # State tensors (placeholders)
        self.dof_pos = torch.zeros(0, device=device)
        self.dof_vel = torch.zeros(0, device=device)
        self.contact_forces = torch.zeros(0, device=device)

        # Viewer
        self.viewer: mujoco.viewer.Handle | None = None
        self.current_world_id: int = 0

        logger.info("=== MjLab Simulator Initialization Completed ===")

    # ----- Configuration Methods -----

    def set_headless(self, headless: bool) -> None:
        super().set_headless(headless)
        self.headless = headless

    def setup(self) -> None:
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self) -> None:
        # Terrain handled during load_assets via mjlab Scene
        return

    def get_supported_scene_formats(self) -> list[str]:
        return []

    # ----- Asset Loading -----

    def load_assets(self) -> None:
        """Load robot assets via mjlab Scene/Entity system."""
        from mjlab.actuator import XmlMotorActuatorCfg  # noqa: PLC0415
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: PLC0415
        from mjlab.scene import Scene, SceneCfg  # noqa: PLC0415
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg  # noqa: PLC0415

        logger.info("=== Loading assets via mjlab ===")

        # Resolve robot XML path
        asset_root = self.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        robot_xml_path = os.path.join(asset_root, self.robot_config.asset.xml_file)
        logger.info(f"Robot XML: {robot_xml_path}")

        # Build initial state config for mjlab (mjlab uses wxyz quaternions)
        init_rot_holosoma = self.robot_config.init_state.rot  # [x, y, z, w]
        init_rot_mjlab = (
            init_rot_holosoma[3],
            init_rot_holosoma[0],
            init_rot_holosoma[1],
            init_rot_holosoma[2],
        )  # [w, x, y, z]

        # Map default_joint_angles to mjlab's regex format
        joint_pos_map = dict(self.robot_config.init_state.default_joint_angles.items())

        init_state_cfg = EntityCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            rot=init_rot_mjlab,
            lin_vel=tuple(self.robot_config.init_state.lin_vel),
            ang_vel=tuple(self.robot_config.init_state.ang_vel),
            joint_pos=joint_pos_map or {".*": 0.0},
            joint_vel={".*": 0.0},
        )

        # Create EntityCfg for the robot
        def robot_spec_fn() -> mujoco.MjSpec:
            return mujoco.MjSpec.from_file(robot_xml_path)

        # Use XML-defined actuators (actuators already defined in the MJCF)
        articulation_cfg = EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),),
            soft_joint_pos_limit_factor=self.robot_config.soft_dof_pos_limit,
        )

        robot_entity_cfg = EntityCfg(
            init_state=init_state_cfg,
            spec_fn=robot_spec_fn,
            articulation=articulation_cfg,
        )

        # Build terrain config
        terrain_cfg = build_terrain_entity_cfg(self.terrain_manager)

        # Build SceneCfg
        scene_cfg = SceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            terrain=terrain_cfg,
            entities={"robot": robot_entity_cfg},
        )

        # Create Scene and compile
        self.mjlab_scene = Scene(scene_cfg, self.device)
        mj_model = self.mjlab_scene.compile()
        self.root_model = mj_model

        # Build SimulationCfg (match mjwarp backend settings)
        warp_cfg = self.simulator_config.mujoco_warp
        nconmax = warp_cfg.nconmax_per_env
        njmax = warp_cfg.njmax_per_env
        if njmax is None:
            njmax = max(nconmax * 6, mj_model.nv * 4)
        mujoco_cfg = MujocoCfg(
            timestep=self.sim_dt,
        )
        sim_cfg = SimulationCfg(mujoco=mujoco_cfg, njmax=njmax, nconmax=nconmax)

        # Create Simulation
        self.mjlab_sim = Simulation(self.num_envs, sim_cfg, mj_model, self.device)
        self.root_data = self.mjlab_sim.mj_data
        self.backend = _MjLabBackendAdapter(self.mjlab_sim)

        # Install warp < 1.12 compatibility patches before entity initialization
        _install_warp_compat_patches()

        # Initialize scene entities with warp model/data
        self.mjlab_scene.initialize(
            self.mjlab_sim.mj_model,
            self.mjlab_sim.wp_model,
            self.mjlab_sim.wp_data,
        )

        # Get robot entity reference
        self.robot_entity = self.mjlab_scene.entities["robot"]

        # Extract robot metadata
        self._extract_robot_metadata()

        # Initialize virtual gantry
        gantry_cfg = self.simulator_config.virtual_gantry
        self.virtual_gantry = create_virtual_gantry(
            sim=self,
            enable=gantry_cfg.enabled,
            attachment_body_names=gantry_cfg.attachment_body_names,
            cfg=gantry_cfg,
        )

        # Initialize bridge system
        self._init_bridge()

        # Setup video recorder (reuse MuJoCo's since mjlab also uses MuJoCo)
        if self.video_config.enabled:
            self.video_recorder = MjLabVideoRecorder(self.video_config, self)
            self.video_recorder.setup_recording()

        logger.info(f"Assets loaded - num_dof: {self.num_dof}, num_bodies: {self.num_bodies}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")

    def _extract_robot_metadata(self) -> None:
        """Extract DOF names, body names, and counts from the robot entity."""
        entity = self.robot_entity
        assert entity is not None

        # Joint names (non-free joints)
        self.dof_names = list(entity.joint_names)
        self.num_dof = len(self.dof_names)

        # Body names
        self.body_names = list(entity.body_names)
        self.num_bodies = len(self.body_names)
        self._body_list = self.body_names

        logger.info(f"Robot DOFs: {self.num_dof}, Bodies: {self.num_bodies}")

    # ----- Environment Creation -----

    def create_envs(self, num_envs, env_origins, base_init_state) -> None:
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        # Create holosoma Scene following SceneInterface protocol
        self.scene = MjLabScene(self.env_origins, self.sim_device)

        # Initialize state tensors
        self.dof_state = torch.zeros(self.num_envs, self.num_dof, 2, device=self.sim_device)
        self.dof_pos = self.dof_state[:, :, 0]
        self.dof_vel = self.dof_state[:, :, 1]
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.sim_device)

        history_length = self.simulator_config.contact_sensor_history_length
        self.contact_forces_history = torch.zeros(
            self.num_envs, history_length, self.num_bodies, 3, device=self.sim_device
        )

        # Initialize root state tensors
        self.robot_root_states = torch.zeros(self.num_envs, 13, device=self.sim_device, dtype=torch.float32)
        self.base_quat = self.robot_root_states[:, 3:7]
        self.all_root_states = self.robot_root_states

        # Initialize rigid body state tensors
        self._rigid_body_pos = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_rot = torch.zeros(
            self.num_envs, self.num_bodies, 4, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_ang_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )

    # ----- Simulation Preparation -----

    def prepare_sim(self) -> None:
        """Prepare simulation: reset entities, setup ObjectRegistry, sync state."""
        assert self.mjlab_sim is not None
        assert self.mjlab_scene is not None
        assert self.robot_entity is not None

        # Reset scene entities to initial state
        self.mjlab_scene.reset()
        self.mjlab_sim.forward()

        # Setup ObjectRegistry (robot-only for now)
        self.object_registry.setup_ranges(self.num_envs, robot_count=1, scene_count=0, individual_count=0)

        # Register robot with initial pose (holosoma uses xyzw quaternions)
        robot_poses = torch.zeros(self.num_envs, 7, device=self.sim_device)
        robot_poses[:, :3] = torch.tensor(self.robot_config.init_state.pos, device=self.sim_device)
        robot_poses[:, 3:7] = torch.tensor(self.robot_config.init_state.rot, device=self.sim_device)  # [x,y,z,w]
        self.object_registry.register_object("robot", ObjectType.ROBOT, 0, robot_poses)
        self.object_registry.finalize_registration()

        # Create state tensor views from mjlab entity data
        self._sync_robot_root_states()
        self._sync_dof_states()
        self._sync_rigid_body_states()

    def _sync_robot_root_states(self) -> None:
        """Sync robot_root_states from mjlab entity data (in-place update)."""
        entity = self.robot_entity
        assert entity is not None

        # root_link_pose_w: (num_envs, 7) in wxyz format
        pose_w = entity.data.root_link_pose_w  # (num_envs, 7) - [x,y,z, w,x,y,z]
        vel_w = entity.data.root_link_vel_w  # (num_envs, 6) - [vx,vy,vz, wx,wy,wz]

        # Update in-place: holosoma format [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        # Position
        self.robot_root_states[:, :3] = pose_w[:, :3]
        # Quaternion: mjlab wxyz -> holosoma xyzw
        self.robot_root_states[:, 3] = pose_w[:, 4]  # qx
        self.robot_root_states[:, 4] = pose_w[:, 5]  # qy
        self.robot_root_states[:, 5] = pose_w[:, 6]  # qz
        self.robot_root_states[:, 6] = pose_w[:, 3]  # qw
        # Velocities
        self.robot_root_states[:, 7:13] = vel_w

    def _sync_dof_states(self) -> None:
        """Sync dof_pos and dof_vel from mjlab entity data."""
        entity = self.robot_entity
        assert entity is not None

        self.dof_pos[:] = entity.data.joint_pos
        self.dof_vel[:] = entity.data.joint_vel

    def _sync_rigid_body_states(self) -> None:
        """Sync rigid body state tensors from mjlab entity data."""
        entity = self.robot_entity
        assert entity is not None

        # body_link_pose_w: (num_envs, num_bodies, 7) - [x,y,z, w,x,y,z]
        body_pose = entity.data.body_link_pose_w
        self._rigid_body_pos[:] = body_pose[:, :, :3]
        # Quaternion wxyz -> xyzw
        self._rigid_body_rot[:, :, 0] = body_pose[:, :, 4]  # qx
        self._rigid_body_rot[:, :, 1] = body_pose[:, :, 5]  # qy
        self._rigid_body_rot[:, :, 2] = body_pose[:, :, 6]  # qz
        self._rigid_body_rot[:, :, 3] = body_pose[:, :, 3]  # qw

        # body_link_vel_w: (num_envs, num_bodies, 6) - [vx,vy,vz, wx,wy,wz]
        body_vel = entity.data.body_link_vel_w
        self._rigid_body_vel[:] = body_vel[:, :, :3]
        self._rigid_body_ang_vel[:] = body_vel[:, :, 3:6]

    # ----- State Refresh -----

    def refresh_sim_tensors(self) -> None:
        """Refresh all state tensors from mjlab simulation data."""
        if self.num_bodies <= 0:
            return

        self._sync_robot_root_states()
        self._sync_dof_states()
        self._sync_rigid_body_states()
        self._update_contact_forces()

    def _update_contact_forces(self) -> None:
        """Update contact forces from mjlab simulation data.

        Uses cfrc_ext from the simulation data which provides body-level
        contact forces directly (MuJoCo Warp computes these).
        """
        assert self.mjlab_sim is not None
        assert self.robot_entity is not None

        # cfrc_ext: (num_envs, nbody, 6) - external forces on bodies [torque(3), force(3)]
        body_ids = self.robot_entity.indexing.body_ids
        cfrc_ext = self.mjlab_sim.data.cfrc_ext[:, body_ids]

        # Shift contact history
        if hasattr(self, "contact_forces_history") and self.contact_forces_history.shape[1] > 1:
            self.contact_forces_history[:, 1:] = self.contact_forces_history[:, :-1].clone()

        # Extract force components (last 3 of the 6-vector)
        self.contact_forces[:] = cfrc_ext[:, :, 3:6]

        # Store current frame in history
        if hasattr(self, "contact_forces_history"):
            self.contact_forces_history[:, 0] = self.contact_forces

    # ----- Control -----

    def apply_torques_at_dof(self, torques: torch.Tensor) -> None:
        """Apply torques via mjlab entity effort targets.

        Sets joint_effort_target so that write_data_to_sim() →
        _apply_actuator_controls() → XmlMotorActuator.compute()
        correctly writes the torques to data.ctrl.
        """
        assert self.robot_entity is not None
        self.robot_entity.data.joint_effort_target[:] = torques

    # ----- Simulation Stepping -----

    def simulate_at_each_physics_step(self) -> None:
        assert self.mjlab_sim is not None
        assert self.mjlab_scene is not None

        if self.virtual_gantry:
            self.virtual_gantry.step()

        self._step_bridge()

        # Write entity data to sim (applies actuator controls)
        self.mjlab_scene.write_data_to_sim()

        # Step physics
        self.mjlab_sim.step()

        # Update entity states from sim
        self.mjlab_scene.update(self.sim_dt)

        # Capture video frame
        if self.video_recorder and self.video_recorder.is_recording:
            self.capture_video_frame()

    # ----- Rendering -----

    def setup_viewer(self) -> None:
        logger.info("=== Setting up MjLab viewer ===")
        if self.headless:
            logger.info("Headless mode - skipping viewer setup")
            self.viewer = None
            return

        assert self.root_model is not None
        assert self.root_data is not None
        self.viewer = mujoco.viewer.launch_passive(self.root_model, self.root_data)
        logger.info("=== Viewer setup completed ===")

    def render(self, sync_frame_time: bool = True) -> None:
        if self.viewer is None:
            return

        assert self.mjlab_sim is not None

        # Sync GPU -> CPU for viewer display
        import mujoco_warp as mjwarp  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        with wp.ScopedDevice(self.mjlab_sim.wp_device):
            mjwarp.get_data_into(
                self.root_data,
                self.mjlab_sim.mj_model,
                self.mjlab_sim.wp_data,
                self.current_world_id,
            )

        if self.simulator_config.viewer.enable_tracking:
            # Track robot root body
            root_body_id = self.robot_entity.indexing.root_body_id
            self.viewer.cam.lookat[:] = self.root_data.xpos[root_body_id]

        self.viewer.sync()

    # ----- Time -----

    def time(self) -> float:
        assert self.mjlab_sim is not None
        # mjlab's sim.data.time is a tensor
        return float(self.mjlab_sim.data.time[0])

    # ----- DOF Properties -----

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)

        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]

            # Apply soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """Find rigid body index in body_names list."""
        if body_name in self.body_names:
            return self.body_names.index(body_name)
        raise RuntimeError(f"Body '{body_name}' not found in body_names: {self.body_names}")

    # ----- Contact Forces -----

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0

    # ----- DOF Forces -----

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        assert self.robot_entity is not None
        forces = self.robot_entity.data.actuator_force
        return forces[env_id]

    # ----- Actor State Access -----

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        resolved_objects = self.object_registry.resolve_indices(indices)
        all_states: list[torch.Tensor] = []

        for obj_name, env_ids in resolved_objects:
            if obj_name != "robot":
                raise NotImplementedError(
                    f"MjLab currently only supports robot state access. "
                    f"Object '{obj_name}' is not supported."
                )
            # Read from robot_root_states (already in holosoma format)
            states_for_envs = self.robot_root_states[env_ids]
            all_states.append(states_for_envs)

        return torch.cat(all_states, dim=0) if all_states else torch.empty(0, 13, device=self.sim_device)

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        assert self.robot_entity is not None
        assert self.mjlab_sim is not None

        resolved_objects = self.object_registry.resolve_indices(indices)
        state_offset = 0

        for obj_name, env_ids in resolved_objects:
            if obj_name != "robot":
                raise NotImplementedError(
                    f"MjLab currently only supports robot state setting. "
                    f"Object '{obj_name}' is not supported."
                )

            num_states = len(env_ids)
            obj_states = states[state_offset : state_offset + num_states]  # (N, 13)
            state_offset += num_states

            # Convert holosoma [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
            # to mjlab [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
            mjlab_root_state = torch.zeros_like(obj_states)
            mjlab_root_state[:, :3] = obj_states[:, :3]  # position
            mjlab_root_state[:, 3] = obj_states[:, 6]  # qw
            mjlab_root_state[:, 4] = obj_states[:, 3]  # qx
            mjlab_root_state[:, 5] = obj_states[:, 4]  # qy
            mjlab_root_state[:, 6] = obj_states[:, 5]  # qz
            mjlab_root_state[:, 7:13] = obj_states[:, 7:13]  # velocities

            self.robot_entity.write_root_state_to_sim(mjlab_root_state, env_ids)

        if write_updates:
            self.mjlab_sim.forward()

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name != "robot":
                raise NotImplementedError(
                    f"MjLab currently only supports robot access. "
                    f"Requested object '{name}' is not supported."
                )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_initial_poses(self, names: list[str], env_ids: EnvIds | None = None) -> ActorPoses:
        for name in names:
            if name != "robot":
                raise NotImplementedError(
                    f"MjLab currently only supports robot initial poses. "
                    f"Requested object '{name}' is not supported."
                )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def write_state_updates(self) -> None:
        """Write deferred state updates to simulation."""
        assert self.mjlab_sim is not None
        self.mjlab_sim.forward()

    # ----- Robot-specific State Setting -----

    def set_actor_root_state_tensor(self, set_env_ids: torch.Tensor | None, root_states: torch.Tensor | None) -> None:
        if root_states is not None and root_states is self.all_root_states:
            self.set_actor_root_state_tensor_robots(set_env_ids, self.robot_root_states[set_env_ids])
        else:
            self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_actor_root_state_tensor_robots(
        self, env_ids: EnvIds | None = None, root_states: torch.Tensor | None = None
    ) -> None:
        assert self.robot_entity is not None
        assert self.mjlab_sim is not None

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if root_states is None:
            root_states = self.robot_root_states[env_ids]
        elif len(root_states) != len(env_ids):
            if len(root_states) == self.num_envs:
                root_states = root_states[env_ids]
            else:
                raise ValueError(
                    f"root_states dimension mismatch: got {len(root_states)}, "
                    f"expected either {len(env_ids)} (pre-sliced) or {self.num_envs} (global)"
                )

        if len(env_ids) == 0 or self.num_dof == 0:
            return

        # Convert holosoma xyzw -> mjlab wxyz quaternion
        mjlab_root_state = torch.zeros(len(env_ids), 13, device=self.sim_device)
        mjlab_root_state[:, :3] = root_states[:, :3]  # position
        mjlab_root_state[:, 3] = root_states[:, 6]  # qw
        mjlab_root_state[:, 4] = root_states[:, 3]  # qx
        mjlab_root_state[:, 5] = root_states[:, 4]  # qy
        mjlab_root_state[:, 6] = root_states[:, 5]  # qz
        mjlab_root_state[:, 7:13] = root_states[:, 7:13]  # velocities

        self.robot_entity.write_root_state_to_sim(mjlab_root_state, env_ids)
        self.mjlab_sim.forward()

    def set_dof_state_tensor(self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None) -> None:
        self.set_dof_state_tensor_robots(env_ids, dof_states)

    def set_dof_state_tensor_robots(
        self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None
    ) -> None:
        assert self.robot_entity is not None
        assert self.mjlab_sim is not None

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if len(env_ids) == 0 or self.num_dof == 0:
            return

        if dof_states is None:
            return

        # dof_states can be [num_envs * num_dof, 2] (IsaacGym format)
        # or [num_envs, num_dof, 2] — full tensor, select env_ids subset
        if dof_states.dim() == 2 and dof_states.shape[0] == len(env_ids) * self.num_dof:
            reshaped = dof_states.view(len(env_ids), self.num_dof, 2)
            positions = reshaped[:, :, 0]
            velocities = reshaped[:, :, 1]
        elif dof_states.dim() == 3:
            positions = dof_states[env_ids, :, 0]
            velocities = dof_states[env_ids, :, 1]
        else:
            logger.warning(f"Unexpected dof_states shape: {dof_states.shape}")
            return

        self.robot_entity.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)
        self.mjlab_sim.forward()

    # ----- Domain Randomization -----

    def prepare_manager_fields(self, **managers) -> None:
        """Scan managers for field requirements and expand them in mjlab sim."""
        if self.mjlab_sim is None:
            return

        from holosoma.simulator.mujoco.fields import collect_required_fields  # noqa: PLC0415

        fields = collect_required_fields(*managers.values())
        if fields:
            logger.info(f"Expanding model fields for DR: {fields}")
            self.mjlab_sim.expand_model_fields(tuple(fields))

    # ----- Debug Visualization -----

    def draw_debug_viz(self) -> None:
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def clear_lines(self) -> None:
        pass

    def draw_sphere(
        self, pos: torch.Tensor, radius: float, color: torch.Tensor, env_id: int, pos_id: int | None = None
    ) -> None:
        pass

    def draw_line(self, start_point: torch.Tensor, end_point: torch.Tensor, color: torch.Tensor, env_id: int) -> None:
        pass

    # ----- Cleanup -----

    def __del__(self) -> None:
        logger.info("=== MjLab Simulator Cleanup ===")
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer = None


class MjLabVideoRecorder(MuJoCoVideoRecorder):
    """Video recorder for MjLab, reusing MuJoCo's recording infrastructure."""

    def __init__(self, config, simulator: MjLab) -> None:
        # MuJoCoVideoRecorder expects a MuJoCo simulator, but we can adapt
        super(MuJoCoVideoRecorder, self).__init__(config, simulator)
        self.simulator: MjLab = simulator

        import mujoco as mj  # noqa: PLC0415

        self._renderer: mj.Renderer | None = None
        self._camera: mj.MjvCamera | None = None

        logger.info(
            f"MjLab video recorder initialized - {'threaded' if config.use_recording_thread else 'synchronous'} mode"
        )

    @property
    def renderer(self):
        if self._renderer is None:
            import threading  # noqa: PLC0415

            import mujoco as mj  # noqa: PLC0415

            logger.debug(f"Creating MuJoCo renderer on thread: {threading.current_thread().name}")
            self._renderer = mj.Renderer(self.simulator.root_model, height=self.config.height, width=self.config.width)
        return self._renderer

    @property
    def camera(self):
        if self._camera is None:
            import mujoco as mj  # noqa: PLC0415

            self._camera = mj.MjvCamera()
            mj.mjv_defaultCamera(self._camera)
        return self._camera

    def _capture_frame_impl(self) -> None:
        """Capture a frame using mjlab's simulation data."""
        import mujoco_warp as mjwarp  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        sim = self.simulator.mjlab_sim
        assert sim is not None

        # Sync GPU -> CPU for rendering (world_id=0)
        render_data = self.simulator.root_data
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.get_data_into(render_data, sim.mj_model, sim.wp_data, 0)

        # Update camera position
        self._update_camera_position(self.camera)

        # Render
        self.renderer.update_scene(render_data, camera=self.camera)
        frame = self.renderer.render()

        if frame is None:
            raise RuntimeError("MuJoCo renderer returned None frame")

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = frame
        else:
            raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

        frame_with_overlay = self._apply_command_overlay(frame_rgb)
        self._add_frame(frame_with_overlay)

    def cleanup(self) -> None:
        super(MuJoCoVideoRecorder, self).cleanup()
        self._renderer = None
        self._camera = None

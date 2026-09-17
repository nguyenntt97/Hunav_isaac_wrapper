#!/usr/bin/env python3
"""
teleop_hunav_sim.py

Contains the TeleopHuNavSim class which combines:
- ROS 2 teleoperation for a differential robot.
- World loading via WorldBuilder.
- Agent management via HuNavManager.
"""
import os as _os
import sys as _sys

_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _os.path.exists(_repo_root) and _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from isaacsim import SimulationApp

def _env_flag(name, default="false"):
    return _os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# LIVESTREAM=1 serves the viewport over WebRTC instead of opening a local window,
# for running without an X display (signalling port 49100, media port 47998).
# It implies headless, but unlike plain headless the UI is kept so the remote
# client has something to drive.
LIVESTREAM = _env_flag("LIVESTREAM") or _os.environ.get(
    "LIVESTREAM", ""
).strip().lower() == "webrtc"


# Extensions this wrapper needs that the stock isaacsim.exp.base.kit stopped
# depending on after Isaac Sim 4.5. This is exactly the set the 4.5-era
# isaacsim.exp.base.kit in this repo used to supply; requesting them here means
# the installed .kit file never has to be patched.
#
# They must be enabled during app startup rather than afterwards:
# the behavior and navigation runtimes only initialise while the app is
# booting, and OmniGraph node types (isaacsim.ros2.bridge.*,
# isaacsim.sensors.physics.*) must be registered before any stage referencing
# them is opened.
#
#   omni.anim.behavior.*     - motion matching: IBehaviorAgent, BehaviorAgentAPI
#   omni.anim.navigation.*   - navmesh baking; agents are not created without one
#   omni.anim.asset          - asset runtime the behavior system builds on
#   isaacsim.ros2.bridge     - ROS2Context/PublishClock/SubscribeTwist OmniGraph
#                              nodes, used by create_ros_clock_action_graph() and
#                              by the carter_ROS robot's built-in graph
#   isaacsim.sensors.physics - IsaacReadIMU, referenced by the carter_ROS USD
#   omni.physx.bundle        - full PhysX suite (scene query, vehicle, etc.)
#
# omni.anim.graph.core and omni.anim.retarget.core used to be here for the
# AnimationGraph path. That path is gone: see behavior_agent.py.
STARTUP_EXTENSIONS = [
    "omni.anim.behavior.bundle",
    "omni.anim.behavior.core",
    "omni.anim.behavior.schema",
    "omni.anim.navigation.bundle",
    "omni.anim.navigation.core",
    "omni.anim.asset",
    "isaacsim.ros2.bridge",
    "isaacsim.sensors.physics",
    "omni.physx.bundle",
]

_ENABLE_ARGS = []
for _ext in STARTUP_EXTENSIONS:
    _ENABLE_ARGS += ["--enable", _ext]

if LIVESTREAM:
    # Must be enabled at startup, not after SimulationApp() returns. Isaac Sim's
    # own streaming launcher lists omni.kit.livestream.app in the [dependencies]
    # of isaacsim.exp.full.streaming.kit so it initialises alongside the
    # renderer. Enabling it later leaves the renderer already up in --no-window
    # mode with no surface, and it spins logging
    # "advanceCurrentFrame: backbuffers are not initialized!" while the client
    # sees a black screen.
    _ENABLE_ARGS += ["--enable", "omni.kit.livestream.app"]
    _LS = "--/exts/omni.kit.livestream.app/primaryStream"
    _ENABLE_ARGS += [
        f"{_LS}/streamType=webrtc",
        f"{_LS}/signalPort=49100",
        f"{_LS}/streamPort=47998",
    ]
    _ip = _os.environ.get("LIVESTREAM_PUBLIC_IP", "").strip()
    if _ip:
        _ENABLE_ARGS += [f"{_LS}/publicIp={_ip}"]




# Start Isaac Sim. HEADLESS is already plumbed through docker/docker-compose.yml.
CONFIG = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": _env_flag("HEADLESS") or LIVESTREAM,
    "renderer": "RaytracedLighting",
    "extra_args": _ENABLE_ARGS,
}
if LIVESTREAM:
    # Matches standalone_examples/api/isaacsim.simulation_app/livestream.py.
    CONFIG["hide_ui"] = False
    CONFIG["window_width"] = 1920
    CONFIG["window_height"] = 1080
    CONFIG["display_options"] = 3286

simulation_app = SimulationApp(CONFIG)

if LIVESTREAM:
    simulation_app.set_setting("/app/window/drawMouse", True)
    _public_ip = _os.environ.get("LIVESTREAM_PUBLIC_IP", "").strip()
    print(
        "\n[hunav] WebRTC livestream enabled -- connect the Isaac Sim WebRTC "
        f"Streaming Client to {_public_ip or '<host>'}:49100 (media UDP 47998)."
        + ("" if _public_ip else " Set LIVESTREAM_PUBLIC_IP if no video arrives.")
        + "\n",
        flush=True,
    )

import math
import os
import signal
import subprocess
from pathlib import Path
from rclpy.node import Node
from geometry_msgs.msg import Twist
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
import omni
import omni.graph.core as og
 
# Import the WorldBuilder and HuNavManager modules.
from .world_builder import WorldBuilder
from .hunav_manager import HuNavManager
from .terrain import apply_flat_ground
from .asset_paths import (
    get_isaac_major,
    is_robot_available,
    robot_usd_relative_path,
)

# Default maximum step an agent can climb when --terrain-follow is on, in metres.
# 0.25 m is about a kerb: on the brownstone terraces (mean 0.43 m, max 0.94 m)
# agents take the shallow tiers and route around the tall ones.
DEFAULT_STEP_HEIGHT = 0.25


# Where the robot is spawned, per world. The origin works for the indoor worlds,
# but brownstone has several coincident ground meshes stacked at z=0 around
# (0, 0) -- pathway, hardscape and sidewalk all with exact triangle-mesh
# colliders. PhysX resolves that degenerate contact by launching the robot
# (measured: ejected at ~23 m/s, 115 m away within 5 s). Anywhere else on the
# park's path network is stable, so brownstone spawns off-origin instead.
DEFAULT_ROBOT_SPAWN = [0.0, 0.0, 0.0]
ROBOT_SPAWN_POSE = {
    "brownstone": [4.0, -43.0, 0.25],
}


def find_package_share_directory():
    """
    Find the package share directory containing worlds, scenarios, config, etc.
    Works both in development and installed package modes.
    """
    # Try to find via ROS2 package first (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        share_dir = pkg_path / "share" / "hunav_isaac_wrapper"
        if share_dir.exists() and (share_dir / "worlds").exists():
            return str(share_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Development mode fallback
    current_file = Path(__file__)
    
    # Check if we're in src/hunav_isaac_wrapper/ (development mode)
    if current_file.parent.parent.name == "src":
        src_dir = current_file.parent.parent
        if (src_dir / "worlds").exists():
            return str(src_dir)
    
    # Last fallback - check current working directory
    cwd = Path.cwd()
    if (cwd / "worlds").exists():
        return str(cwd)
    
    # If all else fails, return the old path calculation
    return os.path.dirname(os.path.dirname(__file__))


def find_robot_config_path(filename):
    """
    Find robot configuration file in development or installed package.
    
    Args:
        filename: Name of the robot config file (e.g., "nova_carter_ros2_sensors.usd")
    
    Returns:
        str: Absolute path to the robot config file
    """
    # Try to find via ROS2 package share directory (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        robot_config = pkg_path / "share" / "hunav_isaac_wrapper" / "config" / "robots" / filename
        if robot_config.exists():
            return str(robot_config)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try development mode (relative to this file)
    current_file_dir = Path(__file__).parent
    workspace_root = current_file_dir.parent.parent
    robot_config = workspace_root / "config" / "robots" / filename
    if robot_config.exists():
        return str(robot_config)
    
    # Try alternative development paths
    dev_paths = [
        current_file_dir.parent / "config" / "robots" / filename,
        Path.cwd() / "src" / "config" / "robots" / filename,
        Path.cwd() / "config" / "robots" / filename,
    ]
    
    for path in dev_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Robot config file not found: {filename}")

class TeleopHuNavSim(Node):
    """
    Combines:
    - Differential robot teleop (subscribing to /cmd_vel)
    - USD map loading (via WorldBuilder)
    - Agent management and update (via HuNavManager)
    """

    def __init__(
        self,
        map_name,
        hunav_config,
        robot_name,
        flat_ground=False,
        terrain_follow=False,
        step_height=DEFAULT_STEP_HEIGHT,
        navmesh_helper=False,
    ):
        super().__init__("hunav_sim")
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.navmesh_helper = navmesh_helper or _env_flag("HUNAV_NAVMESH_HELPER") or _env_flag("NAVMESH_HELPER")

        # Assets root. Isaac Sim 5.0+ raises instead of returning None, so both
        # failure modes have to be handled; every built-in robot and character
        # asset is resolved against this, so there is no way to continue.
        try:
            assets_root_path = get_assets_root_path()
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not resolve the Isaac Sim assets root: {e}. Check network "
                "access to the Isaac asset bucket, or point "
                "/persistent/isaac/asset_root/default at a local Nucleus server."
            ) from e
        if assets_root_path is None:
            raise RuntimeError(
                "Could not resolve the Isaac Sim assets root (Nucleus root not found)."
            )

        # Experimental terrain flags. Either the CLI flag or the env var turns
        # these on, so they can be toggled without going through the launcher.
        self.flat_ground = flat_ground or _env_flag("HUNAV_FLAT_GROUND")
        self.terrain_follow = terrain_follow or _env_flag("HUNAV_TERRAIN_FOLLOW")
        self.step_height = float(
            _os.environ.get("HUNAV_STEP_HEIGHT", step_height)
        )
        if self.flat_ground and self.terrain_follow:
            # Flattening the world leaves nothing to follow.
            print(
                "[hunav] --flat-ground and --terrain-follow are both set; "
                "flat ground wins and terrain following is disabled."
            )
            self.terrain_follow = False

        # Load USD stage
        self.builder = WorldBuilder(base_path=find_package_share_directory())
        map_loaded = False
        if map_name:
            map_loaded = self.builder.load_map(map_name)

        # Flatten the terrain before the physics scene exists, so the colliders
        # PhysX cooks are the ones we actually want.
        if self.flat_ground and map_loaded:
            apply_flat_ground(self.builder.get_stage(), map_name)

        # Create World object
        timestep = 1.0 / 20.0
        self.world = World(
            stage_units_in_meters=1, physics_dt=timestep, rendering_dt=timestep
        )

        if map_name == "empty_world":
            self.world.scene.add_default_ground_plane()

        # Define configuration for each wheeled robot available. The USD paths of
        # the built-in robots moved in Isaac Sim 5.0, so they are resolved through
        # asset_paths rather than hardcoded; carter_ROS uses the bundled asset.
        isaac_major = get_isaac_major()
        robot_configs = {
            "jetbot": {
                "name": "Jetbot",
                "usd_relative_path": robot_usd_relative_path("jetbot", isaac_major),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.0325,
                "wheel_base": 0.118,
            },
            "create3": {
                "name": "Create3",
                "usd_relative_path": robot_usd_relative_path("create3", isaac_major),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.03575,
                "wheel_base": 0.233,
            },
            "carter": {
                "name": "Nova_Carter",
                "usd_relative_path": robot_usd_relative_path("carter", isaac_major),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
            },
            "carter_ROS": {
                "name": "Nova_Carter",
                "usd_relative_path": find_robot_config_path("nova_carter_ros2_sensors.usd"),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
            },
        }

        if robot_name not in robot_configs:
            raise ValueError(f"Unsupported robot_name: {robot_name}")

        if not is_robot_available(robot_name, isaac_major):
            raise ValueError(
                f"Robot '{robot_name}' has no asset in Isaac Sim {isaac_major}.x. "
                "Use 'carter_ROS', which ships with this package."
            )

        robot_config = robot_configs[robot_name]
        
        # Handle absolute vs relative paths for robot USD files
        if os.path.isabs(robot_config["usd_relative_path"]):
            # Absolute path (for custom robots like carter_ROS)
            robot_path = robot_config["usd_relative_path"]
        else:
            # Relative path (for built-in Isaac Sim robots)
            robot_path = os.path.join(assets_root_path, robot_config["usd_relative_path"])

        # Add robot to world
        robot_prim_path = f"/World/{robot_config['name']}"
        self.robot_spawn_pose = ROBOT_SPAWN_POSE.get(map_name, DEFAULT_ROBOT_SPAWN)
        # Logged because a robot that is "missing" is nearly always one that was
        # spawned somewhere else (a stale session using an older spawn table) or
        # one that PhysX ejected on the first step -- see _warn_if_robot_moved.
        print(
            f"[hunav] Robot '{robot_config['name']}' spawning at "
            f"{self.robot_spawn_pose} (world key: '{map_name}'"
            f"{'' if map_name in ROBOT_SPAWN_POSE else ' -> default'})"
        )
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path=robot_prim_path,
                name="Robot",
                wheel_dof_names=robot_config["wheel_dof_names"],
                create_robot=True,
                usd_path=robot_path,
                position=self.robot_spawn_pose,
                orientation=[0, 0, 0, 1],
            )
        )

        # Create differential drive controller
        self.diff_controller = DifferentialController(
            name="diff_drive_controller",
            wheel_radius=robot_config["wheel_radius"],
            wheel_base=robot_config["wheel_base"],
        )

        # ROS2 cmd_vel subscriber
        self.cmd_lin = 0.00
        self.cmd_ang = 0.00
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10
        )

        # Setup HuNavManager
        self.hunav = HuNavManager(
            node=self,
            world=self.world,
            config_file_path=hunav_config,
            robot_prim_path=robot_prim_path,
            robot=self.robot,
            terrain_follow=self.terrain_follow,
            step_height=self.step_height,
        )

        self.create_ros_clock_action_graph()

        self.hunav.initialize_agents()
        self.hunav.initialize_hunav_nodes()

        if self.navmesh_helper:
            self._init_navmesh_helper()

    def _init_navmesh_helper(self):
        """Initialize navmesh visualizer and interactive control window."""
        try:
            try:
                from new_behavior.nav_mesh_plugin import build_and_visualize_navmesh, show_navmesh_window
            except ImportError:
                from hunav_isaac_wrapper.nav_mesh_plugin import build_and_visualize_navmesh, show_navmesh_window

            stage = self.builder.get_stage() if hasattr(self, "builder") else None
            result = build_and_visualize_navmesh(stage=stage)
            print(f"[hunav] NavMesh visual surface created at: {result.get('mesh_path')}", flush=True)
            print(f"[hunav] NavMesh outlines created: {len(result.get('outlines', []))} lines", flush=True)

            # Open interactive UI window if UI is active (local GUI or WebRTC livestream)
            if not _env_flag("HEADLESS") or LIVESTREAM:
                try:
                    self._navmesh_window = show_navmesh_window()
                    print("[hunav] NavMesh interactive control window opened.", flush=True)
                except Exception as ui_err:
                    print(f"[hunav] Note: Could not open NavMesh UI window ({ui_err})", flush=True)
        except Exception as e:
            print(f"[hunav] Warning: Failed to initialize NavMesh helper: {e}", flush=True)

    def _signal_handler(self, signum, frame):
        """
        Shut down on Ctrl+C / SIGTERM.

        Two things make the naive version unreliable. First, this handler is
        installed at the top of __init__ but self.hunav is only assigned at the
        end of it, and the Isaac Sim startup in between takes minutes -- an
        interrupt in that window used to raise AttributeError inside the
        handler. Second, SimulationApp launches Kit with
        --/app/installSignalHandlers=0, so Python's handler is the only one
        there is: if it returns without exiting, the run loop just carries on.
        """
        # A second interrupt means the graceful path is wedged. Leave now.
        if self._shutdown_requested:
            print("\n[hunav] Second interrupt -- exiting immediately.\n", flush=True)
            os._exit(130)
        self._shutdown_requested = True

        print(
            "\n\nCaught shutdown signal, closing app and stopping hunav nodes...\n\n",
            flush=True,
        )

        hunav = getattr(self, "hunav", None)
        if hunav is not None:
            try:
                hunav.close_hunav_nodes()
            except Exception as e:  # never let cleanup block the exit
                print(f"[hunav] error stopping HuNavSim nodes: {e}", flush=True)

        try:
            simulation_app.close()
        except Exception as e:
            print(f"[hunav] error closing Isaac Sim: {e}", flush=True)

        # close() normally terminates the process via Kit's fast-shutdown path,
        # but if it returned we must not fall back into the simulation loop.
        os._exit(130)

    def _cmd_vel_callback(self, msg):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def _on_physics_step(self, dt: float):
        """
        Called automatically by PhysX each physics frame.
        """
        self.hunav.send_agents_msg()

    def create_ros_clock_action_graph(self, graph_path="/World/ROS2"):
        try:
            keys = og.Controller.Keys
            graph_params = {
                # Create the necessary nodes.
                keys.CREATE_NODES: [
                    # Node for generating a tick on playback.
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    # Node for reading the simulation time.
                    (
                        "isaac_read_simulation_time",
                        "isaacsim.core.nodes.IsaacReadSimulationTime",
                    ),
                    # Node to create a ROS2 context.
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    # Node to publish the clock over ROS2.
                    ("ros2_publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                # Connect outputs to inputs:
                keys.CONNECT: [
                    # Connect context output to the publish clock's context input.
                    (
                        "ros2_context.outputs:context",
                        "ros2_publish_clock.inputs:context",
                    ),
                    # Connect tick output to publish clock's execIn.
                    (
                        "on_playback_tick.outputs:tick",
                        "ros2_publish_clock.inputs:execIn",
                    ),
                    # Connect simulation time output to publish clock's timeStamp.
                    (
                        "isaac_read_simulation_time.outputs:simulationTime",
                        "ros2_publish_clock.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    # For the simulation time node. Isaac Sim 5.0 removed
                    # inputs:swhFrameNumber from IsaacReadSimulationTime (it now
                    # exposes referenceTimeNumerator/Denominator); setting it
                    # aborts creation of the whole graph, so /clock is never
                    # published.
                    ("isaac_read_simulation_time.inputs:resetOnStop", True),
                    # For the ROS2PublishClock node.
                    ("ros2_publish_clock.inputs:nodeNamespace", ""),
                    ("ros2_publish_clock.inputs:qosProfile", ""),
                    ("ros2_publish_clock.inputs:queueSize", 10),
                    ("ros2_publish_clock.inputs:timeStamp", 0.0),
                    ("ros2_publish_clock.inputs:topicName", "clock"),
                    # For the ROS2Context node.
                    ("ros2_context.inputs:useDomainIDEnvVar", True),
                    ("ros2_context.inputs:domain_id", 0),
                ],
            }
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                graph_params,
            )
            print(f"Successfully created ROS_Clock action graph at {graph_path}")
        except Exception as e:
            print(f"Error creating ROS_Clock action graph: {e}")

    def _warn_if_robot_moved(self, tolerance=3.0):
        """
        Warn if the robot is not where it was spawned.

        Coincident ground colliders can make PhysX resolve the initial overlap
        explosively -- at the brownstone origin that threw the robot ~115 m at
        ~23 m/s, which looks exactly like "the robot is missing". Catching it
        here turns a confusing hunt into one log line.
        """
        try:
            pos, _ = self.robot.get_world_pose()
        except Exception as e:  # robot not initialised; nothing useful to say
            print(f"[hunav] Could not read robot pose: {e}")
            return
        want = self.robot_spawn_pose
        drift = math.hypot(pos[0] - want[0], pos[1] - want[1])
        if drift > tolerance:
            print(
                f"[hunav] WARNING: robot is {drift:.1f} m from its spawn "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} vs {tuple(want)}). "
                "This usually means PhysX ejected it from overlapping ground "
                "colliders at the spawn point -- pick a different spawn for this "
                "world in ROBOT_SPAWN_POSE."
            )
        else:
            print(
                f"[hunav] Robot settled at "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )

    def run(self):
        self.world.reset()
        # world.reset() starts the timeline, and that is where the navmesh and
        # crowd debug overlays come back: they are authored in the /persistent
        # settings tree, which is copied over the live one on a stage or timeline
        # reset. Rebuilding those overlays happens inside the per-frame agent
        # simulation, so leaving them on costs seconds per frame rather than
        # merely drawing something nobody looks at.
        if self.hunav.driver is not None:
            self.hunav.driver.suppress_debug_geometry()
        self._warn_if_robot_moved()
        self.physx_interface = omni.physx.get_physx_interface()
        self.physx_sub = self.physx_interface.subscribe_physics_step_events(
            self._on_physics_step
        )
        self.hunav.send_agents_msg()
        # One-shot skinning check, once the agents have been driven long enough
        # for the answer to mean anything. Agents that track their goals while
        # their rig never moves look identical to working ones in the logs.
        locomotion_checked = False
        steps = 0
        while simulation_app.is_running() and not self._shutdown_requested:
            self.world.step(render=True)
            steps += 1
            if not locomotion_checked and steps == 250:
                locomotion_checked = True
                try:
                    self.hunav.report_locomotion_health()
                except Exception as exc:
                    print(f"[hunav] locomotion check errored: {exc}", flush=True)
            wheel_action = self.diff_controller.forward([self.cmd_lin, self.cmd_ang])
            self.robot.apply_wheel_actions(wheel_action)

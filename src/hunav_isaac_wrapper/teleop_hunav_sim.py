#!/usr/bin/env python3
"""
teleop_hunav_sim.py

Contains the TeleopHuNavSim class which combines:
- ROS 2 teleoperation for a differential robot.
- World loading via WorldBuilder.
- Agent management via HuNavManager.
"""
import os as _os

from isaacsim import SimulationApp

# Extensions this wrapper needs that the stock isaacsim.exp.base.kit stopped
# depending on after Isaac Sim 4.5. This is exactly the set the 4.5-era
# isaacsim.exp.base.kit in this repo used to supply; requesting them here means
# the installed .kit file never has to be patched.
#
# They must be enabled during app startup rather than afterwards:
# omni.anim.graph.core only initialises its CharacterManager while the app is
# booting, and OmniGraph node types (isaacsim.ros2.bridge.*,
# isaacsim.sensors.physics.*) must be registered before any stage referencing
# them is opened.
#
#   omni.anim.graph.core     - AnimationGraph runtime + ag.get_character()
#   omni.anim.retarget.core  - CreateRetargetAnimationsCommand
#   isaacsim.ros2.bridge     - ROS2Context/PublishClock/SubscribeTwist OmniGraph
#                              nodes, used by create_ros_clock_action_graph() and
#                              by the carter_ROS robot's built-in graph
#   isaacsim.sensors.physics - IsaacReadIMU, referenced by the carter_ROS USD
#   omni.physx.bundle        - full PhysX suite (scene query, vehicle, etc.)
STARTUP_EXTENSIONS = [
    "omni.anim.graph.core",
    "omni.anim.retarget.core",
    "isaacsim.ros2.bridge",
    "isaacsim.sensors.physics",
    "omni.physx.bundle",
]

_ENABLE_ARGS = []
for _ext in STARTUP_EXTENSIONS:
    _ENABLE_ARGS += ["--enable", _ext]


def _env_flag(name, default="false"):
    return _os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# LIVESTREAM=1 serves the viewport over WebRTC instead of opening a local window,
# for running without an X display (signalling port 49100, media port 47998).
# It implies headless, but unlike plain headless the UI is kept so the remote
# client has something to drive.
LIVESTREAM = _env_flag("LIVESTREAM") or _os.environ.get(
    "LIVESTREAM", ""
).strip().lower() == "webrtc"

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
    from isaacsim.core.utils.extensions import enable_extension as _enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)

    # The signalling socket binds to 0.0.0.0, but WebRTC advertises ICE
    # candidates for the media stream and auto-detection picks badly on a
    # multi-homed host (docker bridges, VPN interfaces). Symptom is a client
    # that connects and then shows no video. Pin the address the client should
    # actually reach us on, e.g. LIVESTREAM_PUBLIC_IP=192.168.11.2
    _public_ip = _os.environ.get("LIVESTREAM_PUBLIC_IP", "").strip()
    if _public_ip:
        simulation_app.set_setting(
            "/exts/omni.kit.livestream.app/primaryStream/publicIp", _public_ip
        )

    _enable_extension("omni.kit.livestream.app")
    print(
        "\n[hunav] WebRTC livestream enabled -- connect the Isaac Sim WebRTC "
        f"Streaming Client to {_public_ip or '<host>'}:49100 (media UDP 47998)."
        + ("" if _public_ip else " Set LIVESTREAM_PUBLIC_IP if no video arrives.")
        + "\n",
        flush=True,
    )

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
from .asset_paths import (
    get_isaac_major,
    is_robot_available,
    robot_usd_relative_path,
)

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

    def __init__(self, map_name, hunav_config, robot_name):
        super().__init__("hunav_sim")
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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

        # Load USD stage
        self.builder = WorldBuilder(base_path=find_package_share_directory())
        if map_name:
            self.builder.load_map(map_name)

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
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path=robot_prim_path,
                name="Robot",
                wheel_dof_names=robot_config["wheel_dof_names"],
                create_robot=True,
                usd_path=robot_path,
                position=[0.0, 0.0, 0.0],
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
        )

        self.create_ros_clock_action_graph()

        self.hunav.initialize_agents()
        self.hunav.initialize_hunav_nodes()

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

    def run(self):
        self.world.reset()
        self.physx_interface = omni.physx.get_physx_interface()
        self.physx_sub = self.physx_interface.subscribe_physics_step_events(
            self._on_physics_step
        )
        self.hunav.send_agents_msg()
        while simulation_app.is_running() and not self._shutdown_requested:
            self.world.step(render=True)
            wheel_action = self.diff_controller.forward([self.cmd_lin, self.cmd_ang])
            self.robot.apply_wheel_actions(wheel_action)

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
#   isaacsim.robot.policy.examples
#                            - Go2FlatTerrainPolicy and the PolicyController it
#                              derives from. Extension python modules are only
#                              importable while the extension is enabled, and
#                              the Go2 driver imports it at construction time.
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
    "isaacsim.robot.policy.examples",
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
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
import omni
import omni.graph.core as og
 
# Import the WorldBuilder and HuNavManager modules.
from .world_builder import WorldBuilder
from .hunav_manager import HuNavManager
from .behavior_agent import BehaviorAgentDriver
from .terrain import apply_flat_ground
from .asset_paths import get_isaac_major
from .robots import RENDER_DT, get_spec, make_driver, require_available
from .robots.ros_publishers import RobotStatePublisher
from .perf import get_profiler

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
#
# The z here is the world's own ground clearance. A robot that needs to be
# dropped in from higher up -- a quadruped has to fall into its stance rather
# than start in it -- adds RobotSpec.spawn_z_offset on top, so the per-world
# entries stay valid for every robot.
DEFAULT_ROBOT_SPAWN = [0.0, 0.0, 0.0]
ROBOT_SPAWN_POSE = {
    "brownstone": [4.0, -43.0, 0.25],
}


from .package_paths import (
    find_config_path,
    find_package_share_directory,
    find_robot_config_path,
)


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
        author_mode=False,
    ):
        super().__init__("hunav_sim")
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.navmesh_helper = navmesh_helper or _env_flag("HUNAV_NAVMESH_HELPER") or _env_flag("NAVMESH_HELPER")

        # Authoring mode builds the world and the navmesh and stops there: no
        # characters, no robot, no hunav_loader / hunav_agent_manager
        # subprocesses. The scenario being edited is the one the *next* launch
        # will read, because the YAML is read once at startup and there is no
        # reload path -- so editing alongside a live simulation would be
        # editing something the running agents can never pick up.
        self.author_mode = author_mode or _env_flag("HUNAV_AUTHOR_SCENARIO")
        if self.author_mode:
            # The whole point is the helper window.
            self.navmesh_helper = True
        self.hunav = None
        self.robot = None
        self.driver = None
        self.robot_spec = None
        self.state_publisher = None
        self.scenario_manager = None
        self._navmesh_window = None

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

        # The robot decides how fast physics has to run. A differential drive is
        # happy at the rendering rate; a learned locomotion policy is a closed
        # loop that has to be stepped at the rate it was trained at (200 Hz for
        # the Go2) or the robot never stands up. Rendering stays at 20 Hz either
        # way, so world.step() simply substeps physics when the two differ.
        #
        # Authoring mode never spawns a robot, so it takes the historical rate.
        self.robot_spec = None if self.author_mode else get_spec(robot_name)
        physics_dt = RENDER_DT if self.robot_spec is None else self.robot_spec.physics_dt
        backend = "numpy" if self.robot_spec is None else self.robot_spec.backend
        device = None if self.robot_spec is None else self.robot_spec.device

        # Create World object
        self.world = World(
            stage_units_in_meters=1,
            physics_dt=physics_dt,
            rendering_dt=RENDER_DT,
            backend=backend,
            device=device,
        )

        # send_agents_msg() blocks on a HuNavSim service round-trip and
        # HuNavManager differences agent poses against a hardcoded 1/20 s. Both
        # break if the PhysX callback that drives them starts firing at 200 Hz,
        # so the crowd tick is decimated back to the rendering rate.
        self._hunav_decimation = max(1, int(round(RENDER_DT / physics_dt)))
        self._physics_step_count = 0

        # Off unless HUNAV_PERF=1, in which case every span below starts timing.
        self.perf = get_profiler()
        self._perf_norender = _env_flag("HUNAV_PERF_NORENDER")
        if self._perf_norender:
            print("[perf] HUNAV_PERF_NORENDER=1: stepping with render=False", flush=True)

        if map_name == "empty_world":
            self.world.scene.add_default_ground_plane()

        self.map_name = map_name

        # Everything below this point is the runtime: robot, characters, ROS
        # nodes. Authoring needs none of it.
        if self.author_mode:
            self._build_authoring(map_name)
            return

        spec = self.robot_spec
        isaac_major = get_isaac_major()
        require_available(spec, isaac_major)

        # Only the selected robot's USD is resolved. The old code built every
        # robot's path while constructing a dict literal, so a still-zipped
        # Carter asset made --robot jetbot fail too.
        robot_path = spec.usd_path(
            assets_root_path, find_robot_config_path, isaac_major
        )
        spec = spec.with_resolved_policy(find_config_path)
        self.robot_spec = spec

        world_spawn = ROBOT_SPAWN_POSE.get(map_name, DEFAULT_ROBOT_SPAWN)
        self.robot_spawn_pose = [
            world_spawn[0],
            world_spawn[1],
            world_spawn[2] + spec.spawn_z_offset,
        ]

        # A learned gait is trained against a specific ground friction (1.0 /
        # 1.0 for the Go2); on a slicker floor the feet skate and the policy
        # degrades. Wheeled robots keep whatever the stage authored.
        if spec.driver == "go2_policy":
            from .robots.ground import apply_ground_friction

            try:
                apply_ground_friction(self.builder.get_stage())
            except Exception as exc:
                print(f"[hunav] Could not set ground friction: {exc}", flush=True)

        robot_prim_path = f"/World/{spec.prim_name}"
        # Logged because a robot that is "missing" is nearly always one that was
        # spawned somewhere else (a stale session using an older spawn table) or
        # one that PhysX ejected on the first step -- see _warn_if_robot_moved.
        print(
            f"[hunav] Robot '{spec.prim_name}' spawning at "
            f"{self.robot_spawn_pose} (world key: '{map_name}'"
            f"{'' if map_name in ROBOT_SPAWN_POSE else ' -> default'}), "
            f"physics at {1.0 / spec.physics_dt:.0f} Hz"
        )
        self.driver = make_driver(
            spec, self.world, robot_path, self.robot_spawn_pose
        )
        # HuNavManager only ever calls get_world_pose/get_linear_velocity/
        # get_angular_velocity on this, which every driver implements.
        self.robot = self.driver

        # Robots whose USD ships its own publisher graph (carter_ROS) must not
        # get a second one.
        self.state_publisher = (
            RobotStatePublisher(self, self.driver) if spec.publish_odom_tf else None
        )

        # ROS2 cmd_vel subscriber
        self.cmd_lin = 0.00
        self.cmd_lin_y = 0.00
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
            robot_spec=spec,
        )

        self.create_ros_clock_action_graph()

        self.hunav.initialize_agents()
        self.hunav.initialize_hunav_nodes()

        if self.navmesh_helper:
            self._init_navmesh_helper()

    def _build_authoring(self, map_name):
        """Mode A: bake the navmesh over the whole map and open the editor.

        The navmesh is baked through BehaviorAgentDriver, not through the
        plugin's own build_navmesh. The two disagree materially -- agent radius
        50 vs 60 cm, step height 25 vs 90 cm, slope 20 vs 45 degrees, island
        radius 500 vs 80 cm -- so authoring against the plugin's mesh would
        validate spawns on a surface the simulator never produces.

        The volume comes from the map's own extent rather than from the agent
        poses. Deriving it from the poses, as the runtime does, is circular
        here: you could only ever place an agent inside the box the current
        poses already describe.
        """
        from .behavior_agent import NAVMESH_SETTINGS
        from .scenario import paths as scenario_paths
        from .scenario.bake import (
            ground_z_for_map,
            navmesh_settings_digest,
            sampling_cm_for_extent,
            scene_bounds,
        )
        from .scenario.spec import NavmeshProvenance

        stage = self.builder.get_stage()
        maps_dir = scenario_paths.maps_dir()

        ground_z = ground_z_for_map(map_name, maps_dir)
        bounds = scene_bounds(map_name, maps_dir, ground_z=ground_z)
        if bounds is None:
            print(
                f"[hunav] No map description for '{map_name}' in {maps_dir}; "
                "falling back to the world's own bounds."
            )

        self.driver = BehaviorAgentDriver(stage, get_assets_root_path(), dt=1.0 / 20.0)
        self.driver.configure_navmesh()
        self.driver.ensure_navmesh_volume(bounds=bounds)
        self.driver.bake_navmesh(
            extent=self.driver.navmesh_extent, ground_z=ground_z
        )

        extent = self.driver.navmesh_extent or (0.0, 0.0, 0.0)
        provenance = NavmeshProvenance(
            settings_digest=navmesh_settings_digest(NAVMESH_SETTINGS),
            volume_min=tuple(bounds[0]) if bounds else (0.0, 0.0, 0.0),
            volume_max=tuple(bounds[1]) if bounds else (0.0, 0.0, 0.0),
            ground_z=float(ground_z),
            sampling_cm=float(
                getattr(self.driver, "navmesh_sampling", None)
                or sampling_cm_for_extent(
                    extent, NAVMESH_SETTINGS["agentSamplingDistance"], 1000
                )
            ),
        )

        self._init_navmesh_helper()

        if self._navmesh_window is not None:
            manager = self._navmesh_window._ensure_scenario_manager()
            manager.set_provenance(provenance)
            self.scenario_manager = manager

        print(
            "\n[hunav] Scenario authoring mode.\n"
            f"        Map:       {map_name}\n"
            f"        NavMesh:   {provenance.sampling_cm:.1f} cm sampling, "
            f"digest {provenance.settings_digest}\n"
            f"        Scenarios: {scenario_paths.scenarios_dir()}\n"
            f"        Trees:     {scenario_paths.behavior_trees_dir()}\n"
            "        The bake above covers the whole map. To design the navmesh,\n"
            "        select the walkable meshes in the stage tree, press Assign\n"
            "        Mesh, then Build Navmesh -- that bake is recorded and is what\n"
            "        the run reproduces. Then use 'Agent Spawns & Goals'.\n"
            "        Export writes the YAML and one behavior tree per agent;\n"
            "        relaunch without --author-scenario to run what you wrote.\n",
            flush=True,
        )

    def run_authoring(self):
        """Mode A loop: keep the app responsive, but never start physics.

        world.reset() would start the timeline, and the behavior system would
        begin looking for agents that deliberately do not exist here.
        """
        print("[hunav] Authoring session ready. Ctrl+C to exit.", flush=True)
        while simulation_app.is_running() and not self._shutdown_requested:
            simulation_app.update()
            rclpy.spin_once(self, timeout_sec=0.0)

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
        # Write the profile first. Both exits below are os._exit(), which skips
        # atexit handlers, so a run stopped with Ctrl-C -- which is how a timed
        # measurement run always ends -- would otherwise leave no JSON behind.
        # dump() is idempotent and a no-op when profiling is off.
        try:
            self.perf.dump()
        except Exception:
            pass

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
        # linear.y is carried through for robots that can strafe; the
        # differential drivers ignore it, and Nav2's controllers never set it.
        self.cmd_lin = msg.linear.x
        self.cmd_lin_y = msg.linear.y
        self.cmd_ang = msg.angular.z
        if self.driver is not None:
            self.driver.set_command(self.cmd_lin, self.cmd_lin_y, self.cmd_ang)

    def _on_physics_step(self, dt: float):
        """
        Called automatically by PhysX each physics frame.

        Control that closes a loop around joint state runs every step, at the
        physics rate. The crowd update does not: it blocks on a HuNavSim
        service call and assumes a 20 Hz cadence, so it is decimated back to
        that regardless of how fast physics is running.
        """
        # This fires once per PhysX substep -- ten times per rendered frame for
        # the Go2 -- so the profiler sums these spans across the frame rather
        # than reporting the cost of one substep.
        with self.perf.span("physx_cb_total"):
            with self.perf.span("go2_policy"):
                self.driver.on_physics_step(dt)

            self._physics_step_count += 1
            if self._physics_step_count % self._hunav_decimation == 0:
                with self.perf.span("crowd"):
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
        # Mode A never starts physics or the timeline: there are no agents to
        # step and no robot to drive.
        if self.author_mode:
            self.run_authoring()
            return

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
            with self.perf.frame():
                with self.perf.span("world_step"):
                    # HUNAV_PERF_NORENDER=1 steps physics without rendering.
                    # world.step() is a single opaque call, so this is the only
                    # way to split the renderer from the PhysX solver -- the
                    # difference between the two runs is the render. Diagnostic
                    # only: the viewport does not update.
                    self.world.step(render=not self._perf_norender)
                steps += 1
                if not locomotion_checked and steps == 250:
                    locomotion_checked = True
                    try:
                        self.hunav.report_locomotion_health()
                    except Exception as exc:
                        print(f"[hunav] locomotion check errored: {exc}", flush=True)
                with self.perf.span("driver_render"):
                    self.driver.on_render_step()
                if self.state_publisher is not None:
                    with self.perf.span("ros_publish"):
                        self.state_publisher.publish()
                # The node used to be spun only as a side effect of
                # HuNavManager's blocking service call, which made teleop depend
                # on HuNavSim being alive. Spin it here so /cmd_vel is serviced
                # on its own.
                with self.perf.span("ros_spin"):
                    rclpy.spin_once(self, timeout_sec=0.0)
                # Feed the simulated clock so the report can state the real-time
                # factor. 30 FPS at 0.6x real time and 30 FPS at 1.0x real time
                # are indistinguishable in an FPS counter and are not the same
                # result; the target here is the latter.
                self.perf.note_sim_time(self.world.current_time)

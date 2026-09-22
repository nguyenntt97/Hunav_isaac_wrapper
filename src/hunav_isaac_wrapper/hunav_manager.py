#!/usr/bin/env python3
"""
hunav_manager.py

Contains the HuNavManager class for spawning and managing HuNavSim agents,
driving their locomotion, and handling physics.

Locomotion runs on the Isaac Sim 6 behavior framework (see behavior_agent.py).
HuNavSim's social-force model remains authoritative over where each agent goes;
Isaac's motion matching only decides how the body renders that motion.
"""

import os
import os as _os
import math
import random
import yaml
import numpy as np
import subprocess, signal
from typing import Tuple, List, Optional

# ROS messages
import rclpy
from geometry_msgs.msg import Quaternion, Pose, Point
from hunav_msgs.srv import ComputeAgents
from hunav_msgs.msg import Agent, Agents, AgentBehavior
from std_msgs.msg import Header

# Isaac Sim imports
import omni
import omni.kit.commands
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension

from pxr import Sdf, Gf, UsdGeom, UsdPhysics, PhysxSchema
import carb

from .animation_utils import find_skelroot_path, find_skeleton_path
from .behavior_agent import (
    BehaviorAgentDriver,
    BehaviorAgentError,
    STARTUP_EXTENSIONS as BEHAVIOR_EXTENSIONS,
)
from .scenario.spec import BEHAVIOR_TYPES, FORCE_FACTOR_RANGES, VEL_RANGE
from .perf import get_profiler

# These are requested at app startup via SimulationApp's extra_args (see
# STARTUP_EXTENSIONS in teleop_hunav_sim.py), which is the only point at which
# the behavior and navigation runtimes initialise. The calls below are no-ops in
# that path and only matter when HuNavManager is imported into an app that was
# started some other way.
for _ext in BEHAVIOR_EXTENSIONS:
    enable_extension(_ext)


class HuNavManager:
    """
    Manages HuNavSim agents by reading configuration from a YAML file,
    spawning character assets, driving them through the Isaac Sim 6 behavior
    framework, and handling ROS 2 communications.
    """

    def __init__(
        self,
        node,
        world,
        config_file_path,
        robot_prim_path,
        robot,
        terrain_follow=False,
        step_height=0.25,
        robot_spec=None,
    ):
        self.node = node
        self.stage = world.stage
        # Rendering step, matching TeleopHuNavSim's World(). Used to
        # finite-difference agent velocity, which the behavior API does not
        # report (get_linear_velocity() returns zero for a walking agent).
        #
        # Note this is the *rendering* rate, not necessarily the physics rate:
        # a robot whose controller needs faster physics (the Go2 policy runs at
        # 200 Hz) does not change how often agents are updated. TeleopHuNavSim
        # decimates its physics callback to keep that true.
        self.dt = 1.0 / 20.0
        self.perf = get_profiler()
        # Diagnostic, off by default. Agents are still spawned, attached and
        # ticked by the crowd engine, but no goal is ever issued to them. The
        # difference between a run with this on and one with it off is the cost
        # that the agents' movement actually causes, as opposed to the cost the
        # navmesh and crowd engine carry regardless.
        self._no_drive = _os.environ.get("HUNAV_NO_DRIVE", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if self._no_drive:
            print("[HuNavManager] HUNAV_NO_DRIVE=1: agents will not be given goals",
                  flush=True)
        self.robot_prim_path = robot_prim_path
        self.robot_obj = robot
        # Describes the robot HuNavSim is told about. None keeps the historical
        # one-size-fits-all values.
        self.robot_spec = robot_spec
        self.world = world
        self.config_file_path = config_file_path

        self.assets_root = get_assets_root_path()
        self._usd_context = omni.usd.get_context()

        # HuNavSim's ROS 2 service client
        self.compute_agents_client = self.node.create_client(
            ComputeAgents, "/compute_agents"
        )

        # List of target model assets
        character_root_path = os.path.join(self.assets_root, "Isaac/People/Characters/")

        character_models = [
            "F_Business_02/F_Business_02.usd",
            "F_Medical_01/F_Medical_01.usd",
            "M_Medical_01/M_Medical_01.usd",
            "male_adult_construction_01_new/male_adult_construction_01_new.usd",
            "male_adult_construction_05_new/male_adult_construction_05_new.usd",
            "female_adult_police_01_new/female_adult_police_01_new.usd",
            "female_adult_police_02/female_adult_police_02.usd",
            "female_adult_police_03_new/female_adult_police_03_new.usd",
            "male_adult_police_04/male_adult_police_04.usd",
            "original_female_adult_business_02/female_adult_business_02.usd",
            "original_female_adult_medical_01/female_adult_medical_01.usd",
        ]

        self.target_model_paths = [
            f"{character_root_path}{model}" for model in character_models
        ]
        
        # Mapping from skin ID to character model index
        # This allows agents to specify which character model to use via the 'skin' field
        # in their configuration. Valid options:
        #   0-10: Specific character models (see mapping below)
        #   "random": Random character model selection
        self.skin_to_model_mapping = {
            1: 0,   # F_Business_02
            2: 1,   # F_Medical_01
            3: 2,   # M_Medical_01
            4: 3,   # male_adult_construction_01_new
            5: 4,   # male_adult_construction_05_new
            6: 5,   # female_adult_police_01_new
            7: 6,   # female_adult_police_02
            8: 7,   # female_adult_police_03_new
            9: 8,   # male_adult_police_04
            10: 9,  # original_female_adult_business_02
            11: 10, # original_female_adult_medical_01
        }

        # Data holders. self.agents holds the character prims; the parallel
        # lists hold the SkelRoot each BehaviorAgentAPI was applied to and the
        # live agent handle once the timeline is playing.
        self.agents = []
        self.agent_skelroots = []
        self.agent_handles = []
        self.agent_initial_states = []
        self._hunav_processes = []

        # Locomotion. Created in initialize_agents(); handles are acquired
        # lazily on the first tick, because an agent is not registered with the
        # behavior system until the simulation has started running.
        self.driver = None
        self._agents_active = False

        self.robot_prim = None
        self._last_yaw = {}
        # Agents whose behavior type we have already complained about, so a
        # bad scenario logs once per agent rather than once per tick.
        self._warned_behavior_types = set()
        self._anim_debug_ticks = {}

        if config_file_path is not None:
            self.config = self._load_yaml(config_file_path)
        else:
            self.config = None

        # --- Experimental terrain following -------------------------------
        # HuNavSim's social-force model is 2-D: getUpdatedAgentMsg never sets a
        # Z, so every agent is written to the stage at z=0 each tick. On flat
        # indoor worlds that is correct; on terraced ground (brownstone rises to
        # 0.94 m) agents walk inside the terrain. When enabled, we raycast down
        # at each agent's XY and place them on whatever they are standing over,
        # refusing steps taller than step_height so they route around the tall
        # tiers rather than teleporting up them.
        self.terrain_follow = terrain_follow
        self.step_height = float(step_height)
        # Last accepted ground Z per agent path, so a missed raycast holds
        # position instead of snapping the agent to zero.
        self.agent_ground_z = {}
        # How far above the agent to start the downward probe, in metres.
        self.ground_probe_height = 2.0
        # Per-tick lerp toward the sampled ground, to avoid popping at edges.
        self.ground_smoothing_factor = 0.2
        if self.terrain_follow:
            print(
                f"[hunav] Terrain following enabled (max step {self.step_height:.2f} m)"
            )

    def _load_yaml(self, relative_path):
        full_path = os.path.join(os.path.dirname(__file__), relative_path)
        with open(full_path, "r") as file:
            return yaml.safe_load(file)

    def normalize_angle(self, a: float) -> float:
        value = a
        while value <= -math.pi:
            value += 2 * math.pi
        while value > math.pi:
            value -= 2 * math.pi
        return value

    @staticmethod
    def _ros_subprocess_env():
        """
        Environment for the HuNavSim ROS 2 nodes we spawn.

        They are launched from inside Isaac Sim's interpreter, so they would
        otherwise inherit Isaac's PYTHONPATH. Python nodes (hunav_evaluator)
        then import Isaac's bundled numpy while their own extension modules
        (pandas) were built against the system numpy, aborting with
        "numpy.dtype size changed ... binary incompatibility". Strip the Isaac
        entries so the nodes resolve the system/ROS site-packages instead.
        """
        env = os.environ.copy()
        isaac_root = os.environ.get("ISAAC_PATH", "/isaac-sim")
        pythonpath = [
            entry
            for entry in env.get("PYTHONPATH", "").split(os.pathsep)
            if entry and not entry.startswith(isaac_root)
        ]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env

    def initialize_hunav_nodes(self):
        env = self._ros_subprocess_env()
        process_1 = subprocess.Popen(
            [
                "ros2",
                "run",
                "hunav_agent_manager",
                "hunav_loader",
                "--ros-args",
                "--params-file",
                self.config_file_path,
            ],
            preexec_fn=os.setsid,
            env=env,
        )
        process_2 = subprocess.Popen(
            [
                "ros2",
                "run",
                "hunav_agent_manager",
                "hunav_agent_manager",
                "--ros-args",
                "--params-file",
                self.config_file_path,
            ],
            preexec_fn=os.setsid,
            env=env,
        )
        process_3 = subprocess.Popen(
            ["ros2", "run", "hunav_evaluator", "hunav_evaluator_node"],
            preexec_fn=os.setsid,
            env=env,
        )

        self._hunav_processes = [process_1, process_2, process_3]

    def close_hunav_nodes(self):
        for process in self._hunav_processes:
            print(process)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as e:
                # Already exited (e.g. a node whose package is not built).
                print(f"[HuNavManager] could not signal pid {process.pid}: {e}")
        self._hunav_processes = []

    def initialize_agents(self):
        """
        Read the agent configuration, spawn one character per agent, and make
        each of them a behavior agent.

        Character transforms are owned by the behavior system once the agents
        are live, so each character is payloaded directly at its initial pose
        with no moving parent: a parent transform would compose with the
        engine's own and carry the agent off the map.

        The live agent handles are NOT acquired here. An agent is only
        registered once the simulation is running, so acquisition happens
        lazily on the first tick -- see _ensure_agents_active().
        """
        if self.config is None:
            print("[HuNavManager] No config loaded, skipping agent creation.")
            return

        agent_configs = self.config["hunav_loader"]["ros__parameters"]["agents"]

        self.driver = BehaviorAgentDriver(
            self.stage, self.assets_root, dt=self.dt
        )

        # Bake the navmesh FIRST, while the stage still holds only the world.
        # The baker's memory use scales with resident geometry, and it fails by
        # returning an empty navmesh rather than raising -- so the motion
        # library payload and eight character assets are loaded afterwards.
        self.driver.configure_navmesh()
        bounds = self._agent_activity_bounds() or self._world_bounds()
        self.driver.ensure_navmesh_volume(bounds=bounds)
        # Ground level the agents stand at: the volume's own floor is padded
        # well below it and is not a usable reference for what is walkable.
        ground_z = (bounds[0][2] + self._NAVMESH_Z_BELOW) if bounds else 0.0
        self.driver.bake_navmesh(
            extent=self.driver.navmesh_extent, ground_z=ground_z
        )

        self.driver.load_motion_library()

        asset_cycle = self.target_model_paths.copy()
        random.shuffle(asset_cycle)

        for agent_name in agent_configs:
            agent_cfg = self.config["hunav_loader"]["ros__parameters"][agent_name]

            # Use skin value to select specific character model
            skin_value = agent_cfg["skin"]
            asset_path = self.get_character_model_from_skin(skin_value)

            if asset_path is None:
                # Invalid skin value, fall back to round-robin selection
                self.node.get_logger().warn(
                    f"Invalid skin value '{skin_value}' for agent {agent_name}, "
                    f"falling back to random selection. Valid skin values are 0 (random) or 1-{len(self.target_model_paths)}"
                )
                if len(asset_cycle) == 0:
                    asset_cycle = self.target_model_paths.copy()
                    random.shuffle(asset_cycle)
                asset_path = asset_cycle.pop()
            else:
                if skin_value == 0:
                    self.node.get_logger().info(
                        f"Agent {agent_name} using random skin: {asset_path.split('/')[-2]}"
                    )
                else:
                    self.node.get_logger().info(
                        f"Agent {agent_name} using skin {skin_value}: {asset_path.split('/')[-2]}"
                    )

            init_pose = agent_cfg["init_pose"]
            position = (init_pose["x"], init_pose["y"], init_pose["z"])
            yaw = float(init_pose.get("h", 0.0))

            character = self.driver.spawn_character(
                f"/World/Characters/{agent_name}", asset_path, position, yaw
            )
            self.agents.append(character)
            self.agent_initial_states.append({"position": position, "yaw": yaw})
            self.agent_handles.append(None)

        # Let every payload resolve before anything inspects the hierarchies.
        # attach() needs the SkelRoot, which does not exist until the character
        # asset has finished loading.
        self._pump_app(400)

        for character in self.agents:
            self.agent_skelroots.append(
                self.driver.attach(character, find_skelroot_path)
            )

        print(
            f"[HuNavManager] {len(self.agents)} agents spawned as behavior agents"
        )

        # Set up the robot prim
        if self.robot_prim_path:
            rp = self.stage.GetPrimAtPath(self.robot_prim_path)
            if rp.IsValid():
                self.robot_prim = rp
            else:
                print(
                    f"[HuNavManager] Warning: no valid robot prim at {self.robot_prim_path}"
                )
        else:
            print("[HuNavManager] no robot_prim_path provided")

    @staticmethod
    def _pump_app(iterations):
        """Advance the app so pending asset loads complete."""
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(iterations):
            app.update()

    # Margin around the agents' working area for the navmesh volume, in metres.
    _NAVMESH_MARGIN = 6.0
    # Vertical band the navmesh covers, relative to the agents' ground level.
    _NAVMESH_Z_BELOW = 2.0
    _NAVMESH_Z_ABOVE = 4.0

    def _agent_activity_bounds(self):
        """Extent of everywhere the agents actually go, padded.

        Sizing the navmesh volume to the whole world is what made brownstone
        unbakeable: its bounds span 84 x 128 x 23 m, and the vertical extent is
        almost entirely the sunken road at -19.8 m and empty air. The GPU baker
        exhausts CUDA memory long before that voxelises, and coarsening the
        sampling far enough to fit leaves cells larger than agentMinIslandRadius,
        at which point nothing survives the bake at all.

        The agents only ever occupy their start poses and their goals, so that
        -- plus a margin -- is the region that actually needs to be walkable.
        """
        if self.config is None:
            return None
        params = self.config["hunav_loader"]["ros__parameters"]
        points = []

        for name in params.get("agents", []):
            pose = params.get(name, {}).get("init_pose")
            if pose:
                points.append((float(pose["x"]), float(pose["y"]),
                               float(pose.get("z", 0.0))))

        goals = params.get("global_goals") or {}
        for goal in goals.values():
            try:
                points.append((float(goal["x"]), float(goal["y"]), 0.0))
            except (KeyError, TypeError, ValueError):
                continue

        if not points:
            return None

        margin = self._NAVMESH_MARGIN
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        ground = min(pt[2] for pt in points)
        return (
            (min(xs) - margin, min(ys) - margin, ground - self._NAVMESH_Z_BELOW),
            (max(xs) + margin, max(ys) + margin, ground + self._NAVMESH_Z_ABOVE),
        )

    def _world_bounds(self):
        """XY/Z extent of /World. Fallback when the scenario defines no goals."""
        from pxr import Usd

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        root = self.stage.GetPrimAtPath("/World")
        if not root or not root.IsValid():
            return None
        rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return tuple(rng.GetMin()), tuple(rng.GetMax())

    def _ensure_agents_active(self):
        """Acquire the live agent handles. Returns True once all are live.

        An agent is not registered with the behavior system until the
        simulation has started running, so this is retried each tick until it
        succeeds rather than being done at spawn time.
        """
        if self._agents_active or self.driver is None:
            return self._agents_active

        import omni.anim.behavior.core as bh

        interface = bh.acquire_interface()
        pending = 0
        for index, skelroot in enumerate(self.agent_skelroots):
            if self.agent_handles[index] is not None:
                continue
            if interface.get_agent(skelroot) is None:
                pending += 1
                continue
            agent_cfg = self._agent_cfg(index)
            handle = self.driver.acquire(skelroot)
            handle.agent.set_speed(float(agent_cfg["max_vel"]))
            state = self.agent_initial_states[index]
            self.driver.teleport(handle, state["position"], state["yaw"])
            self.agent_handles[index] = handle

        self._tick_attempts = getattr(self, "_tick_attempts", 0) + 1
        if pending == 0:
            self._agents_active = True
            print(f"[HuNavManager] {len(self.agent_handles)} behavior agents live")
        elif self._tick_attempts == 200:
            raise BehaviorAgentError(
                f"{pending} of {len(self.agent_skelroots)} agents were never "
                "registered with the behavior system after 200 ticks. The usual "
                "cause is a missing navmesh: it bakes only from UsdGeom.Mesh "
                "geometry, so a world whose walkable ground is implicit "
                "(Cube/Plane) produces none and no agent is ever created."
            )
        return self._agents_active

    def _agent_cfg(self, index):
        agent_ref = self.config["hunav_loader"]["ros__parameters"]["agents"][index]
        return self.config["hunav_loader"]["ros__parameters"][agent_ref]

    def verify_locomotion(self, min_driven_ticks=100):
        """Report agents whose body is not actually being animated.

        The failure this guards against was silent: the old retargeting step
        produced a clip whose joints rotated by a mean of 0.08 degrees across
        the whole walk cycle, every structural check passed, and the only
        symptom was agents gliding to their goals in bind pose. Checking that
        prims and clips are *valid* does not catch that, so this checks that a
        leg joint has actually changed pose while the agent was being driven.

        Returns a list of human-readable problems; empty means healthy.
        """
        problems = []
        for index, handle in enumerate(self.agent_handles):
            label = f"agent {index + 1}"
            if handle is None:
                problems.append(f"{label}: no live behavior agent")
                continue
            try:
                height = float(handle.agent.get_height())
            except Exception as exc:
                problems.append(f"{label}: agent handle is dead ({exc})")
                continue
            # A rig the motion library cannot pose reports a degenerate height
            # rather than a plausible human one.
            if not 0.5 < height < 2.5:
                problems.append(
                    f"{label}: implausible body height {height:.2f} m -- the "
                    "motion library is probably not posing this rig"
                )
            if handle.driven_ticks < min_driven_ticks:
                # Not enough evidence yet; say so rather than pass silently.
                problems.append(
                    f"{label}: only driven for {handle.driven_ticks} ticks, "
                    f"need {min_driven_ticks} before the skinning check means "
                    "anything"
                )
            elif not handle._joint_moved:
                problems.append(
                    f"{label}: driven for {handle.driven_ticks} ticks but its "
                    "leg joint never moved -- the character is being slid along "
                    "the ground, not animated (this is the bind-pose failure "
                    "the AnimationGraph path used to produce)"
                )
        return problems

    def report_locomotion_health(self, min_driven_ticks=100):
        """Print the result of verify_locomotion once. Returns True if healthy."""
        problems = self.verify_locomotion(min_driven_ticks)
        if problems:
            print("[HuNavManager] locomotion check FAILED:")
            for problem in problems:
                print(f"  - {problem}")
            return False
        print(
            f"[HuNavManager] locomotion check passed: {len(self.agent_handles)} "
            "agents animating"
        )
        return True

    def reset_agent_states(self):
        """Return every agent to its configured start pose.

        Teleporting is right here and wrong for driving: it places the agent
        exactly, but a stream of teleports reads to the motion matcher as
        discontinuous jumps and never produces a gait.
        """
        if self.driver is None:
            return
        for handle, state in zip(self.agent_handles, self.agent_initial_states):
            if handle is not None:
                self.driver.teleport(handle, state["position"], state["yaw"])
        print("[HuNavManager] agent states reset.")

    def clear_simulation(self):
        self.close_hunav_nodes()
        stage = self._usd_context.get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        for prim in world_prim.GetChildren():
            stage.RemovePrim(prim.GetPath())
        self.agents.clear()
        self.agent_skelroots.clear()
        self.agent_handles.clear()
        self.robot = None
        self.agent_initial_states.clear()
        self.driver = None
        self._agents_active = False

    # Obstacle detection functions
    def generate_lasers(self, num_lasers: int) -> List[Gf.Vec3f]:
        """
        Generate ray directions evenly distributed over 360° in the global frame.
        """
        directions = []
        angle_increment = 360.0 / num_lasers
        for i in range(num_lasers):
            angle_rad = math.radians(i * angle_increment)
            x = math.cos(angle_rad)
            y = math.sin(angle_rad)
            directions.append(Gf.Vec3f(x, y, 0.0))
        return directions

    def get_closest_obstacles(
        self,
        agent_position: Gf.Vec3d,
        max_distance: float,
        sensor_offsets: List[float],
        num_lasers: int = 90,
    ) -> List[Tuple[float, Optional[Gf.Vec3f]]]:
        """
        Cast rays from the agent's sensor origins at multiple heights in fixed directions.
        For each ray direction, iterate over the provided sensor_offsets and select the hit
        with the smallest distance (if any).
        """
        directions = self.generate_lasers(num_lasers)
        scene_query = omni.physx.get_physx_scene_query_interface()
        closest_hits = []
        # Counted rather than derived from num_lasers * len(sensor_offsets), so
        # the figure stays honest if this loop ever gains an early-out.
        rays_cast = 0

        # Iterate over each ray direction
        for direction in directions:
            best_distance = max_distance
            best_hit = None
            hit_found = False
            # Test each sensor offset
            for offset in sensor_offsets:
                sensor_origin = Gf.Vec3f(
                    float(agent_position[0]),
                    float(agent_position[1]),
                    float(agent_position[2]) + offset,
                )
                hit = scene_query.raycast_closest(
                    sensor_origin, direction, max_distance
                )
                rays_cast += 1
                if hit.get("hit", False):
                    hit_found = True
                    distance = hit.get("distance", max_distance)
                    # Keep the closest hit
                    if distance < best_distance:
                        best_distance = distance
                        best_hit = hit.get("position")
            # If at least one hit was found, append the best hit; else, use defaults
            if hit_found:
                closest_hits.append((best_distance, best_hit))
            else:
                closest_hits.append((max_distance, None))

        self.perf.count("raycasts", rays_cast)
        return closest_hits

    def sample_ground_height(self, x, y, previous_z):
        """
        Height of the ground under (x, y), or None if nothing was found.

        Reuses the same scene-query interface as get_closest_obstacles, only
        pointing down instead of sideways. Two things to be careful of:

        - PhysX returns no hits until after World.reset(), so the first ticks
          legitimately miss. Callers must hold the previous Z rather than
          falling back to zero.
        - The agent containers carry RigidBodyAPI, so a ray starting inside an
          agent can hit that agent. We start well above and skip any hit that
          lands on a character.
        """
        scene_query = omni.physx.get_physx_scene_query_interface()
        origin_z = float(previous_z) + self.ground_probe_height
        origin = Gf.Vec3f(float(x), float(y), origin_z)
        direction = Gf.Vec3f(0.0, 0.0, -1.0)
        max_distance = self.ground_probe_height + 20.0

        hit = scene_query.raycast_closest(origin, direction, max_distance)
        if not hit.get("hit", False):
            return None

        body = str(hit.get("rigidBody", "") or hit.get("collision", ""))
        if "/World/Characters" in body:
            # Started inside another agent -- retry from just below that hit.
            hit_z = hit.get("position")[2]
            retry_origin = Gf.Vec3f(float(x), float(y), float(hit_z) - 0.05)
            hit = scene_query.raycast_closest(retry_origin, direction, max_distance)
            if not hit.get("hit", False):
                return None
            body = str(hit.get("rigidBody", "") or hit.get("collision", ""))
            if "/World/Characters" in body:
                return None

        return float(hit.get("position")[2])

    def resolve_agent_z(self, agent_path, x, y, fallback_z):
        """
        Ground-following Z for an agent, honouring the step-height limit.

        Stepping down is always allowed (agents walk off kerbs); stepping up is
        refused beyond self.step_height, so the agent does not pop up the face
        of a tall terrace.

        Note this only clamps the vertical: HuNavSim owns XY and knows nothing
        about the terrain, so refusing a step does not steer the agent away.
        Horizontal avoidance comes from get_closest_obstacles(), whose rays at
        0.05-1.0 m above the agent already hit terrace faces because those are
        colliders, and are fed to HuNavSim as obstacle forces.
        """
        previous_z = self.agent_ground_z.get(agent_path, fallback_z)
        ground_z = self.sample_ground_height(x, y, previous_z)
        if ground_z is None:
            return previous_z

        if ground_z - previous_z > self.step_height:
            # Too tall to climb: stay at the current height.
            return previous_z

        smoothed = previous_z + (ground_z - previous_z) * self.ground_smoothing_factor
        self.agent_ground_z[agent_path] = smoothed
        return smoothed

    def euler_from_quaternion(
        self, x: float, y: float, z: float, w: float
    ) -> Tuple[float, float, float]:
        """
        Converts a quaternion (with w as the scalar part) to Euler roll, pitch, yaw.
        quaternion = [x, y, z, w]
        """
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def send_agents_msg(self):
        """
        Called every physics step to call /compute_agents and update agent transforms.
        """
        if self.robot_prim is None:
            print("[HuNavManager] No robot assigned.")
            return

        # Agents are only registered with the behavior system once the
        # simulation is running, so the handles are acquired on the first ticks
        # rather than at spawn time.
        if not self._ensure_agents_active():
            return

        # Build Agents message
        agents_msg = Agents()
        agents_msg.header = Header()
        now = self.node.get_clock().now().to_msg()
        agents_msg.header.stamp.sec = now.sec
        agents_msg.header.stamp.nanosec = now.nanosec
        agents_msg.header.frame_id = "world"

        # Build robot message
        robot_msg = self._create_robot_msg()

        # For each agent, create and add an Agent message. This is where the
        # per-agent obstacle raycasts happen, so the "rays" span nested inside
        # it is the one to read first.
        with self.perf.span("msg_build"):
            for idx, agent_prim in enumerate(self.agents):
                agent_msg = self._create_agent_msg(agent_prim, idx)
                agents_msg.agents.append(agent_msg)

        # Wait for the compute_agents service and call it
        with self.perf.span("svc_wait"):
            available = self.compute_agents_client.wait_for_service(timeout_sec=2.0)
        if not available:
            print("[HuNavManager] /compute_agents not available.")
            return
        with self.perf.span("svc_call"):
            self._call_compute(agents_msg, robot_msg)

    def _create_robot_msg(self):
        # Retrieve robot pose and velocities from the robot driver. Every driver
        # returns the same shapes as the WheeledRobot this used to be handed:
        # xyz position, wxyz orientation, and world-frame velocities.
        pos, quat = self.robot_obj.get_world_pose()
        lin_vel = self.robot_obj.get_linear_velocity()
        ang_vel = self.robot_obj.get_angular_velocity()

        robot = Agent()
        robot.id = 0
        robot.type = Agent.ROBOT
        robot.skin = 1
        robot.name = "Robot"
        robot.group_id = 0
        # Sized from the robot's own spec where there is one; a Go2 is not the
        # same 0.5 m disc as a Nova Carter.
        robot.radius = 0.5 if self.robot_spec is None else self.robot_spec.hunav_radius
        robot.desired_velocity = (
            1.0
            if self.robot_spec is None
            else self.robot_spec.hunav_desired_velocity
        )
        # float() is mandatory, not cosmetic: rosidl's generated C asserts
        # PyFloat_Check on these fields, and get_linear_velocity()/
        # get_angular_velocity() return backend-dependent scalars (numpy or
        # torch, depending on the isaacsim.core backend) that fail that check
        # and abort the process. _create_agent_msg() already does this.
        robot.linear_vel = float(
            np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2 + lin_vel[2] ** 2)
        )
        robot.angular_vel = float(
            np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)
        )

        # Pose
        robot.position.position.x = float(pos[0])
        robot.position.position.y = float(pos[1])
        robot.position.position.z = float(pos[2])
        robot.position.orientation.w = float(quat[0])
        robot.position.orientation.x = float(quat[1])
        robot.position.orientation.y = float(quat[2])
        robot.position.orientation.z = float(quat[3])
        _, _, yaw = self.euler_from_quaternion(quat[1], quat[2], quat[3], quat[0])
        robot.yaw = float(yaw)

        # Velocities
        robot.velocity.linear.x = float(lin_vel[0])
        robot.velocity.linear.y = float(lin_vel[1])
        robot.velocity.linear.z = float(lin_vel[2])
        robot.velocity.angular.x = float(ang_vel[0])
        robot.velocity.angular.y = float(ang_vel[1])
        robot.velocity.angular.z = float(ang_vel[2])
        robot.cyclic_goals = True
        robot.goal_radius = 0.5
        robot.closest_obs = []
        return robot

    def _behavior_type_id(self, beh, agent_cfg):
        """Map the scenario's behavior name onto the AgentBehavior enum.

        The scenario schema spells the type as a string ("Regular",
        "Impassive", ...) while the message carries the uint8. An unrecognised
        name is a scenario bug that would otherwise present as agents quietly
        ignoring the robot, so it is reported once per agent and falls back to
        Regular rather than to the zero that caused the original problem.
        """
        raw = beh.get("type", "Regular")

        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
            type_id = int(raw)
            if type_id in BEHAVIOR_TYPES.values():
                return type_id
        elif raw in BEHAVIOR_TYPES:
            return BEHAVIOR_TYPES[raw]

        agent_id = agent_cfg.get("id", "?")
        if agent_id not in self._warned_behavior_types:
            self._warned_behavior_types.add(agent_id)
            print(
                f"[HuNavManager] agent {agent_id}: unknown behavior type {raw!r}; "
                f"expected one of {', '.join(BEHAVIOR_TYPES)}. Falling back to "
                "Regular."
            )
        return BEHAVIOR_TYPES["Regular"]

    def _create_agent_msg(self, agent_prim, index):
        agent_ref = self.config["hunav_loader"]["ros__parameters"]["agents"][index]
        agent_cfg = self.config["hunav_loader"]["ros__parameters"][agent_ref]

        agent = Agent()
        agent.id = int(agent_cfg["id"])
        agent.type = Agent.PERSON
        if self.config["hunav_loader"]["ros__parameters"]["simulator"] == "Gazebo":
            agent.skin = agent_cfg["skin"]
        else:
            agent.skin = 0  
        agent.name = f"Agent{index + 1}"
        agent.group_id = int(agent_cfg["group_id"])
        agent.radius = float(agent_cfg["radius"])
        agent.desired_velocity = float(agent_cfg["max_vel"])

        # Read the pose back from the behavior agent, not from the prim: the
        # engine writes agent transforms to Fabric, so xformOp:translate on the
        # character keeps its authored spawn value while the agent walks.
        handle = self.agent_handles[index]
        pos, _quat, lin = self.driver.pose(handle)
        yaw = self.driver.yaw(handle)

        agent.position.position.x = float(pos[0])
        agent.position.position.y = float(pos[1])
        agent.position.position.z = float(pos[2])
        # Orientation is rebuilt from the agent's own facing direction rather
        # than passed through from get_world_rotation(), which also carries the
        # rig's internal axis corrections (this character's forward is -Y).
        agent.position.orientation.x = 0.0
        agent.position.orientation.y = 0.0
        agent.position.orientation.z = float(math.sin(yaw * 0.5))
        agent.position.orientation.w = float(math.cos(yaw * 0.5))
        agent.yaw = float(self.normalize_angle(yaw))

        # Velocity is finite-differenced in the driver; the behavior API's
        # get_linear_velocity() reports zero even for a walking agent.
        ang_z = self.normalize_angle(yaw - self._last_yaw.get(index, yaw)) / self.dt
        self._last_yaw[index] = yaw
        agent.linear_vel = float(math.hypot(lin[0], lin[1]))
        agent.angular_vel = float(abs(ang_z))
        agent.velocity.linear.x = float(lin[0])
        agent.velocity.linear.y = float(lin[1])
        agent.velocity.linear.z = float(lin[2])
        agent.velocity.angular.x = 0.0
        agent.velocity.angular.y = 0.0
        agent.velocity.angular.z = float(ang_z)

        # Goals
        agent.cyclic_goals = agent_cfg["cyclic_goals"]
        agent.goal_radius = float(agent_cfg["goal_radius"])
        
        # Behavior
        beh = agent_cfg["behavior"]
        configuration = int(beh["configuration"])

        # The behaviour *type* is what decides how the robot enters this agent's
        # social-force computation: BEH_REGULAR pushes it in as another human,
        # BEH_IMPASSIVE as an obstacle, and anything the agent manager does not
        # recognise falls through to a branch that leaves the robot out
        # altogether. Sending 0 -- which is what an unset field is, and what this
        # wrapper used to send -- silently selected that last case for every
        # agent in every scenario.
        behavior_type = self._behavior_type_id(beh, agent_cfg)

        # SFM defaults applied when configuration is BEH_CONF_DEFAULT.
        #
        # NOTE: these are the values this wrapper has always used, and they are
        # goal/obstacle swapped relative to hunav_loader's own defaults (which
        # are goal 2.0, obstacle 10.0, matching upstream lightsfm). The wrapper's
        # numbers are the ones the simulation integrates, so they are left alone
        # here rather than changed underneath the four shipped scenarios.
        # Author new scenarios with configuration 1 and the question does not
        # arise -- see new_behavior/scenario_init_design.md.
        DEFAULT_SFM_PARAMS = {
            "goal_force_factor": 10.0,
            "obstacle_force_factor": 2.0,
            "social_force_factor": 5.0,
        }

        def clamp_value(value: float, min_val: float, max_val: float) -> float:
            """Constrain value within specified range."""
            return max(min_val, min(value, max_val))

        authored = {
            "social_force_factor": float(beh["social_force_factor"]),
            "goal_force_factor": float(beh["goal_force_factor"]),
            "obstacle_force_factor": float(beh["obstacle_force_factor"]),
            "other_force_factor": float(beh["other_force_factor"]),
        }

        # Set SFM parameters based on configuration type
        if configuration == 0:  # Default configuration
            sfm_params = DEFAULT_SFM_PARAMS.copy()
            sfm_params["other_force_factor"] = authored["other_force_factor"]
        elif configuration == 1:  # Custom, deliberately unconstrained
            sfm_params = authored
        else:  # Random configurations: clamp to the ranges hunav_loader uses,
               # so the two ends of the pipeline agree on what is legal.
            sfm_params = authored
            for param, (min_val, max_val) in FORCE_FACTOR_RANGES.items():
                if param in sfm_params:
                    sfm_params[param] = clamp_value(sfm_params[param], min_val, max_val)

        # duration/once/vel/dist drive the timed behaviours (Scared, Curious,
        # Surprised, Threatening). They are optional in the scenario schema --
        # the scenarios written before this wrapper sent them omit the keys --
        # so fall back to hunav_loader's own declared defaults.
        vel = clamp_value(float(beh.get("vel", 1.0)), *VEL_RANGE)

        agent.behavior = AgentBehavior(
            type=behavior_type,
            state=1,
            configuration=configuration,
            duration=float(beh.get("duration", 40.0)),
            once=bool(beh.get("once", True)),
            vel=vel,
            dist=float(beh.get("dist", 0.0)),
            social_force_factor=sfm_params["social_force_factor"],
            goal_force_factor=sfm_params["goal_force_factor"],
            obstacle_force_factor=sfm_params["obstacle_force_factor"],
            other_force_factor=sfm_params["other_force_factor"],
        )

        # Obstacle detection
        max_distance = 4.0
        agent.closest_obs = []
        sensor_offsets = [0.05, 0.1, 0.25, 0.5, 1.0]
        with self.perf.span("rays"):
            hits = self.get_closest_obstacles(pos, max_distance, sensor_offsets)
        for hit in hits:
            if hit[1] is not None:
                pt = Point(
                    x=float(hit[1][0]),
                    y=float(hit[1][1]),
                    z=float(hit[1][2]),
                )
            else:
                pt = Point(x=10000.0, y=10000.0, z=10000.0)
            agent.closest_obs.append(pt)
        return agent

    def _call_compute(self, agents_msg, robot_msg):
        try:
            req = ComputeAgents.Request()
            req.current_agents = agents_msg
            req.robot = robot_msg
            future = self.compute_agents_client.call_async(req)

            # This blocks the PhysX step thread for the whole round trip to
            # hunav_agent_manager, so it is timed on its own: it is latency we
            # are waiting on, not work we are doing, and the two have different
            # fixes.
            with self.perf.span("svc_spin"):
                rclpy.spin_until_future_complete(self.node, future)
            if future.done():
                resp = future.result()
                if resp is None:
                    print("[HuNavManager] No response from service.")
                else:
                    with self.perf.span("drive"):
                        self._update_agents(resp.updated_agents)
                return resp
            else:
                print("[HuNavManager] Service response not completed.")
                return None
        except Exception as e:
            print(f"[HuNavManager] Error calling service: {e}")
            return None

    def _update_agents(self, updated_agents):
        """Hand one tick of HuNavSim's output to the motion matcher.

        HuNavSim's social-force model decides where each agent should be; the
        behavior system decides how the body gets there. The agent is given a
        goal a short way along its social-force velocity rather than its exact
        commanded position -- asking it to move to where it already stands
        produces no gait. Tracking error in the spike was a mean of 0.17 m.

        Nothing is written to the character prims here. The engine owns their
        transforms, and writing a competing transform is what used to carry
        agents off the map.
        """
        debug = _os.environ.get("HUNAV_ANIM_DEBUG", "0") != "0"

        for upd in updated_agents.agents:
            idx = upd.id - 1
            if idx < 0 or idx >= len(self.agent_handles):
                continue
            handle = self.agent_handles[idx]
            if handle is None:
                continue

            position = (
                upd.position.position.x,
                upd.position.position.y,
                upd.position.position.z,
            )
            if self.terrain_follow:
                # HuNavSim's model is 2-D and never reports a Z. The navmesh
                # already keeps agents on the walkable surface, so this only
                # matters for the goal's height on terraced ground.
                position = (
                    position[0],
                    position[1],
                    self.resolve_agent_z(
                        self.agents[idx].GetPath(),
                        position[0],
                        position[1],
                        position[2],
                    ),
                )

            velocity = (
                upd.velocity.linear.x,
                upd.velocity.linear.y,
                upd.velocity.linear.z,
            )
            if not self._no_drive:
                self.driver.drive(handle, position, velocity)

            if debug:
                key = self.agent_skelroots[idx]
                n = self._anim_debug_ticks.get(key, 0)
                if n == 0 or n % 100 == 0:
                    speed = math.hypot(velocity[0], velocity[1])
                    actual = handle.agent.get_world_translation()
                    drift = math.hypot(
                        float(actual[0]) - position[0],
                        float(actual[1]) - position[1],
                    )
                    print(
                        f"[anim] {key} |v|={speed:.3f} "
                        f"cmd=({position[0]:.2f},{position[1]:.2f}) "
                        f"actual=({float(actual[0]):.2f},{float(actual[1]):.2f}) "
                        f"drift={drift:.3f}m",
                        flush=True,
                    )
                self._anim_debug_ticks[key] = n + 1

    def get_character_model_from_skin(self, skin_value):
        """
        Get character model path based on skin value.
        
        Args:
            skin_value (int): The skin ID from agent configuration
                            - 0: Random character model selection
                            - 1-11: Specific character models
            
        Returns:
            str: Path to the character model, or None if skin_value is invalid
        """
        # Handle random skin option (skin value 0)
        if skin_value == 0:
            random_index = random.randint(0, len(self.target_model_paths) - 1)
            return self.target_model_paths[random_index]
        
        # Handle specific skin values (1-11 mapped to model indices 0-10)
        if isinstance(skin_value, (int, float)) and skin_value in self.skin_to_model_mapping:
            model_index = self.skin_to_model_mapping[int(skin_value)]
            if 0 <= model_index < len(self.target_model_paths):
                return self.target_model_paths[model_index]
        
        return None

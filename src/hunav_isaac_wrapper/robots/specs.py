#!/usr/bin/env python3
"""
robots/specs.py

The single registry of robots this wrapper can spawn.

Before this module the robot list lived in four hand-maintained places: the
asset table in ``asset_paths.py``, a ``robot_configs`` dict built inline inside
``TeleopHuNavSim.__init__``, and the ``ROBOTS`` list plus its descriptions dict
in ``scripts/main.py``. Adding a robot meant editing all four and hoping they
agreed; the batch-mode default was even a positional index into ``ROBOTS``, so
inserting an entry silently changed which robot an unattended run used.

Everything except the version-sensitive USD paths now lives here. Those stay in
``asset_paths.py``, which is the one place that tracks where an asset moved
between Isaac Sim releases.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ..asset_paths import get_isaac_major, robot_usd_relative_path


# Isaac Sim's stock rendering/agent cadence. HuNavManager's own dt is 1/20 and
# it divides by that constant when differencing agent poses, so this is not a
# free parameter.
RENDER_DT = 1.0 / 20.0

# The Go2 locomotion policy was trained at sim.dt 0.005 with decimation 4
# (Isaac/Samples/Policies/go2/physx_env.yaml). At the wrapper's usual 20 Hz it
# does not stand up. Relaxing this trades gait quality for frame rate.
GO2_PHYSICS_DT = 1.0 / 200.0

# Isaac Sim 6.0's own Go2 policy does not walk on 6.0.1-rc.7 -- it stands, then
# splays and drags under any non-zero command. This package ships its own policy
# instead (config/policies/go2/, with provenance in that directory's README) and
# corrects two upstream mismatches: the default asset is the Newton one rather
# than the PhysX one the policy is trained against, and PolicyController drives
# the joints through PhysX's implicit PD instead of Isaac Lab's explicit
# DCMotor. Hence asset_paths resolving the IsaacLab USD, and actuation="explicit".

# The command range the Go2 policy was trained over
# (physx_env.yaml: commands.base_velocity.ranges). Commands are clamped to it
# rather than passed through, because the policy has never seen anything larger
# and responds to it by falling over.
GO2_MAX_LINEAR = 1.0
GO2_MAX_ANGULAR = 1.0


@dataclass(frozen=True)
class RobotSpec:
    """Everything the wrapper needs to know about one selectable robot."""

    key: str
    """The name used on the command line and in last_launch_config.json."""

    prim_name: str
    """The robot is spawned at ``/World/<prim_name>``. Must be unique."""

    description: str
    """One line, shown in the interactive robot menu."""

    driver: str
    """Which RobotDriver implementation to build: "wheeled" or "go2_policy"."""

    # Exactly one of these says where the robot's USD comes from. Both may be
    # None, in which case the driver supplies its own default.
    asset_key: Optional[str] = None
    """Key into asset_paths._ROBOT_USD -- streamed from the Isaac asset bucket."""

    bundled_usd: Optional[str] = None
    """Filename under config/robots/, shipped with this package."""

    physics_dt: float = RENDER_DT
    """PhysX step. Anything below RENDER_DT makes world.step() substep."""

    spawn_z_offset: float = 0.0
    """Added to the world's spawn z. A legged robot has to be dropped in."""

    spawn_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Initial orientation as wxyz. The wheeled robots have always used
    (0, 0, 0, 1), which is a 180-degree yaw rather than identity; that is left
    alone so their heading does not change under this refactor."""

    backend: str = "numpy"
    device: Optional[str] = None
    """Passed to World(). The Go2 policy runs its network on the GPU and reads
    articulation state as torch tensors, so it needs the torch backend on cuda."""

    # Wheeled robots only.
    wheel_dof_names: Optional[Tuple[str, ...]] = None
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None

    # Fields HuNavManager puts in the robot's Agent message. These used to be
    # hardcoded, so every robot claimed a 0.5 m radius regardless of its size.
    hunav_radius: float = 0.5
    hunav_desired_velocity: float = 1.0

    actuation: str = "implicit"
    """How joint commands reach PhysX, for policy-driven robots.

    ``"implicit"`` leaves PhysX's own PD drives in charge of the position
    targets, which is what ``PolicyController`` does by default.
    ``"explicit"`` zeroes those drives and computes the torque in Python each
    physics step from Isaac Lab's DCMotor model -- the actuator the policy was
    actually trained against. Ignored by the wheeled driver."""

    policy_path: Optional[str] = None
    env_config_path: Optional[str] = None
    """Override the locomotion policy and its env config. Relative paths are
    resolved against config/policies/ in this package; ``None`` takes whatever
    the policy class defaults to (for the Go2, the assets in
    Isaac/Samples/Policies/go2/)."""

    publish_odom_tf: bool = False
    """Publish /odom, /tf and /joint_states from Python. Off for carter_ROS:
    its bundled USD carries its own OmniGraph publishers and would double up."""

    def with_resolved_policy(self, config_resolver):
        """Return a copy whose policy paths are absolute.

        A spec names its policy relative to config/ so the registry stays
        readable and location-independent; the policy classes need real paths.
        Absolute paths and URLs are passed through untouched.
        """
        import dataclasses

        def resolve(value):
            if value is None:
                return None
            if "://" in value or os.path.isabs(value):
                return value
            return config_resolver(value)

        return dataclasses.replace(
            self,
            policy_path=resolve(self.policy_path),
            env_config_path=resolve(self.env_config_path),
        )

    def usd_path(self, assets_root_path, bundled_resolver, isaac_major=None):
        """Resolve this robot's USD, or None if the driver supplies its own.

        ``bundled_resolver`` maps a filename to an absolute path under
        config/robots/ -- it is passed in rather than imported so this module
        stays importable without booting Isaac Sim, which scripts/main.py needs
        in order to build its robot menu.

        Only the selected robot's USD is resolved. Building every robot's path
        up front, as the old robot_configs dict literal did, meant that a
        missing (still zipped) Carter asset broke ``--robot jetbot`` too.

        Raises:
            FileNotFoundError: if a bundled USD is missing (it ships zipped).
        """
        if self.bundled_usd is not None:
            return bundled_resolver(self.bundled_usd)
        if self.asset_key is None:
            return None
        relative = robot_usd_relative_path(self.asset_key, isaac_major)
        if relative is None:
            return None
        return os.path.join(assets_root_path, relative)


_SPECS = (
    RobotSpec(
        key="jetbot",
        prim_name="Jetbot",
        description="Small differential drive robot for basic navigation",
        driver="wheeled",
        asset_key="jetbot",
        wheel_dof_names=("left_wheel_joint", "right_wheel_joint"),
        wheel_radius=0.0325,
        wheel_base=0.118,
    ),
    RobotSpec(
        key="create3",
        prim_name="Create3",
        description="iRobot Create3 educational robot platform",
        driver="wheeled",
        asset_key="create3",
        wheel_dof_names=("left_wheel_joint", "right_wheel_joint"),
        wheel_radius=0.03575,
        wheel_base=0.233,
    ),
    RobotSpec(
        key="carter",
        prim_name="Nova_Carter",
        description="NVIDIA Carter robot for advanced navigation (Isaac Sim 4.5 only)",
        driver="wheeled",
        asset_key="carter",
        wheel_dof_names=("joint_wheel_left", "joint_wheel_right"),
        wheel_radius=0.14,
        wheel_base=0.413,
    ),
    RobotSpec(
        key="carter_ROS",
        prim_name="Nova_Carter",
        description="Carter with full ROS2 Nav2 stack support",
        driver="wheeled",
        bundled_usd="nova_carter_ros2_sensors.usd",
        wheel_dof_names=("joint_wheel_left", "joint_wheel_right"),
        wheel_radius=0.14,
        wheel_base=0.413,
    ),
    RobotSpec(
        key="go2",
        prim_name="Go2",
        description="Unitree Go2 quadruped, flat-terrain locomotion policy (Isaac Sim 6.0+)",
        driver="go2_policy",
        asset_key="go2",
        physics_dt=GO2_PHYSICS_DT,
        # The dog stands about 0.32 m tall. Isaac's own Go2 example spawns it at
        # 0.5 and lets the policy catch it; this is the same idea relative to
        # whatever ground the world puts under it.
        spawn_z_offset=0.45,
        spawn_orientation=(1.0, 0.0, 0.0, 0.0),
        backend="torch",
        device="cuda",
        # Isaac Sim's shipped Go2 policy does not hold a gait in 6.0.1-rc.7, so
        # this ships its own, trained with Isaac Lab against the same asset.
        # See config/policies/go2/README.md for how to regenerate it.
        policy_path="policies/go2/policy.pt",
        env_config_path="policies/go2/env.yaml",
        actuation="explicit",
        # Body is roughly 0.70 x 0.31 m, so half its length rather than the
        # 0.5 m every robot used to claim.
        hunav_radius=0.35,
        hunav_desired_velocity=GO2_MAX_LINEAR,
        publish_odom_tf=True,
    ),
)

ROBOT_SPECS = {spec.key: spec for spec in _SPECS}

# Batch mode and the interactive menu both default to this. Named rather than
# indexed, so adding a robot above cannot quietly change it.
DEFAULT_ROBOT = "carter_ROS"


def robot_names():
    """Selectable robot keys, in menu order."""
    return [spec.key for spec in _SPECS]


def robot_descriptions():
    """{key: description}, for the interactive menu."""
    return {spec.key: spec.description for spec in _SPECS}


def get_spec(key):
    """Look up a RobotSpec by CLI name.

    Raises:
        ValueError: if the name is not a known robot.
    """
    try:
        return ROBOT_SPECS[key]
    except KeyError:
        raise ValueError(
            f"Unsupported robot_name: {key}. Known robots: {', '.join(robot_names())}"
        ) from None


def require_available(spec, isaac_major=None):
    """Raise if this robot has no asset in the running Isaac Sim release.

    Unlike the old ``is_robot_available``, which returned True for any name it
    did not recognise, this only consults the table for robots that actually
    resolve their USD through it.
    """
    if spec.asset_key is None:
        return
    if isaac_major is None:
        isaac_major = get_isaac_major()
    if robot_usd_relative_path(spec.asset_key, isaac_major) is None:
        raise ValueError(
            f"Robot '{spec.key}' has no asset in Isaac Sim {isaac_major}.x. "
            "Use 'carter_ROS', which ships with this package."
        )

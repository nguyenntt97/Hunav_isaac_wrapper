"""
spec.py

The scenario model: dataclasses mirroring exactly what `hunav_loader` declares,
a validator for the invariants HuNav enforces silently, and a YAML emitter.

No Isaac Sim imports. Navmesh-dependent checks take an injected `nav` object so
this module stays importable and testable under a plain interpreter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


# --- HuNav enums, from hunav_msgs/msg/AgentBehavior.msg -------------------

BEHAVIOR_TYPES: Dict[str, int] = {
    "Regular": 1,
    "Impassive": 2,
    "Surprised": 3,
    "Scared": 4,
    "Curious": 5,
    "Threatening": 6,
}
BEHAVIOR_NAMES: Dict[int, str] = {v: k for k, v in BEHAVIOR_TYPES.items()}

BEH_CONF_DEFAULT = 0
BEH_CONF_CUSTOM = 1
BEH_CONF_RANDOM_NORMAL = 2
BEH_CONF_RANDOM_UNIFORM = 3

# Ranges hunav_loader clamps to when configuration != CUSTOM, from
# hunav_loader.cpp. Authoring against these keeps what you write and what runs
# the same thing.
FORCE_FACTOR_RANGES: Dict[str, Tuple[float, float]] = {
    "goal_force_factor": (2.0, 5.0),
    "obstacle_force_factor": (2.0, 50.0),
    "social_force_factor": (5.0, 20.0),
    "other_force_factor": (0.0, 25.0),
}

# What configuration == BEH_CONF_DEFAULT silently overwrites the authored
# values with.
DEFAULT_CONF_FORCES: Dict[str, float] = {
    "goal_force_factor": 2.0,
    "obstacle_force_factor": 10.0,
    "social_force_factor": 5.0,
}

# hunav_loader rewrites behaviour velocity into this band without telling you.
VEL_RANGE: Tuple[float, float] = (0.0, 1.8)


@dataclass
class Problem:
    """One validation failure. `fatal` means the scenario will not run."""

    where: str
    message: str
    fatal: bool = True

    def __str__(self) -> str:
        mark = "ERROR" if self.fatal else "warn "
        return f"[{mark}] {self.where}: {self.message}"


@dataclass
class Pose:
    """Spawn pose. `h` is yaw about +Z in radians, as authored on the prim."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    h: float = 0.0

    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class BehaviorSpec:
    """The `behavior:` block. Field names match the ROS parameter names."""

    type: str = "Regular"
    # CUSTOM by default on purpose: configuration 0 makes hunav_loader replace
    # the force factors with its own defaults, so anything authored is lost.
    configuration: int = BEH_CONF_CUSTOM
    duration: float = 40.0
    once: bool = True
    vel: float = 1.0
    dist: float = 0.0
    goal_force_factor: float = 2.0
    obstacle_force_factor: float = 10.0
    social_force_factor: float = 5.0
    other_force_factor: float = 20.0

    @property
    def type_id(self) -> int:
        """The uint8 the simulator expects. Raises on an unknown name."""
        try:
            return BEHAVIOR_TYPES[self.type]
        except KeyError:
            raise ValueError(
                f"unknown behavior type {self.type!r}; expected one of "
                f"{', '.join(BEHAVIOR_TYPES)}"
            ) from None


@dataclass
class AgentSpec:
    """One agent, mirroring the per-agent block hunav_loader declares."""

    id: int
    name: str
    skin: int = 0
    group_id: int = -1
    max_vel: float = 1.5
    radius: float = 0.4
    goal_radius: float = 0.3
    cyclic_goals: bool = True
    init_pose: Pose = field(default_factory=Pose)
    behavior: BehaviorSpec = field(default_factory=BehaviorSpec)
    goals: List[int] = field(default_factory=list)


@dataclass
class NavmeshProvenance:
    """What the navmesh was when this scenario was authored.

    Recorded so the run-mode bake can be the same bake, and so a scenario whose
    assumptions have gone stale says so instead of quietly misbehaving.
    """

    settings_digest: str = ""
    volume_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_z: float = 0.0
    sampling_cm: float = 0.0


@dataclass
class ScenarioSpec:
    """A whole scene's agents, goals and the navmesh they were authored on."""

    yaml_base_name: str
    map: str
    agents: List[AgentSpec] = field(default_factory=list)
    global_goals: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    simulator: str = "Isaac Sim"
    publish_people: bool = True
    navmesh: Optional[NavmeshProvenance] = None

    # --- construction ---------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioSpec":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw, fallback_base_name=_basename_no_ext(path))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], fallback_base_name: str = "") -> "ScenarioSpec":
        params = raw.get("hunav_loader", {}).get("ros__parameters", {})
        if not params:
            raise ValueError("not a HuNav scenario: no hunav_loader.ros__parameters")

        goals: Dict[int, Tuple[float, float]] = {}
        for key, value in (params.get("global_goals") or {}).items():
            goals[int(key)] = (float(value["x"]), float(value["y"]))

        agents: List[AgentSpec] = []
        for name in params.get("agents", []) or []:
            block = params.get(name)
            if not isinstance(block, dict):
                continue
            agents.append(_agent_from_block(name, block))

        provenance = None
        auth = raw.get("hunav_isaac_authoring", {}).get("ros__parameters", {})
        nav_block = (auth or {}).get("navmesh")
        if isinstance(nav_block, dict):
            provenance = NavmeshProvenance(
                settings_digest=str(nav_block.get("settings_digest", "")),
                volume_min=tuple(nav_block.get("volume_min", (0.0, 0.0, 0.0))),
                volume_max=tuple(nav_block.get("volume_max", (0.0, 0.0, 0.0))),
                ground_z=float(nav_block.get("ground_z", 0.0)),
                sampling_cm=float(nav_block.get("sampling_cm", 0.0)),
            )

        return cls(
            yaml_base_name=str(params.get("yaml_base_name") or fallback_base_name),
            map=str(params.get("map", "")),
            agents=agents,
            global_goals=goals,
            simulator=str(params.get("simulator", "Isaac Sim")),
            publish_people=bool(params.get("publish_people", True)),
            navmesh=provenance,
        )

    # --- lookups --------------------------------------------------------

    def agent_by_id(self, agent_id: int) -> Optional[AgentSpec]:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def sorted_agents(self) -> List[AgentSpec]:
        """Agents in id order.

        Prim traversal and `sorted()` on names both give agent1, agent10,
        agent2 -- which renumbers everyone past nine. Order by id instead.
        """
        return sorted(self.agents, key=lambda a: a.id)

    def goal_xy(self, goal_id: int) -> Optional[Tuple[float, float]]:
        return self.global_goals.get(goal_id)

    # --- editing --------------------------------------------------------

    def renumber_goals(self) -> Dict[int, int]:
        """Compact goal ids to 1..N, rewriting every agent's ring.

        hunav_loader stops scanning `global_goals` at the first missing id, so a
        gap silently deletes every goal above it. Any edit that removes a goal
        has to come back through here.

        Returns the old id -> new id mapping.
        """
        ordered = sorted(self.global_goals)
        mapping = {old: new for new, old in enumerate(ordered, start=1)}

        self.global_goals = {mapping[old]: self.global_goals[old] for old in ordered}
        for agent in self.agents:
            agent.goals = [mapping[g] for g in agent.goals if g in mapping]

        return mapping

    # --- validation -----------------------------------------------------

    def validate(self, nav: Any = None, snap_tolerance: float = 0.5) -> List[Problem]:
        """Every invariant HuNav enforces silently, checked loudly.

        `nav` is optional and duck-typed: anything with `snap(xyz)` and
        `reachable(a, b)` (see `nav_mesh_plugin.sampling.NavmeshSampler`). The
        pure checks always run; the geometric ones are skipped without it.
        """
        problems: List[Problem] = []

        problems += self._check_goal_table()
        problems += self._check_agents()
        problems += self._check_separation()
        if nav is not None:
            problems += self._check_navmesh(nav, snap_tolerance)

        return problems

    def _check_goal_table(self) -> List[Problem]:
        problems: List[Problem] = []
        if not self.global_goals:
            problems.append(Problem("global_goals", "no goals defined"))
            return problems

        ids = sorted(self.global_goals)
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            missing = sorted(set(expected) - set(ids))
            problems.append(
                Problem(
                    "global_goals",
                    f"ids must be contiguous from 1, got {ids}. hunav_loader stops "
                    f"scanning at the first gap, so goals above {missing[0] if missing else ids[-1]} "
                    "would not exist at runtime. Call renumber_goals().",
                )
            )
        return problems

    def _check_agents(self) -> List[Problem]:
        problems: List[Problem] = []

        if not self.agents:
            problems.append(Problem("agents", "no agents defined"))
            return problems

        seen_ids: Dict[int, str] = {}
        seen_names: Dict[str, int] = {}
        for agent in self.agents:
            where = f"{agent.name} (id {agent.id})"

            if agent.id < 1:
                problems.append(Problem(where, f"id must be >= 1, got {agent.id}"))
            if agent.id in seen_ids:
                problems.append(
                    Problem(where, f"duplicate id, already used by {seen_ids[agent.id]}; "
                                   "behavior tree filenames would collide")
                )
            seen_ids[agent.id] = agent.name

            if agent.name in seen_names:
                problems.append(Problem(where, "duplicate agent name"))
            seen_names[agent.name] = agent.id

            if agent.behavior.type not in BEHAVIOR_TYPES:
                problems.append(
                    Problem(
                        where,
                        f"unknown behavior type {agent.behavior.type!r}; the simulator "
                        "would fall through to the branch that ignores the robot "
                        f"entirely. Expected one of {', '.join(BEHAVIOR_TYPES)}.",
                    )
                )

            lo, hi = VEL_RANGE
            if not lo <= agent.behavior.vel <= hi:
                problems.append(
                    Problem(
                        where,
                        f"behavior.vel {agent.behavior.vel} outside [{lo}, {hi}]; "
                        "hunav_loader would rewrite the parameter silently",
                    )
                )

            if agent.max_vel <= 0.0:
                problems.append(Problem(where, f"max_vel must be > 0, got {agent.max_vel}"))
            if agent.radius <= 0.0:
                problems.append(Problem(where, f"radius must be > 0, got {agent.radius}"))
            if agent.goal_radius <= 0.0:
                problems.append(
                    Problem(where, f"goal_radius must be > 0, got {agent.goal_radius}")
                )

            problems += self._check_forces(agent, where)

            if not agent.goals:
                problems.append(Problem(where, "no goals; the agent has nowhere to walk"))
            for goal_id in agent.goals:
                if goal_id not in self.global_goals:
                    problems.append(
                        Problem(
                            where,
                            f"goal {goal_id} is not in global_goals; SetGoal returns "
                            "FAILURE and the agent never receives a destination",
                        )
                    )

        return problems

    def _check_forces(self, agent: AgentSpec, where: str) -> List[Problem]:
        problems: List[Problem] = []
        beh = agent.behavior

        if beh.configuration == BEH_CONF_DEFAULT:
            differing = [
                name
                for name, default in DEFAULT_CONF_FORCES.items()
                if not math.isclose(getattr(beh, name), default, rel_tol=1e-6)
            ]
            if differing:
                problems.append(
                    Problem(
                        where,
                        "configuration 0 makes hunav_loader overwrite "
                        f"{', '.join(differing)} with its defaults "
                        f"({', '.join(f'{k}={v}' for k, v in DEFAULT_CONF_FORCES.items())}); "
                        "use configuration 1 to keep the authored values",
                        fatal=False,
                    )
                )
            return problems

        if beh.configuration == BEH_CONF_CUSTOM:
            # Custom is deliberately unconstrained; nothing to check.
            return problems

        for name, (lo, hi) in FORCE_FACTOR_RANGES.items():
            value = getattr(beh, name)
            if not lo <= value <= hi:
                problems.append(
                    Problem(
                        where,
                        f"{name} {value} outside [{lo}, {hi}] for configuration "
                        f"{beh.configuration}; it would be clamped",
                        fatal=False,
                    )
                )
        return problems

    def _check_separation(self, margin: float = 0.2) -> List[Problem]:
        problems: List[Problem] = []
        agents = self.sorted_agents()
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                gap = math.dist(a.init_pose.xy(), b.init_pose.xy())
                need = a.radius + b.radius + margin
                if gap < need:
                    problems.append(
                        Problem(
                            f"{a.name} / {b.name}",
                            f"spawns {gap:.2f} m apart but need {need:.2f} m; "
                            "the agents interpenetrate at t=0",
                        )
                    )
        return problems

    def _check_navmesh(self, nav: Any, tolerance: float) -> List[Problem]:
        problems: List[Problem] = []

        for goal_id, (gx, gy) in sorted(self.global_goals.items()):
            snapped = nav.snap((gx, gy, self._ground_z()))
            if snapped is None:
                problems.append(
                    Problem(f"goal {goal_id}", f"({gx:.2f}, {gy:.2f}) is not on the navmesh")
                )
            elif math.dist((gx, gy), (snapped[0], snapped[1])) > tolerance:
                problems.append(
                    Problem(
                        f"goal {goal_id}",
                        f"({gx:.2f}, {gy:.2f}) is {math.dist((gx, gy), snapped[:2]):.2f} m "
                        "off the navmesh; agents will never register arrival",
                    )
                )

        for agent in self.sorted_agents():
            where = f"{agent.name} (id {agent.id})"
            spawn = agent.init_pose.xyz()
            snapped = nav.snap(spawn)
            if snapped is None:
                problems.append(Problem(where, "spawn is not on the navmesh"))
                continue
            drift = math.dist(spawn[:2], snapped[:2])
            if drift > tolerance:
                problems.append(
                    Problem(where, f"spawn is {drift:.2f} m off the navmesh")
                )

            # Walk the ring the way the agent will, including the wrap when
            # cyclic_goals is on.
            ring = [self.global_goals[g] for g in agent.goals if g in self.global_goals]
            if not ring:
                continue
            legs = [(spawn[:2], ring[0])]
            legs += [(ring[i], ring[i + 1]) for i in range(len(ring) - 1)]
            if agent.cyclic_goals and len(ring) > 1:
                legs.append((ring[-1], ring[0]))

            for src, dst in legs:
                a3 = (src[0], src[1], self._ground_z())
                b3 = (dst[0], dst[1], self._ground_z())
                if not nav.reachable(a3, b3):
                    problems.append(
                        Problem(
                            where,
                            f"no navmesh path from ({src[0]:.1f}, {src[1]:.1f}) to "
                            f"({dst[0]:.1f}, {dst[1]:.1f}); the agent is stranded on "
                            "a disconnected island",
                        )
                    )
                    break

        return problems

    def _ground_z(self) -> float:
        if self.navmesh is not None:
            return float(self.navmesh.ground_z)
        if self.agents:
            return min(a.init_pose.z for a in self.agents)
        return 0.0

    # --- emission -------------------------------------------------------

    def to_yaml(self) -> str:
        """Render the scenario in the exact shape hunav_loader declares.

        Written by hand rather than through yaml.dump for two reasons: ROS2
        params-file loading is type-strict, so a coordinate that lands on a
        round number must still emit as a float, and key order carries meaning
        for anyone diffing scenarios.
        """
        lines: List[str] = []
        add = lines.append

        add("hunav_loader:")
        add("  ros__parameters:")
        add(f"    yaml_base_name: {self.yaml_base_name}")
        add(f"    simulator: {self.simulator}")
        add(f"    map: {self.map}")
        add(f"    publish_people: {_b(self.publish_people)}")

        add("    global_goals:")
        for goal_id in sorted(self.global_goals):
            gx, gy = self.global_goals[goal_id]
            add(f"      {goal_id}:")
            add(f"        x: {_f3(gx)}")
            add(f"        y: {_f3(gy)}")

        agents = self.sorted_agents()
        add("    agents:")
        for agent in agents:
            add(f"      - {agent.name}")

        for agent in agents:
            beh = agent.behavior
            add(f"    {agent.name}:")
            add(f"      id: {int(agent.id)}")
            add(f"      group_id: {int(agent.group_id)}")
            add(f"      skin: {int(agent.skin)}")
            add(f"      max_vel: {_f(agent.max_vel)}")
            add(f"      radius: {_f(agent.radius)}")
            add(f"      goal_radius: {_f(agent.goal_radius)}")
            add(f"      cyclic_goals: {_b(agent.cyclic_goals)}")
            add("      init_pose:")
            add(f"        x: {_f3(agent.init_pose.x)}")
            add(f"        y: {_f3(agent.init_pose.y)}")
            add(f"        z: {_f3(agent.init_pose.z)}")
            add(f"        h: {_f3(agent.init_pose.h)}")
            add("      behavior:")
            add(f"        type: {beh.type}")
            add(f"        configuration: {int(beh.configuration)}")
            add(f"        duration: {_f(beh.duration)}")
            add(f"        once: {_b(beh.once)}")
            add(f"        vel: {_f(beh.vel)}")
            add(f"        dist: {_f(beh.dist)}")
            add(f"        goal_force_factor: {_f(beh.goal_force_factor)}")
            add(f"        obstacle_force_factor: {_f(beh.obstacle_force_factor)}")
            add(f"        social_force_factor: {_f(beh.social_force_factor)}")
            add(f"        other_force_factor: {_f(beh.other_force_factor)}")
            add("      goals:")
            for goal_id in agent.goals:
                add(f"        - {int(goal_id)}")

        if self.navmesh is not None:
            nav = self.navmesh
            # A sibling node name nothing is called, so ROS2 skips it and the
            # wrapper -- which reads only config["hunav_loader"] -- ignores it.
            add("")
            add("hunav_isaac_authoring:")
            add("  ros__parameters:")
            add("    navmesh:")
            add(f"      settings_digest: {nav.settings_digest}")
            add(f"      volume_min: [{_f3(nav.volume_min[0])}, {_f3(nav.volume_min[1])}, {_f3(nav.volume_min[2])}]")
            add(f"      volume_max: [{_f3(nav.volume_max[0])}, {_f3(nav.volume_max[1])}, {_f3(nav.volume_max[2])}]")
            add(f"      ground_z: {_f3(nav.ground_z)}")
            add(f"      sampling_cm: {_f(nav.sampling_cm)}")

        return "\n".join(lines) + "\n"

    def write_yaml(self, path: str) -> str:
        """Write the scenario, forcing `yaml_base_name` to match the filename.

        They have to agree: `yaml_base_name` is what hunav_agent_manager uses to
        find each agent's behavior tree, so a mismatch loads another scenario's
        trees, or none.
        """
        base = _basename_no_ext(path)
        if self.yaml_base_name != base:
            self.yaml_base_name = base
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_yaml())
        return path


# --- helpers -------------------------------------------------------------


def _agent_from_block(name: str, block: Dict[str, Any]) -> AgentSpec:
    pose_block = block.get("init_pose") or {}
    pose = Pose(
        x=float(pose_block.get("x", 0.0)),
        y=float(pose_block.get("y", 0.0)),
        z=float(pose_block.get("z", 0.0)),
        h=float(pose_block.get("h", 0.0)),
    )

    beh_block = block.get("behavior") or {}
    # The checked-in scenarios predate duration/once/vel/dist being sent, so
    # they omit those keys. Fall back to hunav_loader's own declared defaults
    # rather than inventing new ones.
    behavior = BehaviorSpec(
        type=str(beh_block.get("type", "Regular")),
        configuration=int(beh_block.get("configuration", BEH_CONF_DEFAULT)),
        duration=float(beh_block.get("duration", 40.0)),
        once=bool(beh_block.get("once", True)),
        vel=float(beh_block.get("vel", 1.0)),
        dist=float(beh_block.get("dist", 0.0)),
        goal_force_factor=float(beh_block.get("goal_force_factor", 2.0)),
        obstacle_force_factor=float(beh_block.get("obstacle_force_factor", 10.0)),
        social_force_factor=float(beh_block.get("social_force_factor", 5.0)),
        other_force_factor=float(beh_block.get("other_force_factor", 20.0)),
    )

    return AgentSpec(
        id=int(block.get("id", 0)),
        name=name,
        skin=int(block.get("skin", 0)),
        group_id=int(block.get("group_id", -1)),
        max_vel=float(block.get("max_vel", 1.5)),
        radius=float(block.get("radius", 0.4)),
        goal_radius=float(block.get("goal_radius", 0.3)),
        cyclic_goals=bool(block.get("cyclic_goals", True)),
        init_pose=pose,
        behavior=behavior,
        goals=[int(g) for g in (block.get("goals") or [])],
    )


def _basename_no_ext(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]


def _f3(value: float) -> str:
    """Coordinates, always three decimals and always a float to ROS2."""
    return f"{float(value):.3f}"


def _f(value: float) -> str:
    """A scalar that must stay a float. `2` would parse as int and throw."""
    text = repr(round(float(value), 6))
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _b(value: bool) -> str:
    return "true" if value else "false"

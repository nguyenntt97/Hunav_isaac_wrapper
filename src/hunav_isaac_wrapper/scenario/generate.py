"""
generate.py

Build a complete scenario from a scene: sample goals and spawns on the navmesh,
give each agent a reachable ring, and assign behaviors from a requested mix.

`nav` is duck-typed -- anything with `snap`, `reachable` and
`sample_connected_points` (see `nav_mesh_plugin.sampling.NavmeshSampler`) -- so
this module holds no Isaac Sim import and the assignment logic is testable with
a stub.

Deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import (
    BEHAVIOR_TYPES,
    BEH_CONF_CUSTOM,
    AgentSpec,
    BehaviorSpec,
    Pose,
    ScenarioSpec,
)

# Skins the wrapper maps to specific character assets. 0 means "pick at
# random", which is fine but makes a run unreproducible, so generated
# scenarios cycle the explicit ones instead.
_SKIN_CHOICES = tuple(range(1, 12))


def parse_behavior_mix(text: str) -> Dict[str, int]:
    """Parse `Regular:5,Curious:2,Scared:1` into a count per behavior.

    A bare list of names (`Regular,Scared`) means one of each.
    """
    mix: Dict[str, int] = {}
    for chunk in (part.strip() for part in text.split(",") if part.strip()):
        if ":" in chunk:
            name, _, count = chunk.partition(":")
            name = name.strip()
            try:
                number = int(count)
            except ValueError:
                raise ValueError(f"{chunk!r}: expected Behavior:count") from None
        else:
            name, number = chunk, 1

        if name not in BEHAVIOR_TYPES:
            raise ValueError(
                f"unknown behavior {name!r}; expected one of {', '.join(BEHAVIOR_TYPES)}"
            )
        if number < 0:
            raise ValueError(f"{chunk!r}: count cannot be negative")
        mix[name] = mix.get(name, 0) + number

    if not mix:
        raise ValueError("empty behavior mix")
    return mix


def expand_behavior_mix(mix: Dict[str, int], count: int, rng: random.Random) -> List[str]:
    """Turn a mix into exactly `count` behavior names.

    A mix that does not add up to `count` is filled or trimmed rather than
    rejected: asking for eight agents and naming five behaviors is a reasonable
    thing to do.
    """
    names: List[str] = []
    for name, number in mix.items():
        names.extend([name] * number)

    if not names:
        names = ["Regular"]

    if len(names) < count:
        # Top up with the most-requested behavior, so the stated proportions
        # stay roughly intact.
        filler = max(mix, key=lambda k: mix[k])
        names.extend([filler] * (count - len(names)))
    elif len(names) > count:
        rng.shuffle(names)
        names = names[:count]

    rng.shuffle(names)
    return names


def _order_ring(
    start: Tuple[float, float],
    goals: Sequence[Tuple[int, Tuple[float, float]]],
) -> List[int]:
    """Order a set of goals into a nearest-neighbour tour from `start`.

    A random order makes agents criss-cross the map, which looks wrong and
    makes the scenario harder to reason about. Nearest-neighbour is not optimal
    and does not need to be.
    """
    remaining = list(goals)
    cursor = start
    ring: List[int] = []

    while remaining:
        best = min(remaining, key=lambda item: math.dist(cursor, item[1]))
        ring.append(best[0])
        cursor = best[1]
        remaining.remove(best)

    return ring


def build_scenario(
    map_name: str,
    nav: Any,
    num_agents: int = 8,
    num_goals: int = 15,
    goals_per_agent: int = 5,
    behavior_mix: Optional[Dict[str, int]] = None,
    seed: int = 0,
    yaml_base_name: Optional[str] = None,
    min_goal_separation: float = 3.0,
    min_spawn_separation: float = 2.0,
    min_spawn_to_first_goal: float = 4.0,
    agent_radius: float = 0.4,
    max_vel: float = 1.5,
    goal_radius: float = 0.3,
    cyclic_goals: bool = True,
) -> Tuple[ScenarioSpec, List[str]]:
    """Sample a whole scenario. Returns the spec and any non-fatal notes."""
    rng = random.Random(seed)
    notes: List[str] = []

    if goals_per_agent > num_goals:
        notes.append(
            f"goals_per_agent {goals_per_agent} exceeds num_goals {num_goals}; "
            f"using {num_goals}"
        )
        goals_per_agent = num_goals

    # Goals first, all on one connected island: they are what the agents have
    # to be able to reach, so they define the region worth spawning into.
    goal_points = nav.sample_connected_points(
        num_goals, min_separation=min_goal_separation
    )
    if not goal_points:
        raise RuntimeError(
            "could not sample any goal positions; is the navmesh baked and "
            "does the volume cover the walkable area?"
        )
    if len(goal_points) < num_goals:
        notes.append(
            f"placed {len(goal_points)}/{num_goals} goals at "
            f"{min_goal_separation:.1f} m separation"
        )

    # Goal ids are contiguous from 1 because hunav_loader stops scanning at the
    # first gap. This is the only place they are assigned.
    global_goals: Dict[int, Tuple[float, float]] = {
        index: (float(point[0]), float(point[1]))
        for index, point in enumerate(goal_points, start=1)
    }

    anchor = goal_points[0]
    spawn_points = nav.sample_connected_points(
        num_agents, min_separation=min_spawn_separation, anchor=anchor
    )
    if not spawn_points:
        raise RuntimeError("could not sample any spawn positions on the navmesh")
    if len(spawn_points) < num_agents:
        notes.append(
            f"placed {len(spawn_points)}/{num_agents} spawns at "
            f"{min_spawn_separation:.1f} m separation"
        )

    behaviors = expand_behavior_mix(
        behavior_mix or {"Regular": num_agents}, len(spawn_points), rng
    )

    goal_items = list(global_goals.items())
    agents: List[AgentSpec] = []

    for index, spawn in enumerate(spawn_points):
        agent_id = index + 1
        spawn_xy = (float(spawn[0]), float(spawn[1]))

        # Prefer goals that are not on top of the spawn, so the agent has
        # somewhere to walk on its first tick.
        far_enough = [
            item
            for item in goal_items
            if math.dist(spawn_xy, item[1]) >= min_spawn_to_first_goal
        ]
        pool = far_enough if len(far_enough) >= goals_per_agent else goal_items
        if pool is goal_items and far_enough != goal_items:
            notes.append(
                f"agent{agent_id}: fewer than {goals_per_agent} goals are "
                f"{min_spawn_to_first_goal:.1f} m from its spawn; using the full set"
            )

        chosen = rng.sample(pool, min(goals_per_agent, len(pool)))
        ring = _order_ring(spawn_xy, chosen)

        first_xy = global_goals[ring[0]]
        heading = math.atan2(first_xy[1] - spawn_xy[1], first_xy[0] - spawn_xy[0])

        behavior = BehaviorSpec(
            type=behaviors[index],
            configuration=BEH_CONF_CUSTOM,
        )
        _tune_behavior(behavior, max_vel)

        agents.append(
            AgentSpec(
                id=agent_id,
                name=f"agent{agent_id}",
                skin=_SKIN_CHOICES[index % len(_SKIN_CHOICES)],
                group_id=-1,
                max_vel=max_vel,
                radius=agent_radius,
                goal_radius=goal_radius,
                cyclic_goals=cyclic_goals,
                init_pose=Pose(
                    x=float(spawn[0]),
                    y=float(spawn[1]),
                    z=float(spawn[2]),
                    h=float(heading),
                ),
                behavior=behavior,
                goals=ring,
            )
        )

    spec = ScenarioSpec(
        yaml_base_name=yaml_base_name or f"{map_name}_agents",
        map=map_name,
        agents=agents,
        global_goals=global_goals,
    )

    return spec, notes


def _tune_behavior(behavior: BehaviorSpec, max_vel: float) -> None:
    """Give a behavior the parameters its tree actually reads.

    Left at the dataclass defaults, `dist` is zero, which for the reactive
    behaviors means the robot is never close enough to trigger anything and the
    agent is Regular in all but name.
    """
    kind = behavior.type

    if kind == "Scared":
        behavior.dist = 5.0
        behavior.vel = min(max_vel * 1.2, 1.8)
        behavior.duration = 8.0
        behavior.once = False
        behavior.other_force_factor = 20.0
    elif kind == "Curious":
        behavior.dist = 10.0
        behavior.vel = min(max_vel * 0.8, 1.8)
        behavior.duration = 10.0
        behavior.once = False
    elif kind == "Surprised":
        behavior.dist = 5.0
        behavior.duration = 4.0
        behavior.once = True
    elif kind == "Threatening":
        behavior.dist = 4.5
        behavior.vel = min(max_vel, 1.8)
        behavior.duration = 10.0
        behavior.once = False
    else:
        # Regular and Impassive navigate normally; the difference between them
        # is made in the force computation, not in the tree.
        behavior.vel = min(max_vel, 1.8)

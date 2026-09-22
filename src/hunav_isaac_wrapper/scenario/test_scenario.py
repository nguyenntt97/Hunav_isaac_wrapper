#!/usr/bin/env python3
"""
test_scenario.py

Unit tests for the scenario model, the behavior-tree emitter and the generator.

No Isaac Sim: the navmesh is stubbed, so this runs under any interpreter with
PyYAML. The navmesh-backed half of the verification lives in
`nav_mesh_plugin/test_scenario_sampling.py`.

    python3 src/hunav_isaac_wrapper/scenario/test_scenario.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from hunav_isaac_wrapper.scenario.bt_emit import (  # noqa: E402
    bt_filename,
    emit_all_trees,
    emit_tree,
    verify_trees,
)
from hunav_isaac_wrapper.scenario.generate import (  # noqa: E402
    build_scenario,
    expand_behavior_mix,
    parse_behavior_mix,
)
from hunav_isaac_wrapper.scenario.bake import (  # noqa: E402
    navmesh_settings_digest,
)
from hunav_isaac_wrapper.scenario.spec import (  # noqa: E402
    BEHAVIOR_TYPES,
    BEH_CONF_CUSTOM,
    BEH_CONF_DEFAULT,
    AgentSpec,
    BehaviorSpec,
    NavmeshProvenance,
    Pose,
    ScenarioSpec,
)

PASSED = []
FAILED = []


def check(name):
    def decorator(fn):
        try:
            fn()
        except AssertionError as exc:
            FAILED.append((name, str(exc) or "assertion failed"))
            print(f"  FAIL  {name}: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
        else:
            PASSED.append(name)
            print(f"  ok    {name}", flush=True)
        return fn

    return decorator


# --- fixtures ------------------------------------------------------------


class FlatNavStub:
    """A navmesh that is one big walkable square with a wall at x == 0.

    Everything with x < 0 is one island and x > 0 is another, so reachability
    is a real question rather than always-true.
    """

    def __init__(self, half=20.0, split=False):
        self.half = half
        self.split = split
        self._seq = 0

    def snap(self, point):
        x, y, _z = point
        if abs(x) > self.half or abs(y) > self.half:
            return None
        return (float(x), float(y), 0.0)

    def reachable(self, a, b):
        if self.snap(a) is None or self.snap(b) is None:
            return False
        if self.split:
            return (a[0] < 0) == (b[0] < 0)
        return True

    def sample_connected_points(self, count, min_separation=2.0, anchor=None, **_):
        # A deterministic lattice, so tests do not depend on an RNG.
        points = []
        step = max(min_separation, 1.0)
        side = 1.0 if anchor is None or anchor[0] >= 0 else -1.0
        n = 0
        while len(points) < count and n < 10000:
            row, col = divmod(n, 12)
            x = side * (1.0 + col * step)
            y = -self.half + 1.0 + row * step
            n += 1
            if abs(x) > self.half or abs(y) > self.half:
                continue
            points.append((x, y, 0.0))
        return points


def tiny_spec(behavior="Regular", goals=(1, 2, 3)):
    return ScenarioSpec(
        yaml_base_name="unit_agents",
        map="unit",
        global_goals={1: (5.0, 0.0), 2: (0.0, 5.0), 3: (-5.0, 0.0)},
        agents=[
            AgentSpec(
                id=1,
                name="agent1",
                init_pose=Pose(1.0, 1.0, 0.0, 0.5),
                behavior=BehaviorSpec(type=behavior, configuration=BEH_CONF_CUSTOM),
                goals=list(goals),
            )
        ],
    )


# --- spec ----------------------------------------------------------------


@check("shipped scenarios parse and have no fatal problems")
def _():
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scenarios")
    )
    names = ["brownstone", "hospital", "office", "warehouse"]
    for name in names:
        spec = ScenarioSpec.from_yaml(os.path.join(root, f"{name}_agents.yaml"))
        assert spec.agents, f"{name}: no agents parsed"
        assert spec.global_goals, f"{name}: no goals parsed"
        fatal = [p for p in spec.validate() if p.fatal]
        assert not fatal, f"{name}: {fatal[0]}"


@check("yaml round trip preserves every field")
def _():
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scenarios")
    )
    spec = ScenarioSpec.from_yaml(os.path.join(root, "brownstone_agents.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "brownstone_agents.yaml")
        spec.write_yaml(path)
        again = ScenarioSpec.from_yaml(path)

    assert again.global_goals == spec.global_goals, "goal table drifted"
    assert again.map == spec.map
    assert again.yaml_base_name == spec.yaml_base_name
    for before, after in zip(spec.sorted_agents(), again.sorted_agents()):
        assert before == after, f"{before.name} changed across the round trip"


@check("a navmesh assignment survives the yaml round trip")
def _():
    settings = {
        "cellSize": 0.3, "agentHeight": 2.0, "agentRadius": 0.6,
        "agentMinRadius": None, "agentMaxClimb": 0.9, "agentMaxSlope": 45.0,
        "agentMinIslandRadius": 2.0, "excludeRigidBodies": True, "useGpu": True,
    }
    spec = tiny_spec()
    spec.navmesh = NavmeshProvenance(
        settings_digest=navmesh_settings_digest(settings),
        volume_min=(-53.0, -85.0, -2.0), volume_max=(32.0, 44.95, 4.0),
        ground_z=0.0, sampling_cm=30.0,
        assigned_prims=("/World/brownstone/Paths", "/World/brownstone/Plaza"),
        assigned_mesh_count=214, bake_settings=settings,
    )
    again = ScenarioSpec.from_dict(yaml.safe_load(spec.to_yaml())).navmesh

    assert again.has_assignment, "the run would fall back to deriving its own bake"
    assert again.assigned_prims == spec.navmesh.assigned_prims, again.assigned_prims
    assert again.assigned_mesh_count == 214, again.assigned_mesh_count
    # A bool rendered as 1.0, or None rendered as 0.0, is a different bake.
    assert again.bake_settings == settings, again.bake_settings
    assert navmesh_settings_digest(again.bake_settings) == again.settings_digest


@check("a scenario with no assignment still parses, and says it has none")
def _():
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scenarios")
    )
    spec = ScenarioSpec.from_yaml(os.path.join(root, "brownstone_agents.yaml"))
    assert spec.navmesh is not None, "the authoring block stopped parsing"
    assert not spec.navmesh.has_assignment
    assert spec.navmesh.bake_settings == {}


@check("write_yaml forces yaml_base_name to match the filename")
def _():
    spec = tiny_spec()
    with tempfile.TemporaryDirectory() as tmp:
        spec.write_yaml(os.path.join(tmp, "renamed_agents.yaml"))
    assert spec.yaml_base_name == "renamed_agents", spec.yaml_base_name


@check("coordinates emit as floats even on round numbers")
def _():
    spec = tiny_spec()
    spec.agents[0].init_pose = Pose(-37.0, 2.0, 0.0, 0.0)
    text = spec.to_yaml()
    assert "x: -37.000" in text, "a round coordinate would parse as int in ROS2"
    assert "goal_radius: 0.3" in text
    assert "duration: 40.0" in text, "behavior duration must stay a float"


@check("goal ids must be contiguous from 1")
def _():
    spec = tiny_spec()
    del spec.global_goals[2]
    spec.agents[0].goals = [1, 3]
    problems = [p for p in spec.validate() if "contiguous" in p.message]
    assert problems, "a gap in the goal table was not caught"


@check("renumber_goals closes the gap and rewrites the rings")
def _():
    spec = tiny_spec()
    del spec.global_goals[2]
    spec.agents[0].goals = [1, 3]
    mapping = spec.renumber_goals()

    assert sorted(spec.global_goals) == [1, 2], spec.global_goals
    assert spec.agents[0].goals == [1, 2], spec.agents[0].goals
    assert mapping == {1: 1, 3: 2}, mapping
    assert not [p for p in spec.validate() if p.fatal], "still invalid after renumber"


@check("a goal outside the table is caught")
def _():
    spec = tiny_spec(goals=(1, 9))
    problems = [p for p in spec.validate() if "not in global_goals" in p.message]
    assert problems, "dangling goal reference was not caught"


@check("an unknown behavior name is caught")
def _():
    spec = tiny_spec(behavior="Grumpy")
    problems = [p for p in spec.validate() if "unknown behavior" in p.message]
    assert problems and problems[0].fatal


@check("behavior vel outside the loader's band is caught")
def _():
    spec = tiny_spec()
    spec.agents[0].behavior.vel = 3.0
    problems = [p for p in spec.validate() if "behavior.vel" in p.message]
    assert problems and problems[0].fatal


@check("duplicate agent ids are caught")
def _():
    spec = tiny_spec()
    twin = AgentSpec(
        id=1, name="agent2", init_pose=Pose(9.0, 9.0), goals=[1],
        behavior=BehaviorSpec(configuration=BEH_CONF_CUSTOM),
    )
    spec.agents.append(twin)
    problems = [p for p in spec.validate() if "duplicate id" in p.message]
    assert problems and problems[0].fatal


@check("overlapping spawns are caught")
def _():
    spec = tiny_spec()
    spec.agents.append(
        AgentSpec(
            id=2, name="agent2", init_pose=Pose(1.05, 1.0), goals=[1],
            behavior=BehaviorSpec(configuration=BEH_CONF_CUSTOM),
        )
    )
    problems = [p for p in spec.validate() if "interpenetrate" in p.message]
    assert problems and problems[0].fatal


@check("configuration 0 warns that force factors are discarded")
def _():
    spec = tiny_spec()
    spec.agents[0].behavior.configuration = BEH_CONF_DEFAULT
    spec.agents[0].behavior.goal_force_factor = 9.0
    problems = [p for p in spec.validate() if "overwrite" in p.message]
    assert problems, "silent overwrite was not reported"
    assert not problems[0].fatal, "this is a warning, not an error"


@check("sorted_agents orders by id, not by name")
def _():
    spec = tiny_spec()
    spec.agents = [
        AgentSpec(id=10, name="agent10", goals=[1]),
        AgentSpec(id=2, name="agent2", goals=[1]),
    ]
    assert [a.id for a in spec.sorted_agents()] == [2, 10]


@check("off-navmesh spawns and goals are caught")
def _():
    spec = tiny_spec()
    spec.agents[0].init_pose = Pose(500.0, 500.0)
    problems = [p for p in spec.validate(nav=FlatNavStub()) if "navmesh" in p.message]
    assert problems, "a spawn far off the mesh was accepted"


@check("an unreachable ring is caught")
def _():
    spec = tiny_spec(goals=(1,))
    spec.global_goals = {1: (5.0, 0.0)}
    spec.agents[0].init_pose = Pose(-5.0, 0.0)
    problems = [
        p for p in spec.validate(nav=FlatNavStub(split=True)) if "no navmesh path" in p.message
    ]
    assert problems, "a goal across the divide was accepted"


@check("a valid scenario passes the navmesh checks")
def _():
    spec = tiny_spec(goals=(1, 2))
    spec.global_goals = {1: (5.0, 0.0), 2: (0.0, 5.0)}
    problems = [p for p in spec.validate(nav=FlatNavStub()) if p.fatal]
    assert not problems, problems[0] if problems else ""


# --- behavior trees ------------------------------------------------------


@check("bt filename matches what the agent manager composes")
def _():
    assert bt_filename("brownstone_agents", 3) == "brownstone_agents__agent_3_bt.xml"


@check("every behavior emits a parseable tree")
def _():
    for name in BEHAVIOR_TYPES:
        spec = tiny_spec(behavior=name)
        xml = emit_tree(spec.agents[0])
        root = ET.fromstring(xml)
        assert root.get("main_tree_to_execute") == "DefaultTree", name
        ids = {n.get("ID") for n in root.iter("BehaviorTree")}
        assert "DefaultTree" in ids, name


@check("the emitted ring matches the scenario ring, in order")
def _():
    spec = tiny_spec(goals=(3, 1, 2))
    root = ET.fromstring(emit_tree(spec.agents[0]))
    emitted = [int(n.get("goal_id")) for n in root.iter("SetGoal")]
    assert emitted == [3, 1, 2], emitted


@check("the last SetGoal is inverted so the fallback proceeds")
def _():
    root = ET.fromstring(emit_tree(tiny_spec(goals=(1, 2)).agents[0]))
    inverted = [
        int(g.get("goal_id"))
        for inv in root.iter("Inverter")
        for g in inv.iter("SetGoal")
    ]
    assert inverted == [2], inverted


@check("reactive behaviors substitute literals, never blackboard keys")
def _():
    spec = tiny_spec(behavior="Scared")
    spec.agents[0].behavior.dist = 6.0
    spec.agents[0].behavior.vel = 1.7
    spec.agents[0].behavior.other_force_factor = 22.0
    xml = emit_tree(spec.agents[0])

    root = ET.fromstring(xml)
    node = next(root.iter("ScaredNav"))
    assert node.get("runaway_vel") == "1.700", node.get("runaway_vel")
    assert node.get("scary_force_factor") == "22.000"

    visible = next(root.iter("IsRobotVisible"))
    assert visible.get("distance") == "6.000", visible.get("distance")

    # Only id and dt are ever put on the blackboard, so nothing else may be a key.
    for element in root.iter():
        for key, value in element.attrib.items():
            if value.startswith("{"):
                assert value in ("{id}", "{dt}"), f"{element.tag}.{key}={value}"


@check("Impassive navigates like Regular; the difference is in the forces")
def _():
    regular = ET.fromstring(emit_tree(tiny_spec(behavior="Regular").agents[0]))
    impassive = ET.fromstring(emit_tree(tiny_spec(behavior="Impassive").agents[0]))
    assert not list(impassive.iter("IsRobotVisible")), "Impassive should have no reaction branch"
    assert len(list(regular.iter("RegularNav"))) == len(list(impassive.iter("RegularNav")))


@check("emit_all_trees writes one file per agent and self-checks")
def _():
    spec = tiny_spec()
    spec.agents.append(
        AgentSpec(
            id=7, name="agent7", init_pose=Pose(9.0, 9.0), goals=[2, 3],
            behavior=BehaviorSpec(type="Curious", configuration=BEH_CONF_CUSTOM),
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        written = emit_all_trees(spec, tmp)
        assert len(written) == 2, written
        assert os.path.basename(written[1]) == "unit_agents__agent_7_bt.xml"
        assert verify_trees(spec, tmp) == []


@check("verify_trees catches a ring edited without regenerating")
def _():
    spec = tiny_spec()
    with tempfile.TemporaryDirectory() as tmp:
        emit_all_trees(spec, tmp)
        spec.agents[0].goals = [3, 2, 1]
        problems = verify_trees(spec, tmp)
    assert problems and "does not match" in problems[0], problems


@check("verify_trees catches a missing tree")
def _():
    spec = tiny_spec()
    with tempfile.TemporaryDirectory() as tmp:
        problems = verify_trees(spec, tmp)
    assert problems and "no tree" in problems[0], problems


# --- generation ----------------------------------------------------------


@check("behavior mix parses counts and bare names")
def _():
    assert parse_behavior_mix("Regular:5,Curious:2") == {"Regular": 5, "Curious": 2}
    assert parse_behavior_mix("Regular,Scared") == {"Regular": 1, "Scared": 1}
    for bad in ("Grumpy:1", "Regular:x", ""):
        try:
            parse_behavior_mix(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


@check("a mix is expanded to exactly the agent count")
def _():
    import random

    rng = random.Random(0)
    names = expand_behavior_mix({"Regular": 2, "Scared": 1}, 8, rng)
    assert len(names) == 8, len(names)
    names = expand_behavior_mix({"Regular": 20}, 3, rng)
    assert len(names) == 3


@check("a generated scenario validates against the same navmesh")
def _():
    nav = FlatNavStub()
    spec, _notes = build_scenario(
        "unit", nav, num_agents=6, num_goals=10, goals_per_agent=4,
        behavior_mix={"Regular": 3, "Curious": 2, "Scared": 1}, seed=0,
    )
    problems = [p for p in spec.validate(nav=nav) if p.fatal]
    assert not problems, problems[0]
    assert len(spec.agents) == 6
    assert sorted(spec.global_goals) == list(range(1, 11))


@check("generation is deterministic for a seed")
def _():
    nav = FlatNavStub()
    kwargs = dict(num_agents=5, num_goals=8, goals_per_agent=3, seed=42)
    a, _ = build_scenario("unit", nav, **kwargs)
    b, _ = build_scenario("unit", nav, **kwargs)
    assert a.to_yaml() == b.to_yaml(), "same seed produced a different scenario"

    c, _ = build_scenario("unit", nav, **{**kwargs, "seed": 43})
    assert c.to_yaml() != a.to_yaml(), "a different seed produced the same scenario"


@check("generated agents face their first goal")
def _():
    nav = FlatNavStub()
    spec, _ = build_scenario("unit", nav, num_agents=4, num_goals=8, goals_per_agent=3)
    for agent in spec.agents:
        goal = spec.global_goals[agent.goals[0]]
        expected = math.atan2(
            goal[1] - agent.init_pose.y, goal[0] - agent.init_pose.x
        )
        assert abs(agent.init_pose.h - expected) < 1e-6, agent.name


@check("reactive behaviors get a trigger distance, not zero")
def _():
    nav = FlatNavStub()
    spec, _ = build_scenario(
        "unit", nav, num_agents=4, num_goals=8,
        behavior_mix={"Scared": 1, "Curious": 1, "Surprised": 1, "Threatening": 1},
    )
    for agent in spec.agents:
        assert agent.behavior.dist > 0.0, f"{agent.name} ({agent.behavior.type})"
        assert agent.behavior.configuration == BEH_CONF_CUSTOM


@check("generated scenario emits trees that pass verification")
def _():
    nav = FlatNavStub()
    spec, _ = build_scenario("unit", nav, num_agents=5, num_goals=9, goals_per_agent=3)
    with tempfile.TemporaryDirectory() as tmp:
        emit_all_trees(spec, tmp)
        assert verify_trees(spec, tmp) == []


def main():
    print("\n" + "=" * 70)
    print("SCENARIO MODEL / BT EMITTER / GENERATOR")
    print("=" * 70)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed\n")
    if FAILED:
        for name, why in FAILED:
            print(f"  FAILED {name}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

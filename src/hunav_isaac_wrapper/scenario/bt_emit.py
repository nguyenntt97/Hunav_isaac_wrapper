"""
bt_emit.py

Emit one behavior tree XML per agent, deterministically.

`hunav_agent_manager` loads `<wrapper_src>/behavior_trees/{yaml_base_name}__agent_{id}_bt.xml`
for every agent. A missing or malformed file logs an error and that agent gets
no tree at all -- it spawns and never moves. The goal ring lives in these files
too, as `SetGoal` nodes, because the wrapper never populates `Agent.goals`.
So the YAML alone is not a scenario; these are.

Why not reuse the shipped templates verbatim: `BTScaredNav.xml` and friends
reference blackboard keys (`{duration}`, `{once}`, `{maxvel}`, `{dist}`,
`{forcefactor}`, `{stopdist}`, `{frontdist}`) that nothing ever sets --
`initializeBehaviorTree` puts only `id` and `dt` on the blackboard. Per-agent
values therefore have to be substituted as literal attributes, which is exactly
what the checked-in generated trees do.

Why not the LLM generator: `hunav_behavior_tree_generator` hardcodes Gazebo
workspace paths in its config, needs a reachable model endpoint, and is
non-deterministic. It stays available for bespoke trees.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from .spec import AgentSpec, ScenarioSpec


# Trigger distances the shipped templates use when a behavior needs one and the
# scenario leaves `behavior.dist` at zero. Taken from the templates themselves
# so a default-configured agent behaves the way the stock trees do.
DEFAULT_TRIGGER_DIST: Dict[str, float] = {
    "Surprised": 5.0,
    "Scared": 5.0,
    "Curious": 10.0,
    "Threatening": 4.5,
}

# Behaviors with no action node of their own. Impassive is not a tree at all:
# it is the same navigation as Regular, and the difference is made in
# computeForces by `behavior.type`, which puts the robot in the obstacle set.
NAV_ONLY_BEHAVIORS = ("Regular", "Impassive")


_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<root main_tree_to_execute="DefaultTree" BTCPP_format="4">
  <TreeNodesModel>
    <!-- Conditions -->
    <Condition ID="IsGoalReached">
      <input_port name="agent_id" type="int">1</input_port>
    </Condition>
    <Condition ID="IsRobotVisible">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="distance" type="double" default="5.0"/>
    </Condition>
    <Condition ID="TimeExpiredCondition">
      <input_port name="seconds" type="double" default="5.0"/>
      <input_port name="ts" type="double" default="0.1"/>
      <input_port name="only_once" type="bool" default="true"/>
    </Condition>

    <!-- Actions -->
    <Action ID="UpdateGoal">
      <input_port name="agent_id" type="int">1</input_port>
    </Action>
    <Action ID="SetGoal">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="goal_id" type="int"/>
    </Action>
    <Action ID="RegularNav">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="time_step" type="double" default="0.1"/>
    </Action>
    <Action ID="SurprisedNav">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="time_step" type="double" default="0.1"/>
      <input_port name="beh_duration" type="double" default="5.0"/>
      <input_port name="only_once" type="bool" default="true"/>
    </Action>
    <Action ID="CuriousNav">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="time_step" type="double" default="0.1"/>
      <input_port name="beh_duration" type="double" default="5.0"/>
      <input_port name="only_once" type="bool" default="true"/>
      <input_port name="agent_vel" type="double" default="1.0"/>
      <input_port name="stop_distance" type="double" default="0.5"/>
    </Action>
    <Action ID="ScaredNav">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="time_step" type="double" default="0.1"/>
      <input_port name="beh_duration" type="double" default="5.0"/>
      <input_port name="only_once" type="bool" default="true"/>
      <input_port name="runaway_vel" type="double" default="1.5"/>
      <input_port name="scary_force_factor" type="double" default="1.0"/>
    </Action>
    <Action ID="ThreateningNav">
      <input_port name="agent_id" type="int">1</input_port>
      <input_port name="time_step" type="double" default="0.1"/>
      <input_port name="beh_duration" type="double" default="5.0"/>
      <input_port name="only_once" type="bool" default="true"/>
      <input_port name="goal_dist" type="double" default="1.0"/>
    </Action>
  </TreeNodesModel>
"""

_FOOTER = "</root>\n"


def bt_filename(yaml_base_name: str, agent_id: int) -> str:
    """The exact filename hunav_agent_manager composes at bt_node.cpp:311."""
    return f"{yaml_base_name}__agent_{int(agent_id)}_bt.xml"


def _num(value: float) -> str:
    return f"{float(value):.3f}"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _set_goals_block(agent: AgentSpec, indent: str) -> List[str]:
    """The `SetGoal` chain that gives the agent its ring.

    Each goal is pushed once through `RunOnce`. The last is wrapped in an
    `Inverter` so the whole Sequence reports FAILURE on every tick and the
    enclosing Fallback always falls through to the navigation branch -- the
    shape the stock generated trees use.
    """
    lines = [f'{indent}<Sequence name="SetGoals">']
    inner = indent + "  "

    for position, goal_id in enumerate(agent.goals):
        last = position == len(agent.goals) - 1
        pad = inner + "  " if last else inner

        if last:
            lines.append(f"{inner}<Inverter>")
        lines.append(f"{pad}<RunOnce>")
        lines.append(f'{pad}  <SetGoal agent_id="{{id}}" goal_id="{int(goal_id)}"/>')
        lines.append(f"{pad}</RunOnce>")
        if last:
            lines.append(f"{inner}</Inverter>")

    lines.append(f"{indent}</Sequence>")
    return lines


def _reaction_block(agent: AgentSpec, indent: str) -> List[str]:
    """The behavior-specific branch, with every port substituted as a literal."""
    beh = agent.behavior
    kind = beh.type
    if kind in NAV_ONLY_BEHAVIORS:
        return []

    trigger = beh.dist if beh.dist > 0.0 else DEFAULT_TRIGGER_DIST.get(kind, 5.0)
    inner = indent + "  "

    lines = [f'{indent}<Sequence name="{kind}Reaction">']
    lines.append(f'{inner}<IsRobotVisible agent_id="{{id}}" distance="{_num(trigger)}"/>')
    lines.append(f"{inner}<Inverter>")
    lines.append(
        f'{inner}  <TimeExpiredCondition seconds="{_num(beh.duration)}" '
        f'ts="{{dt}}" only_once="{_bool(beh.once)}"/>'
    )
    lines.append(f"{inner}</Inverter>")

    common = (
        f'agent_id="{{id}}" time_step="{{dt}}" '
        f'beh_duration="{_num(beh.duration)}" only_once="{_bool(beh.once)}"'
    )
    if kind == "Surprised":
        lines.append(f"{inner}<SurprisedNav {common}/>")
    elif kind == "Scared":
        lines.append(
            f"{inner}<ScaredNav {common} "
            f'runaway_vel="{_num(beh.vel)}" '
            f'scary_force_factor="{_num(beh.other_force_factor)}"/>'
        )
    elif kind == "Curious":
        # How close the agent gets before stopping to look. Its own footprint
        # plus the robot's is the floor; anything less and it walks into it.
        stop_distance = max(agent.radius + 0.5, 0.5)
        lines.append(
            f"{inner}<CuriousNav {common} "
            f'agent_vel="{_num(beh.vel)}" '
            f'stop_distance="{_num(stop_distance)}"/>'
        )
    elif kind == "Threatening":
        lines.append(
            f"{inner}<ThreateningNav {common} "
            f'goal_dist="{_num(max(agent.radius * 2.0, 1.0))}"/>'
        )

    lines.append(f"{indent}</Sequence>")
    return lines


def emit_tree(agent: AgentSpec) -> str:
    """Render one agent's behavior tree XML."""
    if agent.behavior.type not in (
        *NAV_ONLY_BEHAVIORS,
        "Surprised",
        "Scared",
        "Curious",
        "Threatening",
    ):
        raise ValueError(
            f"{agent.name}: no tree template for behavior {agent.behavior.type!r}"
        )

    body: List[str] = []
    body.append("")
    body.append('<BehaviorTree ID="DefaultTree">')
    body.append('  <Fallback name="MainFallback">')

    if agent.goals:
        body.append("    <!-- Goal ring -->")
        body.extend(_set_goals_block(agent, "    "))

    reaction = _reaction_block(agent, "    ")
    if reaction:
        body.append(f"    <!-- {agent.behavior.type} reaction to the robot -->")
        body.extend(reaction)

    body.append("    <!-- Navigation loop -->")
    body.append('    <Sequence name="RegularNavigation">')
    body.append("      <Inverter>")
    body.append('        <IsGoalReached agent_id="{id}"/>')
    body.append("      </Inverter>")
    body.append('      <RegularNav agent_id="{id}" time_step="{dt}"/>')
    body.append("    </Sequence>")
    body.append("    <!-- Advance to the next goal -->")
    body.append('    <UpdateGoal agent_id="{id}"/>')
    body.append("  </Fallback>")
    body.append("</BehaviorTree>")
    body.append("")

    return _HEADER + "\n".join(body) + _FOOTER


def emit_all_trees(spec: ScenarioSpec, bt_dir: str) -> List[str]:
    """Write one tree per agent into `bt_dir`. Returns the paths written."""
    os.makedirs(bt_dir, exist_ok=True)

    written: List[str] = []
    for agent in spec.sorted_agents():
        path = os.path.join(bt_dir, bt_filename(spec.yaml_base_name, agent.id))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(emit_tree(agent))
        written.append(path)

    problems = verify_trees(spec, bt_dir)
    if problems:
        raise RuntimeError(
            "emitted behavior trees failed self-check:\n  "
            + "\n  ".join(problems)
        )

    return written


def verify_trees(spec: ScenarioSpec, bt_dir: str) -> List[str]:
    """Re-parse what was written and check it against the scenario.

    This is the check that catches the drift the whole module exists to
    prevent: a tree present for every agent, and every `SetGoal` naming a goal
    the loader will actually serve.
    """
    problems: List[str] = []

    for agent in spec.sorted_agents():
        path = os.path.join(bt_dir, bt_filename(spec.yaml_base_name, agent.id))
        if not os.path.isfile(path):
            problems.append(f"{agent.name}: no tree at {path}")
            continue

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            problems.append(f"{agent.name}: {os.path.basename(path)} does not parse: {exc}")
            continue

        tree_ids = {node.get("ID") for node in root.iter("BehaviorTree")}
        wanted = root.get("main_tree_to_execute")
        if wanted not in tree_ids:
            problems.append(
                f"{agent.name}: main_tree_to_execute={wanted!r} is not defined in the file"
            )

        emitted_goals = [int(n.get("goal_id")) for n in root.iter("SetGoal")]
        if emitted_goals != [int(g) for g in agent.goals]:
            problems.append(
                f"{agent.name}: tree ring {emitted_goals} does not match "
                f"scenario ring {agent.goals}"
            )
        for goal_id in emitted_goals:
            if goal_id not in spec.global_goals:
                problems.append(
                    f"{agent.name}: SetGoal goal_id={goal_id} is not in global_goals"
                )

    return problems

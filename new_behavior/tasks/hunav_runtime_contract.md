# Reference: what actually drives a HuNav agent at runtime

Traced from `/workspace/hunav_ws/src/hunav_sim` on 2026-09-17. These are the facts any
scenario-authoring, behavior-initialization or goal-editing work has to respect. Each one
fails **silently** if violated — no exception, no warning, just an agent that stands
still or ignores the robot.

Consumed by [../scenario_init_design.md](../scenario_init_design.md).

---

## 1. Three artifacts feed an agent, and the YAML is the weakest of them

| artifact | produced by | consumed by | what it controls |
|---|---|---|---|
| `src/scenarios/{base}.yaml` | us | `hunav_loader` | SFM parameters, `global_goals` table, spawn poses (read by the wrapper directly) |
| `src/behavior_trees/{base}__agent_{id}_bt.xml` | `hunav_behavior_tree_generator` (LLM) | `hunav_agent_manager` | **the agent's actual behavior and goal ring** |
| `hunav_msgs/Agent` | `hunav_manager.py` per tick | `compute_agents` service | live pose, radius, desired velocity, behavior enum, force factors |

## 2. The behavior tree is the controller

`bt_node` composes the path at
[bt_node.cpp:311-321](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp):

```
<wrapper_src>/behavior_trees/{yaml_base_name}__agent_{id}_bt.xml
```

`yaml_base_name` comes from the loader parameter that [main.py:1035](../../src/scripts/main.py#L1035)
passes. The directory is the **source** tree, not the install space — `share_to_src_path()`
([bt_node.cpp:22-50](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp))
maps `…/install/<pkg>/share/<pkg>` back to `…/src/<pkg>`. The `"Isaac Sim"` simulator
string selects `hunav_isaac_wrapper` ([bt_node.cpp:107-109](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp)).

**A missing or malformed tree logs an error and the agent gets no tree at all.** It spawns
and never moves. Rename a scenario without regenerating trees and every agent freezes.

## 3. Goals reach the agent only through the tree

`hunav_manager.py` sets `cyclic_goals` and `goal_radius` but never populates
`Agent.goals` ([hunav_manager.py:845-847](../../src/hunav_isaac_wrapper/hunav_manager.py#L845-L847)),
so `initializeAgents` builds an **empty goal deque**
([agent_manager.cpp:1141-1148](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/agent_manager.cpp)).

Goals arrive from `SetGoal` nodes in the tree, resolved against the `global_goals` table
that `hunav_loader` serves over its `GetParameters` service
([bt_functions.cpp:327-351](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_functions.cpp)).
An id not in the table returns `FAILURE`, not an error.

The generated trees encode the ring literally — `brownstone_agents__agent_1_bt.xml` holds
`SetGoal 15, 11, 13, 7` then `12` wrapped in `<Inverter>` so the sequence fails through to
the navigation loop. That mirrors agent1's `goals: [15, 11, 13, 7, 12]` in the YAML, but
**nothing enforces the match**. The YAML `goals:` list is read only by the tree generator.
Edit YAML without regenerating and the agent walks the old ring.

## 4. Goal ids must be contiguous from 1

[hunav_loader.cpp:309-316](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/hunav_loader.cpp):

```cpp
for (int goal_id = 1; ; ++goal_id) {
  if (!this->has_parameter(x_key) || !this->has_parameter(y_key)) break;
```

Remove goal 7 from a 15-goal table and goals 8–15 cease to exist for every consumer.
Deleting a goal means **renumbering** and rewriting every ring and tree that referenced
anything above it.

## 5. `behavior.type` never reaches the simulator today

[hunav_manager.py:894-901](../../src/hunav_isaac_wrapper/hunav_manager.py#L894-L901):

```python
agent.behavior = AgentBehavior(
    # type=int(beh["type"]),
    state=1,
    configuration=configuration,
    # duration=float(beh["duration"]),
    # once=beh["once"],
    # vel=float(beh["vel"]),
    # dist=float(beh["dist"]),
```

`type` defaults to `0`, which is not a valid `BEH_*` value — they run 1…6. In
`computeForces` that reaches the `default:` branch
([agent_manager.cpp:1380-1396](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/agent_manager.cpp)),
which computes social forces **with the robot excluded entirely** — neither `BEH_REGULAR`
(robot added as another human) nor `BEH_IMPASSIVE` (robot added as an obstacle).

Every agent in every Isaac scenario currently ignores the robot socially, whatever the
YAML says. `duration`, `once`, `vel` and `dist` are dark the same way.

Enum, from `hunav_msgs/msg/AgentBehavior.msg`:

```
type:           1 Regular  2 Impassive  3 Surprised  4 Scared  5 Curious  6 Threatening
state:          0 BEH_NO_ACTIVE  1 BEH_ACTIVE_1  2 BEH_ACTIVE_2
configuration:  0 DEFAULT  1 CUSTOM  2 RANDOM_NORMAL  3 RANDOM_UNIFORM
```

The wrapper sends `state=1`, which is what gates the `switch` at
[agent_manager.cpp:1378](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/agent_manager.cpp).

## 6. `configuration: 0` discards your force factors

Two clamp tables run, and they disagree:

| field | wrapper ([hunav_manager.py:854-865](../../src/hunav_isaac_wrapper/hunav_manager.py#L854-L865)) | `hunav_loader` ([hunav_loader.cpp:110-180](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/hunav_loader.cpp)) |
|---|---|---|
| `goal_force_factor` | default 10.0, clamp (5, 10) | default 2.0, clamp [2, 5]; conf 0 → 2.0 |
| `obstacle_force_factor` | default 2.0, clamp (0.5, 5) | default 10.0, clamp [2, 50]; conf 0 → 10.0 |
| `social_force_factor` | default 5.0, clamp (5, 20) | default 5.0, clamp [5, 20]; conf 0 → 5.0 |
| `other_force_factor` | passthrough | default 20.0, clamp [0, 25] |

The wrapper's values are what the SFM integrates — it builds the `Agent` msg directly. The
loader's only reach RViz and the tree generator. Also: `behavior.vel` is silently rewritten
into `[0.0, 1.8]` by the loader.

All four checked-in scenarios use `configuration: 0`, so their authored
`goal_force_factor: 10.0` / `obstacle_force_factor: 2.0` are replaced by defaults on both
sides. **Author with `configuration: 1` (`BEH_CONF_CUSTOM`) if written values should
survive.**

## 7. The shipped BT templates reference blackboard keys nothing sets

`BTScaredNav.xml`, `BTCuriousNav.xml`, `BTSurprisedNav.xml` and `BTThreateningNav.xml` use
ports `{duration}`, `{once}`, `{maxvel}`, `{dist}`, `{forcefactor}`, `{stopdist}`,
`{frontdist}`. `initializeBehaviorTree` puts only `id` and `dt` on the blackboard
([bt_node.cpp:307-308](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp)).

Any emitter must substitute per-agent values as **literal XML attributes**. That is why
the generated `*__agent_N_bt.xml` files hardcode their numbers.

## 8. The shipped tree generator is not wired for Isaac

`hunav_behavior_tree_generator` produced the checked-in trees, but its
[config.py](/workspace/hunav_ws/src/hunav_sim/hunav_behavior_tree_generator/hunav_behavior_tree_generator/config/config.py)
hardcodes `WORKSPACE_NAME = "hunav_gz_classic_ws"` / `package_name = "hunav_gazebo_wrapper"`,
with `# Possible implementation for Isaac Sim or Webots` as a placeholder in the `match`.
`BT_OUTPUT_DIR` and `SCENARIOS_DIR` therefore resolve outside this workspace. It also
requires a reachable LLM endpoint (`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`) and is
non-deterministic.

## 9. Nothing reloads — the YAML is read once

`initialize_hunav_nodes` ([hunav_manager.py:211-245](../../src/hunav_isaac_wrapper/hunav_manager.py#L211-L245))
spawns `hunav_loader`, `hunav_agent_manager` and `hunav_evaluator` as subprocesses with
`--params-file` at startup; agents are built from the same YAML at
[hunav_manager.py:327-329](../../src/hunav_isaac_wrapper/hunav_manager.py#L327-L329).
Any scenario edit requires a relaunch.

Coordinates need no transform: `init_pose` `(x, y, z)` goes straight to the stage
transform, and `h` is yaw about +Z in radians written as `xformOp:rotateXYZ.z`
([behavior_agent.py:345-347](../../src/hunav_isaac_wrapper/behavior_agent.py#L345-L347)).
The character rig's own forward is −Y and it handles that internally — `h` must not be
pre-compensated.

## 10. Navmesh facts that constrain authoring

- The bake volume is derived from the `init_pose` and `global_goals` already in the YAML
  ([hunav_manager.py:379-420](../../src/hunav_isaac_wrapper/hunav_manager.py#L379-L420)).
  Correct for running a scenario, circular for authoring one.
- The bake happens before characters load, at
  [hunav_manager.py:285-292](../../src/hunav_isaac_wrapper/hunav_manager.py#L285-L292).
- Two settings tables again: `DEFAULT_RECAST_SETTINGS`
  ([core.py:263-278](../nav_mesh_plugin/core.py#L263-L278)) vs `NAVMESH_SETTINGS`
  ([behavior_agent.py:113-122](../../src/hunav_isaac_wrapper/behavior_agent.py#L113-L122)) —
  step height 90 cm vs 25 cm, slope 45° vs 20°, island radius 80 cm vs 500 cm. The window's
  "Build Navmesh" button uses the former and will replace the mesh live agents are using.
- `omni.anim.navigation.core` exports `query_closest_point`, `query_random_point` and
  `query_shortest_path`. `core.py` wraps the latter two only; **`query_closest_point` is
  available and unused** — it is the correct snap primitive.
- `NavMeshExcludeAPI` exists in `omni.anim.navigation.schema`. Authoring-only meshes
  (pins, `/World/navmeshmesh`) need it or a rebake bakes them in as obstacles;
  `excludeRigidBodies: True` does not cover them, they have no rigid bodies.
- `agentMinIslandRadius: 500.0` leaves brownstone with several disconnected islands, and
  `query_random_point` samples across all of them by area. Reachability must be checked
  with `find_paths`, not assumed.

## 11. `new_behavior/nav_mesh_plugin` and `src/hunav_isaac_wrapper/nav_mesh_plugin` are duplicates

Byte-identical. [teleop_hunav_sim.py:433-435](../../src/hunav_isaac_wrapper/teleop_hunav_sim.py#L433-L435)
imports the former and falls back to the latter; `find_packages()` in
[setup.py](../../src/setup.py) installs only the latter. Edits to `new_behavior/`'s copy
alone never reach an installed run.

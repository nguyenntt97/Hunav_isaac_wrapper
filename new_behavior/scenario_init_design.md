# Design: static scenario & behavior initialization

How a scene gets a complete, self-consistent set of HuNav agents: spawn poses, goal
rings, behavior types and social-force parameters, all validated against the baked
navmesh before the simulator ever starts.

Scope is **static**. The agent roster is fixed at launch, as HuNav's own model requires.
There is no emitter, no spawn scheduling, no mid-run population change. Read
[tasks/hunav_runtime_contract.md](tasks/hunav_runtime_contract.md) first for the traced
evidence this design rests on, and [design.md](design.md) for the locomotion layer it sits
on top of.

---

## What "behavior control" actually is in HuNav

This is the part the earlier draft got wrong, so it is worth stating precisely. Three
artifacts feed an agent at runtime, and they are **not** the one file it looks like.

### 1. The per-agent behavior tree — the real controller

`hunav_agent_manager` loads one XML per agent:

```
<wrapper_src>/behavior_trees/{yaml_base_name}__agent_{id}_bt.xml
```

built at [bt_node.cpp:311-321](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp),
under a directory resolved by mapping the *installed* share path back to source
([bt_node.cpp:117](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp)) —
so these live in `src/behavior_trees/`, not the install space. A missing or malformed
file logs an error and the agent gets **no tree at all**; it spawns and never moves.

### 2. The goal ring — only reaches the agent through the tree

`Agent.goals` is a `geometry_msgs/Pose[]`, and the wrapper never populates it
([hunav_manager.py:845-847](../src/hunav_isaac_wrapper/hunav_manager.py#L845-L847) sets
`cyclic_goals` and `goal_radius` and stops). `initializeAgents` therefore builds an
agent with an **empty goal deque**
([agent_manager.cpp:1141-1148](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/agent_manager.cpp)).

Goals arrive instead from `SetGoal` nodes inside the tree, resolved against the
`global_goals` table that `hunav_loader` serves over `GetParameters`. The existing
generated trees encode the ring literally — `brownstone_agents__agent_1_bt.xml` has
`SetGoal 15, 11, 13, 7`, then `12` wrapped in an `Inverter` so the sequence fails and
falls through to the navigation loop. That matches agent1's `goals: [15, 11, 13, 7, 12]`
in the YAML, but **nothing enforces the match**. The YAML `goals:` list is consumed only
by the tree generator. Edit the YAML without regenerating the tree and the agent keeps
walking the old ring.

### 3. The YAML — parameters, and a contiguity trap

`hunav_loader` declares the schema at
[hunav_loader.cpp:72-280](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/hunav_loader.cpp).
Its `global_goals` scan **breaks at the first gap**
([hunav_loader.cpp:309-316](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/hunav_loader.cpp)):

```cpp
for (int goal_id = 1; ; ++goal_id) {
  if (!this->has_parameter(x_key) || !this->has_parameter(y_key)) break;
```

Delete goal 7 from a 15-goal table and goals 8–15 silently cease to exist. Every
`SetGoal` referencing them returns `FAILURE`. **Goal ids must be contiguous from 1.**

### The blocker: behavior type never reaches the simulator

[hunav_manager.py:894-901](../src/hunav_isaac_wrapper/hunav_manager.py#L894-L901):

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

`type` defaults to `0`, which is not a valid `BEH_*` value (they run 1…6). In
`computeForces` that hits the `default:` branch
([agent_manager.cpp:1380-1396](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/agent_manager.cpp)),
which computes social forces **with the robot excluded entirely** — neither `BEH_REGULAR`
(robot pushed in as another human) nor `BEH_IMPASSIVE` (robot pushed in as an obstacle).

Today, every agent in every Isaac scenario ignores the robot in its social-force model,
regardless of what the YAML says. `duration`, `once`, `vel` and `dist` are dark the same
way. Until this is fixed, "initializing behavior" is unobservable.

---

## Design goals

1. **One model, three artifacts.** A single in-memory `ScenarioSpec` is the source of
   truth; the YAML and the N behavior trees are both *emitted* from it, never edited
   independently.
2. **Validate before launch, not after.** Every invariant HuNav enforces silently
   (contiguous goal ids, `vel` clamping, force-factor ranges, tree-per-agent) becomes a
   loud check in the authoring step.
3. **Reachability is a scenario property.** A spawn that cannot path to its goals is a
   broken scenario, and the navmesh can tell us at authoring time.
4. **Deterministic given a seed.** A scenario is reproducible from
   `(map, n_agents, behavior mix, seed)`.
5. **No new runtime surface.** `hunav_manager` keeps reading one YAML at startup. The
   authoring tool runs before or beside the sim, never mutating a live one.

## Non-goals

- Emission scheduling, agent lifetime, population dynamics. Fixed roster.
- Replacing `hunav_behavior_tree_generator`'s LLM path. That stays available for
  bespoke trees; this emits the ordinary ones deterministically.
- Changing the ROS interface or the YAML schema. We fill the existing schema correctly.

---

## The two modes

### Mode A — author

Open the map, bake the navmesh, place and parameterise agents, export.
**Produces:** `src/scenarios/{base}.yaml` **and** `src/behavior_trees/{base}__agent_{id}_bt.xml`
for every agent — both, always. Exporting only the YAML leaves agents walking the ring
baked into the stale trees (contract §3), and renaming a scenario without regenerating
freezes every agent (contract §2).
**Does not run:** no characters, no `hunav_loader` / `hunav_agent_manager` subprocesses,
no robot.

### Mode B — run

Load a scenario, spawn, simulate, test. Exactly what the wrapper does today.

### This split already exists — extend it, don't rebuild it

`interactive_config_selection()` in [main.py:600-610](../src/scripts/main.py#L600-L610)
already offers *Create new agents yaml (RViz panel)* against *Use existing agents yaml* /
*Use last launch configuration*. Mode A is implemented today by `hunav_rviz2_panel`: it
loads `src/maps/{map}.yaml` into `nav2 map_server`, places agents and goals on the 2D
occupancy grid, and writes both artifacts — the scenario to
[`actor_panel.cpp:5474`](/workspace/hunav_ws/src/hunav_sim/hunav_rviz2_panel/src/actor_panel.cpp)
and the trees to
[`actor_panel.cpp:5970`](/workspace/hunav_ws/src/hunav_sim/hunav_rviz2_panel/src/actor_panel.cpp).

So the architecture is right and already present. What this design changes is the
*surface* of Mode A: a 2D occupancy grid cannot see elevation, stairs, thresholds or the
navmesh the agents will actually be steered on, which is the whole reason hand-placed
poses end up off-mesh. Mode A becomes an Isaac-native authoring session over the baked
navmesh, and slots into the same menu as a third option beside the RViz panel rather than
replacing it outright. (Note if you keep the RViz path: its Docker fallback is hardcoded
to `/workspace/hunav_isaac_ws/src/...`, which is not this workspace.)

### The contract between the modes: both bakes must be the same navmesh

This is the load-bearing requirement, and the easiest one to get wrong. A spawn validated
as walkable in Mode A is worthless if Mode B bakes a different mesh at the same
coordinates. Three things currently guarantee a different mesh:

1. **Two settings tables.** `NavmeshInterface.build_navmesh` applies
   `DEFAULT_RECAST_SETTINGS` — agent radius 60 cm, step 90 cm, slope 45°, island radius
   80 cm. The driver applies `NAVMESH_SETTINGS` — 50 cm, 25 cm, 20°, 500 cm. Those are not
   small differences: a 45° ramp is walkable in one and not the other, and the 80 cm island
   floor admits dozens of specks the 500 cm floor discards. **Mode A must bake through
   `BehaviorAgentDriver`, not through the plugin's `build_navmesh`.**
2. **The bake is a path-dependent procedure, not a function call.**
   [behavior_agent.py:554-599](../src/hunav_isaac_wrapper/behavior_agent.py#L554-L599)
   hides all non-walkable geometry and bakes ground-only *first*, coarsening ×2 up to three
   attempts, and only then tries the full scene. Its own comment records why: a failed bake
   **poisons the process**, so every later bake returns empty too. Mode A must run the same
   sequence, and must not let a user press "Build Navmesh" first — one failed exploratory
   bake and nothing correct can be baked afterwards without a restart.
3. **`ground_z` is derived from the agent poses** — `bounds[0][2] + _NAVMESH_Z_BELOW` at
   [hunav_manager.py:288-292](../src/hunav_isaac_wrapper/hunav_manager.py#L288-L292) — and
   in Mode A those poses do not exist yet. Take `ground_z` from the world instead, in
   **both** modes: the map YAML's `origin[2]`, or the lowest large horizontal mesh in the
   stage. That removes the last input the two modes cannot share.

Volume extent is the one input that may legitimately differ — Mode A needs the whole map
(1.1), Mode B sizes to the poses. On these maps that is harmless: brownstone is
1700 × 2599 px at 0.05 m/px = 85 × 130 m, so `largest_cm / _MAX_NAVMESH_CELLS_PER_AXIS`
= 13.0 cm, below the 20 cm floor, and both modes bake at 20 cm. Record the extent anyway
(below) so it stops being an accident.

**Persist the bake inputs in the scenario.** A sibling top-level key, which ROS2 ignores
because no node is named for it and the wrapper ignores because it reads only
`config["hunav_loader"]`:

```yaml
hunav_isaac_authoring:
  ros__parameters:
    navmesh:
      settings_digest: "<sha1 of NAVMESH_SETTINGS>"
      volume: {min: [x, y, z], max: [x, y, z]}
      ground_z: 0.0
      sampling_cm: 20.0
```

Mode B reuses these verbatim instead of recomputing, and warns loudly when
`settings_digest` no longer matches the running code — that is the moment a scenario
authored last month silently stopped meaning what it meant.

### Mode B re-validates on load

Cheap, and it closes the loop: after baking, run `validate()` (Phase 2) against the fresh
navmesh before spawning a single character. Every spawn on-mesh, every goal on-mesh, every
ring reachable, a tree present per agent id. Refuse to launch and name the offending agent
rather than spawning eight characters that stand still. This is the check that catches
bake divergence, a hand-edited YAML, and a scenario renamed without its trees.

---

## Phase 0 — unblock behavior (prerequisite)

Nothing downstream is observable until this lands. It is small and independently
valuable.

**0.1 Map the behavior name to the enum.** New module
`src/hunav_isaac_wrapper/behavior_spec.py`:

```python
BEHAVIOR_TYPES = {"Regular": 1, "Impassive": 2, "Surprised": 3,
                  "Scared": 4, "Curious": 5, "Threatening": 6}
```

Unknown name raises. A typo currently degrades to "ignores the robot" in silence.

**0.2 Send the whole behavior block.** Uncomment `type`, `duration`, `once`, `vel`,
`dist` in `_build_agent_msg`, sourcing `type` through the map above.

**0.3 Reconcile the two clamp tables.** They currently disagree and both run:

| field | wrapper `DEFAULT_SFM_PARAMS` / `SFM_CONSTRAINTS` ([hunav_manager.py:854-865](../src/hunav_isaac_wrapper/hunav_manager.py#L854-L865)) | `hunav_loader` ([hunav_loader.cpp:110-165](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/hunav_loader.cpp)) |
|---|---|---|
| `goal_force_factor` | default 10.0, clamp (5, 10) | default 2.0, clamp [2, 5]; **conf 0 forces 2.0** |
| `obstacle_force_factor` | default 2.0, clamp (0.5, 5) | default 10.0, clamp [2, 50]; **conf 0 forces 10.0** |
| `social_force_factor` | default 5.0, clamp (5, 20) | default 5.0, clamp [5, 20]; conf 0 forces 5.0 |
| `other_force_factor` | passthrough | default 20.0, clamp [0, 25] |

The wrapper's values are what the SFM actually integrates (it builds the `Agent` msg
directly); the loader's only reach RViz and the tree generator. Note the consequence for
the checked-in scenarios: every agent has `configuration: 0`, so its authored
`goal_force_factor: 10.0 / obstacle_force_factor: 2.0` are replaced by defaults on both
sides. **Author with `configuration: 1` (`BEH_CONF_CUSTOM`) so written values survive.**
Make `ScenarioSpec` emit `1` by default and validate against the loader's ranges, which
are the tighter and more physically motivated pair.

**0.4 Gate.** Launch `brownstone_agents` with one agent set `Impassive` and one
`Regular`. `/human_states` reports `behavior.type` 2 and 1 respectively (currently both
0), and the RViz marker suffix from
[bt_node.cpp:692-713](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp)
reads `/IMPASSIVE` and `/REGULAR`.

---

## Phase 1 — a sampling surface that is not derived from the answer

`_agent_activity_bounds()` ([hunav_manager.py:379-420](../src/hunav_isaac_wrapper/hunav_manager.py#L379-L420))
sizes the NavMeshVolume from the `init_pose` and `global_goals` already in the YAML. For
running a scenario that is the right call — it is what made brownstone bakeable at all.
For *authoring* one it is circular: you can only place agents inside the box the current
poses already describe.

**1.1 A second bounds source, used only when authoring.** `scene_bounds(map_name)` from
`src/maps/{map}.yaml` — `origin` + `resolution` × PNG pixel dimensions gives the walkable
extent in the same metric frame the scenario uses (confirmed: `init_pose` goes straight
to the stage transform at
[hunav_manager.py:327-329](../src/hunav_isaac_wrapper/hunav_manager.py#L327-L329), no
offset). Fall back to `/World` prim bounds when no map YAML exists.

**1.2 `nav_mesh_plugin/sampling.py`**, wrapping the native queries:

- `snap(xyz) -> xyz | None` — `navmesh.query_closest_point(target=...)`. The binding
  exports it; it is simply unused by `core.py` today. Do not hand-roll triangle
  projection.
- `sample_points(n, min_sep, max_tries) -> list[xyz]` — `query_random_point` plus
  rejection. It takes no clearance argument, so separation is ours to enforce; return
  fewer points **and say so** rather than looping forever. (`get_random_points`
  ([core.py:435-462](nav_mesh_plugin/core.py#L435-L462)) currently under-delivers
  silently — fix that too.)
- `reachable(a, b) -> bool` — `find_paths` returning ≥2 waypoints. With
  `agentMinIslandRadius: 500.0` brownstone keeps several disconnected islands, and
  `query_random_point` samples across all of them by area. Without this check, scattered
  agents land where no path to their goals exists.

**1.3 Keep authoring geometry out of the bake.** Any pin or marker authored as a
`UsdGeom.Mesh` is bakeable geometry — `excludeRigidBodies: True` does not help, pins have
no rigid bodies. Apply `NavMeshExcludeAPI` (from `omni.anim.navigation.schema`) to
`/World/HuNavSpawns` and `/World/HuNavGoals`, and to `/World/navmeshmesh` while we are
here. Otherwise a rebake carves a hole at every spawn point.

**1.4 Do not let the helper re-bake under a live sim.** The plugin's
`DEFAULT_RECAST_SETTINGS` ([core.py:263-278](nav_mesh_plugin/core.py#L263-L278)) and the
driver's `NAVMESH_SETTINGS` ([behavior_agent.py:113-122](../src/hunav_isaac_wrapper/behavior_agent.py#L113-L122))
disagree materially — step height 90 cm vs 25 cm, slope 45° vs 20°, island radius
**80 cm vs 500 cm**. `ScenarioManager` must take the *existing* navmesh and refuse to
bake when `omni.anim.behavior.core` has live agents.

---

## Phase 2 — the scenario model

`src/hunav_isaac_wrapper/scenario/spec.py`. Dataclasses mirroring exactly what
`hunav_loader` declares — no more, no less.

```python
@dataclass
class BehaviorSpec:
    type: str = "Regular"          # BEHAVIOR_TYPES key
    configuration: int = 1         # BEH_CONF_CUSTOM; see 0.3
    duration: float = 40.0
    once: bool = True
    vel: float = 1.0               # clamped to [0.0, 1.8] by the loader
    dist: float = 0.0
    goal_force_factor: float = 2.0
    obstacle_force_factor: float = 10.0
    social_force_factor: float = 5.0
    other_force_factor: float = 20.0

@dataclass
class AgentSpec:
    id: int; name: str; skin: int; group_id: int = -1
    max_vel: float = 1.5; radius: float = 0.4; goal_radius: float = 0.3
    cyclic_goals: bool = True
    init_pose: Pose                # x, y, z, h  (h = yaw about +Z, radians)
    behavior: BehaviorSpec
    goals: list[int]               # ids into ScenarioSpec.global_goals

@dataclass
class ScenarioSpec:
    yaml_base_name: str; simulator: str = "Isaac Sim"; map: str
    publish_people: bool = True
    global_goals: dict[int, tuple[float, float]]   # contiguous from 1
    agents: list[AgentSpec]
```

`h` is yaw about +Z in radians — that is what `spawn_character` writes as
`xformOp:rotateXYZ.z` ([behavior_agent.py:345-347](../src/hunav_isaac_wrapper/behavior_agent.py#L345-L347)).
Note the character rig's own forward is −Y; the rig handles that internally and `h` must
not be pre-compensated for it.

### `validate(spec, navmesh) -> list[Problem]`

Empty list means launchable. Each check exists because something fails silently without
it:

| check | silent failure it prevents |
|---|---|
| goal ids contiguous from 1 | loader truncates the table at the gap |
| every `agent.goals` id ∈ `global_goals` | `SetGoal` returns FAILURE, agent never gets a goal |
| agent ids unique and ≥1 | BT filename collision |
| `behavior.type` ∈ `BEHAVIOR_TYPES` | falls to the robot-ignoring `default:` branch |
| `0.0 ≤ vel ≤ 1.8` | loader rewrites the parameter behind your back |
| force factors in range for `configuration` | values replaced by defaults |
| `snap(init_pose)` within tolerance | agent spawns off-navmesh, cannot be steered |
| every goal snaps | agent walks at an unreachable point forever |
| `reachable(spawn, goals[0])` and around the ring | agent stuck on a disconnected island |
| pairwise spawn separation > `r_i + r_j + margin` | agents interpenetrate at t=0 |
| a BT file exists per agent id | agent loads no tree and never moves |

### `to_yaml()`

Formats **every** coordinate and force as `%.3f`. ROS2 params-file loading is type-strict:
a snapped coordinate that lands on exactly `-37` dumped as `-37` parses as `int` where the
loader declares `double`. Goal ids and `id`/`skin`/`group_id` stay `int`. `yaml_base_name`
is rewritten to match the output filename — it is what
[main.py:1035](../src/scripts/main.py#L1035) passes as `hunav_loader.yaml_base_name`, and
it is the prefix `bt_node` uses to find the trees.

---

## Phase 3 — the behavior-tree emitter

`src/hunav_isaac_wrapper/scenario/bt_emit.py`. Deterministic and template-based.

**3.1 Why not the shipped templates verbatim.** `BTScaredNav.xml` and friends reference
blackboard ports — `{duration}`, `{once}`, `{maxvel}`, `{dist}`, `{forcefactor}`,
`{stopdist}`, `{frontdist}` — that nothing ever sets. `initializeBehaviorTree` puts only
`id` and `dt` on the blackboard
([bt_node.cpp:307-308](/workspace/hunav_ws/src/hunav_sim/hunav_agent_manager/src/bt_node.cpp)).
The emitter must substitute the agent's own values as **literal XML attributes**. This is
why the generated `brownstone_agents__agent_N_bt.xml` files hardcode their numbers, and
the pattern to follow.

**3.2 Why not the LLM generator.** `hunav_behavior_tree_generator` exists and produced the
checked-in trees, but its
[config.py](/workspace/hunav_ws/src/hunav_sim/hunav_behavior_tree_generator/hunav_behavior_tree_generator/config/config.py)
hardcodes `WORKSPACE_NAME = "hunav_gz_classic_ws"` / `package_name = "hunav_gazebo_wrapper"`
with a `# Possible implementation for Isaac Sim or Webots` placeholder, so
`BT_OUTPUT_DIR` and `SCENARIOS_DIR` resolve outside this workspace. It also needs a
reachable LLM endpoint and is non-deterministic. Keep it for bespoke trees; add the
`"Isaac Sim"` case to its `match` so it works when wanted. Ordinary Regular/Scared/Curious
rings should not need an LLM round trip to be reproducible.

**3.3 Emit.** Per agent: the shared `TreeNodesModel` header, `<include path="BTRegularNav.xml"/>`,
a `SetGoal` chain from `agent.goals` following the established shape — n−1 plain
`<RunOnce>`, the last wrapped in `<Inverter>` so the sequence fails through to the
navigation loop — then the behavior's own `Fallback` with literals substituted. Write to
`src/behavior_trees/{yaml_base_name}__agent_{id}_bt.xml`.

**3.4 Self-check.** Re-parse each emitted file; assert one exists per agent id and that
every `SetGoal/@goal_id` is a key of `global_goals`. This is the check that catches the
YAML/tree drift described at the top.

---

## Phase 4 — entry points

All three drive the same `ScenarioSpec`; none is a second source of truth.

**4.1 `scripts/init_scenario.py` — the primary path.**

```
python init_scenario.py --map brownstone --agents 8 \
    --behaviors Regular:5,Curious:2,Scared:1 \
    --goals 15 --goals-per-agent 5 --seed 0 \
    --out src/scenarios/brownstone_agents.yaml
```

Bake over `scene_bounds` → sample goal positions with separation → sample spawns →
assign each agent a ring of `k` goals, nearest-first with a minimum spawn↔first-goal
distance so nobody starts on top of their target → assign behaviors from the mix →
validate → emit YAML + N trees. Runs headless under `/isaac-sim/python.sh`.

**4.2 The viewport pin layer — an override, not the author.** Load a validated spec, draw
a pin per spawn and goal, let the user drag, read back, re-validate, re-emit. Now it is a
*view* over a model that is correct before it opens. Necessary corrections to the earlier
draft: sort spawns by `AgentSpec.id`, not prim name (`/World/HuNavSpawns/*` traverses
`agent1, agent10, agent2…`); read pose via `ComputeLocalToWorldTransform` and flatten
off-axis tilt, since the viewport gizmo may author `xformOp:orient` where
`spawn_character` only ever writes `rotateXYZ`; deleting a goal pin **renumbers** the
table to stay contiguous and rewrites every affected agent's ring and tree.

How that lands inside the existing helper is section 4.5 below.

**4.3 Be explicit that export needs a relaunch.** `initialize_hunav_nodes`
([hunav_manager.py:211-245](../src/hunav_isaac_wrapper/hunav_manager.py#L211-L245))
spawns `hunav_loader` and `hunav_agent_manager` once with `--params-file`, and agents are
built from the YAML at
[hunav_manager.py:327-329](../src/hunav_isaac_wrapper/hunav_manager.py#L327-L329). There
is no reload path and this design does not add one. The UI says so.

**4.4 The plugin is already a single copy.** An earlier draft of this document claimed
`new_behavior/nav_mesh_plugin` and `src/hunav_isaac_wrapper/nav_mesh_plugin` were
duplicates that needed deduplicating. They are not: `src/hunav_isaac_wrapper/nav_mesh_plugin`
is a symlink to `new_behavior/nav_mesh_plugin`, which is why `diff -rq` reports no
differences. Both import spellings already reach the same files and there is nothing to
fix. New modules go in `new_behavior/nav_mesh_plugin/` and are visible under both names.

### 4.5 Fitting into the existing navmesh helper

The helper already exists and is reached by `--navmesh-helper` /
`HUNAV_NAVMESH_HELPER=1`. This section says exactly how the authoring layer attaches to
it, because two properties of the current wiring rule out the obvious approach.

**It opens too late to author the run you are in.** `_init_navmesh_helper()` is called at
[teleop_hunav_sim.py:426-427](../src/hunav_isaac_wrapper/teleop_hunav_sim.py#L426-L427),
*after* `initialize_agents()` and `initialize_hunav_nodes()` on lines 423-424. By the time
the window exists, characters are spawned from the YAML and `hunav_loader` /
`hunav_agent_manager` already own the scenario, with no reload path (contract §9). Editing
pins in that window can only ever produce a file for the **next** launch.

So Mode A needs its own construction path — `--author-scenario`, wired as a third option
in the existing launcher menu: build the world, bake through the driver (see the bake
contract above), open the helper, and **skip** `initialize_agents()` and
`initialize_hunav_nodes()` entirely. No characters, no ROS subprocesses, no live agents to
fight over the navmesh. Running the helper in a Mode B launch stays supported, but the pin
section renders read-only there and says why.

`TeleopHuNavSim.__init__` is a single 180-line sequence that always resolves the assets
root, loads the map, builds the `World`, **spawns a robot** (raising on an unsupported
name, [teleop_hunav_sim.py:353-360](../src/hunav_isaac_wrapper/teleop_hunav_sim.py#L353-L360)),
creates the differential controller and `cmd_vel` subscriber, constructs `HuNavManager`,
authors the ROS clock graph, then initialises agents and nodes. Mode A needs the first
three steps and nothing after them. Split that constructor into `_build_world()` and
`_build_runtime()` before adding the flag — a boolean threaded through the existing body
will leave a half-initialised node whose `run()` loop still references `self.hunav` and
`self.robot`.

**It already re-bakes with the wrong settings.** `build_navmesh()` behind the "Build
Navmesh" button ([ui_window.py:77-88](nav_mesh_plugin/ui_window.py#L77-L88)) applies
`DEFAULT_RECAST_SETTINGS`, which differs from the driver's table on step height, slope and
island radius (contract §10). In a normal launch that silently replaces the mesh the live
agents are steering on. Disable that button, and "Reset / Clear NavMesh", whenever
`omni.anim.behavior.core` reports live agents.

**What to reuse rather than rebuild:**

| existing helper piece | reuse as |
|---|---|
| `visualize_random_points` → `/World/Points` ([core.py:534-546](nav_mesh_plugin/core.py#L534-L546)) | scatter preview before committing spawns to pins |
| Start/End Prim drag targets + `get_specific_path` ([ui_window.py:125-148](nav_mesh_plugin/ui_window.py#L125-L148)) | already a reachability probe — repoint at a spawn pin and a goal pin to show *why* `validate()` rejected a pair |
| `visualize_path` → `/World/Path` | render the full goal ring for the selected agent |
| `make_outline` walls | the boundary the user is placing against; already drawn on startup |
| the `s_red`/`s_yellow`/`s_green`/`s_done` button-state convention | validation state: red = invalid spec, yellow = unvalidated edits, green = launchable |

**Wiring:**

- Add one collapsible "Agent Spawns & Goals" section to `NavmeshWindow._build_ui()`. Do
  not open a second window — the helper is already the navmesh tool and a second one would
  need its own `NavmeshInterface`.
- `NavmeshWindow.__init__` constructs `NavmeshInterface()` with no stage argument
  ([ui_window.py:38](nav_mesh_plugin/ui_window.py#L38)). `ScenarioManager` must take
  **that instance**, not build its own, or the two will disagree about `built` and about
  which stage they are on.
- `clear_and_reset_navmesh()` ([ui_window.py:150-161](nav_mesh_plugin/ui_window.py#L150-L161))
  clears `/World/navmeshmesh`, `/World/Outline`, `/World/Points`, `/World/Path`. Extend
  `clear_visualizations()` to take `/World/HuNavSpawns` and `/World/HuNavGoals` too —
  otherwise pins outlive the navmesh they were snapped to and the next validation pass
  reads stale positions against a mesh that no longer exists.
- Every pin prim gets `NavMeshExcludeAPI` at creation (1.3), so the helper's own "Build
  Navmesh" cannot bake a hole at each spawn.

**The helper is the override surface, not the entry point.** `init_scenario.py` (4.1) runs
headless and produces a validated scenario on its own; the window is where you inspect it,
nudge four agents off a doorway, and re-emit. Keeping that order means the tool is never
the only thing standing between you and a broken scenario.

**If Mode A also edits the map geometry**, saving the stage is a third artifact and needs
its own rules. Moving an obstacle changes the navmesh, which invalidates every spawn and
goal already placed — so a stage edit must force a re-bake and a re-`validate()` before
export, not after. Save to `src/worlds/{map}.usd` via `omni.usd` `save_as` with the pin
scopes **deactivated or removed first**; `brownstone.usd.bak` shows the stage has been
edited before, and a pin serialised into the world would be baked as an obstacle on every
subsequent run. Keeping map editing out of scope for the first iteration is a defensible
call — placement and parameters are the stated need, and the USD is separately editable in
the Isaac UI.

---

## Phase 5 — verification

**Unit** (no Isaac): spec → YAML → reparse round trip; validator rejects each invariant in
the table above with a targeted fixture; emitted trees parse and their `SetGoal` ids
resolve.

**Navmesh** (`/isaac-sim/python.sh`, extends `test_navmesh_tool.py` — renumber, the
existing harness ends at `[8/8]`): a point lifted into the air snaps back to ground
elevation; `sample_points(50, min_sep=2.0)` returns 50 points all ≥2.0 m apart and all
snapping to themselves; `reachable()` is False across a wall and True around it.

**Bake parity between modes** — the test that protects the Mode A guarantee. Bake in Mode
A, record every navmesh vertex; bake in Mode B from the exported scenario; assert
`query_closest_point` returns the same point within a cell for all N spawns and goals.
A drift here means everything Mode A validated was validated against a mesh that does not
exist at run time.

**In-sim round trip** — the only test that proves the feature:

1. `init_scenario.py --map brownstone --agents 4 --behaviors Regular:2,Impassive:1,Scared:1 --seed 0`
2. `./launch_hunav_isaac.sh --config brownstone_agents.yaml --flat-ground --navmesh-helper --batch`
3. Assert `Behavior Tree for agent … loaded!` appears once per agent id — the absence of
   this line is the silent failure mode.
4. Assert `/human_states` reports `behavior.type` ∈ {1, 2, 4} matching the mix, not 0.
5. Assert every agent's distance to its first goal decreases over 30 s.
6. Strip render of the four agents walking to `/workspace/Hunav_isaac_wrapper/debug/`.

Step 4 is the one that would have caught the Phase 0 bug, and step 3 the one that catches
YAML/tree drift.

---

## Implementation status

Built and verified on 2026-09-18.

| phase | where | verified by |
|---|---|---|
| 0 — behavior plumbing | [hunav_manager.py](../src/hunav_isaac_wrapper/hunav_manager.py) `_behavior_type_id`, `_create_agent_msg` | `/human_states` reports type 1/4/5 for a Regular/Scared/Curious mix; previously 0 for every agent |
| 1 — sampling | [nav_mesh_plugin/sampling.py](nav_mesh_plugin/sampling.py), [pins.py](nav_mesh_plugin/pins.py), [scenario/bake.py](../src/hunav_isaac_wrapper/scenario/bake.py) | `test_scenario_sampling.py`, 15 checks |
| 2 — scenario model | [scenario/spec.py](../src/hunav_isaac_wrapper/scenario/spec.py), [paths.py](../src/hunav_isaac_wrapper/scenario/paths.py) | `test_scenario.py`, 32 checks |
| 3 — tree emitter | [scenario/bt_emit.py](../src/hunav_isaac_wrapper/scenario/bt_emit.py) | same suite; plus 6/6 trees loaded by `hunav_agent_manager` in a live run |
| 4 — entry points | [scripts/init_scenario.py](../src/scripts/init_scenario.py), [nav_mesh_plugin/scenario_manager.py](nav_mesh_plugin/scenario_manager.py), `--author-scenario` | end-to-end generate → launch → agents walk |

Two things went differently from the plan and the code reflects the corrected version:

- **`query_closest_point` returns `(point, island_id)`, not a point.** Unpacking it wrong
  makes every snap return `None`, and every spawn read as off-navmesh. The island id is a
  bonus: it is the navmesh's own connectivity component, so `reachable()` is an integer
  comparison rather than a pathfinding call. `path_length()` checks it first, because
  `query_shortest_path` across a gap returns a partial path to the boundary rather than
  nothing — which would read as a short walk instead of an impossible one.
- **§4.4 was wrong about duplicate plugin directories.** See the corrected note there.

Not built: saving edited map geometry (§4.5's last note). Placement and parameters were
the stated need and the USD is separately editable in the Isaac UI.

## Open decision

**Whether the goal ring is per-agent or shared.** CrowdES pairs one origin with one goal.
HuNav gives each agent an ordered ring over a shared table with `cyclic_goals`, which is a
different model: agents patrol. The sampler in 4.1 assumes rings drawn from a shared pool,
which reproduces the existing scenarios. A "commute" mode — one goal each, `cyclic_goals:
false`, agents idle on arrival — is a one-line variant of the same sampler if that is
closer to the benchmark you want.

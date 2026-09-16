# What changed, and why the current implementation fails

All findings below were taken from the installed Isaac Sim
(`/isaac-sim`, version `6.0.1-rc.7+release.42383.32955d8d.gl`) and from the public
Omniverse asset buckets, not from documentation prose. Each is marked **[verified]**
with the check that produced it, or **[unverified]** where it is an inference.

---

## 1. The extension that this wrapper is written against no longer exists

**[verified]** `omni.anim.people` is absent from the install — no match under
`/isaac-sim/extscache`, `/isaac-sim/exts`, or anywhere below `/isaac-sim`.

It was split into three families, all of which *are* installed:

| family | extensions | role |
|---|---|---|
| Behavior simulation | `omni.anim.behavior.core` `.schema` `.tree` `.ui` `.bundle` | character motion, tasks, avoidance |
| Scene/agent orchestration | `omni.metropolis.pipeline` `.schema` `.utils` `.agent_registry` | actor lifecycle, config, triggers |
| Replicator agent | `isaacsim.replicator.agent.core` `.ui` `.schema` | the user-facing "spawn N people" tool |

**[verified]** `isaacsim.replicator.agent.core` contains **no** retargeting code and
**no** animation-graph code (`grep -rl` for `retarget`, `AnimationGraph`,
`animation_graph` across its Python returns nothing). The same is true of
`omni.metropolis.pipeline`. All character motion now lives in
`omni.anim.behavior.core`.

The legacy `omni.anim.graph.core` (110.1.2) *is* still installed and still functions —
it is legacy-but-present, not removed. The wrapper is not calling a dead API; it is
calling a supported API in a way the 6.x asset and retargeting pipeline no longer
serves well.

---

## 2. Animation moved from blend trees to motion matching

### What the wrapper does today

[`animation_utils.py`](../src/hunav_isaac_wrapper/animation_utils.py) authors, per agent:

```
<SkelRoot>/AnimationGraph            (AnimationGraph)
  ├── Blend                          inputs:pose0 → IdleLoop, inputs:pose1 → WalkLoop
  │                                  inputs:blendWeight → speed
  ├── IdleLoop  (AnimationClip)      → /World/Characters/IdleLoop
  ├── WalkLoop  (AnimationClip)      → /World/Characters/WalkLoop
  └── speed     (ReadVariable)       reads anim:graph:variable:speed
```

`/World/Characters/{Idle,Walk}Loop` are produced at launch by
`setup_anim_retargeting()`, which runs `CreateRetargetAnimationsCommand` twice to map
NVIDIA's 81-joint `biped_demo` rig onto the character's 101-joint rig. Speed comes
from HuNavSim and is written with `char.set_variable("speed", v)`.

### What Isaac Sim 6 does

**[verified]** from `omni.anim.behavior.schema`'s `generatedSchema.usda` and from the
shipped test scenes under
`omni.anim.behavior.core-110.1.4+.../data/tests/usd/`:

```usda
def "Human" (
    prepend payload = @.../Characters/Human/Default.usd@
)
{
    over "SkelRoot" (
        prepend apiSchemas = ["BehaviorAgentAPI"]
    )
    {
        rel behavior:motionLibrary = </World/Humans/HumanMotionLibrary>
        custom string[] behavior:navMeshAreasAllowed = []

        over "Root"
        {
            uniform token   controlRig:forwardAxis  = "Z"
            uniform token   controlRig:upAxis       = "Y"
            uniform token[] controlRig:retargetTags = ["", "Hips", "RightLeg", ...]
        }
    }
}

def "HumanMotionLibrary" (
    prepend payload = @.../MotionLibrary/Human/HumanMotionLibrary.usd@
) { }
```

There is no animation graph, no blend node, and **no runtime retargeting command**.
Retargeting is expressed *declaratively* as per-joint `controlRig:retargetTags` on the
skeleton. The engine matches the motion library's clips to the rig through those tags.

**[verified]** the motion library is a payload-composed clip database. Groups in
`Assets/Isaac/6.0/Isaac/People/MotionLibrary/BuiltinActions/`:

```
MoveWalk  MoveWalkSlow  MoveJog  MoveCarryObject
Idle      Dodge         Fall     Sit  Ride  PickupPlaceObject
```

with many directional variants per group (`WalkTurns_15`, `WalkArc`,
`WalkSlowBackward`, `JogForward`, `IdleTired`, …). That is a motion-matching database,
not a pair of loops to cross-fade.

### The architectural consequence

| | 4.x / this wrapper | Isaac Sim 6 |
|---|---|---|
| Clip selection | we pick, and blend two | engine matches from a database |
| Retargeting | runtime command, per launch | authored tags on the skeleton, offline |
| Speed | a graph variable we set | `IBehaviorAgent.set_speed()` |
| Turning | we slerp `xformOp:orient` ourselves | turn clips selected by the matcher |
| Foot contact | none — feet slide | `contactJoints` / `contactJointTags` in `BehaviorMotionLibrary` |

---

## 3. The runtime API is `IBehaviorAgent`

**[verified]** by reading the symbol and docstring table out of
`omni/anim/behavior/core/bindings/_omni_anim_behavior_core.cpython-312-x86_64-linux-gnu.so`.

Acquired with `omni.anim.behavior.core.acquire_interface().get_agent(prim_path)`, where
`prim_path` is "the prim path of the SkelRoot that has applied a
`BehaviorSchema.BehaviorAgentAPI`".

**[verified]** `get_agent` is on the `IBehaviorSystem` interface, **not** on the module —
`omni.anim.behavior.core.get_agent` does not exist. That interface carries exactly two
methods: `get_agent` and `set_random_seed` (seeding is system-wide, not per agent).

Relevant to this wrapper:

| call | note |
|---|---|
| `teleport(target, facing)` | `facing` may be a direction **or** an object |
| `move_to(...)` | navmesh-pathed movement |
| `set_speed(v)` / `get_speed()` | **stage units per second** — see the correction below |
| `get_linear_velocity()` | **[verified]** returns `(0,0,0)` even while walking — difference positions instead |
| `get_world_transform(t, r, use_previous_frame)` | |
| `idle(...)`, `look_at(...)`, `follow(target, distance)` | |
| `dodge(direction, motion_scale)`, `fall(...)`, `sit(target, snap_to_seat)` | |
| `set_obstacle_avoidance_enabled(b)`, `set_auto_avoidance_enabled(b)` | **must be disabled** — see design.md |
| `set_navmesh_areas_allowed(areas)` | `None` clears the constraint |
| `set_enabled(b)` | "whether agent updates are enabled" |
| `set_random_seed(n)` | determinism for reproducible runs |
| `get_task_status()`, `get_task_name()`, `get_task_elapsed_time()` | |
| `get_height()`, `get_radius()`, `get_reach()`, `get_arm_length()` | body metrics, useful for HuNav agent radius |

Enums and events:

- `BehaviorRootAnimation` = `IGNORE_ROOT_ANIMATION` | `USE_ROOT_ANIMATION` | `APPLY_ROOT_ANIMATION_TO_HIPS`
  — **this is the externally-driven-agent control point.** It decides whether the clip
  moves the character or the character's imposed motion drives the clip.
- `BehaviorTaskStatus` = `RUNNING` | `SUCCEEDED` | `FAILED` **[verified]** at runtime; an
  earlier reading of `IN_PROGRESS`/`SUCCESS`/`CANCELED` off raw binary strings was wrong
- `BEHAVIOR_TASK_ID_INVALID` = -1; task id `0` is the implicit idle task and never
  emits `TaskEnded`
- Global events: `..._AGENT_CREATED`, `..._AGENT_DESTROYED`, `..._TASK_STARTED`,
  `..._TASK_ENDED`, `..._OBJECT_ATTACHED`, `..._OBJECT_DETACHED`,
  `..._AGENT_AVOIDANCE_TRIGGERED`

Tunables (from `omni.anim.behavior.core`'s `extension.toml`) worth knowing about:

```
enableAutoAvoidance        = true      # turn OFF: HuNav owns avoidance
enableRagdollPhysics       = false
walkSpeedControlRatio      = 0.1
agentResponseTime          = 0.3       # latency before an agent reacts to a new command
agentResponseTimeMaxRandomOffset = 0.1
navBlockedTimeout          = 1.0
idleAnimationTags = ["Idle","IdleCautious","IdleGroove","IdleTaunt","IdleTilt","IdleTired","IdleTough"]
```

`agentResponseTime = 0.3` is the number to watch: at a 10–30 Hz HuNav tick a 300 ms
reaction delay will visibly lag the social-force trajectory unless it is lowered.

---

## 4. Why the current implementation produces a T-pose

### The measurement

Driving the character's animation graph with the **raw** source clips deforms the mesh
violently (wrong rig, but proof the graph → clip → skeleton → skin chain is intact).
Driving it with the **retargeted** clips gives a perfect bind pose. Rendered evidence
is in [`../debug/`](../debug/):

```
01_side_by_side_reference_vs_ours.png   NVIDIA's biped posed, ours T-posed
02_nvidia_reference_animates.png        reference animates → environment can skin
03_ours_retargeted_clips_TPOSE.png      retargeted clips → bind pose
04_ours_RAW_clips_deforms.png           raw clips → deformation
```

Measuring the retargeted walk clip's joint rotation over its full duration:

```
ROT_CHANGE_over_clip:  max = 3.99°   mean = 0.08°   joints exceeding 5° = 0
```

A real walk cycle swings limbs through tens of degrees. **The retarget output is a
static pose.** It has the right joint count, the right sample count and unit scales —
it simply contains no motion. That is the T-pose.

> A caution for whoever picks this up: an earlier check compared quaternion
> *components* between time samples, saw them differ, and concluded the clips were
> fine. They differ by fractions of a degree. Judge animation on rendered pixels or on
> angular metrics, never on raw component deltas.

### The version-specific cause

**[verified]** by HTTP status against
`omniverse-content-production.s3-us-west-2.amazonaws.com`:

| asset | 4.5 | 5.0 | **6.0** | 6.1 |
|---|---|---|---|---|
| `Isaac/People/Characters/Biped_Setup.usd` | 200 | 200 | **404** | 200 |
| `Isaac/People/Characters/biped_demo/` | present | present | **absent** | present |
| `Isaac/People/MotionLibrary/HumanMotionLibrary.usd` | — | — | **200** | 200 |

The installed Isaac is **6.0.1**, and 6.0 is the single release whose bucket is missing
the retarget source rig. [`asset_paths.py:114`](../src/hunav_isaac_wrapper/asset_paths.py#L114)
already works around this by falling back to the 4.5 bucket — so a **4.5-era source
rig** is being fed to a **6.0 retargeter**, and the output is dead.

**Correction to record:** that function's docstring states "Isaac Sim 5.0+ removed both
`Biped_Setup.usd` and the `biped_demo/` subtree … there is no replacement in the current
buckets". Both halves are wrong. 5.0 and 6.1 have the file; 6.0 alone does not; and the
replacement is `MotionLibrary/`, which appears in exactly the releases where
`Biped_Setup` vanished. Fix the comment whichever path is chosen.

### The good news for migration

**[verified]** by opening `Assets/Isaac/6.0/Isaac/People/Characters/F_Business_02/F_Business_02.usd`
with `pxr.Usd` — the character this wrapper already spawns:

```
metersPerUnit = 1.0        upAxis = Z
SkelRoot  /Root/female_adult_business_02/ManRoot/female_adult_business_02
Skeleton  .../female_adult_business_02   101 joints
controlRig:retargetTags    101 entries, 56 non-empty
controlRig:forwardAxis     "MINUS Y"
controlRig:upAxis          "Z"
```

**The stock Isaac People characters already carry the control-rig contract the behavior
system needs.** They are metres and Z-up, matching our stage. Migration does not require
new character assets or an asset-authoring pass — it requires applying an API schema and
a relationship.

(Note: a plain `strings | grep retargetTags` on that file finds nothing, because it is
binary USD crate. Query it through `pxr.Usd`.)

---

## 5. Behavior trees: two layers, not two competitors

`https://docs.isaacsim.omniverse.nvidia.com/6.1.0/.../ext_behavior_tree_gen/context_files_and_schemas.html`
documents `omni.ai.behavior_tree_gen` — an **authoring-time LLM tool** that consumes
JSON "context files" (`id`, `semantic_description`, `metadata`,
`supported_interactions`, `entity_type`, validated by `actor_metadata_schema.json` /
`object_metadata_schema.json`) and writes a behavior tree. It is not a runtime.

The runtime beneath it, **[verified]** from the installed extensions:

- `omni.behavior.tree.core` (110.1.9, titled *EXPERIMENTAL*) — JSON format
  `schemaVersion 2.0.0`, with `nodeLibraries` / `localBlackboard` / `root` /
  `portOverrides`. Builtin nodes: `Sequence` `Selector` `Parallel` `SetBlackboard`
  `PushQueue` `Wait` `LogMessage` `DispatchEvent`; modifiers `CheckEvent` `PopQueue`
  `Retry` `Repeat` `ForceStatus` `InvertStatus` `Delay` `Timeout` `Cooldown`
  `CheckBlackboard` `RandomFloat` `RandomInt` `RandomChoice`.
- `omni.anim.behavior.tree` (110.1.3) — the character node pack:
  `MoveTo` `MoveAlong` `Teleport` `SetSpeed` `Idle` `LookAt` `Follow` `Dodge` `Fall`
  `Sit` `PickupObject` `PlaceObject` `ReleaseObject` `ReachHand` `PoseHand` `Reset`
  `RandomNavMeshPoint`, conditions `CheckAgentDistance` `CheckSpeed` `CheckTaskName`
  `CheckNavMeshPathExists`.

### Decision: do not adopt Isaac's behavior tree

| | Isaac 6 BT | HuNav BT |
|---|---|---|
| Engine | `omni.behavior.tree.core` (experimental) | BehaviorTree.CPP v4 |
| Format | JSON `schemaVersion 2.0.0` | XML `BTCPP_format="4"` |
| Process | inside Isaac Sim | `hunav_agent_manager`, a ROS 2 node |
| Interface | `behavior_tree` field in the IRA YAML | `/compute_agents` service, per tick |
| Vocabulary | task/object interaction | social navigation |

HuNavSim's node set — `IsAnyoneLookingAtMe`, `ConversationFormation`, `IsRobotFacingAgent`,
`ThreateningNav`, `ScaredNav`, `BlockRobot`, `SetGroupWalk` — has no counterpart in
Isaac's. Those nodes are the reason this project uses HuNavSim at all. The two formats
share no schema and no migration path is shipped.

The useful observation is the opposite one: **Isaac's character BT nodes are thin
wrappers over `IBehaviorAgent`.** `MoveTo` → `move_to`, `Teleport` → `teleport`,
`SetSpeed` → `set_speed`. The BT is one optional consumer of that API, and we can call
the API directly and skip it entirely.

So the division of responsibility is:

- **HuNav's behavior tree decides** — goals, social reactions, emotional state, velocity.
- **`omni.anim.behavior.core` executes** — which clips play, foot placement, turning.
- **Isaac's behavior tree and the LLM generator are unused.**

The BT layer therefore needs **no migration work at all**. Only animation does.

---

## 6. Corrections and constraints found while implementing

### Units are stage units, not centimetres

**[verified]** `get_height()` returns **1.654** for `F_Business_02` on this metres stage.
The NVIDIA sample scenes report ~180 only because those stages use
`metersPerUnit = 0.01`. HuNavSim's m/s goes to `set_speed()` unchanged — an earlier note
in these documents said to multiply by 100, and that was wrong.

### The engine writes to Fabric, not USD

**[verified]** `xformOp:translate` on a walking agent's character prim keeps its authored
spawn value. Agent poses must be read back through `get_world_translation()` /
`get_world_rotation()` (a `Float4` in x, y, z, w order), never off the prim. This is why
`_create_agent_msg` no longer reads the stage.

### `CreateBehaviorAgentCommand` is unusable for this flow

**[verified]** it probes the asset for a `UsdSkel.Root` immediately after creating the
prim, before the payload has loaded, and fails with *"asset missing UsdSkel.Root"*. Spawn
the character, wait for the load, then call `ApplyBehaviorAgentAPICommand`.

### A navmesh is required, and it is the hard part

- With no navmesh, `get_agent()` returns `None` forever and the plugin logs only
  *"Behavior System disabled because no navmesh is available."*
- The baker **ignores implicit geometry**. A `UsdGeom.Cube` ground bakes nothing; a
  `UsdGeom.Mesh` ground bakes. This is why `terrain.py`'s flat-ground proxy is now a Mesh.
- Agent parameters come from **carb settings**, not from the stage's
  `customLayerData["navmeshSettings"]`. The plugin reads the layer data at stage-open and
  caches it, so writing it afterwards changes nothing — and the live defaults are from the
  centimetre era (`agentMinHeight 200`, `agentSamplingDistance 20`), which on a metres
  stage describe a 200 m tall agent that no surface can accommodate.
- The GPU baker exhausts CUDA memory far below any plausible voxel budget, and fails by
  producing an **empty navmesh** after logging `CUDA error: out of memory` — it does not
  raise. Measured on an RTX 5090 with 25 GB free:

  | volume | sampling | bake |
  |---|---|---|
  | 4 x 4 m | 0.2 m | ✓ |
  | 10 x 10 m | 0.2 m | ✗ |
  | 10 x 10 m | 0.5 m | ✓ |
  | 30 x 45 m | 0.5 m | ✗ |
  | 30 x 45 m | 1.0 m | ✓ |

- The CPU path (`navMesh/useGpu = false`) produces **no navmesh at all** in this build.
- **The memory use scales with total scene geometry, not with the navmesh volume.**
  Brownstone (1268 meshes) fails at every sampling distance and every volume size; the
  same scene with only its ground visible bakes at 2.73 m. So `behavior_agent.py` bakes in
  two passes: the full scene first, then — if that comes back empty — a retry with every
  mesh that rises out of the walkable band hidden, restoring visibility afterwards. Trees,
  buildings and roofs cannot be walked on, and HuNavSim does obstacle avoidance from its
  own raycasts, so their absence from the navmesh costs nothing here.
- **The bake must be given frames to see visibility changes.** Hiding meshes and baking in
  the same breath leaves the baker looking at the old, full scene -- which is precisely
  why an isolated bake succeeded in a test script (which happened to pump frames between
  the two steps) and failed inside the simulator for fourteen launch attempts. Pump the
  app between hiding and baking.
- **A failed bake poisons the process.** After a bake that OOMs, every later bake in the
  same process returns an empty navmesh, including one that would have succeeded alone.
  So the isolated bake is attempted *first*, not as a fallback -- trying the full scene
  first destroys the attempt that works.
- The volume is sized to where the agents actually go (start poses plus `global_goals`,
  padded) rather than to the whole world. Brownstone's world bounds span 84 x 128 x 23 m,
  almost all of it the sunken road at -19.8 m and empty air; the agent activity area is
  61 x 105 x 6 m. Predicted sampling: warehouse 0.89 m, office 1.40 m, hospital 1.48 m,
  brownstone 2.73 m.

The coarse navmesh is acceptable here because HuNavSim still does the navigation. The
navmesh exists to enable the behavior system and to let `move_to` find a local path to a
goal that is only half a second away.

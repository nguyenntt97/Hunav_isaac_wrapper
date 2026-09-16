# Design: locomotion on the Isaac Sim 6 behavior framework

Revision of character locomotion in `hunav_isaac_wrapper` so that it follows the Isaac
Sim 6 behavior framework. Read [differences.md](differences.md) first for the evidence
this design rests on.

> **Status:** implemented in `behavior_agent.py`. The animation-graph path has been
> **removed**, not kept behind a flag — see goal 2.

## Design goals

1. **HuNavSim stays authoritative.** The social-force model and its behavior trees
   decide where agents go. Isaac decides only how the body renders that motion.
2. **Replace, do not coexist.** An earlier draft of this document proposed keeping the
   animation-graph path selectable behind an `--animation-backend` flag. That was
   rejected in favour of replacing it outright: the old path is a 4.x design whose
   retargeting step is broken on 6.0, so carrying it forward would mean maintaining a
   code path known not to work. There is consequently no backend flag.
3. **The failure is loud.** The old code retargeted, got a dead clip, and reported
   success. The replacement verifies its own output and refuses to pretend — see
   `verify_locomotion()`.
4. **Follow the existing conventions of this repo.** Terrain behaviour stays behind the
   existing `--flat-ground` / `--terrain-follow` toggles.

## Non-goals

- Adopting Isaac's behavior tree (`omni.behavior.tree.core`) or its LLM tree generator.
  See [differences.md § Behavior trees](differences.md#5-behavior-trees-two-layers-not-two-competitors).
- Adopting `isaacsim.replicator.agent` for spawning. It is a scenario-authoring tool
  whose YAML and lifecycle overlap what `hunav_loader` already owns.
- Changing the ROS 2 interface, the scenario schema, or the world/terrain code.

---

## 1. Structure

Today [`hunav_manager.py`](../src/hunav_isaac_wrapper/hunav_manager.py) (1060 lines)
mixes four concerns: ROS plumbing, agent spawning, terrain probing, and animation.
Animation is threaded through `initialize_agents()` and `_update_agents()` with module
state (`self.bound_animations`, `self.flag_anim`, `self.retarget_flag`,
`self.animationDict`). Adding a second animation system inline would double that
tangle.

Instead, extract the animation concern behind one small interface:

```
src/hunav_isaac_wrapper/
  locomotion/
    __init__.py        get_backend(name) -> LocomotionBackend
    base.py            LocomotionBackend, AgentHandle  (the interface + contract)
    anim_graph.py      current behaviour, lifted out of hunav_manager verbatim
    behavior_agent.py  NEW: omni.anim.behavior.core
  animation_utils.py   unchanged; imported only by locomotion/anim_graph.py
  hunav_manager.py     calls the interface, owns no animation state
```

### The interface

```python
class LocomotionBackend:
    """How an agent's body is driven. One instance per simulation."""

    name: str
    startup_extensions: tuple[str, ...]   # enabled at SimulationApp boot

    def prepare_stage(self, stage, assets_root: str) -> None:
        """Stage-wide, once, before any agent is spawned."""

    def attach_agent(self, stage, container_prim, skelroot_prim,
                     agent_cfg: dict) -> AgentHandle:
        """Make one spawned character drivable. Raises on failure."""

    def verify(self) -> list[str]:
        """Post-setup self-check. Returns human-readable problems; empty == healthy."""

    def update_agent(self, handle: AgentHandle, *, position, orientation,
                     linear_velocity, dt: float) -> None:
        """Called once per HuNav tick with that agent's new pose."""

    def teardown(self) -> None: ...
```

`hunav_manager` then loses every `ag.*` call and every animation attribute; its
`_update_agents()` loop reduces to computing the pose (including the existing
`terrain_follow` Z resolution) and handing it to `backend.update_agent(...)`.

This is worth doing even if the new backend is ultimately rejected: it is what makes
the two paths comparable at all, and it is what lets `verify()` exist.

---

## 2. The `behavior_agent` backend

### `prepare_stage()`

Add the motion library once, as a payload, mirroring the shipped test scenes:

```python
lib = stage.DefinePrim("/World/HumanMotionLibrary")
lib.GetPayloads().AddPayload(
    f"{assets_root}/Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
)
```

**[verified]** this asset exists for 6.0 (`.../Assets/Isaac/6.0/Isaac/People/MotionLibrary/HumanMotionLibrary.usd`
returns 200), so no version fallback is needed and
[`asset_paths.py`](../src/hunav_isaac_wrapper/asset_paths.py)'s legacy-bucket escape
hatch is not involved.

### `attach_agent()`

```python
from pxr import Usd
import BehaviorSchema

skelroot = find_skelroot_path(agent_prim)          # reuse animation_utils
prim = stage.GetPrimAtPath(skelroot)
BehaviorSchema.BehaviorAgentAPI.Apply(prim)
prim.CreateRelationship("behavior:motionLibrary").SetTargets(["/World/HumanMotionLibrary"])
prim.CreateAttribute("behavior:navMeshAreasAllowed",
                     Sdf.ValueTypeNames.StringArray, custom=True).Set([])
```

No `controlRig:*` authoring is required: **[verified]** `F_Business_02.usd` in the 6.0
bucket already ships `controlRig:retargetTags` (101 entries, 56 non-empty),
`controlRig:forwardAxis = "MINUS Y"`, `controlRig:upAxis = "Z"` on its 101-joint
skeleton, and is metres / Z-up like our stage. `attach_agent()` should *assert* those
are present and fail loudly if a character model lacks them, rather than silently
producing another dead rig.

The agent handle is then acquired after the timeline starts:

```python
import omni.anim.behavior.core as bh
agent = bh.get_agent(str(skelroot))
agent.set_obstacle_avoidance_enabled(False)   # HuNav owns avoidance
agent.set_auto_avoidance_enabled(False)       # ditto
agent.set_random_seed(agent_cfg["id"])        # reproducible runs
```

**Disabling both avoidance systems is a deliberate decision, not an oversight.**
HuNavSim's social-force model already computes inter-agent and agent-robot repulsion.
Leaving Isaac's avoidance on means two controllers fighting over the same pose each
tick, which will read as jitter or as agents refusing to converge on HuNav's
trajectory. If Isaac's avoidance is ever wanted, it belongs behind its own flag and
HuNav's must be disabled in exchange.

### `update_agent()`

Two candidate strategies. **This is the main open question** — see below.

**(A) Per-tick teleport.** HuNav's pose is imposed exactly; motion matching is asked to
find clips that fit the imposed root motion.

```python
agent.set_speed(speed_mps)                    # stage units/s -- NOT cm/s
agent.teleport(carb.Float3(x, y, z), facing_dir)
```

*Pro:* HuNav's trajectory is reproduced to the millimetre — important for a navigation
benchmark. *Risk:* `teleport` is semantically a discontinuous jump; the matcher may not
build a continuous gait from a stream of them, giving a foot-sliding idle instead of a
walk.

**(B) Per-tick `move_to` at HuNav's next position.** Each tick issues a short goal and
lets the engine walk there.

```python
agent.set_speed(speed_mps * 100.0)
agent.move_to(next_position, facing_dir)
```

*Pro:* the natural input to a motion matcher; proper gait, turning and foot contact.
*Risk:* `move_to` is navmesh-pathed, so Isaac may route around obstacles differently
than HuNav did, and the agent's true position drifts from what HuNav believes. Also
`agentResponseTime = 0.3` s will lag a 10–30 Hz tick unless lowered.

**RESOLVED — (B) shipped.** The Phase 1 spike settled this on rendered frames:

| strategy | tracking error | gait |
|---|---|---|
| (A) per-tick `teleport()` | 0.000 m | **none** — idle pose, legs static |
| (B) per-tick `move_to()` | mean 0.173 m, max 0.247 m | **walk cycle** |

`teleport()` reproduces the commanded path exactly, but the matcher reads a stream of
teleports as discontinuous jumps and never selects a gait. (B)'s goal comes from
HuNavSim's own velocity — `position + velocity * lookahead` — so no future trajectory is
needed. `move_to(target, auto_brake)` takes no facing argument; the agent turns to face
its direction of travel by itself. See
[sample/behavior_agent_demo.py](sample/behavior_agent_demo.py).

Either way the *existing* orientation-smoothing slerp in `_update_agents()` should be
**removed** for this backend. It exists to compensate for the blend tree having no turn
clips; the motion library has `WalkTurns_*`, `WalkArc` and `JogTurns_*`, and hand-slerping
on top of a matcher that is already choosing turn animations will fight it.

Likewise `char.set_world_transform()` and the `HUNAV_DRIVE_CHARACTER` escape hatch in
[`hunav_manager.py:983`](../src/hunav_isaac_wrapper/hunav_manager.py#L983) are
animation-graph-specific and do not carry over.

### `verify()`

The check that would have caught the present bug in one launch:

- the motion library payload resolved and has clips under `BuiltinActions/`;
- every agent's skeleton has non-empty `controlRig:retargetTags`;
- `bh.get_agent(path)` returns non-`None` for every agent;
- after N ticks with non-zero commanded speed, sampled joint rotation across the rig
  exceeds a threshold (a few degrees) — i.e. **the body is actually moving**.

The last item is the important one and applies equally to the legacy backend. The
present `_verify_retargeted_animations()`
([hunav_manager.py:475](../src/hunav_isaac_wrapper/hunav_manager.py#L475)) only checks
the clip prims are *valid*, which a dead clip is.

---

## 3. Extension and flag wiring

`STARTUP_EXTENSIONS` in
[`teleop_hunav_sim.py:45`](../src/hunav_isaac_wrapper/teleop_hunav_sim.py#L45) becomes
backend-dependent — these must be enabled at boot, as that file's existing comment
documents:

```python
COMMON_EXTENSIONS = ("isaacsim.ros2.bridge", "isaacsim.sensors.physics", "omni.physx.bundle")

# anim_graph backend
("omni.anim.graph.core", "omni.anim.retarget.core")

# behavior_agent backend
("omni.anim.behavior.core", "omni.anim.behavior.schema", "omni.anim.navigation.core")
```

CLI, alongside the existing experimental flags at
[`main.py:415`](../src/scripts/main.py#L415):

```
--animation-backend {anim_graph,behavior_agent}   default: anim_graph
```

Default stays `anim_graph` until the new backend passes the gates below. The same
four-site plumbing as `--flat-ground` (argparse → call site → signature → constructor).

---

## 4. Staged migration

Each stage is independently useful and independently revertable.

| stage | work | gate |
|---|---|---|
| **0. Cheap probe** | Repoint `biped_setup_url()` at the **6.1** bucket for `major >= 5`; fix its inaccurate docstring. One line. | Re-measure the retargeted clip: does `ROT_CHANGE_over_clip` exceed a few degrees? If yes, animation is restored today and the rest becomes optional rather than urgent. |
| **1. Spike** | Standalone script, no ROS: one `F_Business_02`, motion library, `BehaviorAgentAPI`, driven by a synthetic circular trajectory. Tests strategy (A) then (B). | A rendered video in [`../debug/`](../debug/) showing a walk cycle. Answers both open questions. |
| **2. Refactor** | Extract `locomotion/`; move current code into `anim_graph.py` unchanged. | All four scenarios behave exactly as before. Pure refactor, no behaviour change. |
| **3. Implement** | `behavior_agent.py` + flag wiring + `verify()`. | `--animation-backend behavior_agent` walks in `brownstone_agents`. |
| **4. Compare** | Both backends on the same scenario and seed. | Positional divergence from HuNav's commanded trajectory quantified; visual quality compared side by side in `../debug/`. |
| **5. Switch** | Flip the default if stage 4 supports it. | `warehouse_agents` regression passes — this code path is shared by all four scenarios and nothing may depend on anything brownstone-specific. |

Stage 0 is worth doing first regardless: it is a one-line change that may restore
animation immediately, and it makes the migration a considered choice rather than a
forced one.

---

## Open questions

Both are answered by the stage 1 spike; neither blocks starting.

1. **Does motion matching accept externally-imposed root motion?**
   Strategy (A) depends on it. `BehaviorRootAnimation`'s three modes
   (`IGNORE_ROOT_ANIMATION` / `USE_ROOT_ANIMATION` / `APPLY_ROOT_ANIMATION_TO_HIPS`) and
   the `walkSpeedControlRatio` setting both suggest this is supported, but **[unverified]** —
   no shipped test drives an agent from an external controller. Find where that enum is
   set (it is not on `IBehaviorAgent`'s surface; likely the motion library or an
   authored attribute) before committing to (A).

2. **Is a baked navmesh required?**
   `omni.anim.behavior.core` depends on `omni.anim.navigation.core`, and the shipped test
   scenes carry `navmeshSettings` in `customLayerData`. Whether an agent driven purely by
   `teleport` still needs one is **[unverified]**. If it does, brownstone needs a navmesh
   bake step — which would also want the outstanding vegetation-collider fix, since trees
   currently have no colliders and would not register as obstacles.

## Risks

- **Migration may not fix the T-pose.** It removes the failing component
  (`CreateRetargetAnimationsCommand`) rather than repairing it, which is a strong
  argument, but it is not a proof. Stage 1 is deliberately placed before any refactor so
  this is settled cheaply.
- **`omni.behavior.tree.core` is labelled EXPERIMENTAL** in its own `extension.toml`.
  `omni.anim.behavior.core` is not so labelled, but it is young. Keeping the legacy
  backend selectable is the mitigation.
- **Units (corrected).** `set_speed`/`get_speed` are in stage units per second, not
  centimetres; `get_height()` returns 1.654 on this metres stage. The text below was
  wrong and is kept only to mark the correction. The stage, HuNav
  and every other part of this wrapper are metres. Convert at the backend boundary, in
  one place, and say so in the code.
- **Headless availability.** Whether `omni.anim.behavior.core` initialises correctly in
  this container's headless/livestream configuration is **[unverified]**; stage 1
  establishes it before any dependency is taken on it.

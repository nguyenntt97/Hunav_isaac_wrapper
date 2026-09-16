# Phase 1 spike — runnable samples

Standalone scripts that prove out the Isaac Sim 6 behavior framework before any
of it lands in the wrapper. No ROS, no HuNavSim, no brownstone world. Verified on
Isaac Sim 6.0.1 (`6.0.1-rc.7+release.42383.32955d8d.gl`) in this container.

```bash
/isaac-sim/python.sh behavior_agent_demo.py            # both drive strategies
/isaac-sim/python.sh behavior_agent_demo.py --mode moveto_tick
/isaac-sim/python.sh navmesh_ground_check.py           # the Cube-vs-Mesh finding
```

Renders land in [`../../debug/`](../../debug/). Each run takes roughly 3–6 minutes,
most of it asset download and shader compilation on the first pass.

## `behavior_agent_demo.py`

Builds a stage, makes a character a behavior agent, and drives it along a
synthetic trajectory that stands in for HuNavSim's per-tick output (straight for
4 s, then a constant-radius arc, at 20 Hz).

Three drive strategies, selectable with `--mode`:

| mode | tracking error | result |
|---|---|---|
| `teleport` | **0.000 m** | **idle pose** — legs static, no gait |
| `moveto_tick` | mean 0.173 m, max 0.247 m | **walk cycle** ✅ |
| `moveto_once` | mean 4.563 m | walks, but ignores the commanded path |

`moveto_tick` is the one to build Phase 2 on: it re-issues `move_to()` every tick
aimed `LOOKAHEAD = 0.5 s` along the trajectory. The agent walks toward that point
and the motion matcher selects a gait, while staying within ~0.25 m of where
HuNavSim wanted it.

`teleport` reproduces the commanded path perfectly, but the matcher reads a
stream of teleports as discontinuous jumps and never leaves idle. Compare
`debug/06_spikeA_teleport_strip.png` against `debug/08_spikeB_moveto_tick_strip.png`.

## `navmesh_ground_check.py`

Isolates the finding that cost the most time. The behavior system produces **no
agent without a navmesh**, and `get_agent()` returns `None` forever with nothing
logged to explain it.

```
reference (NVIDIA follow.usda)   navmesh ✓   (control)
UsdGeom.Cube  ground             navmesh ✗
UsdGeom.Mesh  ground             navmesh ✓
UsdGeom.Mesh  ground, x100 settings  navmesh ✓
```

The geometry type is the discriminator, not the settings magnitudes. **This
affects the wrapper directly:** `terrain.py`'s `--flat-ground` proxy is built as a
`UsdGeom.Cube` (`_add_ground_plane`), so it has to become a triangulated Mesh
before the behavior backend can work in flat-ground mode.

## Things that cost time, recorded so they don't again

- **`get_agent` is on the interface, not the module** —
  `bh.acquire_interface().get_agent(path)`. `omni.anim.behavior.core.get_agent`
  does not exist.
- **Speeds and body metrics are in stage units, not centimetres.**
  `get_height()` returns `1.654` on a metres stage. The NVIDIA samples report
  ~180 only because those stages are authored in cm. Pass HuNav's m/s straight
  through — do not scale by 100.
- **`CreateBehaviorAgentCommand` is unusable here.** It probes the asset for a
  `UsdSkel.Root` immediately after creating the prim, before the payload has
  loaded, and fails with *"asset missing UsdSkel.Root"*. Spawn the character,
  wait for the load, then call `ApplyBehaviorAgentAPICommand`.
- **`move_to(target, auto_brake)` takes no facing argument.** The agent faces its
  direction of travel on its own.
- **Colours in `navmeshSettings` must be `Gf.Vec3f`.** A bare Python tuple
  serialises without a typename and the resulting `.usda` will not reopen
  (`Unrecognized value typename 'color'`).
- **The spawn hierarchy did not matter.** Outer `SkelRoot` + reference (what the
  wrapper does today), `Xform` + typeless child, and a bare payload all behaved
  identically — all three failed before the navmesh existed, and the working
  setup does not depend on which is used. The existing `find_skelroot_path()`
  helper locates the right prim in every case.
- **Judge gait on the rendered strip, not on pixel-difference sums.** The demo
  prints a frame-churn number as a hint, but this investigation has twice been
  misled by numeric proxies that looked like motion.

## What still isn't answered

The demo runs one agent on flat ground. Before Phase 2 is finished:

- multiple agents sharing one motion library
- a navmesh baked over brownstone's real terrain (and the vegetation-collider gap —
  trees currently have no colliders, so they would not register as obstacles)
- behaviour under `--flat-ground` once the proxy is a Mesh, and under `--terrain-follow`

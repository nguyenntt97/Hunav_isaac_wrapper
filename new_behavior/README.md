# Adapting the HuNav Isaac wrapper to the Isaac Sim 6 behavior framework

Design notes for revising this wrapper's character locomotion so it follows the
framework Isaac Sim 6 actually ships, instead of emulating the Isaac Sim 4.x
`omni.anim.people` design that no longer exists.

| | |
|---|---|
| Status | Design proposal — nothing here is implemented yet |
| Target | Isaac Sim 6.0.1 (`6.0.1-rc.7+release.42383.32955d8d.gl`), forward-compatible with 6.1 |
| Trigger | Agents render in bind pose ("T-pose"); root cause traced to runtime retargeting |
| Scope | Character animation and locomotion only. HuNavSim, its behavior trees, ROS 2 and the world/terrain work are untouched |

## Read in this order

1. **[differences.md](differences.md)** — what changed between Isaac Sim 4.x and 6.x,
   what this wrapper does today, and why the current approach fails. Every claim is
   marked *verified* or *unverified* with how it was checked.
2. **[design.md](design.md)** — the proposed revision: a pluggable locomotion backend,
   the new `behavior_agent.py` module, the USD authoring contract, and the staged
   migration with its verification gates.

## The one-paragraph summary

Isaac Sim 6 replaced `omni.anim.people` with `omni.anim.behavior.core`, a
**motion-matching** system. Characters are marked with `BehaviorSchema.BehaviorAgentAPI`
and pointed at a `HumanMotionLibrary`; a C++ `IBehaviorAgent` interface then drives
them. This wrapper still builds a hand-authored `AnimationGraph` and retargets two
clips at runtime through a source biped — the 4.x design. On 6.0 that retarget
silently produces a near-static clip (measured: **max 3.99°, mean 0.08° of joint
rotation across the entire walk cycle**), which is the T-pose. The proposal is to add
a second locomotion backend built on `omni.anim.behavior.core`, selected by a
`--animation-backend` flag alongside the existing `--flat-ground` / `--terrain-follow`
experimental toggles, keeping the current path intact until the new one is proven.

**HuNavSim stays the brain.** Isaac's own behavior tree and its LLM tree generator are
deliberately *not* adopted — see [differences.md § Behavior trees](differences.md#5-behavior-trees-two-layers-not-two-competitors).

## What is not decided yet

Two things need an experiment before the design can be finalised, both called out in
[design.md § Open questions](design.md#open-questions):

- whether per-tick `teleport()` or per-tick `move_to()` is the right way to hand
  HuNav-computed poses to a motion-matched agent;
- whether `omni.anim.behavior.core` requires a baked navmesh even when the agent's
  path is imposed externally.

Neither blocks starting the work; both are answered by the Stage 1 spike.

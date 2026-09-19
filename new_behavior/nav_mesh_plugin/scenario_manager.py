"""
scenario_manager.py

Ties the scenario model to the stage: draws a pin per spawn and goal, reads the
pins back after the user has dragged them, and exports both artifacts.

The model is the source of truth, not the pins. A scenario is generated and
validated first, and the pins are a view over it -- so the tool is never the
only thing standing between a user and a broken scenario.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import omni.usd
from pxr import Usd

from .core import NavmeshInterface
from .sampling import NavmeshSampler
from . import pins

# The plugin is importable as new_behavior.nav_mesh_plugin (development) and as
# hunav_isaac_wrapper.nav_mesh_plugin (installed, via symlink). The scenario
# package always lives under hunav_isaac_wrapper.
try:
    from hunav_isaac_wrapper.scenario import paths as scenario_paths
    from hunav_isaac_wrapper.scenario.bt_emit import emit_all_trees, verify_trees
    from hunav_isaac_wrapper.scenario.generate import build_scenario, parse_behavior_mix
    from hunav_isaac_wrapper.scenario.spec import (
        NavmeshProvenance,
        Problem,
        ScenarioSpec,
    )
except ImportError:  # pragma: no cover - only when src/ is not on sys.path
    import sys

    _src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
    )
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from hunav_isaac_wrapper.scenario import paths as scenario_paths
    from hunav_isaac_wrapper.scenario.bt_emit import emit_all_trees, verify_trees
    from hunav_isaac_wrapper.scenario.generate import build_scenario, parse_behavior_mix
    from hunav_isaac_wrapper.scenario.spec import (
        NavmeshProvenance,
        Problem,
        ScenarioSpec,
    )


class ScenarioManager:
    """Author agent spawns, goals and behaviors against a baked navmesh."""

    def __init__(
        self,
        adapter: Optional[NavmeshInterface] = None,
        stage: Optional[Usd.Stage] = None,
        seed: Optional[int] = None,
    ):
        # Share the caller's adapter rather than building a second one: two
        # adapters disagree about whether a mesh is built and about which stage
        # they are on.
        self.adapter = adapter or NavmeshInterface(stage=stage)
        self.stage = stage or self.adapter.stage or omni.usd.get_context().get_stage()
        self.adapter.stage = self.stage
        self.sampler = NavmeshSampler(self.adapter, seed=seed)
        self.spec: Optional[ScenarioSpec] = None
        self.provenance: Optional[NavmeshProvenance] = None

    # --- navmesh readiness ----------------------------------------------

    @property
    def navmesh_ready(self) -> bool:
        return self.sampler.ready

    def require_navmesh(self) -> bool:
        if self.navmesh_ready:
            return True
        print(
            "[ScenarioManager] no navmesh. Bake through the simulator's driver "
            "first -- baking from this window uses different settings and would "
            "produce a mesh the agents are not steered on."
        )
        return False

    def set_provenance(self, provenance: NavmeshProvenance) -> None:
        """Record what the navmesh was, so the run can reproduce this bake."""
        self.provenance = provenance
        if self.spec is not None:
            self.spec.navmesh = provenance

    # --- loading --------------------------------------------------------

    def load_scenario(self, yaml_path: str) -> ScenarioSpec:
        """Load a scenario from disk and draw it."""
        self.spec = ScenarioSpec.from_yaml(yaml_path)
        if self.provenance is not None and self.spec.navmesh is None:
            self.spec.navmesh = self.provenance
        self.draw()
        print(
            f"[ScenarioManager] loaded {os.path.basename(yaml_path)}: "
            f"{len(self.spec.agents)} agents, {len(self.spec.global_goals)} goals"
        )
        return self.spec

    def generate_scenario(self, map_name: str, **kwargs) -> Tuple[ScenarioSpec, List[str]]:
        """Sample a fresh scenario on the current navmesh and draw it."""
        if not self.require_navmesh():
            raise RuntimeError("cannot generate a scenario without a baked navmesh")

        spec, notes = build_scenario(map_name, self.sampler, **kwargs)
        if self.provenance is not None:
            spec.navmesh = self.provenance
        self.spec = spec
        self.draw()

        for note in notes:
            print(f"[ScenarioManager] {note}")
        return spec, notes

    # --- drawing --------------------------------------------------------

    def draw(self) -> None:
        """Rebuild every pin from the current spec."""
        if self.spec is None:
            return

        pins.clear_pins(self.stage)
        pins.ensure_scopes(self.stage)

        for agent in self.spec.sorted_agents():
            pose = agent.init_pose
            pins.create_spawn_pin(
                agent.name, (pose.x, pose.y, pose.z), pose.h, stage=self.stage
            )

        for goal_id, (gx, gy) in sorted(self.spec.global_goals.items()):
            pins.create_goal_pin(goal_id, (gx, gy, self._ground_z()), stage=self.stage)

    def clear(self) -> List[str]:
        """Remove every pin from the stage. The spec is left alone."""
        return pins.clear_pins(self.stage)

    def set_visible(self, visible: bool) -> None:
        pins.set_pins_visible(visible, self.stage)

    # --- reading back ---------------------------------------------------

    def sync_from_stage(self) -> List[str]:
        """Pull dragged pin positions back into the spec.

        Goals whose pins were deleted are dropped and the table is renumbered,
        because hunav_loader stops scanning at the first missing id. Rings that
        referenced a deleted goal are rewritten to match.
        """
        if self.spec is None:
            return ["no scenario loaded"]

        notes: List[str] = []

        spawn_poses = pins.read_spawn_pins(self.stage)
        for agent in self.spec.agents:
            pose = spawn_poses.get(agent.name)
            if pose is None:
                notes.append(f"{agent.name}: spawn pin missing; keeping its stored pose")
                continue
            agent.init_pose.x, agent.init_pose.y, agent.init_pose.z, agent.init_pose.h = pose

        goal_positions = pins.read_goal_pins(self.stage)
        if goal_positions:
            removed = sorted(set(self.spec.global_goals) - set(goal_positions))
            self.spec.global_goals = {
                goal_id: xy for goal_id, xy in sorted(goal_positions.items())
            }
            if removed:
                mapping = self.spec.renumber_goals()
                notes.append(
                    f"goals {removed} were deleted; renumbered the table to "
                    f"1..{len(self.spec.global_goals)} and rewrote every ring "
                    f"({mapping})"
                )
            for agent in self.spec.agents:
                dropped = [g for g in agent.goals if g not in self.spec.global_goals]
                if dropped:
                    agent.goals = [g for g in agent.goals if g in self.spec.global_goals]
                    notes.append(f"{agent.name}: dropped missing goals {dropped}")
        else:
            notes.append("no goal pins on stage; keeping the stored goal table")

        return notes

    # --- editing --------------------------------------------------------

    def snap_all(self) -> List[str]:
        """Move every spawn and goal onto the walkable surface."""
        if self.spec is None or not self.require_navmesh():
            return ["nothing to snap"]

        notes: List[str] = []
        ground = self._ground_z()

        for agent in self.spec.sorted_agents():
            pose = agent.init_pose
            snapped = self.sampler.snap((pose.x, pose.y, pose.z))
            if snapped is None:
                notes.append(f"{agent.name}: no walkable surface anywhere near its spawn")
                continue
            moved = math.dist((pose.x, pose.y), snapped[:2])
            pose.x, pose.y, pose.z = snapped
            if moved > 0.01:
                notes.append(f"{agent.name}: moved {moved:.2f} m onto the navmesh")

        for goal_id, (gx, gy) in sorted(self.spec.global_goals.items()):
            snapped = self.sampler.snap((gx, gy, ground))
            if snapped is None:
                notes.append(f"goal {goal_id}: no walkable surface nearby")
                continue
            moved = math.dist((gx, gy), snapped[:2])
            self.spec.global_goals[goal_id] = (snapped[0], snapped[1])
            if moved > 0.01:
                notes.append(f"goal {goal_id}: moved {moved:.2f} m onto the navmesh")

        self.draw()
        return notes or ["everything was already on the navmesh"]

    def scatter_spawns(self, min_separation: float = 2.0) -> List[str]:
        """Re-sample every spawn position, keeping goals and behaviors."""
        if self.spec is None or not self.require_navmesh():
            return ["nothing to scatter"]

        agents = self.spec.sorted_agents()
        anchor = None
        if self.spec.global_goals:
            first = self.spec.global_goals[min(self.spec.global_goals)]
            anchor = (first[0], first[1], self._ground_z())

        points = self.sampler.sample_connected_points(
            len(agents), min_separation=min_separation, anchor=anchor
        )
        if len(points) < len(agents):
            return [
                f"only found {len(points)} spawn positions for {len(agents)} agents "
                f"at {min_separation:.1f} m separation; nothing was changed"
            ]

        for agent, point in zip(agents, points):
            agent.init_pose.x, agent.init_pose.y, agent.init_pose.z = (
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )
            if agent.goals:
                goal = self.spec.global_goals.get(agent.goals[0])
                if goal:
                    agent.init_pose.h = math.atan2(
                        goal[1] - agent.init_pose.y, goal[0] - agent.init_pose.x
                    )

        self.draw()
        return [f"scattered {len(agents)} spawns at {min_separation:.1f} m separation"]

    # --- validation & export --------------------------------------------

    def validate(self) -> List[Problem]:
        if self.spec is None:
            return [Problem("scenario", "nothing loaded")]
        nav = self.sampler if self.navmesh_ready else None
        return self.spec.validate(nav=nav)

    def export(
        self,
        scenario_path: Optional[str] = None,
        bt_dir: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Write the scenario YAML and one behavior tree per agent.

        Both, always. A YAML without its trees leaves every agent loading the
        previous scenario's ring, or no tree at all.
        """
        if self.spec is None:
            raise RuntimeError("no scenario to export")

        self.sync_from_stage()

        problems = self.validate()
        fatal = [p for p in problems if p.fatal]
        if fatal and not force:
            raise RuntimeError(
                "scenario is not launchable; nothing was written:\n  "
                + "\n  ".join(str(p) for p in fatal)
            )

        if scenario_path is None:
            scenario_path = scenario_paths.scenario_path(self.spec.yaml_base_name)
        if bt_dir is None:
            bt_dir = scenario_paths.behavior_trees_dir()

        os.makedirs(os.path.dirname(scenario_path), exist_ok=True)
        self.spec.write_yaml(scenario_path)
        written_trees = emit_all_trees(self.spec, bt_dir)

        print(
            f"[ScenarioManager] wrote {scenario_path} and "
            f"{len(written_trees)} behavior trees to {bt_dir}"
        )
        for problem in problems:
            if not problem.fatal:
                print(f"[ScenarioManager] {problem}")

        return {
            "scenario": scenario_path,
            "trees": written_trees,
            "problems": problems,
        }

    # --- helpers --------------------------------------------------------

    def _ground_z(self) -> float:
        if self.provenance is not None:
            return float(self.provenance.ground_z)
        if self.spec is not None and self.spec.navmesh is not None:
            return float(self.spec.navmesh.ground_z)
        if self.spec is not None and self.spec.agents:
            return min(a.init_pose.z for a in self.spec.agents)
        return 0.0

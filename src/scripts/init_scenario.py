#!/usr/bin/env python3
"""
init_scenario.py

Mode A, headless: build a complete HuNav scenario for a map and write both
artifacts it takes to run one -- the scenario YAML and one behavior tree per
agent.

    /isaac-sim/python.sh src/scripts/init_scenario.py \
        --map brownstone --agents 8 \
        --behaviors Regular:5,Curious:2,Scared:1 \
        --goals 15 --ring 5 --seed 0

The navmesh is baked through BehaviorAgentDriver -- the same path, settings and
ground height the simulator uses -- over the whole map rather than over the
poses being written. Every spawn and goal is then checked to be on that mesh
and every ring checked to be walkable, so a scenario that gets written is one
that runs.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir))
_ROOT = os.path.abspath(os.path.join(_SRC, os.pardir))
for _path in (_SRC, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a HuNav scenario on a map's navmesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--map", required=True, help="Map name under src/maps and src/worlds")
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default 8)")
    parser.add_argument("--goals", type=int, default=15, help="Size of the global goal table (default 15)")
    parser.add_argument("--ring", type=int, default=5, help="Goals per agent (default 5)")
    parser.add_argument(
        "--behaviors",
        default="Regular",
        help="Behavior mix, e.g. Regular:5,Curious:2,Scared:1 (default: all Regular)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0)")
    parser.add_argument(
        "--name",
        default=None,
        help="Scenario base name (default: <map>_agents). Also the behavior tree prefix.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Scenario path (default: src/scenarios/<name>.yaml)",
    )
    parser.add_argument(
        "--flat-ground",
        action="store_true",
        help="Flatten the terrain before baking, matching the launcher's flag",
    )
    parser.add_argument("--max-vel", type=float, default=1.5, help="Agent max_vel (default 1.5)")
    parser.add_argument("--radius", type=float, default=0.4, help="Agent radius (default 0.4)")
    parser.add_argument(
        "--spawn-separation", type=float, default=2.0,
        help="Minimum distance between spawns, in metres (default 2.0)",
    )
    parser.add_argument(
        "--goal-separation", type=float, default=3.0,
        help="Minimum distance between goals, in metres (default 3.0)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Give reactive behaviors only_once semantics (default: they repeat)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate and validate, but write nothing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Write even if validation found blocking problems",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        "omni.anim.navigation.core",
        "omni.anim.navigation.schema",
        "omni.anim.behavior.core",
        "omni.anim.behavior.schema",
    ):
        try:
            manager.set_extension_enabled_immediate(extension, True)
        except Exception as exc:
            print(f"[init_scenario] could not enable {extension}: {exc}", flush=True)

    for _ in range(30):
        app.update()

    try:
        status = _run(args, app)
    except Exception as exc:
        import traceback

        print(f"\n[init_scenario] FAILED: {exc}\n", file=sys.stderr, flush=True)
        traceback.print_exc()
        app.close()
        return 1

    app.close()
    return status


def _run(args, app) -> int:
    from isaacsim.storage.native import get_assets_root_path

    from hunav_isaac_wrapper.behavior_agent import BehaviorAgentDriver, NAVMESH_SETTINGS
    from hunav_isaac_wrapper.scenario import paths as scenario_paths
    from hunav_isaac_wrapper.scenario.bake import (
        ground_z_for_map,
        navmesh_settings_digest,
        sampling_cm_for_extent,
        scene_bounds,
    )
    from hunav_isaac_wrapper.scenario.bt_emit import emit_all_trees
    from hunav_isaac_wrapper.scenario.generate import build_scenario, parse_behavior_mix
    from hunav_isaac_wrapper.scenario.spec import NavmeshProvenance
    from hunav_isaac_wrapper.terrain import apply_flat_ground
    from hunav_isaac_wrapper.world_builder import WorldBuilder

    # The plugin is reachable under either spelling depending on how the
    # package was installed; the two are the same directory.
    try:
        from hunav_isaac_wrapper.nav_mesh_plugin.core import NavmeshInterface
        from hunav_isaac_wrapper.nav_mesh_plugin.sampling import NavmeshSampler
    except ImportError:
        from new_behavior.nav_mesh_plugin.core import NavmeshInterface
        from new_behavior.nav_mesh_plugin.sampling import NavmeshSampler

    src_dir = scenario_paths.wrapper_src_dir()
    maps_dir = scenario_paths.maps_dir()

    print(f"\n[init_scenario] Loading map '{args.map}' from {src_dir}/worlds", flush=True)
    builder = WorldBuilder(base_path=src_dir)
    if not builder.load_map(args.map):
        raise RuntimeError(f"map '{args.map}' not found under {src_dir}/worlds")

    stage = builder.get_stage()

    if args.flat_ground:
        apply_flat_ground(stage, args.map)
        print("[init_scenario] Terrain flattened.", flush=True)

    for _ in range(60):
        app.update()

    # --- bake, exactly the way the simulator will ------------------------
    ground_z = ground_z_for_map(args.map, maps_dir)
    bounds = scene_bounds(args.map, maps_dir, ground_z=ground_z)
    if bounds is None:
        print(
            f"[init_scenario] No map description for '{args.map}' in {maps_dir}; "
            "the navmesh volume will cover the world bounds instead.",
            flush=True,
        )
    else:
        extent = tuple(bounds[1][i] - bounds[0][i] for i in range(3))
        print(
            f"[init_scenario] Map extent {extent[0]:.1f} x {extent[1]:.1f} m, "
            f"ground at z={ground_z:.2f}",
            flush=True,
        )

    driver = BehaviorAgentDriver(stage, get_assets_root_path(), dt=1.0 / 20.0)
    driver.configure_navmesh()
    driver.ensure_navmesh_volume(bounds=bounds)
    driver.bake_navmesh(extent=driver.navmesh_extent, ground_z=ground_z)

    for _ in range(30):
        app.update()

    adapter = NavmeshInterface(stage=stage)
    sampler = NavmeshSampler(adapter, seed=args.seed)
    if not sampler.ready:
        raise RuntimeError(
            "navmesh bake produced nothing; there is no walkable surface to "
            "place agents on"
        )

    verts, _faces = adapter.get_navmesh_polygons()
    print(f"[init_scenario] NavMesh ready: {len(verts)} vertices.", flush=True)

    volume_extent = driver.navmesh_extent or (0.0, 0.0, 0.0)
    provenance = NavmeshProvenance(
        settings_digest=navmesh_settings_digest(NAVMESH_SETTINGS),
        volume_min=tuple(bounds[0]) if bounds else (0.0, 0.0, 0.0),
        volume_max=tuple(bounds[1]) if bounds else (0.0, 0.0, 0.0),
        ground_z=float(ground_z),
        sampling_cm=float(
            getattr(driver, "navmesh_sampling", None)
            or sampling_cm_for_extent(
                volume_extent, NAVMESH_SETTINGS["agentSamplingDistance"], 1000
            )
        ),
    )

    # --- generate --------------------------------------------------------
    mix = parse_behavior_mix(args.behaviors)
    base_name = args.name or f"{args.map}_agents"

    print(
        f"[init_scenario] Sampling {args.agents} agents, {args.goals} goals, "
        f"ring of {args.ring}, mix {mix}, seed {args.seed}",
        flush=True,
    )

    spec, notes = build_scenario(
        map_name=args.map,
        nav=sampler,
        num_agents=args.agents,
        num_goals=args.goals,
        goals_per_agent=args.ring,
        behavior_mix=mix,
        seed=args.seed,
        yaml_base_name=base_name,
        min_goal_separation=args.goal_separation,
        min_spawn_separation=args.spawn_separation,
        max_vel=args.max_vel,
        agent_radius=args.radius,
    )
    spec.navmesh = provenance

    if args.once:
        for agent in spec.agents:
            agent.behavior.once = True

    for note in notes:
        print(f"[init_scenario] note: {note}", flush=True)

    # --- validate --------------------------------------------------------
    problems = spec.validate(nav=sampler)
    fatal = [p for p in problems if p.fatal]
    for problem in problems:
        print(f"[init_scenario] {problem}", flush=True)

    print(
        f"\n[init_scenario] {len(spec.agents)} agents, "
        f"{len(spec.global_goals)} goals, "
        f"{len(fatal)} blocking problem(s), "
        f"{len(problems) - len(fatal)} warning(s)",
        flush=True,
    )

    behaviours = {}
    for agent in spec.agents:
        behaviours[agent.behavior.type] = behaviours.get(agent.behavior.type, 0) + 1
    print(f"[init_scenario] behavior mix: {behaviours}", flush=True)

    if args.dry_run:
        print("\n[init_scenario] --dry-run: nothing written.\n", flush=True)
        return 1 if fatal else 0

    if fatal and not args.force:
        print(
            "\n[init_scenario] Refusing to write a scenario that will not run. "
            "Loosen --spawn-separation / --goal-separation, ask for fewer "
            "agents, or pass --force.\n",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # --- write both artifacts -------------------------------------------
    out_path = args.out or scenario_paths.scenario_path(base_name)
    bt_dir = scenario_paths.behavior_trees_dir()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spec.write_yaml(out_path)
    trees = emit_all_trees(spec, bt_dir)

    print(
        f"\n[init_scenario] Wrote:\n"
        f"  scenario  {out_path}\n"
        f"  trees     {len(trees)} files in {bt_dir}\n"
        f"            {os.path.basename(trees[0])} .. {os.path.basename(trees[-1])}\n"
        f"\nRun it:\n"
        f"  ./launch_hunav_isaac.sh --config {os.path.basename(out_path)}"
        f"{' --flat-ground' if args.flat_ground else ''} --batch\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

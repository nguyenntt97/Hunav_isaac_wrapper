#!/usr/bin/env python3
"""
test_scenario_sampling.py

Verification for the navmesh-backed half of scenario authoring: snapping,
separation-constrained sampling, reachability, and the pin round trip.

The stage is a room split by a full-height wall with no gap, so "reachable" is
a real question rather than always true.

    /isaac-sim/python.sh new_behavior/nav_mesh_plugin/test_scenario_sampling.py
"""

import math
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

_ext_manager = omni.kit.app.get_app().get_extension_manager()
for _ext in ("omni.anim.navigation.core", "omni.anim.navigation.schema"):
    try:
        _ext_manager.set_extension_enabled_immediate(_ext, True)
    except Exception as _exc:  # pragma: no cover
        print(f"[test] could not enable {_ext}: {_exc}", flush=True)

for _ in range(30):
    app.update()

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "src")))

from nav_mesh_plugin import pins  # noqa: E402
from nav_mesh_plugin.core import NavmeshInterface  # noqa: E402
from nav_mesh_plugin.sampling import NavmeshSampler  # noqa: E402

PASSED = []
FAILED = []


def check(name, fn):
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


def _quad(mesh, half_x, half_y, z):
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half_x, -half_y, z),
            Gf.Vec3f(half_x, -half_y, z),
            Gf.Vec3f(half_x, half_y, z),
            Gf.Vec3f(-half_x, half_y, z),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-half_x, -half_y, z), Gf.Vec3f(half_x, half_y, z)])


def _box(stage, path, hw, hh, hz):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-hw, -hh, 0.0), Gf.Vec3f(hw, -hh, 0.0),
            Gf.Vec3f(hw, hh, 0.0), Gf.Vec3f(-hw, hh, 0.0),
            Gf.Vec3f(-hw, -hh, hz), Gf.Vec3f(hw, -hh, hz),
            Gf.Vec3f(hw, hh, hz), Gf.Vec3f(-hw, hh, hz),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]
    )
    mesh.CreateExtentAttr([Gf.Vec3f(-hw, -hh, 0.0), Gf.Vec3f(hw, hh, hz)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def build_stage():
    """A 40 x 20 m room divided by a solid wall at x == 0."""
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    _quad(ground, 20.0, 10.0, 0.0)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Wall spans the full depth, so the two halves really are disconnected.
    _box(stage, "/World/Divider", 0.5, 11.0, 3.0)

    for _ in range(30):
        app.update()
    return stage, ground.GetPrim()


def main():
    print("\n" + "=" * 70, flush=True)
    print("SCENARIO SAMPLING / PINS VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    stage, ground_prim = build_stage()

    adapter = NavmeshInterface(stage=stage)
    # Assign the whole room, not just the floor. The bake is restricted to what
    # is assigned (ov_navmesh semantics), so an unassigned wall is not an
    # obstacle -- it is simply absent, and the two halves join up. Obstacles
    # have to be part of the assignment for them to obstruct.
    assert adapter.load_mesh(stage.GetPrimAtPath("/World")), "could not load the room"
    for _ in range(30):
        app.update()

    assert adapter.build_navmesh(
        settings={
            "cellSize": 0.2,
            "agentHeight": 2.0,
            "agentRadius": 0.5,
            "agentMaxClimb": 0.25,
            "agentMaxSlope": 20.0,
        }
    ), "navmesh bake failed"
    for _ in range(30):
        app.update()

    sampler = NavmeshSampler(adapter, seed=0)
    verts, _ = adapter.get_navmesh_polygons()
    print(f"\n  navmesh: {len(verts)} vertices\n", flush=True)

    def snap_ready():
        assert sampler.ready, "sampler reports no navmesh after a successful bake"

    def snap_from_air():
        # The claim the authoring tool rests on: a pin lifted off the floor
        # comes back down onto the walkable surface.
        snapped = sampler.snap((10.0, 0.0, 25.0))
        assert snapped is not None, "no closest point for a point above the floor"
        assert abs(snapped[2]) < 1.0, f"snapped to z={snapped[2]:.3f}, expected ~0"
        assert math.dist((10.0, 0.0), snapped[:2]) < 1.0, snapped

    def snap_is_idempotent():
        first = sampler.snap((7.0, 3.0, 5.0))
        second = sampler.snap(first)
        assert second is not None
        assert math.dist(first[:2], second[:2]) < 1e-3, (first, second)

    def on_navmesh_rejects_outside():
        assert not sampler.on_navmesh((500.0, 500.0, 0.0), tolerance=0.5)
        assert sampler.on_navmesh(sampler.snap((5.0, 5.0, 0.0)))

    def sampling_respects_separation():
        points = sampler.sample_points(20, min_separation=2.0)
        assert len(points) == 20, f"only placed {len(points)}/20"
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                gap = math.dist(a[:2], b[:2])
                assert gap >= 2.0 - 1e-6, f"two points only {gap:.3f} m apart"

    def sampling_reports_shortfall():
        # 40 points at 8 m separation does not fit in a 40 x 20 room; the
        # sampler must return short rather than spin.
        points = sampler.sample_points(40, min_separation=8.0, max_tries_per_point=40)
        assert len(points) < 40, "impossible request was somehow satisfied"

    def sampled_points_are_walkable():
        for point in sampler.sample_points(10, min_separation=1.0):
            assert sampler.on_navmesh(point, tolerance=0.3), point

    def wall_blocks_reachability():
        left = sampler.snap((-15.0, 0.0, 0.0))
        right = sampler.snap((15.0, 0.0, 0.0))
        assert left is not None and right is not None
        assert left[0] < 0 and right[0] > 0, (left, right)
        assert sampler.reachable(left, sampler.snap((-10.0, 4.0, 0.0))), "same side unreachable"
        assert not sampler.reachable(left, right), "path crossed a solid wall"

    def connected_sampling_stays_on_one_island():
        anchor = sampler.snap((-15.0, 0.0, 0.0))
        points = sampler.sample_connected_points(8, min_separation=2.0, anchor=anchor)
        assert len(points) == 8, f"only placed {len(points)}/8"
        for point in points:
            assert sampler.reachable(anchor, point), point
            assert point[0] < 0.0, f"sampled across the wall: {point}"

    def path_length_is_none_when_blocked():
        left = sampler.snap((-15.0, 0.0, 0.0))
        right = sampler.snap((15.0, 0.0, 0.0))
        assert sampler.path_length(left, right) is None
        near = sampler.path_length(left, sampler.snap((-10.0, 0.0, 0.0)))
        assert near is not None and near >= 4.0, near

    def pins_round_trip():
        pins.clear_pins(stage)
        pins.create_spawn_pin("agent1", (3.0, -4.0, 0.0), 1.25, stage=stage)
        pins.create_goal_pin(7, (-6.0, 2.0, 0.0), stage=stage)

        spawns = pins.read_spawn_pins(stage)
        assert "agent1" in spawns, spawns
        x, y, z, yaw = spawns["agent1"]
        assert abs(x - 3.0) < 1e-4 and abs(y + 4.0) < 1e-4, (x, y)
        assert abs(yaw - 1.25) < 1e-4, f"heading drifted: {yaw}"

        goals = pins.read_goal_pins(stage)
        assert 7 in goals, goals
        assert abs(goals[7][0] + 6.0) < 1e-4, goals[7]

    def pins_survive_negative_yaw():
        pins.create_spawn_pin("agent2", (0.0, 0.0, 0.0), -2.142, stage=stage)
        _, _, _, yaw = pins.read_spawn_pins(stage)["agent2"]
        assert abs(yaw - (-2.142)) < 1e-4, yaw

    def pins_are_excluded_from_baking():
        prim = stage.GetPrimAtPath(pins.SPAWN_SCOPE)
        assert prim and prim.IsValid(), "spawn scope missing"
        applied = [str(s) for s in prim.GetAppliedSchemas()]
        assert any("NavMeshExclude" in s for s in applied), (
            f"pins are not excluded from the bake; applied schemas: {applied}"
        )

    def rebaking_with_pins_does_not_lose_area():
        # The concrete failure NavMeshExcludeAPI prevents: pin geometry being
        # voxelised as an obstacle, punching a hole at every spawn.
        before, _ = adapter.get_navmesh_polygons()
        area_before = len(before)

        assert adapter.build_navmesh(
            settings={
                "cellSize": 0.2,
                "agentHeight": 2.0,
                "agentRadius": 0.5,
                "agentMaxClimb": 0.25,
                "agentMaxSlope": 20.0,
            }
        ), "re-bake with pins on stage failed"
        for _ in range(30):
            app.update()

        after, _ = adapter.get_navmesh_polygons()
        assert len(after) >= area_before * 0.98, (
            f"navmesh shrank from {area_before} to {len(after)} vertices with "
            "pins on stage -- they were baked as obstacles"
        )

    def pins_clear():
        pins.clear_pins(stage)
        assert not pins.read_spawn_pins(stage)
        assert not pins.read_goal_pins(stage)

    for name, fn in [
        ("sampler sees the baked navmesh", snap_ready),
        ("a point in the air snaps down to the floor", snap_from_air),
        ("snapping an already-snapped point is a no-op", snap_is_idempotent),
        ("on_navmesh rejects a point off the map", on_navmesh_rejects_outside),
        ("sampling respects minimum separation", sampling_respects_separation),
        ("an impossible request returns short, not forever", sampling_reports_shortfall),
        ("sampled points are on the walkable surface", sampled_points_are_walkable),
        ("a solid wall blocks reachability", wall_blocks_reachability),
        ("connected sampling stays on one island", connected_sampling_stays_on_one_island),
        ("path_length is None across the wall", path_length_is_none_when_blocked),
        ("pin position and heading round trip", pins_round_trip),
        ("negative headings round trip", pins_survive_negative_yaw),
        ("pins carry NavMeshExcludeAPI", pins_are_excluded_from_baking),
        ("re-baking with pins on stage keeps the area", rebaking_with_pins_does_not_lose_area),
        ("clearing removes every pin", pins_clear),
    ]:
        check(name, fn)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed\n", flush=True)
    if FAILED:
        for name, why in FAILED:
            print(f"  FAILED {name}: {why}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except Exception as exc:
        import traceback

        print(f"\n[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        app.close()
        sys.exit(1)
    app.close()
    sys.exit(status)

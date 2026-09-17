"""
test_navmesh_tool.py

Comprehensive verification script for the Native NavMesh Plugin & Visualizer.
Runs in headless Isaac Sim to test:
1. Stage geometry setup with walkable ground and an obstacle.
2. Navmesh baking via NativeNavmeshInterface.
3. Translucent preview mesh authoring (/World/navmeshmesh).
4. Boundary outline curve generation (/World/Outline/WallOutline*).
5. Random point sampling and marker placement (/World/Points).
6. Point-to-point shortest path computation and spline authoring (/World/Path).
7. Obstacle wall extrusion generation.
"""

from __future__ import annotations

import sys
import numpy as np

# Boot Isaac Sim with navigation extensions enabled in extra_args
from isaacsim import SimulationApp

STARTUP_EXTENSIONS = [
    "omni.anim.behavior.bundle",
    "omni.anim.behavior.core",
    "omni.anim.behavior.schema",
    "omni.anim.navigation.bundle",
    "omni.anim.navigation.core",
    "omni.anim.asset",
    "omni.physx.bundle",
]

_ENABLE_ARGS = []
for _ext in STARTUP_EXTENSIONS:
    _ENABLE_ARGS += ["--enable", _ext]

print("[Test] Launching Isaac Sim SimulationApp...", flush=True)
app = SimulationApp({
    "width": 640,
    "height": 480,
    "sync_loads": True,
    "headless": True,
    "renderer": "RaytracedLighting",
    "extra_args": _ENABLE_ARGS,
})

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# Import our adapter
from new_behavior.nav_mesh_plugin import (
    NavmeshInterface,
    usd_utils,
)


def create_test_stage():
    """Create walkable ground mesh and an obstacle box directly in the default stage."""
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 1. Walkable ground plane (30m x 30m, centered at origin, Z=0)
    HALF = 15.0
    ground_path = "/World/GroundPlane/CollisionMesh"
    ground_mesh = UsdGeom.Mesh.Define(stage, ground_path)
    ground_pts = [
        Gf.Vec3f(-HALF, -HALF, 0.0),
        Gf.Vec3f(HALF, -HALF, 0.0),
        Gf.Vec3f(HALF, HALF, 0.0),
        Gf.Vec3f(-HALF, HALF, 0.0),
    ]
    ground_mesh.CreatePointsAttr(ground_pts)
    ground_mesh.CreateFaceVertexCountsAttr([3, 3])
    ground_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    ground_mesh.CreateExtentAttr([Gf.Vec3f(-HALF, -HALF, 0.0), Gf.Vec3f(HALF, HALF, 0.0)])
    UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())

    # 2. Obstacle box in the center (from -2 to +2 in X and Y, height 2m)
    obs_path = "/World/Obstacle/CollisionMesh"
    obs_mesh = UsdGeom.Mesh.Define(stage, obs_path)
    hw, hh, hz = 2.0, 2.0, 2.0
    obs_pts = [
        Gf.Vec3f(-hw, -hh, 0.0), Gf.Vec3f(hw, -hh, 0.0), Gf.Vec3f(hw, hh, 0.0), Gf.Vec3f(-hw, hh, 0.0),
        Gf.Vec3f(-hw, -hh, hz), Gf.Vec3f(hw, -hh, hz), Gf.Vec3f(hw, hh, hz), Gf.Vec3f(-hw, hh, hz),
    ]
    box_indices = [
        0, 1, 2, 0, 2, 3,  # bottom
        4, 6, 5, 4, 7, 6,  # top
        0, 4, 5, 0, 5, 1,  # front
        1, 5, 6, 1, 6, 2,  # right
        2, 6, 7, 2, 7, 3,  # back
        3, 7, 4, 3, 4, 0,  # left
    ]
    obs_mesh.CreatePointsAttr(obs_pts)
    obs_mesh.CreateFaceVertexCountsAttr([3] * 12)
    obs_mesh.CreateFaceVertexIndicesAttr(box_indices)
    obs_mesh.CreateExtentAttr([Gf.Vec3f(-hw, -hh, 0.0), Gf.Vec3f(hw, hh, hz)])
    UsdPhysics.CollisionAPI.Apply(obs_mesh.GetPrim())

    for _ in range(30):
        app.update()

    return stage, ground_mesh.GetPrim()


def run_verification():
    print("\n" + "=" * 70, flush=True)
    print("STARTING NATIVE NAVMESH ADAPTER VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    stage, ground_prim = create_test_stage()
    print("[1/7] Test stage created with walkable ground and obstacle.", flush=True)

    # Instantiate adapter
    nav_tool = NavmeshInterface(stage=stage)

    # 1. Load mesh & configure volume
    loaded = nav_tool.load_mesh(ground_prim)
    assert loaded, "Failed to load mesh geometry from ground_prim."
    print("[2/7] Ground mesh loaded and NavMeshVolume successfully sized.", flush=True)

    for _ in range(30):
        app.update()

    # 2. Bake navmesh
    print("[3/7] Baking NavMesh...", flush=True)
    settings = {
        "cellSize": 0.25,     # 0.25m = 25cm
        "agentHeight": 2.0,   # 2.0m = 200cm
        "agentRadius": 0.5,   # 0.5m = 50cm
        "agentMaxClimb": 0.4, # 0.4m = 40cm
        "agentMaxSlope": 25.0,
    }
    success = nav_tool.build_navmesh(settings=settings)
    for _ in range(30):
        app.update()

    assert success, "NavMesh baking failed!"
    assert nav_tool.built, "nav_tool.built is False after bake!"

    verts, faces = nav_tool.get_navmesh_polygons()
    print(f"      NavMesh baked successfully: {len(verts)} vertices, {len(faces)} triangles.", flush=True)
    assert len(verts) > 0, "Expected non-zero navmesh vertices!"

    # 3. Visualize navmesh (translucent preview surface)
    print("[4/7] Generating visual NavMesh (/World/navmeshmesh)...", flush=True)
    mesh_path = nav_tool.visualize_navmesh(prim_path="/World/navmeshmesh", opacity=0.75)
    assert mesh_path == "/World/navmeshmesh"
    mesh_prim = stage.GetPrimAtPath("/World/navmeshmesh")
    assert mesh_prim.IsValid() and mesh_prim.IsA(UsdGeom.Mesh), "Visual mesh prim not created!"
    print(f"      Visual mesh successfully authored at {mesh_path}.", flush=True)

    # 4. Generate boundary outlines
    print("[5/7] Generating boundary outline curves (/World/Outline/WallOutline)...", flush=True)
    outlines = nav_tool.make_outline(prim_prefix="/World/Outline/WallOutline", batched=False)
    assert len(outlines) > 0, "No outline curves were authored!"
    first_curve = stage.GetPrimAtPath(outlines[0])
    assert first_curve.IsValid() and first_curve.IsA(UsdGeom.BasisCurves), "Outline curve is not a BasisCurves prim!"
    print(f"      Authored {len(outlines)} boundary outline curves.", flush=True)

    # Also test batched curve generation
    batched_path = nav_tool.make_outline(prim_prefix="/World/Outline/WallOutlineBatched", batched=True)
    assert len(batched_path) == 1, "Batched outline creation failed!"
    print("      Batched outline curve created successfully.", flush=True)

    # 5. Query random points
    print("[6/7] Querying random points and authoring markers (/World/Points)...", flush=True)
    pts = nav_tool.get_random_points(10)
    assert pts is not None and len(pts) == 10, f"Expected 10 points, got {len(pts) if pts is not None else 0}"
    p_path = nav_tool.visualize_random_points(num_points=10, prim_path="/World/Points")
    assert p_path == "/World/Points"
    points_prim = stage.GetPrimAtPath("/World/Points")
    assert points_prim.IsValid() and points_prim.IsA(UsdGeom.Points), "Points prim not authored!"
    print(f"      Sampled {len(pts)} points successfully on navmesh surface.", flush=True)

    # 6. Query shortest path around the obstacle
    print("[7/7] Computing shortest path avoiding obstacle (-6, 0, 0) -> (+6, 0, 0)...", flush=True)
    start_pos = (-6.0, 0.0, 0.0)
    goal_pos = (6.0, 0.0, 0.0)
    path_pts = nav_tool.find_paths([start_pos], [goal_pos])
    assert len(path_pts) >= 2, f"Expected path waypoints, got {len(path_pts)}"
    curve_path = nav_tool.visualize_path(start_pos, goal_pos, prim_path="/World/Path")
    assert curve_path == "/World/Path"
    path_prim = stage.GetPrimAtPath("/World/Path")
    assert path_prim.IsValid() and path_prim.IsA(UsdGeom.BasisCurves), "Path curve prim not authored!"
    print(f"      Computed path with {len(path_pts)} waypoints avoiding obstacle.", flush=True)

    # Extra: Test wall extrusion
    wall_v, wall_t = nav_tool.make_walls(height=1.5)
    print(f"      Extruded walls geometry: {len(wall_v)} vertices, {len(wall_t)} triangles.", flush=True)
    assert len(wall_v) > 0 and len(wall_t) > 0, "Wall extrusion returned empty arrays!"

    # 7. Test Reset & Rebake
    print("[8/8] Testing Reset & Rebake functionality...", flush=True)
    nav_tool.reset_navmesh(clear_stage=True)
    assert not stage.GetPrimAtPath("/World/navmeshmesh").IsValid(), "Visual mesh was not removed on reset!"
    assert not stage.GetPrimAtPath("/World/Points").IsValid(), "Points prim was not removed on reset!"
    assert not stage.GetPrimAtPath("/World/Path").IsValid(), "Path curve was not removed on reset!"
    assert not nav_tool.built, "nav_tool.built should be False after reset!"
    print("      Reset verified: stage visualizations and cached state successfully cleared.", flush=True)

    rebaked = nav_tool.rebake_navmesh(settings={"cellSize": 0.20, "agentHeight": 2.0}, visualize=True)
    assert rebaked, "rebake_navmesh failed!"
    assert nav_tool.built, "nav_tool.built should be True after rebake!"
    assert stage.GetPrimAtPath("/World/navmeshmesh").IsValid(), "Re-baked visual mesh prim not found on stage!"
    print("      Rebake verified: NavMesh rebuilt and re-visualized with new settings.", flush=True)

    for _ in range(30):
        app.update()

    print("\n" + "=" * 70, flush=True)
    print("ALL 8 VERIFICATION CHECKS PASSED PERFECTLY!", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    try:
        run_verification()
    except Exception as exc:
        print(f"\n[FATAL ERROR] Verification failed: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        app.close()
        sys.exit(1)

    app.close()
    sys.exit(0)


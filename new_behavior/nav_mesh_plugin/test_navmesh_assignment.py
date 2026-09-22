#!/usr/bin/env python3
"""
test_navmesh_assignment.py

Does "Assign Mesh" actually restrict "Build Navmesh"?

The regression this guards: assigning a few meshes only resized the
NavMeshVolume, and omni.anim.navigation.core then voxelised everything visible
inside that box. Assigning three footpaths laid out in a triangle produced a
navmesh over the whole lawn between them, because the lawn is inside their
combined bounding box.

Standalone Isaac Sim script, same shape as test_navmesh_tool.py:
    /isaac-sim/python.sh test_navmesh_assignment.py
Exits 0 on success, 1 on failure.
"""

import sys

from isaacsim import SimulationApp

_EXTENSIONS = ("omni.anim.navigation.bundle", "omni.anim.navigation.core", "omni.physx.bundle")
_extra = []
for _ext in _EXTENSIONS:
    _extra += ["--enable", _ext]

simulation_app = SimulationApp(
    {
        "width": 800,
        "height": 600,
        "sync_loads": True,
        "headless": True,
        "renderer": "RaytracedLighting",
        "extra_args": _extra,
    }
)

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, "/workspace/Hunav_isaac_wrapper")
from new_behavior.nav_mesh_plugin.core import NavmeshInterface  # noqa: E402

FAILURES = []

# The platforms stand 0.5 m proud of the ground. Any navmesh vertex below this
# came from the ground, which is never assigned.
PLATFORM_TOP = 0.5
GROUND_CUTOFF = 0.25


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not condition:
        FAILURES.append(name)


def define_box(stage, path, centre, size):
    """A closed box whose top face winds counter-clockwise seen from above.

    Winding is not cosmetic here: Recast decides walkability from the face
    normal, so a box built with the top face reversed bakes to nothing at all.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    cx, cy, cz = centre
    points = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    faces = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateExtentAttr(
        [Gf.Vec3f(cx - hx, cy - hy, cz - hz), Gf.Vec3f(cx + hx, cy + hy, cz + hz)]
    )
    return mesh


def pump(iterations=30):
    """The baker acts on what the app has observed, not on the USD edits alone."""
    for _ in range(iterations):
        simulation_app.update()


def build_stage():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    # A ground slab far larger than the assignment, so "did the ground bake?"
    # is unambiguous in both extent and height.
    define_box(stage, "/World/Ground", (0, 0, -0.05), (60, 60, 0.1))
    define_box(stage, "/World/PlatformA", (3, 0, 0.25), (4, 4, PLATFORM_TOP))
    define_box(stage, "/World/PlatformB", (-3, 0, 0.25), (4, 4, PLATFORM_TOP))
    pump()
    return stage


def assign(adapter, paths):
    omni.usd.get_context().get_selection().set_selected_prim_paths(paths, True)
    return adapter.get_selected_prim()


def navmesh_vertices(adapter):
    verts, _faces = adapter.get_navmesh_polygons()
    return np.asarray(verts)


def test_assignment_restricts_bake(stage):
    print("\n--- assignment restricts the bake ---", flush=True)
    adapter = NavmeshInterface(stage=stage)
    check("assign returns True", assign(adapter, ["/World/PlatformA", "/World/PlatformB"]))
    check("resolved two meshes", len(adapter.input_meshes) == 2,
          f"got {[p.GetPath().pathString for p in adapter.input_meshes]}")
    pump()

    check("bake succeeds", adapter.build_navmesh())
    verts = navmesh_vertices(adapter)
    check("navmesh is not empty", verts.size > 0)
    if verts.size == 0:
        return

    on_ground = int((verts[:, 2] < GROUND_CUTOFF).sum())
    on_platform = int((verts[:, 2] >= GROUND_CUTOFF).sum())
    # The regression itself.
    check("no navmesh on the unassigned ground", on_ground == 0, f"{on_ground} vertices below z={GROUND_CUTOFF}")
    check("navmesh covers the assigned platforms", on_platform > 0, f"{on_platform} vertices")

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    # Platforms span x[-5, 5] y[-2, 2]; the volume is padded well beyond that,
    # so this distinguishes "bounded by the assignment" from "bounded by the box".
    check("navmesh stays within the assigned footprint",
          lo[0] >= -5.01 and hi[0] <= 5.01 and lo[1] >= -2.01 and hi[1] <= 2.01,
          f"x[{lo[0]:.2f}, {hi[0]:.2f}] y[{lo[1]:.2f}, {hi[1]:.2f}]")
    adapter.reset_navmesh(clear_stage=True)
    pump()


def test_no_assignment_bakes_everything(stage):
    """The runtime path assigns nothing and must keep its old behaviour."""
    print("\n--- no assignment leaves the bake unrestricted ---", flush=True)
    adapter = NavmeshInterface(stage=stage)
    omni.usd.get_context().get_selection().clear_selected_prim_paths()
    adapter.input_meshes = []
    adapter.ensure_navmesh_volume(((-30, -30, -1), (30, 30, 1)))
    pump()

    check("bake succeeds", adapter.build_navmesh())
    verts = navmesh_vertices(adapter)
    check("ground bakes when nothing is assigned",
          verts.size > 0 and int((verts[:, 2] < GROUND_CUTOFF).sum()) > 0,
          f"{0 if verts.size == 0 else int((verts[:, 2] < GROUND_CUTOFF).sum())} vertices on ground")
    adapter.reset_navmesh(clear_stage=True)
    pump()


def test_instanced_assignment(stage):
    """Assigned meshes living under an instance must survive the hide pass.

    Visibility cannot be authored on an instance proxy, only on the prototype,
    so matching the keep-set by prim path hid the assignment along with
    everything else and the bake came back empty.
    """
    print("\n--- instanced assignment ---", flush=True)
    UsdGeom.Xform.Define(stage, "/World/Proto")
    define_box(stage, "/World/Proto/Slab", (0, 0, 0.25), (4, 4, PLATFORM_TOP))
    for name, x in (("InstA", 3.0), ("InstB", -3.0)):
        xform = UsdGeom.Xform.Define(stage, f"/World/{name}")
        xform.AddTranslateOp().Set(Gf.Vec3d(x, 8.0, 0.0))
        xform.GetPrim().GetReferences().AddInternalReference(Sdf.Path("/World/Proto"))
        xform.GetPrim().SetInstanceable(True)
    pump()

    adapter = NavmeshInterface(stage=stage)
    check("assign returns True", assign(adapter, ["/World/InstA", "/World/InstB"]))
    resolved = [p for p in adapter.input_meshes if p and p.IsValid()]
    check("instanced meshes resolved", len(resolved) == 2,
          f"got {[p.GetPath().pathString for p in resolved]}")
    proxies = [p for p in resolved if p.IsInstanceProxy()]
    check("resolved through instance proxies", len(proxies) == 2, f"{len(proxies)} proxies")
    pump()

    check("bake succeeds", adapter.build_navmesh())
    verts = navmesh_vertices(adapter)
    # Before the prototype-aware keep-set this was empty.
    check("instanced assignment produces a navmesh", verts.size > 0,
          f"{0 if verts.size == 0 else len(verts)} vertices")
    if verts.size:
        on_ground = int((verts[:, 2] < GROUND_CUTOFF).sum())
        check("no navmesh on the unassigned ground", on_ground == 0, f"{on_ground} vertices")
    adapter.reset_navmesh(clear_stage=True)
    pump()


def main():
    stage = build_stage()
    test_assignment_restricts_bake(stage)
    test_no_assignment_bakes_everything(stage)
    test_instanced_assignment(stage)

    print("\n" + "=" * 60, flush=True)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}", flush=True)
    else:
        print("ALL CHECKS PASSED", flush=True)
    return 1 if FAILURES else 0


status = main()
simulation_app.close()
sys.exit(status)

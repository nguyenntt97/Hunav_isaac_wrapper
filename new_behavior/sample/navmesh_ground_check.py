#!/usr/bin/env python3
"""
navmesh_ground_check.py -- isolates the finding that cost the most time.

The behavior system will not produce an agent without a navmesh, and the
navmesh baker ignores implicit geometry. get_agent() simply returns None
forever, with nothing logged to say why.

This script builds the same ground plane three ways and reports whether a
navmesh baked:

    UsdGeom.Cube, metre-scale settings  -> NO navmesh
    UsdGeom.Mesh, metre-scale settings  -> navmesh
    UsdGeom.Mesh, 100x settings         -> navmesh

The third case shows navmeshSettings magnitudes are not the discriminator --
the geometry type is.

It also opens NVIDIA's shipped follow.usda first, as a control, to confirm
get_navmesh() reports true on a scene that is known to work.

Run:
    /isaac-sim/python.sh navmesh_ground_check.py

Relevance to this repo: terrain.py's --flat-ground proxy is currently a
UsdGeom.Cube (see _add_ground_plane), so it must become a Mesh before the
behavior backend can work in flat-ground mode.
"""

from isaacsim import SimulationApp

_enable = []
for _e in ["omni.anim.navigation.bundle", "omni.anim.navigation.core",
           "omni.anim.behavior.bundle", "omni.anim.behavior.core",
           "omni.anim.behavior.schema", "omni.physx.bundle"]:
    _enable += ["--enable", _e]
app = SimulationApp({"width": 640, "height": 480, "sync_loads": True,
                     "headless": True, "renderer": "RaytracedLighting",
                     "extra_args": _enable})

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

import omni.anim.navigation.core as nav

def log(key, value):
    print(f"[navcheck] {key} = {value}", flush=True)

navigation = nav.acquire_interface()
ctx = omni.usd.get_context()

# ---- control: NVIDIA's own scene -----------------------------------------
REFERENCE = ("/isaac-sim/extscache/"
             "omni.anim.behavior.core-110.1.4+110.1.1.lx64.r.cp312.u7f4"
             "/data/tests/usd/follow/follow.usda")
ctx.open_stage(REFERENCE)
stage = None
for _ in range(1200):
    app.update()
    stage = ctx.get_stage()
    if stage is not None and stage.GetPrimAtPath("/World/Humans").IsValid():
        break
log("reference_opened", stage is not None)
log("reference_navmesh", navigation.get_navmesh() is not None)


def build_and_bake(mesh_ground, settings_scale, tag):
    ctx.new_stage()
    stage = None
    for _ in range(600):
        app.update()
        stage = ctx.get_stage()
        if stage is not None:
            break

    root_layer = stage.GetRootLayer()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    custom = dict(root_layer.customLayerData)
    custom["navmeshSettings"] = {
        "agentMaxFloorSlope": 20.0,
        "agentMaxRadius": 0.5 * settings_scale,
        "agentMaxStepHeight": 0.25 * settings_scale,
        "agentMinHeight": 2.0 * settings_scale,
        "agentMinIslandRadius": 5.0 * settings_scale,
        "agentMinRadius": 0.2 * settings_scale,
        "agentSamplingDistance": 0.2 * settings_scale,
        "excludeRigidBodies": True,
        "areas": {
            "0": {"areaName": "Walkable",
                  "color": Gf.Vec3f(0.2, 0.8, 1.0), "defaultCost": 1.0},
            "1": {"areaName": "NotWalkable",
                  "color": Gf.Vec3f(1.0, 0.0, 0.0), "defaultCost": -1.0},
        },
    }
    root_layer.customLayerData = custom

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())
    UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1000.0)

    HALF = 30.0
    if mesh_ground:
        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/GroundPlane/CollisionMesh"))
        mesh.CreatePointsAttr([Gf.Vec3f(-HALF, -HALF, 0.0), Gf.Vec3f(HALF, -HALF, 0.0),
                               Gf.Vec3f(HALF, HALF, 0.0), Gf.Vec3f(-HALF, HALF, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateExtentAttr([Gf.Vec3f(-HALF, -HALF, 0.0), Gf.Vec3f(HALF, HALF, 0.0)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    else:
        cube = UsdGeom.Cube.Define(stage, Sdf.Path("/World/GroundPlane"))
        cube.GetSizeAttr().Set(2.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
        xform.AddScaleOp().Set(Gf.Vec3f(HALF, HALF, 0.5))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    for _ in range(120):
        app.update()

    omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                              parent_prim_path=Sdf.Path("/World"),
                              position=Gf.Vec3d(0, 0, 0))
    for prim in stage.TraverseAll():
        if prim.GetTypeName() == "NavMeshVolume":
            scale = prim.GetAttribute("xformOp:scale")
            if scale and scale.IsValid():
                scale.Set(Gf.Vec3f(40.0, 40.0, 6.0))
    for _ in range(120):
        app.update()

    navigation.start_navmesh_baking_and_wait()
    for _ in range(200):
        app.update()
    log(tag, navigation.get_navmesh() is not None)


build_and_bake(mesh_ground=False, settings_scale=1.0, tag="cube_ground_metres")
build_and_bake(mesh_ground=True, settings_scale=1.0, tag="mesh_ground_metres")
build_and_bake(mesh_ground=True, settings_scale=100.0, tag="mesh_ground_x100_settings")

log("RESULT", "COMPLETED")
app.close()

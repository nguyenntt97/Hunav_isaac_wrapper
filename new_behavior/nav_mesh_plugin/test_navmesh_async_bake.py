#!/usr/bin/env python3
"""Does the async bake run on Kit's loop without breaking it?

Regression guard for a crash the UI hit: "Build Navmesh" ran the bake inside
the button's clicked_fn, which is inside omni.ui's draw pass, and the bake
pumps app.update() so hidden meshes reach the baker. That re-entered the draw
and segfaulted Kit in ImGui::PopID(). Deferring it to a coroutine moved the
re-entrancy rather than removing it -- app.update() also drives Kit's asyncio
loop, so pumping from a task killed every other pending task with "Cannot
enter into task ... while another task is being executed".

This reproduces that path: a coroutine on the async engine bakes with meshes
assigned, so the restriction hides prims and has to settle first. Alongside it
runs a bystander task standing in for the Kit widgets that died, and the loop's
exception handler is captured -- a bake that re-enters the loop fails here.

    /isaac-sim/python.sh new_behavior/nav_mesh_plugin/test_navmesh_async_bake.py

Exits 0 on success, 1 on failure.
"""
import sys

from isaacsim import SimulationApp

_EXTENSIONS = ("omni.anim.navigation.bundle", "omni.anim.navigation.core", "omni.physx.bundle")
_extra = []
for _ext in _EXTENSIONS:
    _extra += ["--enable", _ext]
simulation_app = SimulationApp({
    "width": 800, "height": 600, "sync_loads": True, "headless": True,
    "renderer": "RaytracedLighting", "extra_args": _extra,
})

import asyncio  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, "/workspace/Hunav_isaac_wrapper")
from new_behavior.nav_mesh_plugin.core import NavmeshInterface  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def define_box(stage, path, centre, size):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    cx, cy, cz = centre
    pts = [(cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
           (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz), (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz)]
    faces = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pts])
    mesh.GetFaceVertexCountsAttr().Set([4] * 6)
    mesh.GetFaceVertexIndicesAttr().Set(faces)
    return mesh.GetPrim()


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
define_box(stage, "/World/Ground", (0, 0, -0.25), (60, 60, 0.5))
define_box(stage, "/World/PlatformA", (-3, 0, 0.25), (4, 4, 0.5))
define_box(stage, "/World/PlatformB", (3, 0, 0.25), (4, 4, 0.5))
for _ in range(20):
    simulation_app.update()

# Background tasks standing in for the Kit widgets that died last time: if the
# bake re-enters the loop, these are what break.
bystander_ticks = []
bystander_error = []


async def bystander():
    try:
        while True:
            await omni.kit.app.get_app().next_update_async()
            bystander_ticks.append(1)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        bystander_error.append(repr(exc))


loop_errors = []
asyncio.get_event_loop().set_exception_handler(
    lambda _loop, ctxt: loop_errors.append(str(ctxt.get("exception") or ctxt.get("message")))
)

bys = asyncio.ensure_future(bystander())
nvi = NavmeshInterface(stage=stage)
nvi.load_mesh(stage.GetPrimAtPath("/World/PlatformA"))
nvi.input_meshes += [stage.GetPrimAtPath("/World/PlatformB")]

result = {}


async def run_bake():
    result["ok"] = await nvi.build_navmesh_async()


task = asyncio.ensure_future(run_bake())
for _ in range(400):
    simulation_app.update()
    if task.done():
        break
bys.cancel()
for _ in range(5):
    simulation_app.update()

print("\n=== async bake on Kit's loop ===", flush=True)
check("the bake task finished", task.done(), "still pending" if not task.done() else "")
if task.done() and task.exception() is not None:
    check("the bake raised nothing", False, repr(task.exception()))
else:
    check("the bake raised nothing", True)
check("the bake produced a navmesh", bool(result.get("ok")), f"returned {result.get('ok')}")
check("other loop tasks kept running", len(bystander_ticks) > 5,
      f"{len(bystander_ticks)} ticks")
check("no bystander task was broken", not bystander_error, "; ".join(bystander_error))
check("the event loop reported no errors", not loop_errors, "; ".join(loop_errors[:3]))

verts, _ = nvi.get_navmesh_polygons()
check("restriction still applied", len(verts) > 0 and float(verts[:, 2].min()) > 0.25,
      f"{len(verts)} verts, min z={float(verts[:,2].min()) if len(verts) else 'n/a'}")

print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL CHECKS PASSED'}", flush=True)
simulation_app.close()
import os  # noqa: E402
os._exit(1 if FAILURES else 0)

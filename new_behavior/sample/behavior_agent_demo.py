#!/usr/bin/env python3
"""
behavior_agent_demo.py -- Phase 1 spike, cleaned up.

Standalone demonstration of the Isaac Sim 6 behavior framework driving a
character from an *externally computed* trajectory, the way HuNavSim drives
agents. No ROS, no HuNavSim, no brownstone world.

This is the reference for the Phase 2 implementation: every call below was
verified to work on Isaac Sim 6.0.1 in this container.

Run:
    /isaac-sim/python.sh behavior_agent_demo.py                # both strategies
    /isaac-sim/python.sh behavior_agent_demo.py --mode teleport
    /isaac-sim/python.sh behavior_agent_demo.py --mode moveto_tick

Output: GIFs and frame strips in ../../debug/.

WHAT THIS ESTABLISHED
---------------------
1. move_to() produces a walk cycle; teleport() does not.

       teleport      tracking err 0.000 m   but renders an IDLE pose
       moveto_tick   tracking err ~0.17 m   real walk cycle

   teleport() reproduces the commanded path exactly but the motion matcher
   reads it as a sequence of discontinuous jumps and never selects a gait.
   Re-issuing move_to() each tick at the trajectory position a short time
   ahead keeps the agent within ~0.25 m of the commanded path AND walks.

2. A navmesh is REQUIRED, and it only bakes off real Mesh geometry.
   An implicit UsdGeom.Cube ground produces no navmesh and therefore no
   agent -- get_agent() just returns None forever, with no error logged.
   See navmesh_ground_check.py for that experiment in isolation.

3. Speeds and body metrics are in STAGE UNITS, not centimetres.
   get_height() returns 1.654 on this metres stage. (The NVIDIA sample
   scenes report ~180 only because those stages are authored in cm.)
   So pass HuNav's m/s straight through -- do not scale by 100.

4. get_agent() lives on the interface, not the module:
       bh.acquire_interface().get_agent(skelroot_path)

5. CreateBehaviorAgentCommand is NOT usable for this flow. It probes the
   asset for a UsdSkel.Root immediately after creating the prim, before the
   payload has finished loading, and fails with
   "asset missing UsdSkel.Root". Spawn the character yourself, wait for the
   load, then call ApplyBehaviorAgentAPICommand.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["teleport", "moveto_tick", "moveto_once", "both"],
                    default="both")
parser.add_argument("--speed", type=float, default=1.2, help="m/s")
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()

from isaacsim import SimulationApp

# These must be enabled at boot, not after SimulationApp() returns -- the same
# constraint the wrapper's STARTUP_EXTENSIONS comment already documents.
STARTUP_EXTENSIONS = [
    "omni.anim.behavior.bundle",
    "omni.anim.behavior.core",      # IBehaviorSystem / IBehaviorAgent
    "omni.anim.behavior.schema",    # BehaviorAgentAPI
    "omni.anim.navigation.bundle",
    "omni.anim.navigation.core",    # navmesh baking
    "omni.anim.asset",
    "omni.anim.retarget.core",
    "omni.physx.bundle",
]
_enable = []
for _e in STARTUP_EXTENSIONS:
    _enable += ["--enable", _e]

app = SimulationApp({"width": 1280, "height": 720, "sync_loads": True,
                     "headless": True, "renderer": "RaytracedLighting",
                     "extra_args": _enable})

import math
import os
import sys

import carb
import numpy as np
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

import omni.anim.behavior.core as bh
import omni.anim.navigation.core as nav

DEBUG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "debug"))
os.makedirs(DEBUG_DIR, exist_ok=True)

SPEED = args.speed
DT = 1.0 / 20.0          # matches the wrapper's 20 Hz physics/render step
LOOKAHEAD = 0.5          # seconds ahead to aim move_to()

def log(key, value):
    print(f"[demo] {key} = {value}", flush=True)


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------
assets_root = get_assets_root_path()
CHARACTER_URL = os.path.join(
    assets_root, "Isaac/People/Characters/F_Business_02/F_Business_02.usd")
MOTION_LIBRARY_URL = os.path.join(
    assets_root, "Isaac/People/MotionLibrary/HumanMotionLibrary.usd")

stage = omni.usd.get_context().get_stage()
root_layer = stage.GetRootLayer()

stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("upAxis", "Z")
stage.SetMetadata("startTimeCode", 0)
stage.SetMetadata("endTimeCode", 100)
stage.SetMetadata("timeCodesPerSecond", 24)

# Navmesh parameters live in the root layer's customLayerData, in stage units.
# Colours must be Gf.Vec3f: a bare Python tuple serialises without a typename
# and the resulting .usda will not reopen.
custom = dict(root_layer.customLayerData)
custom["navmeshSettings"] = {
    "agentMaxFloorSlope": 20.0,
    "agentMaxRadius": 0.5,
    "agentMaxStepHeight": 0.25,
    "agentMinHeight": 2.0,
    "agentMinIslandRadius": 5.0,
    "agentMinRadius": 0.2,
    "agentSamplingDistance": 0.2,
    "excludeRigidBodies": True,
    "areas": {
        "0": {"areaName": "Walkable",
              "color": Gf.Vec3f(0.2, 0.8, 1.0), "defaultCost": 1.0},
        "1": {"areaName": "NotWalkable",
              "color": Gf.Vec3f(1.0, 0.0, 0.0), "defaultCost": -1.0},
    },
}
root_layer.customLayerData = custom

world_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
stage.SetDefaultPrim(world_prim.GetPrim())
UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1200.0)
UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight")).CreateIntensityAttr(2500.0)

# Ground MUST be a Mesh. The navmesh baker ignores implicit geometry
# (UsdGeom.Cube / Plane), and with no navmesh there is no agent.
HALF = 30.0
ground = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/GroundPlane/CollisionMesh"))
ground.CreatePointsAttr([Gf.Vec3f(-HALF, -HALF, 0.0), Gf.Vec3f(HALF, -HALF, 0.0),
                         Gf.Vec3f(HALF, HALF, 0.0), Gf.Vec3f(-HALF, HALF, 0.0)])
ground.CreateFaceVertexCountsAttr([3, 3])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
ground.CreateExtentAttr([Gf.Vec3f(-HALF, -HALF, 0.0), Gf.Vec3f(HALF, HALF, 0.0)])
ground.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.5, 0.45)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# Motion library and character, both as payloads.
stage.DefinePrim("/World/HumanMotionLibrary").GetPayloads().AddPayload(MOTION_LIBRARY_URL)
character = stage.DefinePrim("/World/Characters/agent1")
character.GetPayloads().AddPayload(CHARACTER_URL)

# Let the payloads resolve before anything inspects the hierarchy.
for _ in range(400):
    app.update()

# --------------------------------------------------------------------------
# Navmesh
# --------------------------------------------------------------------------
omni.kit.commands.execute("CreateNavMeshVolumeCommand",
                          parent_prim_path=Sdf.Path("/World"),
                          position=Gf.Vec3d(0, 0, 0))
for prim in stage.TraverseAll():
    if prim.GetTypeName() == "NavMeshVolume":
        scale = prim.GetAttribute("xformOp:scale")
        if scale and scale.IsValid():
            scale.Set(Gf.Vec3f(40.0, 40.0, 6.0))

navigation = nav.acquire_interface()
navigation.start_navmesh_baking_and_wait()
for _ in range(200):
    app.update()

# --------------------------------------------------------------------------
# Make the character a behavior agent
# --------------------------------------------------------------------------
sys.path.insert(0, "/workspace/Hunav_isaac_wrapper/src")
from hunav_isaac_wrapper.animation_utils import find_skelroot_path  # noqa: E402

skelroot_path = str(find_skelroot_path(character))
log("skelroot", skelroot_path)

omni.kit.commands.execute(
    "ApplyBehaviorAgentAPICommand",
    skelroot_prim_paths=[Sdf.Path(skelroot_path)],
    motion_library_prim_path=Sdf.Path("/World/HumanMotionLibrary"),
    motion_library_skeleton_rig="Human",
)
for _ in range(200):
    app.update()

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
world.reset()
timeline = omni.timeline.get_timeline_interface()
timeline.set_looping(True)
timeline.play()

behavior = bh.acquire_interface()
agent = None
for _ in range(200):
    world.step(render=True)
    app.update()
    agent = behavior.get_agent(skelroot_path)
    if agent is not None:
        break

if agent is None:
    log("RESULT", "FAILED -- no agent. Check the ground is a Mesh and the "
                  "navmesh baked.")
    app.close()
    sys.exit(1)

log("agent_height_m", agent.get_height())     # ~1.65 -- stage units, not cm
log("agent_radius_m", agent.get_radius())

# HuNavSim's social-force model already handles inter-agent and agent-robot
# repulsion. Leaving Isaac's avoidance on means two controllers writing the
# same pose every tick.
agent.set_obstacle_avoidance_enabled(False)
agent.set_auto_avoidance_enabled(False)
agent.set_speed(SPEED)                        # m/s, NOT cm/s
log("speed_after_set", agent.get_speed())

render_product = rep.create.render_product("/OmniverseKit_Persp", (640, 640))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach(render_product)


# --------------------------------------------------------------------------
# Trajectory: stands in for HuNavSim's per-tick output
# --------------------------------------------------------------------------
def trajectory(t):
    """Straight for 4 s, then a constant-radius arc. Returns (x, y, yaw)."""
    if t < 4.0:
        return SPEED * t, 0.0, 0.0
    s = t - 4.0
    radius = 3.0
    ang = SPEED * s / radius
    return (SPEED * 4.0 + radius * math.sin(ang),
            radius * (1.0 - math.cos(ang)),
            ang)


def drive(mode, steps):
    """Drive the agent for `steps` ticks. Returns (frames, tracking_error)."""
    agent.teleport(carb.Float3(0.0, 0.0, 0.0), carb.Float3(1.0, 0.0, 0.0))
    agent.set_speed(SPEED)
    for _ in range(20):
        world.step(render=True)

    frames, errors = [], []
    for k in range(steps):
        t = k * DT
        x, y, yaw = trajectory(t)

        if mode == "teleport":
            # Exact, but the matcher sees discontinuous jumps and stays idle.
            agent.teleport(carb.Float3(x, y, 0.0),
                           carb.Float3(math.cos(yaw), math.sin(yaw), 0.0))
        elif mode == "moveto_tick":
            # THIS IS THE ONE THAT WALKS. Aim a little ahead of the commanded
            # pose; the agent walks toward it and the matcher picks a gait.
            # move_to(target, auto_brake) -- there is no facing argument, the
            # agent faces its direction of travel by itself.
            ax, ay, _ = trajectory(t + LOOKAHEAD)
            agent.move_to(carb.Float3(ax, ay, 0.0), False)
        elif mode == "moveto_once":
            # Baseline: one distant goal, engine-pathed. Walks, but ignores
            # the commanded trajectory entirely (~4.5 m mean divergence).
            if k == 0:
                agent.move_to(carb.Float3(12.0, 0.0, 0.0), True)

        world.step(render=True)

        translation = agent.get_world_translation()
        actual_x, actual_y = float(translation[0]), float(translation[1])
        errors.append(math.hypot(actual_x - x, actual_y - y))

        # Chase camera, close enough to see the legs.
        set_camera_view(eye=[actual_x - 2.6, actual_y - 2.6, 1.7],
                        target=[actual_x, actual_y, 0.85])

        if k % 4 == 0 and k >= 20:
            frames.append(Image.fromarray(np.asarray(annotator.get_data())[:, :, :3]))

    return frames, errors


def save(frames, errors, mode, tag):
    log(f"{mode}_tracking_err_m",
        f"mean={np.mean(errors):.3f} max={np.max(errors):.3f}")
    if not frames:
        return
    frames[0].save(os.path.join(DEBUG_DIR, f"{tag}.gif"), save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    strip = frames[8:14]
    sheet = Image.new("RGB", (640 * len(strip), 640))
    for i, image in enumerate(strip):
        sheet.paste(image, (640 * i, 0))
    sheet.save(os.path.join(DEBUG_DIR, f"{tag}_strip.png"))
    # Pixel churn between consecutive frames: a rough proxy for limb motion.
    # Idle sits near ~500; a walk cycle is several times that. Judge on the
    # strip, though -- pixel counts alone have fooled this investigation before.
    churn = [int(np.abs(np.asarray(frames[i], dtype=np.int16)
                        - np.asarray(frames[i + 1], dtype=np.int16)).sum() // 1000)
             for i in range(len(frames) - 1)]
    log(f"{mode}_framediff", f"mean={int(np.mean(churn))} "
                             f"min={min(churn)} max={max(churn)}")


MODES = {"teleport": "demo_teleport",
         "moveto_tick": "demo_moveto_tick",
         "moveto_once": "demo_moveto_once"}
selected = MODES.items() if args.mode == "both" else [(args.mode, MODES[args.mode])]
for mode, tag in selected:
    save(*drive(mode, args.steps), mode, tag)

log("RESULT", "COMPLETED -- see " + DEBUG_DIR)
app.close()

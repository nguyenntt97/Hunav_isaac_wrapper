#!/usr/bin/env python3
"""
behavior_agent.py

Character locomotion on the Isaac Sim 6 behavior framework.

Isaac Sim 6 removed ``omni.anim.people`` and replaced it with
``omni.anim.behavior.core``, a motion-matching system: characters are marked
with ``BehaviorSchema.BehaviorAgentAPI``, pointed at a motion library, and
driven through an ``IBehaviorAgent`` handle. This module is the whole of that
integration; it replaces the AnimationGraph and runtime-retargeting code the
wrapper used to carry, which produced a static clip on 6.0 and left every agent
in bind pose.

HuNavSim stays authoritative: its social-force model decides where each agent
should be, and this module only translates that into something the motion
matcher can walk.

Facts established by the Phase 1 spike (new_behavior/sample/) that the code
below depends on -- each cost a debugging cycle, so they are recorded here:

* A navmesh is required. Without one ``get_agent()`` returns None forever and
  nothing is logged to say why.
* The navmesh only bakes off ``UsdGeom.Mesh``. Implicit geometry (Cube, Plane)
  is ignored -- see ``terrain.py``, whose flat-ground proxy is a Mesh for this
  reason.
* Speeds and body metrics are in stage units, not centimetres. On this metres
  stage ``get_height()`` returns ~1.65. Pass HuNav's m/s straight through.
* ``get_agent`` is on the interface, not the module:
  ``omni.anim.behavior.core.acquire_interface().get_agent(path)``.
* ``CreateBehaviorAgentCommand`` cannot be used: it probes the asset for a
  UsdSkel.Root before the payload finishes loading. Spawn, wait, then apply.
* The engine writes agent transforms to **Fabric, not USD**. ``xformOp:translate``
  on the character prim stays at its authored value while the agent walks, so
  agent poses must be read back through ``get_world_translation()`` /
  ``get_world_rotation()``, never off the prim.
* ``get_linear_velocity()`` reports zero for a walking agent, so velocity is
  finite-differenced here instead.
* ``move_to(target, auto_brake)`` takes no facing argument; the agent turns to
  face its direction of travel by itself.
"""

import math
import os as _os

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

# NOTE: this module can only be imported *after* SimulationApp has booted Kit.
# Importing pxr before that segfaults the process. That is why
# teleop_hunav_sim.py carries its own literal copy of the extension list rather
# than importing STARTUP_EXTENSIONS from here -- it needs the names in order to
# boot, which is necessarily before this module can be loaded. Keep the two
# lists in step; hunav_manager re-enables these at import time as a safety net.
#
# omni.* is imported inside the functions that use it, so that a missing
# extension surfaces at the call that needs it rather than at import.

# Kept in step with terrain.GROUND_PLANE_PATH; imported by path rather than by
# module to avoid a circular import (terrain imports this module).
FLAT_GROUND_PROXY_PATH = "/World/FlatGroundProxy"

MOTION_LIBRARY_PATH = "/World/HumanMotionLibrary"
MOTION_LIBRARY_ASSET = "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"

# The rig name the motion library is retargeted for. The stock Isaac People
# characters already carry the matching ``controlRig:retargetTags`` (56 tagged
# joints of 101 on F_Business_02), which is what makes this migration cheap.
SKELETON_RIG = "Human"

# Extensions that must be enabled during SimulationApp startup for any of this
# to work; teleop_hunav_sim.py folds these into STARTUP_EXTENSIONS.
STARTUP_EXTENSIONS = (
    "omni.anim.behavior.bundle",
    "omni.anim.behavior.core",
    "omni.anim.behavior.schema",
    "omni.anim.navigation.bundle",
    "omni.anim.navigation.core",
    "omni.anim.asset",
)

# How far ahead of the agent to place the move_to goal, in seconds of travel.
# The goal is derived from HuNavSim's reported velocity, so this is the only
# free parameter in the hand-off. Too small and the agent arrives and idles;
# too large and it cuts corners on tight turns. 0.5 s held tracking error to
# a mean of 0.17 m / max 0.25 m in the spike.
DEFAULT_LOOKAHEAD = 0.5

# Below this speed (m/s) the agent is told nothing and falls back to the motion
# library's idle animations, rather than being nudged toward a goal it is
# already standing on.
IDLE_SPEED = 0.05

# Navmesh bake parameters, in stage units (metres). Chosen to suit adult
# pedestrians: agentMinRadius/agentMaxRadius bracket HuNav's agent radii, and
# agentMaxStepHeight matches the wrapper's --step-height default.
# UNITS: centimetres, NOT stage units.
#
# This was previously wrong and it is the reason the simulator ran at ~0.1 FPS.
# omni.anim.navigation.core's navMesh/config/* settings are centimetre-valued --
# its own extension.toml marks the block "unit: cm [default in kit]" and ships
# agentMinHeight = 200, agentMinRadius = 20, agentSamplingDistance = 20, which
# are only sensible for an adult pedestrian read as cm. The plugin converts to
# stage units itself (it links UsdGeomGetStageMetersPerUnit); it does not expect
# the caller to have done it.
#
# Passing metres therefore made every navmesh dimension 100x too small, so the
# bake ran at 1/100th the intended cell size -- 10,000x the cells per unit area.
# On brownstone that meant ~4000 cells per axis instead of ~545, which is both
# why the GPU baker kept exhausting CUDA memory and why per-frame agent
# navigation cost seconds. Values below are the Kit defaults, which are already
# tuned for adult pedestrians, except agentMinIslandRadius (500 cm = 5 m, to
# discard the small disconnected patches a park scene produces).
NAVMESH_SETTINGS = {
    "agentMaxFloorSlope": 20.0,     # degrees -- unitless, not scaled
    "agentMaxRadius": 50.0,         # 0.5 m
    "agentMaxStepHeight": 25.0,     # 0.25 m
    "agentMinHeight": 200.0,        # 2.0 m
    "agentMinIslandRadius": 500.0,  # 5.0 m
    "agentMinRadius": 20.0,         # 0.2 m
    "agentSamplingDistance": 20.0,  # 0.2 m
    "excludeRigidBodies": True,
}


def _navmesh_overrides():
    """Per-key overrides from HUNAV_NAVMESH_<KEY>, for tuning runs.

    The per-frame cost of omni.anim.navigation.core is the dominant term in the
    frame on a map-sized navmesh -- a native profile of brownstone put 99.8% of
    main-thread samples inside that plugin, called from the crowd simulation on
    the Kit update tick, and the cost barely moved between 1 and 8 agents. The
    sampling distance is the knob that decides how many cells that work covers,
    so it needs to be settable without editing this file.

    Coarsening is not free: raising agentSamplingDistance drops narrow walkable
    strips out of the bake entirely, and on this map the footpaths are close to
    that limit. Any override must be checked against the baked mesh, not just
    the frame rate.

        HUNAV_NAVMESH_AGENTSAMPLINGDISTANCE=50 (centimetres)
    """
    out = {}
    for key in NAVMESH_SETTINGS:
        raw = _os.environ.get(f"HUNAV_NAVMESH_{key.upper()}")
        if raw is None or not raw.strip():
            continue
        try:
            out[key] = (
                raw.strip().lower() in ("1", "true", "yes", "on")
                if isinstance(NAVMESH_SETTINGS[key], bool)
                else float(raw)
            )
        except ValueError:
            print(f"[behavior] ignoring bad HUNAV_NAVMESH_{key.upper()}={raw!r}")
    return out


def effective_navmesh_settings():
    """NAVMESH_SETTINGS with any environment overrides applied."""
    settings = dict(NAVMESH_SETTINGS)
    settings.update(_navmesh_overrides())
    return settings


# The GPU navmesh baker allocates per axis-cell and can exhaust CUDA memory.
# The earlier figure here was 40 cells per axis, measured back when the sampling
# distance was being passed in metres: those experiments were really running at
# 100x the cell count they appeared to, so the limit they found was 100x too
# low. With the units corrected, brownstone is known to bake at 4000 cells per
# axis (that is what the 2.73 "metre" bake actually was). This keeps a wide
# margin under that while still allowing a far finer mesh than before.
_MAX_NAVMESH_CELLS_PER_AXIS = 1000

# How many times to halve the resolution when a bake comes back empty.
_NAVMESH_COARSEN_ATTEMPTS = 3

# Height above the lowest ground in the volume within which a mesh counts as
# walkable surface rather than an obstacle, in metres. Used only to decide what
# to hide during a fallback bake.
_NAVMESH_GROUND_BAND = 0.5


class BehaviorAgentError(RuntimeError):
    """Raised when the behavior framework cannot drive a character.

    Deliberately fatal. The failure this replaces was silent: retargeting
    produced an empty clip, every check passed, and the problem only showed up
    as agents gliding in bind pose.
    """


class AgentHandle:
    """One driven character. Created by :meth:`BehaviorAgentDriver.acquire`."""

    __slots__ = ("skelroot_path", "agent", "_last_pos", "_velocity",
                 "_joint_index", "_joint_sample", "_joint_moved",
                 "_joint_checks", "driven_ticks")

    def __init__(self, skelroot_path, agent):
        self.skelroot_path = skelroot_path
        self.agent = agent
        self._last_pos = None
        self._velocity = (0.0, 0.0, 0.0)
        # Skinning liveness. A character can track its goal perfectly while its
        # rig never moves -- that is exactly what the old AnimationGraph path
        # did, and no structural check caught it. These track whether a joint
        # has ever actually changed pose.
        self._joint_index = None
        self._joint_sample = None
        self._joint_moved = False
        self._joint_checks = 0
        self.driven_ticks = 0


class BehaviorAgentDriver:
    """Owns the motion library, the navmesh, and every driven character."""

    def __init__(self, stage, assets_root, lookahead=DEFAULT_LOOKAHEAD,
                 auto_brake=False, dt=1.0 / 20.0):
        self.stage = stage
        self.assets_root = assets_root
        self.lookahead = float(lookahead)
        self.auto_brake = bool(auto_brake)
        self.dt = float(dt)
        self.handles = {}
        self._attached = []
        self._navmesh_baked = False
        self.navmesh_extent = None
        self.navmesh_sampling = None

    # -- stage setup -------------------------------------------------------

    def configure_navmesh(self):
        """Author the navmesh settings. Call before baking.

        Split from load_motion_library() so the navmesh can be baked while the
        stage is still as empty as possible -- the baker's memory use scales
        with resident geometry, and the motion library is a large payload.
        """
        layer = self.stage.GetRootLayer()
        custom = dict(layer.customLayerData)
        settings = dict(NAVMESH_SETTINGS)
        # Colours must be typed: a bare Python tuple serialises without a
        # typename and the resulting .usda will not reopen.
        settings["areas"] = {
            "0": {"areaName": "Walkable",
                  "color": Gf.Vec3f(0.2, 0.8, 1.0), "defaultCost": 1.0},
            "1": {"areaName": "NotWalkable",
                  "color": Gf.Vec3f(1.0, 0.0, 0.0), "defaultCost": -1.0},
        }
        custom["navmeshSettings"] = settings
        layer.customLayerData = custom

        # customLayerData alone is NOT enough. The navigation plugin reads it
        # when a stage is opened and caches the result, so writing it afterwards
        # -- which is what happens here, because the world USD is opened before
        # this runs -- leaves the live values untouched. Those defaults are from
        # the centimetre era (agentMinHeight 200, agentSamplingDistance 20), and
        # on a metres stage they describe a 200 m tall agent: no surface can ever
        # qualify as walkable, the bake silently produces nothing, and the
        # behavior system then reports "disabled because no navmesh is
        # available". Write the carb settings too; those are authoritative at
        # bake time.
        self._apply_navmesh_settings()

    def load_motion_library(self):
        """Payload the motion library. Call after the navmesh is baked."""
        library = self.stage.DefinePrim(MOTION_LIBRARY_PATH)
        if not library.HasAuthoredPayloads():
            url = f"{self.assets_root.rstrip('/')}/{MOTION_LIBRARY_ASSET}"
            library.GetPayloads().AddPayload(url)
            print(f"[behavior] motion library: {url}")

    # Both the non-persistent and the /persistent tree are written for every
    # flag. The plugins read the non-persistent key, but the persistent one is
    # what a stage load or a timeline reset copies back over it, so setting only
    # the former lets the defaults creep back the moment the simulation starts.
    _DEBUG_GEOMETRY_SETTINGS = (
        # omni.anim.behavior.core ships its crowd-simulation debug overlays ON.
        # Those are GPU distance fields over the navigable area, and they are the
        # difference between a process where the navmesh bakes and one where it
        # dies in cudaMalloc with 22 GB free.
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentBounds", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showObstacleBounds", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentDistanceField", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showObstacleDistanceField", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAreaDistanceField", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showBorderDistanceField", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showClearanceField", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showVisibilityLines", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentPaths", False),
        ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentGoals", False),
        # omni.anim.navigation.core's own navmesh line overlay. This is the one
        # that actually costs the frame: NavigationController::simulateAgents()
        # takes a NavigationDebugLines* and fills it every tick, and
        # NavigationDebugLines::sort() then sorts a line list covering the whole
        # navmesh. Measured on brownstone (65 x 109 m navmesh, 8 agents): six of
        # six py-spy samples landed in that sort, at ~24 s per frame.
        ("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False),
        ("/exts/omni.anim.navigation.core/navMesh/config/vizGeomEnable", False),
        ("/exts/omni.anim.navigation.core/navMesh/config/vizSurfaceEnable", False),
        ("/exts/omni.anim.navigation.core/navMesh/config/vizOutlineEnable", False),
        ("/exts/omni.anim.navigation.core/navMesh/config/vizOutlineBorderOnly", False),
    )

    @staticmethod
    def suppress_debug_geometry():
        """Turn off every navmesh/crowd debug overlay, in both setting trees.

        Nothing in this wrapper renders these overlays, but building them is not
        free: the navmesh line list is rebuilt and sorted inside the per-frame
        agent simulation, which on a map-sized navmesh costs seconds per frame.

        Safe to call repeatedly, and it must be -- see the note on the persistent
        tree above.
        """
        import carb.settings

        carb_settings = carb.settings.get_settings()
        for key, value in BehaviorAgentDriver._DEBUG_GEOMETRY_SETTINGS:
            carb_settings.set(key, value)
            carb_settings.set("/persistent" + key, value)

        # HUNAV_CROWD_PROFILING=1 turns on omni.anim.behavior.core's own
        # crowd-simulation profiler, which reports its per-frame breakdown under
        # "Behavior System: Crowd Simulation Profiling:". Both plugins are
        # stripped of everything but their dynamic symbols, so a sampling
        # profiler can localise a stall to the library but cannot name the
        # function inside it; this is the instrumentation that can.
        profiling = _os.environ.get("HUNAV_CROWD_PROFILING", "0") != "0"
        for key in ("/exts/omni.anim.behavior.core/enableCrowdSimulationProfiling",
                    "/persistent/exts/omni.anim.behavior.core/"
                    "enableCrowdSimulationProfiling"):
            carb_settings.set(key, profiling)
        if profiling:
            print("[behavior] crowd-simulation profiling ON")

        # Read the settings back. They are written to two trees and restored
        # from the persistent one on a stage or timeline reset, so "we called
        # the setter" is not evidence that the overlay is off -- and the one
        # that matters costs ~24 s per frame when it is not. One line at
        # startup turns a silent seconds-per-frame regression into an
        # obvious one.
        still_on = [
            key
            for key, value in BehaviorAgentDriver._DEBUG_GEOMETRY_SETTINGS
            if value is False and carb_settings.get(key)
        ]
        if still_on:
            print(
                "[behavior] WARNING: debug overlays still enabled after "
                "suppression: " + ", ".join(still_on),
                flush=True,
            )
        else:
            print("[behavior] debug overlays confirmed off", flush=True)

    @staticmethod
    def _apply_navmesh_settings():
        """Push NAVMESH_SETTINGS into carb. See that dict for units."""
        import carb.settings

        prefix = "/exts/omni.anim.navigation.core/navMesh/config"
        carb_settings = carb.settings.get_settings()
        settings = effective_navmesh_settings()
        overrides = _navmesh_overrides()
        if overrides:
            print(f"[behavior] navmesh overrides from environment: {overrides}")
        for key, value in settings.items():
            if key == "excludeRigidBodies":
                carb_settings.set(f"{prefix}/{key}", bool(value))
            else:
                carb_settings.set(f"{prefix}/{key}", float(value))
        # navMesh/useGpu and navMesh/viewNavMesh live one level up from config/
        # and are read from the persistent tree. Both are exposed for tuning
        # runs; unset leaves the plugin's own defaults (useGpu on, view off).
        for name, env in (("useGpu", "HUNAV_NAVMESH_USEGPU"),
                          ("maxVerticesPerTile", "HUNAV_NAVMESH_MAXVERTICESPERTILE")):
            raw = _os.environ.get(env)
            if raw is None or not raw.strip():
                continue
            key = f"/exts/omni.anim.navigation.core/navMesh/{name}"
            if name == "useGpu":
                val = raw.strip().lower() in ("1", "true", "yes", "on")
            else:
                try:
                    val = int(raw)
                except ValueError:
                    print(f"[behavior] ignoring bad {env}={raw!r}")
                    continue
            carb_settings.set(key, val)
            carb_settings.set("/persistent" + key, val)
            print(f"[behavior] navmesh {name} = {val} (from {env})")

        # omni.anim.behavior.core ships its crowd-simulation debug overlays ON:
        # showAgentDistanceField, showObstacleDistanceField, showAreaDistanceField
        # and showBorderDistanceField all default to true. Those are GPU distance
        # fields over the navigable area, and they are the difference between a
        # process where the navmesh bakes and one where it dies in cudaMalloc
        # with 22 GB free -- a standalone process that never loads the behavior
        # extension bakes the same stage fine. Nothing here renders them, so turn
        # them off.
        BehaviorAgentDriver.suppress_debug_geometry()

        print(
            "[behavior] navmesh settings (centimetres): "
            f"minRadius={settings['agentMinRadius']} "
            f"minHeight={settings['agentMinHeight']} "
            f"sampling={settings['agentSamplingDistance']}"
            " (crowd debug overlays off)"
        )

    def spawn_character(self, prim_path, asset_url, position, yaw):
        """Add one character to the stage at its initial pose.

        The prim is payloaded directly, with no moving parent: the engine owns
        the character's transform once it is a behavior agent, and a parent
        transform would compose with it and carry the agent off the map.
        """
        prim = self.stage.DefinePrim(prim_path)
        prim.GetPayloads().AddPayload(asset_url)

        # The character assets ship their own xformOps, so set the existing
        # attributes rather than adding a second, conflicting op.
        translate = prim.GetAttribute("xformOp:translate")
        if translate and translate.IsValid():
            translate.Set(Gf.Vec3d(*position))
        else:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
        rotate = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate and rotate.IsValid():
            rotate.Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))
        return prim

    def ensure_navmesh_volume(self, bounds=None):
        """Create the NavMeshVolume covering the walkable world, if absent."""
        import omni.kit.commands

        existing = [p for p in self.stage.TraverseAll()
                    if p.GetTypeName() == "NavMeshVolume"]
        if not existing:
            omni.kit.commands.execute(
                "CreateNavMeshVolumeCommand",
                parent_prim_path=Sdf.Path("/World"),
                position=Gf.Vec3d(0, 0, 0))
            existing = [p for p in self.stage.TraverseAll()
                        if p.GetTypeName() == "NavMeshVolume"]
        if not existing:
            raise BehaviorAgentError(
                "CreateNavMeshVolumeCommand produced no NavMeshVolume; agents "
                "cannot be created without a navmesh.")

        if bounds is not None:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
            centre = Gf.Vec3d((min_x + max_x) * 0.5, (min_y + max_y) * 0.5,
                              (min_z + max_z) * 0.5)
            # CreateNavMeshVolumeCommand authors an extent of +/-0.5, so the
            # scale is the volume's FULL size, not its half-size. Getting this
            # wrong silently covers only a quarter of the intended footprint.
            scale = Gf.Vec3f(max((max_x - min_x) + 4.0, 1.0),
                             max((max_y - min_y) + 4.0, 1.0),
                             max((max_z - min_z) + 4.0, 6.0))
        else:
            centre, scale = Gf.Vec3d(0, 0, 0), Gf.Vec3f(40.0, 40.0, 6.0)

        volume = existing[0]
        for name, value in (("xformOp:translate", centre),
                            ("xformOp:scale", scale)):
            attr = volume.GetAttribute(name)
            if attr and attr.IsValid():
                attr.Set(value)
        print(f"[behavior] navmesh volume at {tuple(centre)} scale {tuple(scale)}")
        self.navmesh_extent = (float(scale[0]), float(scale[1]), float(scale[2]))
        return volume

    def _isolate_ground(self, ground_z):
        """Hide everything that is not walkable surface. Returns what to restore.

        The baker's memory use scales with the total geometry it voxelises, not
        with the navmesh volume: brownstone's 1268 meshes exhaust CUDA memory at
        any sampling distance and any volume size, while the same scene with
        only its ground visible bakes fine. Trees, buildings and roofs cannot be
        walked on, so hiding them for the duration of the bake costs nothing --
        HuNavSim does obstacle avoidance itself, from its own raycasts.
        """
        from pxr import Usd

        # Relative to the ground the agents stand on -- NOT to the navmesh
        # volume's floor, which is padded well below it. Getting that wrong
        # hides the ground itself and the bake then has nothing to work with.
        band_top = float(ground_z) + _NAVMESH_GROUND_BAND

        # When a flat-ground proxy exists it already *is* the walkable surface
        # for the whole world, so keeping anything else only adds geometry for
        # the baker to choke on. Keeping the ~1000 at-grade path and hardscape
        # meshes as well was still too much for brownstone; keeping the proxy
        # alone bakes.
        proxy = self.stage.GetPrimAtPath(FLAT_GROUND_PROXY_PATH)
        proxy_only = bool(proxy and proxy.IsValid())

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_])
        hidden = []
        # TraverseAll(), not Traverse(): the latter does not descend into
        # instancing prototypes, and brownstone's vegetation -- by far the
        # densest geometry in the scene, and the geometry most likely to blow up
        # the baker -- is instanced.
        for prim in self.stage.TraverseAll():
            if not prim.IsActive():
                continue
            path_str = prim.GetPath().pathString
            if path_str.startswith(("/World/Go2", "/World/Nova_Carter", "/World/Carter", "/World/Jetbot", "/World/Create3")):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if proxy_only and path_str.startswith(
                    FLAT_GROUND_PROXY_PATH):
                # The proxy is authored invisible; the baker needs to see it.
                UsdGeom.Imageable(prim).MakeVisible()
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if not proxy_only:
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                # Keep anything whose whole extent lies within the walkable
                # band -- floors and paths. Anything rising out of the band is
                # an obstacle as far as walking goes.
                if not rng.IsEmpty() and rng.GetMax()[2] <= band_top:
                    continue
            imageable.MakeInvisible()
            hidden.append(prim)
        return hidden

    @staticmethod
    def _pump(iterations):
        """Advance the app so pending stage edits reach the renderer."""
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(iterations):
            app.update()

    @staticmethod
    def _dump_settings_if_asked():
        """HUNAV_NAV_DEBUG=<path> dumps the settings the baker will see.

        The navmesh baker fails inside the full simulator but succeeds on the
        same stage in a standalone process, while reporting a CUDA
        out-of-memory with ~22 GB of VRAM free. The allocation is therefore a
        function of some input that differs between the two processes; this
        dumps the candidates so the two can be diffed.
        """
        import json
        import os

        target = os.environ.get("HUNAV_NAV_DEBUG")
        if not target:
            return
        import carb.settings

        carb_settings = carb.settings.get_settings()
        roots = ("/exts/omni.anim.navigation.core",
                 "/persistent/exts/omni.anim.navigation.core",
                 "/exts/omni.anim.behavior.core",
                 "/rtx/hydra", "/rtx/sceneDb", "/app/renderer",
                 "/physics", "/app/hydraEngine")
        dump = {}
        for root in roots:
            try:
                value = carb_settings.get(root)
            except Exception as exc:
                value = f"<error {exc}>"
            dump[root] = value
        try:
            with open(target, "w") as handle:
                json.dump(dump, handle, indent=2, default=str, sort_keys=True)
            print(f"[behavior] settings dumped to {target}")
        except Exception as exc:
            print(f"[behavior] could not dump settings: {exc}")

    def _restore(self, hidden):
        for prim in hidden:
            UsdGeom.Imageable(prim).MakeVisible()
        # The flat-ground proxy is deliberately invisible in normal rendering;
        # it was only made visible so the baker could see it.
        proxy = self.stage.GetPrimAtPath(FLAT_GROUND_PROXY_PATH)
        if proxy and proxy.IsValid():
            UsdGeom.Imageable(proxy).MakeInvisible()

    def bake_authored_navmesh(self, provenance):
        """Reproduce the navmesh an authoring session designed.

        ``provenance`` is a NavmeshProvenance carrying an assignment: the prims
        the author picked in the navmesh helper, the volume that bake used and
        the settings it ran with.

        The native baker has no "bake only these meshes" input -- it voxelises
        whatever is visible inside the volume -- so an assignment is reproduced
        only by restoring the same volume, the same settings and the same
        visibility. That is precisely what the helper's own bake does, so this
        calls it rather than reimplementing it: a second implementation would
        drift from the first, and the symptom would be a navmesh that differs
        from the authored one in ways nothing reports.

        Returns True when the authored mesh baked. False means the caller must
        fall back to deriving its own bake, which produces a *different* mesh --
        one the scenario's spawns and goals were never validated against.
        """
        from .nav_mesh_plugin.core import NavmeshInterface

        adapter = NavmeshInterface(stage=self.stage)
        resolved = adapter.assign_paths(list(provenance.assigned_prims))
        if not resolved:
            print(
                "[behavior] the scenario's assigned navmesh prims resolved to no "
                "mesh on this stage; the authored navmesh cannot be reproduced."
            )
            return False

        expected = int(provenance.assigned_mesh_count or 0)
        if expected and resolved != expected:
            print(
                f"[behavior] WARNING: the assignment resolved to {resolved} mesh(es) "
                f"but {expected} were assigned when the scenario was authored. The "
                "usual cause is launching with different world flags than the "
                "authoring session used -- --flat-ground hides raised meshes, and "
                "a hidden mesh is not baked."
            )

        settings = dict(provenance.bake_settings or {})
        override = _navmesh_overrides().get("agentSamplingDistance")
        if override is not None:
            # The tuning knob still wins when it is set explicitly -- it exists
            # for frame-rate work -- but it stops this being the authored mesh,
            # and coarsening drops narrow walkable strips out of the bake, so it
            # cannot pass silently. NAVMESH_SETTINGS is in centimetres and the
            # helper's cellSize is in metres.
            settings["cellSize"] = float(override) / 100.0
            print(
                f"[behavior] WARNING: HUNAV_NAVMESH_AGENTSAMPLINGDISTANCE={override} "
                "overrides the authored sampling distance. This run is no longer "
                "on the navmesh the scenario was authored against; unset it to "
                "bake the designed mesh."
            )

        adapter.set_navmesh_volume_box(provenance.volume_min, provenance.volume_max)
        size = tuple(float(hi) - float(lo)
                     for lo, hi in zip(provenance.volume_min, provenance.volume_max))

        # 250 frames to settle, not the helper's 6: a run changes the visibility
        # of every mesh in the world at once during start-up, and baking before
        # that reaches the baker bakes the full scene instead of the assignment.
        if not adapter.build_navmesh(settings=settings, restrict_to_assigned=True,
                                     settle_frames=250):
            print(
                "[behavior] the authored navmesh bake produced nothing. Note that a "
                "failed bake poisons every later bake in this process, so a fallback "
                "is unlikely to succeed either -- fix the scenario's assignment "
                "rather than reading the next bake as healthy."
            )
            return False

        self._navmesh_baked = True
        self.navmesh_extent = size
        cell = float(settings.get("cellSize") or 0.0)
        self.navmesh_sampling = cell * 100.0 if cell < 15.0 else cell
        print(
            f"[behavior] authored navmesh baked: {resolved} assigned mesh(es) at "
            f"{self.navmesh_sampling:.1f} cm sampling, volume "
            f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m"
        )
        return True

    def bake_navmesh(self, extent=None, ground_z=None):
        """Bake the navmesh and block until it is done.

        ``extent`` is the (x, y, z) size of the navmesh volume in metres, used
        to pick a sampling distance the baker can actually afford. ``ground_z``
        is the height the agents stand at; when given, a failed bake is retried
        with everything above the walkable band hidden. Coarsens and retries if
        a bake comes back empty, because the GPU baker signals exhaustion by
        producing nothing rather than by raising.
        """
        import carb.settings
        import omni.anim.navigation.core as nav

        interface = nav.acquire_interface()
        carb_settings = carb.settings.get_settings()
        key = "/exts/omni.anim.navigation.core/navMesh/config/agentSamplingDistance"

        # Via effective_navmesh_settings(), not NAVMESH_SETTINGS: the bake sets
        # this key itself after _apply_navmesh_settings() has run, so reading
        # the raw dict here would silently discard an environment override and
        # bake at the default while reporting the override as applied.
        base = effective_navmesh_settings()["agentSamplingDistance"]
        if extent:
            # Keep the cell count per axis inside what the baker can allocate.
            # ``extent`` is in stage units (metres) but the sampling distance is
            # in centimetres, so the two have to be brought into the same unit
            # before they are compared. Mixing them is what silently drove the
            # bake to a 100x-too-fine mesh.
            largest_cm = max(float(e) for e in extent) * 100.0
            base = max(base, largest_cm / _MAX_NAVMESH_CELLS_PER_AXIS)

        self._dump_settings_if_asked()

        def attempt_bake(label):
            sampling = base
            for attempt in range(_NAVMESH_COARSEN_ATTEMPTS):
                # Re-assert every time: opening a stage reverts these to the
                # plugin's centimetre-era defaults.
                self._apply_navmesh_settings()
                carb_settings.set(key, float(sampling))
                interface.start_navmesh_baking_and_wait()
                if interface.get_navmesh() is not None:
                    self._navmesh_baked = True
                    self.navmesh_sampling = sampling
                    print(
                        f"[behavior] navmesh baked at {sampling:.1f} cm sampling"
                        f" ({label})"
                    )
                    return True
                print(
                    f"[behavior] navmesh bake produced nothing at "
                    f"{sampling:.1f} cm sampling ({label}, attempt "
                    f"{attempt + 1}/{_NAVMESH_COARSEN_ATTEMPTS}); coarsening"
                )
                sampling *= 2.0
            return False

        # Bake against the walkable surface alone FIRST, not as a fallback.
        #
        # The baker allocates absurdly on dense scenes -- it reports "CUDA error:
        # out of memory" with ~22 GB of VRAM free -- and, crucially, a failed
        # bake poisons the process: every subsequent bake then returns an empty
        # navmesh too, including one that would have succeeded on its own. So a
        # full-scene attempt is not a free thing to try first; it destroys the
        # attempt that works. Measured on brownstone: isolate-then-bake succeeds
        # at 2.73 m on the first try with no CUDA error, while the identical
        # bake after three failed full-scene attempts fails at every sampling
        # distance.
        if ground_z is not None:
            hidden = self._isolate_ground(ground_z)
            print(
                f"[behavior] baking navmesh with {len(hidden)} non-walkable "
                "meshes hidden"
            )
            # Let the visibility changes propagate before baking. Without this
            # the baker still sees the full scene -- which is the whole reason
            # the isolated bake worked in a test script (which pumped frames
            # between the two steps) and failed inside the simulator.
            self._pump(250)
            try:
                if attempt_bake("ground only"):
                    return
            finally:
                self._restore(hidden)
                self._pump(60)

        # Only now try the whole scene, for worlds where isolation found nothing
        # usable. If the isolated bake already failed this is unlikely to help,
        # but it costs one attempt and covers worlds with no flat-ground proxy.
        if attempt_bake("full scene"):
            return

        raise BehaviorAgentError(
            "navmesh bake produced nothing at any sampling distance up to "
            f"{base * 2 ** _NAVMESH_COARSEN_ATTEMPTS:.1f} cm, with and without "
            "non-walkable geometry hidden. Without a navmesh the behavior system "
            "disables itself and no agent is ever created. Check that the "
            "walkable ground is UsdGeom.Mesh (implicit Cube/Plane geometry is "
            "ignored by the baker) and that the NavMeshVolume covers it. Note "
            "that the GPU baker fails by returning an empty navmesh after "
            "logging 'CUDA error: out of memory', and that the CPU path "
            "(navMesh/useGpu = False) produces nothing at all in this build."
        )

    # -- per-agent setup ---------------------------------------------------

    def attach(self, character_prim, find_skelroot):
        """Apply BehaviorAgentAPI to a spawned character.

        ``find_skelroot`` is injected so this module does not depend on the
        rest of the package; in practice it is
        ``animation_utils.find_skelroot_path``.
        """
        skelroot_path = str(find_skelroot(character_prim))
        prim = self.stage.GetPrimAtPath(skelroot_path)
        if not prim or not prim.IsValid() or prim.GetTypeName() != "SkelRoot":
            raise BehaviorAgentError(
                f"{character_prim.GetPath()}: no SkelRoot found "
                f"(resolved to {skelroot_path!r}). The character asset may not "
                "have finished loading before attach() was called.")

        self._assert_control_rig(prim, character_prim)

        import omni.kit.commands

        omni.kit.commands.execute(
            "ApplyBehaviorAgentAPICommand",
            skelroot_prim_paths=[Sdf.Path(skelroot_path)],
            motion_library_prim_path=Sdf.Path(MOTION_LIBRARY_PATH),
            motion_library_skeleton_rig=SKELETON_RIG)
        self._attached.append(skelroot_path)
        return skelroot_path

    def _assert_control_rig(self, skelroot_prim, character_prim):
        """Fail loudly if the character lacks the retarget tags.

        Motion matching maps the library's clips onto the rig through
        ``controlRig:retargetTags``. A character without them produces an agent
        that never poses -- the same silent failure mode as the old retarget
        path, which is exactly what this check exists to prevent.
        """
        from pxr import Usd

        for prim in Usd.PrimRange(skelroot_prim):
            if prim.GetTypeName() != "Skeleton":
                continue
            tags = prim.GetAttribute("controlRig:retargetTags").Get()
            tagged = sum(1 for tag in (tags or []) if tag)
            if tagged == 0:
                raise BehaviorAgentError(
                    f"{character_prim.GetPath()}: skeleton {prim.GetPath()} has "
                    "no controlRig:retargetTags. Motion matching cannot pose "
                    "this rig, and it would render in bind pose. Use a stock "
                    "Isaac People character, or run AutoSetupControlRigCommand "
                    "on the asset first.")
            return tagged
        raise BehaviorAgentError(
            f"{character_prim.GetPath()}: no Skeleton under {skelroot_prim.GetPath()}.")

    def set_random_seed(self, seed):
        """Seed the behavior system for reproducible runs.

        Seeding is system-wide, not per agent: set_random_seed lives on
        IBehaviorSystem, not IBehaviorAgent.
        """
        import omni.anim.behavior.core as bh

        bh.acquire_interface().set_random_seed(int(seed))

    def acquire(self, skelroot_path, agent_id=None, max_steps=200, step=None):
        """Get the live IBehaviorAgent. Only valid once the timeline is playing.

        ``step`` is called between attempts (the wrapper passes World.step);
        the agent is not registered until the simulation has ticked.
        """
        import omni.anim.behavior.core as bh

        interface = bh.acquire_interface()
        agent = None
        for _ in range(max_steps):
            agent = interface.get_agent(skelroot_path)
            if agent is not None:
                break
            if step is not None:
                step()
        if agent is None:
            raise BehaviorAgentError(
                f"no behavior agent at {skelroot_path} after {max_steps} steps. "
                "The usual cause is a missing navmesh: it bakes only from "
                "UsdGeom.Mesh geometry, so a world whose ground is implicit "
                "(Cube/Plane) produces none, and agents are never created.")

        # HuNavSim's social-force model already computes inter-agent and
        # agent-robot repulsion. Leaving Isaac's avoidance on puts two
        # controllers on the same pose every tick.
        agent.set_obstacle_avoidance_enabled(False)
        agent.set_auto_avoidance_enabled(False)

        handle = AgentHandle(skelroot_path, agent)
        # Prefer a leg joint: it is the one that unambiguously distinguishes a
        # walk cycle from a character being slid along the ground.
        try:
            index = agent.get_joint_index("LeftLeg")
            handle._joint_index = index if index is not None and index >= 0 else 0
        except Exception:
            handle._joint_index = 0
        self.handles[skelroot_path] = handle
        return handle

    _JOINT_SAMPLE_EVERY = 10

    def _sample_joint(self, handle):
        """Note whether this agent's rig has ever changed pose."""
        if handle._joint_moved or handle._joint_index is None:
            return
        handle._joint_checks += 1
        if handle._joint_checks % self._JOINT_SAMPLE_EVERY:
            return

        # This is a diagnostic. It must never be able to take down the run, so
        # any failure disables the sampler for this agent instead of raising.
        try:
            import carb

            translation = carb.Float3(0.0, 0.0, 0.0)
            rotation = carb.Float4(0.0, 0.0, 0.0, 1.0)
            # Out-parameter form: returns a bool, writes into the arguments.
            if not handle.agent.get_joint_local_transform(
                    handle._joint_index, translation, rotation):
                return
            snapshot = (round(float(rotation[0]), 5), round(float(rotation[1]), 5),
                        round(float(rotation[2]), 5), round(float(rotation[3]), 5))
        except Exception:
            handle._joint_index = None
            return

        if handle._joint_sample is not None and snapshot != handle._joint_sample:
            handle._joint_moved = True
        handle._joint_sample = snapshot

    # -- per-tick driving --------------------------------------------------

    def teleport(self, handle, position, yaw):
        """Place an agent instantly. Used for spawn and reset, not for driving.

        Teleporting every tick tracks the commanded path exactly but renders as
        an idle pose: the motion matcher reads a stream of teleports as
        discontinuous jumps and never selects a gait. Driving goes through
        :meth:`drive`.
        """
        import carb

        handle.agent.teleport(
            carb.Float3(float(position[0]), float(position[1]), float(position[2])),
            carb.Float3(math.cos(yaw), math.sin(yaw), 0.0))
        handle._last_pos = None
        handle._velocity = (0.0, 0.0, 0.0)

    def drive(self, handle, position, velocity):
        """Hand one tick of HuNavSim's output to the motion matcher.

        ``position`` is where HuNavSim says the agent should be and ``velocity``
        its social-force velocity, both in metres. The goal is placed a short
        way along that velocity so the agent has somewhere to walk to; asking it
        to move to where it already stands produces no gait.
        """
        import carb

        speed = math.hypot(float(velocity[0]), float(velocity[1]))
        agent = handle.agent

        if speed < IDLE_SPEED:
            # Nothing to chase: let the motion library's idle actions play.
            return

        handle.driven_ticks += 1
        agent.set_speed(speed)
        goal_x = float(position[0]) + float(velocity[0]) * self.lookahead
        goal_y = float(position[1]) + float(velocity[1]) * self.lookahead
        agent.move_to(carb.Float3(goal_x, goal_y, float(position[2])),
                      self.auto_brake)

    def pose(self, handle):
        """Read an agent's actual pose back for HuNavSim.

        Returns ``(position, quaternion_xyzw, velocity)`` in metres.

        The engine writes agent transforms to Fabric rather than USD, so the
        character prim's xformOp attributes keep their authored values while the
        agent walks; these must come from the agent itself. ``get_linear_velocity()``
        also reports zero for a walking agent, so velocity is differenced here.
        """
        agent = handle.agent
        translation = agent.get_world_translation()
        rotation = agent.get_world_rotation()          # (x, y, z, w)
        position = (float(translation[0]), float(translation[1]), float(translation[2]))

        if handle._last_pos is None:
            velocity = (0.0, 0.0, 0.0)
        else:
            previous = handle._last_pos
            velocity = tuple((position[i] - previous[i]) / self.dt for i in range(3))
        handle._last_pos = position
        handle._velocity = velocity
        try:
            self._sample_joint(handle)
        except Exception:
            # Diagnostics never break the simulation loop.
            handle._joint_index = None

        quaternion = (float(rotation[0]), float(rotation[1]),
                      float(rotation[2]), float(rotation[3]))
        return position, quaternion, velocity

    def yaw(self, handle):
        """Facing angle in radians, from the agent's own facing direction."""
        facing = handle.agent.get_facing_direction()
        return math.atan2(float(facing[1]), float(facing[0]))


def make_ground_collider_mesh(stage, path, min_xy, max_xy, height, cell=2.0):
    """Author a flat, collidable ground quad as a subdivided Mesh.

    A Mesh specifically, because the navmesh baker ignores implicit geometry: a
    Cube or Plane ground yields no navmesh and therefore no agents at all.

    Subdivided into ``cell``-metre quads rather than authored as two big
    triangles, because PhysX cooks huge triangles badly -- "TriangleMesh:
    triangles are too big, reduce their size to increase simulation stability"
    -- and this surface is raycast against every tick by both
    HuNavManager.sample_ground_height and get_closest_obstacles.
    """
    min_x, min_y = float(min_xy[0]), float(min_xy[1])
    max_x, max_y = float(max_xy[0]), float(max_xy[1])
    z = float(height)

    columns = max(1, int(math.ceil((max_x - min_x) / float(cell))))
    rows = max(1, int(math.ceil((max_y - min_y) / float(cell))))
    step_x = (max_x - min_x) / columns
    step_y = (max_y - min_y) / rows

    points = []
    for row in range(rows + 1):
        y = min_y + row * step_y
        for column in range(columns + 1):
            points.append(Gf.Vec3f(min_x + column * step_x, y, z))

    counts, indices = [], []
    stride = columns + 1
    for row in range(rows):
        for column in range(columns):
            bottom_left = row * stride + column
            bottom_right = bottom_left + 1
            top_left = bottom_left + stride
            top_right = top_left + 1
            counts.extend((3, 3))
            indices.extend((bottom_left, bottom_right, top_right))
            indices.extend((bottom_left, top_right, top_left))

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateExtentAttr([Gf.Vec3f(min_x, min_y, z), Gf.Vec3f(max_x, max_y, z)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh.GetPrim()

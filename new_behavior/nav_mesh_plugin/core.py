"""
core.py

Native NavMesh adapter providing 1:1 API equivalence to the original
ov_navmesh Core.NavmeshInterface.

Backed natively by Isaac Sim 6's omni.anim.navigation.core (INavMesh C++ runtime)
and pure USD geometry generation via usd_utils.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import carb
import carb.settings
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
import omni.usd

from . import usd_utils

try:
    import omni.anim.navigation.core as nav
except ImportError:
    nav = None


# Default settings matching pyrecast specification
# ov_navmesh's settings dict, kept key-for-key so existing callers keep working.
# Only the first block maps onto anything omni.anim.navigation.core exposes; see
# UNSUPPORTED_SETTINGS below for the rest.
DEFAULT_RECAST_SETTINGS: Dict[str, float] = {
    # --- supported: these reach the native baker ---
    "cellSize": 0.3,         # meters  -> agentSamplingDistance
    "agentHeight": 2.0,      # meters  -> agentMinHeight
    "agentRadius": 0.6,      # meters  -> agentMaxRadius
    "agentMaxClimb": 0.9,    # meters  -> agentMaxStepHeight
    "agentMaxSlope": 45.0,   # degrees -> agentMaxFloorSlope
    # --- native-only, no ov_navmesh equivalent ---
    # The native baker takes a radius *range*, not one radius. Its shipped
    # defaults are agentMinRadius 20 / agentMaxRadius 50 cm, so the default here
    # reproduces that 0.4 ratio rather than inventing one.
    "agentMinRadius": None,       # meters; None -> agentRadius * 0.4
    "agentMinIslandRadius": 2.0,  # meters; native default is 200 cm
    "excludeRigidBodies": True,
    # The GPU baker fails by returning an *empty* navmesh after logging
    # "CUDA error: out of memory"; the CPU path is the fallback. See
    # behavior_agent.bake_navmesh.
    "useGpu": True,
    # --- accepted for ov_navmesh compatibility, but ignored ---
    "cellHeight": 0.2,
    "regionMinSize": 8.0,    # deprecated alias, converted to agentMinIslandRadius
    "regionMergeSize": 20.0,
    "edgeMaxLen": 12.0,
    "edgeMaxError": 1.3,
    "vertsPerPoly": 6.0,
    "detailSampleDist": 6.0,
    "detailSampleMaxError": 1.0,
    "partitionType": 0.0,
}

# Recast internals omni.anim.navigation.core does not expose. They were silently
# accepted and dropped, which made a settings change look like it had no effect;
# passing one now says so once.
UNSUPPORTED_SETTINGS = frozenset({
    "cellHeight",
    "regionMergeSize",
    "edgeMaxLen",
    "edgeMaxError",
    "vertsPerPoly",
    "detailSampleDist",
    "detailSampleMaxError",
    "partitionType",
})

# Settings that cause severe CPU sorting stalls when left enabled
DEBUG_OVERLAY_SETTINGS = (
    ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showVisibilityLines", False),
    ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentPaths", False),
    ("/exts/omni.anim.behavior.core/displayCrowdSimulation/showAgentGoals", False),
    ("/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False),
    ("/exts/omni.anim.navigation.core/navMesh/config/vizGeomEnable", False),
    ("/exts/omni.anim.navigation.core/navMesh/config/vizSurfaceEnable", False),
    ("/exts/omni.anim.navigation.core/navMesh/config/vizOutlineEnable", False),
    ("/exts/omni.anim.navigation.core/navMesh/config/vizOutlineBorderOnly", False),
)


class NavmeshInterface:
    """1:1 functional drop-in replacement for ov_navmesh NavmeshInterface.

    Operates directly against Isaac Sim's native navigation subsystem.
    """

    def __init__(self, up_axis: Optional[str] = None, stage: Optional[Usd.Stage] = None):
        self.stage = stage or omni.usd.get_context().get_stage()

        global nav
        if nav is None:
            try:
                import omni.anim.navigation.core as _nav
                nav = _nav
            except ImportError:
                print("[NavMeshAdapter] Warning: omni.anim.navigation.core could not be imported yet.")

        self.inav = nav.acquire_interface() if nav else None
        self._navmesh = self.inav.get_navmesh() if self.inav else None
        self.built = self._navmesh is not None

        self.input_prim = None
        self.input_meshes = []
        # Settings the volume padding is derived from; build_navmesh refreshes
        # this with whatever the caller merged in.
        self.settings = dict(DEFAULT_RECAST_SETTINGS)
        self.input_vert = None
        self.input_tri = None
        self.random_points = None
        self.wall_outline = []
        self.contour_verts = np.empty((0, 3), dtype=np.float32)
        self.contour_edges = []
        self.navmesh_v = np.empty((0, 3), dtype=np.float32)
        self.navmesh_t = np.empty((0, 3), dtype=np.int32)
        self.side_triangles = []
        self.wall_v = np.empty((0, 3), dtype=np.float32)
        self.wall_t = np.empty((0, 3), dtype=np.int32)

        # Coordinate system alignment
        if up_axis is not None:
            self.z_up = (up_axis.upper() == "Z")
        elif self.stage:
            stage_up = UsdGeom.GetStageUpAxis(self.stage)
            self.z_up = (stage_up == UsdGeom.Tokens.z)
        else:
            self.z_up = True

    def _convert_up_axis(self, vertices, inverse: bool = False):
        """Preserve original coordinate convention when operating between Y-up and Z-up."""
        if not self.z_up:
            return vertices

        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.size == 0:
            return vertices

        orig_shape = vertices.shape
        flat_verts = vertices.reshape(-1, 3)
        v_copy = np.empty_like(flat_verts)

        if inverse:
            # Y-up to Z-up
            v_copy[:, 0] = flat_verts[:, 0]
            v_copy[:, 1] = -flat_verts[:, 2]
            v_copy[:, 2] = flat_verts[:, 1]
        else:
            # Z-up to Y-up
            v_copy[:, 0] = flat_verts[:, 0]
            v_copy[:, 1] = flat_verts[:, 2]
            v_copy[:, 2] = -flat_verts[:, 1]

        return v_copy.reshape(orig_shape)

    def suppress_debug_geometry(self):
        """Disable internal line-sorting debug overlays that harm frame rate."""
        carb_settings = carb.settings.get_settings()
        for key, val in DEBUG_OVERLAY_SETTINGS:
            carb_settings.set(key, val)
            carb_settings.set(f"/persistent{key}", val)

    def _volume_padding(self) -> Tuple[float, float, float]:
        """(xy, up, down) padding around the assignment, in metres.

        ov_navmesh has no volume at all -- its bake is bounded purely by the
        geometry you assign. The volume here is an artefact of the native baker,
        so it should hug the selection rather than impose a box of its own. It
        cannot hug it exactly, though: a surface only qualifies as walkable if
        it has ``agentHeight`` of clearance above it, so the box has to carry
        that headroom or the bake silently returns nothing. The old fixed +4 m
        XY / 6 m Z was approximating that without saying so.
        """
        settings = getattr(self, "settings", None) or DEFAULT_RECAST_SETTINGS
        agent_height = float(settings.get("agentHeight", 2.0))
        agent_radius = float(settings.get("agentRadius", 0.6))
        max_climb = float(settings.get("agentMaxClimb", 0.9))
        margin = 0.5
        return agent_radius + margin, agent_height + margin, max_climb + margin

    def ensure_navmesh_volume(self, bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None):
        """Ensure a NavMeshVolume exists on stage covering the target geometry."""
        if not self.stage:
            self.stage = omni.usd.get_context().get_stage()

        existing = [p for p in self.stage.TraverseAll() if p.GetTypeName() == "NavMeshVolume"]
        if not existing:
            try:
                import omni.kit.commands
                omni.kit.commands.execute(
                    "CreateNavMeshVolumeCommand",
                    parent_prim_path=Sdf.Path("/World"),
                    position=Gf.Vec3d(0, 0, 0),
                )
                existing = [p for p in self.stage.TraverseAll() if p.GetTypeName() == "NavMeshVolume"]
            except Exception as e:
                print(f"[NavMeshAdapter] Note on NavMeshVolume creation: {e}")

        if not existing:
            # Fallback direct definition
            try:
                import NavSchema
                vol_path = Sdf.Path("/World/NavMeshVolume")
                volume = NavSchema.NavMeshVolume.Define(self.stage, vol_path)
                existing = [volume.GetPrim()]
            except Exception:
                pass

        if not existing:
            return None

        volume = existing[0]

        if bounds is not None:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
            pad_xy, pad_up, pad_down = self._volume_padding()
            # Asymmetric in Z, so the centre is not the bbox centre.
            lo_z, hi_z = min_z - pad_down, max_z + pad_up
            centre = Gf.Vec3d(
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                (lo_z + hi_z) * 0.5,
            )
            scale = Gf.Vec3f(
                max((max_x - min_x) + 2.0 * pad_xy, 2.0 * pad_xy),
                max((max_y - min_y) + 2.0 * pad_xy, 2.0 * pad_xy),
                max(hi_z - lo_z, pad_up + pad_down),
            )
            for name, val in (("xformOp:translate", centre), ("xformOp:scale", scale)):
                attr = volume.GetAttribute(name)
                if attr and attr.IsValid():
                    attr.Set(val)

        return volume

    def load_mesh(self, prim: Usd.Prim):
        """Extract mesh geometry from `prim` and position the NavMeshVolume."""
        self.input_prim = prim
        self.stage = prim.GetStage() if prim else omni.usd.get_context().get_stage()

        points, faces = usd_utils.parent_and_children_as_mesh(prim)
        self.input_vert = points
        self.input_tri = faces
        # The meshes the assignment actually resolves to. Restricting a bake
        # needs these prims; the triangle soup above cannot identify them.
        self.input_meshes = usd_utils.find_stage_meshes(self.stage, [prim])

        if len(points) > 0:
            # Bound the resolved meshes, not the prim that was passed in: an
            # Xform root is not Boundable, so compute_bounds returns None for it
            # and the volume is left at whatever size it happened to have.
            bounds = usd_utils.compute_bounds(self.input_meshes or [prim], stage=self.stage)
            if bounds:
                self.ensure_navmesh_volume(bounds)
            return True
        return False

    def get_selected_prim(self) -> bool:
        """Select prims currently active in the Omniverse context and configure the volume."""
        self.stage = omni.usd.get_context().get_stage()
        usd_context = omni.usd.get_context()
        selection = usd_context.get_selection()
        selected_paths = selection.get_selected_prim_paths()

        if not selected_paths:
            print("[NavMeshAdapter] No prim selected.")
            return False

        self.input_prim = [self.stage.GetPrimAtPath(p) for p in selected_paths]
        # Resolve once and keep the prims: get_all_stage_mesh would re-walk the
        # same hierarchy and still only hand back triangles.
        self.input_meshes = usd_utils.find_stage_meshes(self.stage, self.input_prim)
        points, faces = usd_utils.get_mesh(self.input_meshes)
        self.input_vert = points
        self.input_tri = faces

        if len(self.input_vert) == 0:
            print("[NavMeshAdapter] No mesh geometry found in selected prims.")
            return False

        bounds = usd_utils.compute_bounds(self.input_meshes, stage=self.stage)
        if bounds:
            self.ensure_navmesh_volume(bounds)
        else:
            self.ensure_navmesh_volume()

        self._warn_inverted()
        return True

    def _warn_inverted(self):
        """Say at assign time which meshes cannot bake as they stand.

        Otherwise the only symptom is a bake that comes back smaller than the
        selection, which reads as the assignment being ignored and sends people
        hunting through agent radius and sampling distance -- neither of which
        can help a surface that faces the wrong way.
        """
        try:
            inverted = self._inverted_assigned()
        except Exception:
            return
        if not inverted:
            return
        names = ", ".join(p.GetPath().name for p in inverted[:3])
        print(
            f"[NavMeshAdapter] Note: {len(inverted)} of {len(self.input_meshes)} assigned "
            f"mesh(es) are wound inside-out ({names}{' ...' if len(inverted) > 3 else ''}). "
            "Every face points down, so the baker reads them as ceilings and no "
            "agent radius or sampling distance will make them walkable. The bake "
            "reverses them on the fly; fix the winding in the asset to make it stick."
        )

    @staticmethod
    def _pump(iterations: int = 6):
        """Advance the app so pending visibility edits reach the baker.

        Baking immediately after MakeInvisible() bakes the *old* visibility --
        the same trap behavior_agent.bake_navmesh documents.

        Never call build_navmesh (and so this) from an omni.ui callback: a
        Button's clicked_fn runs inside the draw pass, and pumping from there
        re-enters it and unbalances ImGui's id stack, which segfaults Kit in
        ImGui::PopID(). ui_window._defer exists to put the bake on the next
        frame instead.
        """
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            for _ in range(iterations):
                app.update()
        except Exception:
            pass

    # Visualization prims are real UsdGeom.Mesh geometry with no
    # NavMeshExcludeAPI, so a bake that can see them voxelises the *previous*
    # navmesh into the new one.
    _VISUALIZATION_PREFIXES = ("/World/navmeshmesh", "/World/Outline", "/World/Points", "/World/Path")

    def _assigned_paths(self) -> List[str]:
        """Prim paths of the meshes the assignment resolved to."""
        return [p.GetPath().pathString for p in self.input_meshes if p and p.IsValid()]

    # The settings that actually reach the native baker. The rest of
    # DEFAULT_RECAST_SETTINGS is ov_navmesh compatibility padding that
    # _configure_bake drops, and recording it would make a scenario file look
    # like it controls things it does not.
    RECORDED_SETTINGS = (
        "cellSize", "agentHeight", "agentRadius", "agentMinRadius",
        "agentMaxClimb", "agentMaxSlope", "agentMinIslandRadius",
        "excludeRigidBodies", "useGpu",
    )

    def _find_volume(self) -> Optional[Usd.Prim]:
        """The NavMeshVolume already on stage, without creating one."""
        if not self.stage:
            return None
        for prim in self.stage.TraverseAll():
            if prim.GetTypeName() == "NavMeshVolume":
                return prim
        return None

    def navmesh_volume_box(self) -> Optional[Tuple[Tuple[float, float, float],
                                                   Tuple[float, float, float]]]:
        """The volume's world box as ((min), (max)), or None.

        This is the box the baker actually saw -- padding included -- not the
        bounds of the assignment ensure_navmesh_volume derived it from. A run
        reproducing this bake has to apply it verbatim; handing the assignment
        bounds back to ensure_navmesh_volume would pad an already padded box and
        bake a wider mesh than the one that was designed.
        """
        volume = self._find_volume()
        if volume is None:
            return None
        translate = volume.GetAttribute("xformOp:translate")
        scale = volume.GetAttribute("xformOp:scale")
        if not (translate and translate.IsValid() and scale and scale.IsValid()):
            return None
        centre, size = translate.Get(), scale.Get()
        if centre is None or size is None:
            return None
        # scale is the volume's FULL size: the command authors an extent of
        # +/-0.5, so halving it is what turns a scale into a box.
        half = [float(v) * 0.5 for v in size]
        mid = [float(v) for v in centre]
        return (
            tuple(mid[i] - half[i] for i in range(3)),
            tuple(mid[i] + half[i] for i in range(3)),
        )

    def set_navmesh_volume_box(self, vmin, vmax) -> bool:
        """Put the volume exactly on this box, with no padding of our own."""
        volume = self.ensure_navmesh_volume()
        if volume is None:
            return False
        centre = Gf.Vec3d(*[(float(a) + float(b)) * 0.5 for a, b in zip(vmin, vmax)])
        scale = Gf.Vec3f(*[max(float(b) - float(a), 1e-3) for a, b in zip(vmin, vmax)])
        for name, value in (("xformOp:translate", centre), ("xformOp:scale", scale)):
            attr = volume.GetAttribute(name)
            if attr and attr.IsValid():
                attr.Set(value)
        return True

    def assign_paths(self, paths: List[str]) -> int:
        """Re-establish an assignment from recorded prim paths.

        The resolution `get_selected_prim` does, driven by paths instead of the
        live selection, so a run can restrict its bake to what an authoring
        session assigned. Returns how many meshes resolved -- compare it against
        what the scenario recorded before trusting the bake.
        """
        if not self.stage:
            self.stage = omni.usd.get_context().get_stage()

        prims, missing = [], []
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                prims.append(prim)
            else:
                missing.append(path)
        if missing:
            print(
                f"[NavMeshAdapter] Warning: {len(missing)} assigned prim path(s) are "
                f"not on this stage ({', '.join(missing[:3])}"
                f"{' ...' if len(missing) > 3 else ''}); they cannot be baked."
            )

        self.input_prim = prims
        self.input_meshes = usd_utils.find_stage_meshes(self.stage, prims) if prims else []
        return len(self.input_meshes)

    def describe_bake(self) -> Dict[str, Any]:
        """What this bake was, in the form a scenario file records.

        Read back after the bake rather than taken from the caller's arguments:
        the defaults `_configure_bake` merges in are as much a part of the bake
        as the overrides, and a scenario that records only the overrides cannot
        reproduce it once those defaults change.
        """
        roots = self.input_prim
        if roots is None:
            roots = []
        elif isinstance(roots, Usd.Prim):
            roots = [roots]

        box = self.navmesh_volume_box()
        return {
            "assigned_prims": [p.GetPath().pathString for p in roots if p and p.IsValid()],
            "assigned_mesh_count": len(self.input_meshes),
            "bake_settings": {k: self.settings.get(k)
                              for k in self.RECORDED_SETTINGS if k in self.settings},
            "volume_min": box[0] if box else None,
            "volume_max": box[1] if box else None,
        }

    def _keep_set(self) -> Tuple[set, set]:
        """Paths to keep visible during a restricted bake.

        Returns ``(mesh_paths, prototype_paths)``. The second set exists because
        visibility cannot be authored on an instance proxy -- only on the prim
        inside the prototype, which every instance shares. An assigned mesh that
        lives under an instance is therefore kept by its prototype path, which
        is what a TraverseAll() walk will actually encounter.
        """
        mesh_paths, prototype_paths = set(), set()
        for prim in self.input_meshes:
            if not prim or not prim.IsValid():
                continue
            mesh_paths.add(prim.GetPath().pathString)
            if not prim.IsInstanceProxy():
                continue
            in_prototype = prim.GetPrimInPrototype()
            if not (in_prototype and in_prototype.IsValid()):
                continue
            prototype_paths.add(in_prototype.GetPath().pathString)
            # An instance proxy is read-only, so its visibility is really the
            # visibility of the prim the prototype composes from -- and *that*
            # prim is on the stage proper, where TraverseAll will find and hide
            # it. Hiding it takes every instance down with it and the bake comes
            # back empty. The prim stack is the only way back to those source
            # paths; prototype paths themselves are never traversed.
            for spec in in_prototype.GetPrimStack():
                prototype_paths.add(spec.path.pathString)
        return mesh_paths, prototype_paths

    def _hide_unassigned(self) -> List[Usd.Prim]:
        """Hide every visible mesh outside the assignment. Returns what to restore.

        omni.anim.navigation.core has no "bake only these meshes" input: it
        voxelises whatever is visible inside the NavMeshVolume. Assigning a few
        meshes only resizes that volume, so the ground they sit on -- and
        anything else the volume happens to span -- bakes as walkable too.
        Hiding the rest for the duration of the bake is how behavior_agent.py
        already restricts its own bake, and it is the only lever the native
        baker exposes.
        """
        mesh_paths, prototype_paths = self._keep_set()
        if not mesh_paths:
            return []

        shared = []
        hidden = []
        # TraverseAll(), not Traverse(): the latter skips instancing prototypes,
        # and instanced vegetation is exactly the geometry that pollutes a bake.
        # Note this yields prototype prims, never instance proxies -- which is
        # why _keep_set() translates assigned proxies into prototype paths.
        for prim in self.stage.TraverseAll():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = prim.GetPath().pathString
            if path in mesh_paths:
                continue
            if path in prototype_paths:
                # Shared prototype: some instances are assigned and some are
                # not, and one visibility opinion covers them all. Keeping it
                # bakes the unassigned instances too; say so rather than let it
                # look like the restriction silently failed.
                shared.append(path)
                continue
            if any(path.startswith(prefix) for prefix in self._VISUALIZATION_PREFIXES):
                imageable = UsdGeom.Imageable(prim)
                if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                    imageable.MakeInvisible()
                    hidden.append(prim)
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            imageable.MakeInvisible()
            hidden.append(prim)

        if shared:
            print(
                "[NavMeshAdapter] Warning: assigned mesh(es) share an instancing "
                f"prototype with unassigned ones ({', '.join(sorted(shared)[:3])}"
                f"{' ...' if len(shared) > 3 else ''}). Visibility is per-prototype, "
                "so those unassigned instances will bake too. Uninstance them to "
                "restrict the bake exactly."
            )
        return hidden

    @staticmethod
    def _restore(hidden: List[Usd.Prim]):
        for prim in hidden:
            try:
                UsdGeom.Imageable(prim).MakeVisible()
            except Exception:
                pass

    # A mesh is treated as inside-out when essentially none of its area faces
    # up and a real share of it faces down. Both halves matter: a vertical wall
    # also has no up-facing area, but almost no down-facing area either, and
    # flipping it would be meaningless. Raised walkways in brownstone measure
    # 20-33% up-facing, so they stay well clear of this.
    _INVERTED_UP_FRACTION = 0.02
    _INVERTED_DOWN_FRACTION = 0.30

    @staticmethod
    def _facing_areas(prim: Usd.Prim) -> Tuple[float, float, float]:
        """(up, down, total) triangle area of `prim`, in world space.

        Recast decides walkability from the face normal, so this is the number
        that says whether a mesh can become navmesh at all -- independent of
        agent radius or sampling, which only trim a surface that already faces
        the right way.
        """
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            return 0.0, 0.0, 0.0

        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = np.array([xform.Transform(Gf.Vec3d(*p)) for p in points], dtype=float)
        counts = np.asarray(counts, dtype=int)
        indices = np.asarray(indices, dtype=int)

        up = down = total = 0.0
        offset = 0
        for count in counts:
            face = indices[offset:offset + count]
            offset += count
            for k in range(1, count - 1):
                a, b, c = world[face[0]], world[face[k]], world[face[k + 1]]
                normal = np.cross(b - a, c - a)
                area = float(np.linalg.norm(normal)) / 2.0
                total += area
                if normal[2] > 0:
                    up += area
                elif normal[2] < 0:
                    down += area
        return up, down, total

    def _inverted_assigned(self) -> List[Usd.Prim]:
        """Assigned meshes whose faces all point the wrong way."""
        inverted = []
        for prim in self.input_meshes:
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                continue
            try:
                up, down, total = self._facing_areas(prim)
            except Exception:
                continue
            if total <= 0.0:
                continue
            if (up / total) < self._INVERTED_UP_FRACTION and (down / total) > self._INVERTED_DOWN_FRACTION:
                inverted.append(prim)
        return inverted

    def _flip_inverted(self) -> List[Usd.Prim]:
        """Temporarily reverse inside-out assigned meshes. Returns what to undo.

        An inside-out mesh reads as a ceiling, so the bake silently returns
        nothing for it and no agent radius or sampling distance can recover it
        -- brownstone ships one such footpath, a mirrored duplicate authored
        without reversing its winding.

        The reversal is written to the *session* layer, so it never reaches the
        asset on disk, and it is undone in the caller's finally. Note the baker
        ignores UsdGeom's `orientation` attribute: setting leftHanded still
        bakes nothing, so the indices themselves have to be reversed.
        """
        inverted = self._inverted_assigned()
        if not inverted:
            return []

        flipped, blocked = [], []
        session = self.stage.GetSessionLayer()
        for prim in inverted:
            # Visibility can be authored on a prototype, but geometry cannot be
            # authored on an instance proxy at all.
            if prim.IsInstanceProxy():
                blocked.append(prim.GetPath().pathString)
                continue
            try:
                mesh = UsdGeom.Mesh(prim)
                counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
                indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
                reversed_indices, offset = [], 0
                for count in counts:
                    reversed_indices.extend(reversed(indices[offset:offset + count]))
                    offset += count
                with Usd.EditContext(self.stage, session):
                    mesh.GetFaceVertexIndicesAttr().Set(reversed_indices)
                flipped.append(prim)
            except Exception as exc:
                blocked.append(f"{prim.GetPath().pathString} ({exc})")

        if flipped:
            names = ", ".join(p.GetPath().name for p in flipped[:3])
            print(
                f"[NavMeshAdapter] {len(flipped)} assigned mesh(es) are wound "
                f"inside-out ({names}{' ...' if len(flipped) > 3 else ''}); every face "
                "points down, so the baker reads them as ceilings. Baking a "
                "reversed copy for this bake only -- the asset on disk is not "
                "modified. Fix the winding in the source asset to make this stick."
            )
        if blocked:
            print(
                f"[NavMeshAdapter] Warning: {len(blocked)} inside-out assigned "
                f"mesh(es) could not be corrected ({', '.join(blocked[:3])}"
                f"{' ...' if len(blocked) > 3 else ''}); they will produce no navmesh. "
                "Geometry cannot be overridden on an instance proxy -- uninstance "
                "them or fix the winding in the source asset."
            )
        return flipped

    def _restore_winding(self, flipped: List[Usd.Prim]):
        """Drop the session-layer reversal authored by _flip_inverted."""
        if not flipped:
            return
        session = self.stage.GetSessionLayer()
        for prim in flipped:
            try:
                with Usd.EditContext(self.stage, session):
                    UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Clear()
            except Exception:
                pass
            # Clear() drops the value but leaves an empty property spec behind.
            # It carries no opinion, so composition is already correct -- but it
            # would pile up in the session layer one bake after another, so take
            # the spec out too.
            try:
                prim_spec = session.GetPrimAtPath(prim.GetPath())
                if prim_spec is not None and "faceVertexIndices" in prim_spec.properties:
                    prim_spec.RemoveProperty(prim_spec.properties["faceVertexIndices"])
            except Exception:
                pass

    @staticmethod
    def _warn_unsupported(settings: Optional[Dict[str, Any]]):
        """Name any Recast-only keys the caller passed that cannot be honoured.

        Only keys explicitly passed are reported -- the defaults carry all eight
        for ov_navmesh compatibility, and warning about those would be noise.
        """
        if not settings:
            return
        ignored = sorted(UNSUPPORTED_SETTINGS.intersection(settings))
        if ignored:
            print(
                f"[NavMeshAdapter] Ignoring {', '.join(ignored)}: these are Recast "
                "internals that omni.anim.navigation.core does not expose. Supported "
                "keys are cellSize, agentHeight, agentRadius, agentMinRadius, "
                "agentMaxClimb, agentMaxSlope, agentMinIslandRadius, excludeRigidBodies, useGpu."
            )

    @staticmethod
    def _island_radius_cm(settings: Dict[str, Any], to_cm) -> float:
        """Smallest island to keep, in centimetres.

        Prefers the native parameter. ``regionMinSize`` is ov_navmesh's name for
        a different quantity -- a linear voxel count that Recast squares into an
        area (regionMinSize**2 * cellSize**2) -- so when only that is given it is
        converted by area-equivalence, r = sqrt(area / pi), instead of the
        arbitrary x10 this used to apply.
        """
        explicit = settings.get("agentMinIslandRadius")
        if explicit is not None and "regionMinSize" not in settings:
            return to_cm(explicit)
        if "regionMinSize" in settings:
            region = float(settings["regionMinSize"])
            cell = float(settings.get("cellSize", 0.3))
            side = region * cell                      # metres
            return to_cm(side / math.sqrt(math.pi))   # equal-area radius
        return to_cm(explicit if explicit is not None else 2.0)

    def build_navmesh(
        self,
        settings: Optional[Dict[str, Any]] = None,
        restrict_to_assigned: bool = True,
        fix_inverted: bool = True,
        settle_frames: int = 6,
    ) -> bool:
        """Configure parameters, suppress debug geometry, and trigger synchronous baking.

        Translates user-facing settings (given in meters, matching original ov_navmesh)
        into centimetres as required by omni.anim.navigation.core's carb settings.

        Args:
            settings: Recast-style overrides, in metres.
            restrict_to_assigned: Bake only the meshes passed to
                ``get_selected_prim``/``load_mesh``. The native baker has no
                such input, so this is implemented by hiding everything else
                for the duration of the bake. Pass False to bake the whole
                volume, which is what this did before and what the runtime
                bake in behavior_agent.py wants.
            fix_inverted: Bake a reversed copy of any assigned mesh that is
                wound inside-out, which the baker would otherwise read as a
                ceiling and skip in silence. Session-layer only; the asset is
                never modified. Pass False to bake exactly what the scene says.
            settle_frames: Frames to pump before baking, so the hide reaches the
                baker. Six is enough for an authoring session, where the stage is
                already drawn; a run bakes during start-up with a thousand-odd
                meshes changing visibility at once, and baking too early there
                bakes the *old* visibility -- the full scene -- which is exactly
                the trap behavior_agent.bake_navmesh documents at 250 frames.
        """
        if not self._acquire():
            return False
        self._configure_bake(settings)
        hidden = self._begin_restriction(restrict_to_assigned)
        flipped = self._flip_inverted() if fix_inverted else []
        if hidden or flipped:
            self._pump(settle_frames)
        try:
            self.inav.start_navmesh_baking_and_wait()
        finally:
            self._restore_winding(flipped)
            if hidden:
                self._restore(hidden)
            if hidden or flipped:
                self._pump(max(2, settle_frames // 4))
        return self._conclude_bake()

    async def build_navmesh_async(
        self,
        settings: Optional[Dict[str, Any]] = None,
        restrict_to_assigned: bool = True,
        fix_inverted: bool = True,
    ) -> bool:
        """The same bake, awaiting frames instead of pumping them.

        Anything running on Kit's asyncio loop must use this. ``app.update()``
        drives that loop (omni.kit.async_engine calls ``loop.run_once()`` on
        each update event), so pumping from inside a coroutine re-enters the
        loop while that coroutine is still on it -- every other pending task
        then dies with "Cannot enter into task ... while another task is being
        executed", and the loop itself ends up raising IndexError out of an
        empty ready-deque.

        The bake call is left blocking: it joins a worker rather than pumping,
        so it does not re-enter the loop, and awaiting it instead would mean
        polling is_navmesh_baking() across frames with a start-up race to get
        wrong for no gain.
        """
        import omni.kit.app

        if not self._acquire():
            return False
        self._configure_bake(settings)
        hidden = self._begin_restriction(restrict_to_assigned)
        flipped = self._flip_inverted() if fix_inverted else []

        async def _settle(frames: int = 6):
            for _ in range(frames):
                await omni.kit.app.get_app().next_update_async()

        try:
            if hidden or flipped:
                await _settle()
            self.inav.start_navmesh_baking_and_wait()
        finally:
            self._restore_winding(flipped)
            if hidden:
                self._restore(hidden)
            if hidden or flipped:
                await _settle(2)
        return self._conclude_bake()

    def _acquire(self) -> bool:
        """Make sure the navigation interface is in hand."""
        if not self.inav:
            if nav:
                self.inav = nav.acquire_interface()
            else:
                print("[NavMeshAdapter] Error: navigation core interface not available.")
                return False
        return True

    def _configure_bake(self, settings: Optional[Dict[str, Any]]) -> None:
        """Push settings to carb and make sure the volume exists."""
        merged_settings = dict(DEFAULT_RECAST_SETTINGS)
        if settings:
            merged_settings.update(settings)
        self.settings = merged_settings
        self._warn_unsupported(settings)

        self.suppress_debug_geometry()

        carb_settings = carb.settings.get_settings()
        prefix = "/exts/omni.anim.navigation.core/navMesh/config"

        # Scale translation: inputs in meters (< 15.0) are converted to centimetres
        def to_cm(val: float) -> float:
            v = float(val)
            return v * 100.0 if v < 15.0 else v

        cell_size_cm = to_cm(merged_settings["cellSize"])
        agent_height_cm = to_cm(merged_settings["agentHeight"])
        agent_radius_cm = to_cm(merged_settings["agentRadius"])
        agent_climb_cm = to_cm(merged_settings["agentMaxClimb"])
        agent_slope_deg = float(merged_settings["agentMaxSlope"])

        # The native baker takes a radius range. Its shipped defaults are
        # agentMinRadius 20 / agentMaxRadius 50 cm, so falling back to 0.4 of the
        # max reproduces the ratio NVIDIA ships rather than inventing one.
        min_radius = merged_settings.get("agentMinRadius")
        min_radius_cm = agent_radius_cm * 0.4 if min_radius is None else to_cm(min_radius)

        island_radius_cm = self._island_radius_cm(merged_settings, to_cm)

        carb_settings.set(f"{prefix}/agentSamplingDistance", cell_size_cm)
        carb_settings.set(f"{prefix}/agentMinHeight", agent_height_cm)
        carb_settings.set(f"{prefix}/agentMaxRadius", agent_radius_cm)
        carb_settings.set(f"{prefix}/agentMinRadius", min_radius_cm)
        carb_settings.set(f"{prefix}/agentMaxStepHeight", agent_climb_cm)
        carb_settings.set(f"{prefix}/agentMaxFloorSlope", agent_slope_deg)
        carb_settings.set(f"{prefix}/agentMinIslandRadius", island_radius_cm)
        carb_settings.set(f"{prefix}/excludeRigidBodies",
                          bool(merged_settings.get("excludeRigidBodies", True)))
        # useGpu lives outside the config subtree.
        carb_settings.set("/exts/omni.anim.navigation.core/navMesh/useGpu",
                          bool(merged_settings.get("useGpu", True)))

        # Ensure volume exists if not already present
        self.ensure_navmesh_volume()

        print(f"[NavMeshAdapter] Baking NavMesh (sampling={cell_size_cm:.1f}cm, height={agent_height_cm:.1f}cm, radius={agent_radius_cm:.1f}cm)...")

    def _begin_restriction(self, restrict_to_assigned: bool) -> List[Usd.Prim]:
        """Hide everything outside the assignment; returns what was hidden.

        The caller has to let the hide settle (pump or await) before baking,
        and must restore the returned prims afterwards.
        """
        if not restrict_to_assigned:
            return []
        hidden = self._hide_unassigned()
        if hidden:
            print(f"[NavMeshAdapter] Restricting bake to {len(self._assigned_paths())} "
                  f"assigned mesh(es); {len(hidden)} other mesh(es) hidden.")
        return hidden

    def _conclude_bake(self) -> bool:
        """Pick up whatever the bake produced and report it."""
        self._navmesh = self.inav.get_navmesh()
        self.built = (self._navmesh is not None)

        if self.built:
            print("[NavMeshAdapter] NavMesh baked successfully.")
        else:
            print("[NavMeshAdapter] NavMesh baking returned no mesh.")

        return self.built

    def get_navmesh_polygons(self, area: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract walkable navmesh triangles.

        Returns:
            vertices: (N, 3) float32 numpy array
            faces: (M, 3) int32 numpy array
        """
        if not self._navmesh:
            if self.inav:
                self._navmesh = self.inav.get_navmesh()
                self.built = self._navmesh is not None
            if not self._navmesh:
                return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        draw_verts = self._navmesh.get_draw_triangles(area)
        if not draw_verts or len(draw_verts) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        flat_pts = []
        for p in draw_verts:
            if hasattr(p, "x"):
                flat_pts.append((p.x, p.y, p.z))
            else:
                flat_pts.append(p)

        verts = np.asarray(flat_pts, dtype=np.float32).reshape(-1, 3)
        faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)

        self.navmesh_v = verts
        self.navmesh_t = faces
        return self.navmesh_v, self.navmesh_t

    def get_navmesh_triangles(self) -> np.ndarray:
        """Return triangle indices for the navmesh surface."""
        _, t = self.get_navmesh_polygons()
        return t

    def get_navmesh_contours(self) -> Tuple[np.ndarray, List[List[int]]]:
        """Extract paired boundary lines representing obstacle and perimeter edges."""
        if not self._navmesh:
            if self.inav:
                self._navmesh = self.inav.get_navmesh()
                self.built = self._navmesh is not None
            if not self._navmesh:
                return np.empty((0, 3), dtype=np.float32), []

        draw_lines = self._navmesh.get_draw_lines()
        if not draw_lines or len(draw_lines) == 0:
            return np.empty((0, 3), dtype=np.float32), []

        flat_pts = []
        for p in draw_lines:
            if hasattr(p, "x"):
                flat_pts.append((p.x, p.y, p.z))
            else:
                flat_pts.append(p)

        verts = np.asarray(flat_pts, dtype=np.float32).reshape(-1, 3)
        edges = [[i, i + 1] for i in range(0, len(verts) - 1, 2)]

        self.contour_verts = verts
        self.contour_edges = edges
        return self.contour_verts, self.contour_edges

    def get_navmesh_raw_contours(self) -> Tuple[np.ndarray, List[List[int]]]:
        """Alias for get_navmesh_contours for API compatibility."""
        return self.get_navmesh_contours()

    def make_outline(
        self,
        prim_prefix: str = "/World/Outline/WallOutline",
        color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        width: float = 0.05,
        batched: bool = False,
    ) -> List[str]:
        """Draw BasisCurves on stage along the navmesh boundary contours."""
        verts, edges = self.get_navmesh_contours()
        if len(verts) == 0 or not edges:
            print("[NavMeshAdapter] No contour edges found to outline.")
            return []

        created = []
        self.wall_outline = []

        if batched:
            segments = [(tuple(verts[i]), tuple(verts[j])) for i, j in edges]
            curve_path = f"{prim_prefix}s"
            usd_utils.create_batched_curves(segments, prim_path=curve_path, color=color, width=width, stage=self.stage)
            created.append(curve_path)
        else:
            for idx, (i, j) in enumerate(edges):
                A = tuple(verts[i])
                B = tuple(verts[j])
                self.wall_outline.append([A, B])
                curve_path = f"{prim_prefix}{idx}"
                usd_utils.create_curve([A, B], prim_path=curve_path, width=width, color=color, stage=self.stage)
                created.append(curve_path)

        return created

    def make_walls(
        self,
        vertices: Optional[np.ndarray] = None,
        edges: Optional[List[List[int]]] = None,
        height: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrude vertical walls along contour edges."""
        if vertices is None or edges is None:
            vertices, edges = self.get_navmesh_contours()

        if len(vertices) == 0 or not edges:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        verts_np = np.asarray(vertices, dtype=np.float64)
        extruded_verts = np.copy(verts_np)
        # Add height to up-axis (Z-up vs Y-up)
        if self.z_up:
            extruded_verts[:, 2] += height
        else:
            extruded_verts[:, 1] += height

        side_triangles = []
        for i, j in edges:
            A = verts_np[i]
            B = verts_np[j]
            A_prime = extruded_verts[i]
            B_prime = extruded_verts[j]
            side_triangles.extend([[A, B, B_prime], [A, B_prime, A_prime]])

        self.side_triangles = side_triangles

        # Weld unique vertices
        flat_verts = [v for tri in side_triangles for v in tri]
        faces = [[i, i + 1, i + 2] for i in range(0, len(flat_verts), 3)]
        faces = np.asarray(faces, dtype=np.int32)

        unique_v, inverse_idx = np.unique(flat_verts, axis=0, return_inverse=True)
        welded_faces = inverse_idx[faces.flatten()].reshape(faces.shape)

        self.wall_v = np.asarray(unique_v, dtype=np.float32)
        self.wall_t = np.asarray(welded_faces, dtype=np.int32)
        return self.wall_v, self.wall_t

    def get_random_points(self, num_points: int = 10) -> Optional[np.ndarray]:
        """Query randomly sampled navigable points from the navmesh."""
        if not self._navmesh:
            if self.inav:
                self._navmesh = self.inav.get_navmesh()
                self.built = self._navmesh is not None
            if not self._navmesh:
                print("[NavMeshAdapter] Navmesh not built.")
                return None

        points = []
        for _ in range(num_points):
            try:
                pt = self._navmesh.query_random_point()
                if pt is not None:
                    if hasattr(pt, "x"):
                        points.append((pt.x, pt.y, pt.z))
                    else:
                        points.append(pt)
            except Exception as e:
                print(f"[NavMeshAdapter] Error querying random point: {e}")
                break

        if not points:
            return None

        self.random_points = np.asarray(points, dtype=np.float32)
        return self.random_points

    def find_paths(
        self,
        starts: Union[List[Any], np.ndarray],
        ends: Union[List[Any], np.ndarray],
    ) -> np.ndarray:
        """Compute shortest path between start and end coordinates.

        Returns waypoints array of shape (N, 3).
        """
        if not self._navmesh:
            if self.inav:
                self._navmesh = self.inav.get_navmesh()
                self.built = self._navmesh is not None
            if not self._navmesh:
                print("[NavMeshAdapter] Navmesh not built.")
                return np.empty((0, 3), dtype=np.float32)

        # Unpack start and end positions
        s_raw = starts[0] if isinstance(starts, (list, np.ndarray, tuple)) and len(starts) > 0 and isinstance(starts[0], (list, np.ndarray, tuple)) else starts
        e_raw = ends[0] if isinstance(ends, (list, np.ndarray, tuple)) and len(ends) > 0 and isinstance(ends[0], (list, np.ndarray, tuple)) else ends

        s_tuple = (float(s_raw[0]), float(s_raw[1]), float(s_raw[2]))
        e_tuple = (float(e_raw[0]), float(e_raw[1]), float(e_raw[2]))

        try:
            path_obj = self._navmesh.query_shortest_path(start_pos=s_tuple, end_pos=e_tuple)
            if path_obj is None:
                return np.empty((0, 3), dtype=np.float32)

            pts = path_obj.get_points()
            if not pts:
                return np.empty((0, 3), dtype=np.float32)

            path_pnts = []
            for p in pts:
                if hasattr(p, "x"):
                    path_pnts.append((p.x, p.y, p.z))
                else:
                    path_pnts.append(p)

            return np.asarray(path_pnts, dtype=np.float32)
        except Exception as e:
            print(f"[NavMeshAdapter] Path query failed: {e}")
            return np.empty((0, 3), dtype=np.float32)

    # --- High-level visualization helpers ---

    def visualize_navmesh(
        self,
        prim_path: str = "/World/navmeshmesh",
        color: Tuple[float, float, float] = (0.0512, 0.7749, 0.9458),
        opacity: float = 0.89,
    ) -> Optional[str]:
        """Create or update a translucent preview surface mesh of the navigable surface."""
        verts, faces = self.get_navmesh_polygons()
        if len(verts) == 0:
            print("[NavMeshAdapter] No navmesh geometry to visualize.")
            return None

        usd_utils.create_mesh(
            prim_path=prim_path,
            points=verts,
            indices=faces,
            color=color,
            opacity=opacity,
            use_prevsrf=True,
            stage=self.stage,
        )
        return prim_path

    def visualize_random_points(
        self,
        num_points: int = 10,
        prim_path: str = "/World/Points",
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        point_width: float = 0.2,
    ) -> Optional[str]:
        """Query random points and author red markers at /World/Points."""
        pnts = self.get_random_points(num_points)
        if pnts is None or len(pnts) == 0:
            return None
        usd_utils.create_geompoints(pnts, prim_path=prim_path, color=color, point_width=point_width, stage=self.stage)
        return prim_path

    def visualize_path(
        self,
        start,
        end,
        prim_path: str = "/World/Path",
        color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        width: float = 0.15,
    ) -> Optional[str]:
        """Query shortest path and author a green spline ribbon at /World/Path."""
        path_pnts = self.find_paths([start], [end])
        if len(path_pnts) < 2:
            print("[NavMeshAdapter] No valid path found between start and end.")
            return None
        usd_utils.create_curve(path_pnts, prim_path=prim_path, color=color, width=width, stage=self.stage)
        return prim_path

    # --- Reset & Rebake Management ---

    def clear_visualizations(
        self,
        mesh_path: str = "/World/navmeshmesh",
        outline_path: str = "/World/Outline",
        points_path: str = "/World/Points",
        path_path: str = "/World/Path",
    ) -> List[str]:
        """Remove all authored visual geometry and material prims from the USD stage."""
        if not self.stage:
            self.stage = omni.usd.get_context().get_stage()

        removed = []
        for path_str in [mesh_path, outline_path, points_path, path_path]:
            if usd_utils.remove_prim(path_str, stage=self.stage):
                removed.append(path_str)

        # Also remove preview surface material if authored
        clean_name = mesh_path.replace("/", "_").strip("_")
        mtl_path = f"/World/Looks/PreviewSurface_{clean_name}"
        if usd_utils.remove_prim(mtl_path, stage=self.stage):
            removed.append(mtl_path)

        if removed:
            print(f"[NavMeshAdapter] Cleared stage visualizations: {', '.join(removed)}")
        return removed

    def reset_navmesh(self, clear_stage: bool = True, clear_cache: bool = True) -> bool:
        """Reset the baked navmesh state and optionally clear all USD visual geometry.

        Args:
            clear_stage: If True, deletes /World/navmeshmesh, /World/Outline, /World/Points, /World/Path.
            clear_cache: If True, clears navigation cache directory in Isaac Sim.

        Returns:
            True if reset succeeded.
        """
        if clear_stage:
            self.clear_visualizations()

        if clear_cache and self.inav:
            try:
                self.inav.clear_cache_dir()
            except Exception as e:
                print(f"[NavMeshAdapter] Note on clearing cache dir: {e}")

        # Invalidate internal cache
        self._navmesh = None
        self.built = False
        self.navmesh_v = np.empty((0, 3), dtype=np.float32)
        self.navmesh_t = np.empty((0, 3), dtype=np.int32)
        self.contour_verts = np.empty((0, 3), dtype=np.float32)
        self.contour_edges = []
        self.wall_outline = []
        self.random_points = None

        print("[NavMeshAdapter] Navmesh state reset successfully.")
        return True

    def rebake_navmesh(
        self,
        settings: Optional[Dict[str, Any]] = None,
        visualize: bool = True,
        color: Tuple[float, float, float] = (0.0512, 0.7749, 0.9458),
        opacity: float = 0.89,
        draw_outlines: bool = True,
    ) -> bool:
        """One-call method to reset previous bake, apply new settings, and re-visualize."""
        self.reset_navmesh(clear_stage=True)
        success = self.build_navmesh(settings=settings)
        if success and visualize:
            self.visualize_navmesh(color=color, opacity=opacity)
            if draw_outlines:
                self.make_outline(batched=True)
        return success


# Alias for explicit naming
NativeNavmeshInterface = NavmeshInterface


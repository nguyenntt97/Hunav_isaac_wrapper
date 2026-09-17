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
DEFAULT_RECAST_SETTINGS: Dict[str, float] = {
    "cellSize": 0.3,         # meters
    "cellHeight": 0.2,       # meters
    "agentHeight": 2.0,      # meters
    "agentRadius": 0.6,      # meters
    "agentMaxClimb": 0.9,    # meters
    "agentMaxSlope": 45.0,   # degrees
    "regionMinSize": 8.0,
    "regionMergeSize": 20.0,
    "edgeMaxLen": 12.0,
    "edgeMaxError": 1.3,
    "vertsPerPoly": 6.0,
    "detailSampleDist": 6.0,
    "detailSampleMaxError": 1.0,
    "partitionType": 0.0,
}

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
            centre = Gf.Vec3d(
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                (min_z + max_z) * 0.5,
            )
            scale = Gf.Vec3f(
                max((max_x - min_x) + 4.0, 2.0),
                max((max_y - min_y) + 4.0, 2.0),
                max((max_z - min_z) + 4.0, 6.0),
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

        if len(points) > 0:
            bounds = usd_utils.compute_bounds([prim], stage=self.stage)
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
        points, faces = usd_utils.get_all_stage_mesh(self.stage, self.input_prim)
        self.input_vert = points
        self.input_tri = faces

        if len(self.input_vert) == 0:
            print("[NavMeshAdapter] No mesh geometry found in selected prims.")
            return False

        bounds = usd_utils.compute_bounds(self.input_prim, stage=self.stage)
        if bounds:
            self.ensure_navmesh_volume(bounds)
        else:
            self.ensure_navmesh_volume()

        return True

    def build_navmesh(self, settings: Optional[Dict[str, Any]] = None) -> bool:
        """Configure parameters, suppress debug geometry, and trigger synchronous baking.

        Translates user-facing settings (given in meters, matching original ov_navmesh)
        into centimetres as required by omni.anim.navigation.core's carb settings.
        """
        if not self.inav:
            if nav:
                self.inav = nav.acquire_interface()
            else:
                print("[NavMeshAdapter] Error: navigation core interface not available.")
                return False

        merged_settings = dict(DEFAULT_RECAST_SETTINGS)
        if settings:
            merged_settings.update(settings)

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

        carb_settings.set(f"{prefix}/agentSamplingDistance", cell_size_cm)
        carb_settings.set(f"{prefix}/agentMinHeight", agent_height_cm)
        carb_settings.set(f"{prefix}/agentMaxRadius", agent_radius_cm)
        carb_settings.set(f"{prefix}/agentMinRadius", agent_radius_cm * 0.4)
        carb_settings.set(f"{prefix}/agentMaxStepHeight", agent_climb_cm)
        carb_settings.set(f"{prefix}/agentMaxFloorSlope", agent_slope_deg)
        carb_settings.set(f"{prefix}/agentMinIslandRadius", float(merged_settings.get("regionMinSize", 8.0)) * 10.0)
        carb_settings.set(f"{prefix}/excludeRigidBodies", True)

        # Ensure volume exists if not already present
        self.ensure_navmesh_volume()

        # Trigger bake
        print(f"[NavMeshAdapter] Baking NavMesh (sampling={cell_size_cm:.1f}cm, height={agent_height_cm:.1f}cm, radius={agent_radius_cm:.1f}cm)...")
        success = self.inav.start_navmesh_baking_and_wait()

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


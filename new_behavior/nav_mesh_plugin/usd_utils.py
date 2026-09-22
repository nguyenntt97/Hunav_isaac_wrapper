"""
usd_utils.py

USD geometry and material authoring utilities for the NavMesh plugin.
Provides robust mesh extraction, triangle conversion, bounding box computation,
translucent preview surface material binding, and BasisCurves / Points generation.
Compatible with Python 3.12 and USD 24+ in Isaac Sim 6.x.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
import omni.usd


def traverse_instanced_children(prim: Usd.Prim):
    """Yield all child prims beneath `prim`, including instance proxies."""
    if not prim or not prim.IsValid():
        return
    for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child
        for subchild in traverse_instanced_children(child):
            yield subchild


def convert_to_triangle_mesh(face_vertex_indices, face_vertex_counts):
    """Convert arbitrary polygon face vertex indices and counts into triangulated faces.

    Uses fan triangulation for polygons with > 3 vertices.
    """
    if not face_vertex_counts or not face_vertex_indices:
        return np.empty((0, 3), dtype=np.int32)

    faces = []
    start = 0
    for count in face_vertex_counts:
        end = start + count
        face = face_vertex_indices[start:end]
        if count == 3:
            faces.append(face)
        elif count > 3:
            # Fan triangulation: v0 connects to v(i) and v(i+1)
            v0 = face[0]
            for i in range(1, count - 1):
                faces.append([v0, face[i], face[i + 1]])
        start = end

    if not faces:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(faces, dtype=np.int32)


def meshconvert(prim: Usd.Prim):
    """Extract local vertices and faces from a UsdGeom.Mesh, triangulate, and transform to world coordinates."""
    if not prim or not prim.IsA(UsdGeom.Mesh):
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    mesh = UsdGeom.Mesh(prim)
    tris = mesh.GetFaceVertexIndicesAttr().Get()
    if not tris:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float64)
    tris_cnt = mesh.GetFaceVertexCountsAttr().Get()
    if not tris_cnt:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    points_attr = mesh.GetPointsAttr()
    local_points = points_attr.Get()
    if not local_points or len(local_points) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float64)

    points_np = np.array(local_points, dtype=np.float64)
    num_points = len(local_points)
    ones = np.ones((num_points, 1), dtype=np.float64)
    points_h = np.hstack((points_np, ones))

    # Compute world transform
    xform_cache = UsdGeom.XformCache()
    world_transform = xform_cache.GetLocalToWorldTransform(prim)
    matrix_np = np.array(world_transform, dtype=np.float64).reshape((4, 4))
    world_points = np.dot(points_h, matrix_np)[:, :3]

    tri_list = convert_to_triangle_mesh(tris, tris_cnt)
    return tri_list, world_points


def get_mesh(objs):
    """Collect world vertices and triangulated faces across an iterable of UsdGeom.Mesh prims."""
    points = []
    faces = []
    for obj in objs:
        if not obj or not obj.IsValid() or not obj.IsA(UsdGeom.Mesh):
            continue
        f, p = meshconvert(obj)
        if len(p) == 0 or len(f) == 0:
            continue
        f_offset = len(points)
        points.extend(p)
        faces.extend(f + f_offset)

    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)
    return np.array(points, dtype=np.float64), np.array(faces, dtype=np.int32)


def parent_and_children_as_mesh(parent_prim: Usd.Prim):
    """Retrieve all mesh geometry from parent_prim and any visible descendants."""
    if not parent_prim or not parent_prim.IsValid():
        return np.empty((0, 3)), np.empty((0, 3))
    if UsdGeom.Imageable(parent_prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        return np.empty((0, 3)), np.empty((0, 3))
    if parent_prim.IsA(UsdGeom.Mesh):
        return get_mesh([parent_prim])

    found_meshes = []
    for x in traverse_instanced_children(parent_prim):
        if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if x.IsA(UsdGeom.Mesh):
            found_meshes.append(x)

    return get_mesh(found_meshes)


def find_stage_meshes(stage: Usd.Stage, prims: list[Usd.Prim]) -> list[Usd.Prim]:
    """Resolve a selection to the visible UsdGeom.Mesh prims it contains.

    These are ov_navmesh's selection rules, kept deliberately identical
    (siborg/create/navmesh/usd_utils.py:get_all_stage_mesh):

      * every selected prim contributes, not just the first;
      * descend with ``Usd.TraverseInstanceProxies()``, so instanced meshes are
        found rather than silently skipped;
      * only ``UsdGeom.Mesh`` counts -- implicit gprims (Cube, Plane, Capsule)
        are ignored, which is also all the native baker will voxelise;
      * skip anything whose *computed* visibility is invisible, so hiding a
        parent excludes its whole subtree.

    Returned separately from the geometry because restricting a bake needs the
    prims themselves, not the triangle soup.
    """
    if not prims:
        prims = [stage.GetPseudoRoot()]

    found_meshes = []
    for prim in prims:
        if not prim or not prim.IsValid():
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if prim.IsA(UsdGeom.Mesh):
            found_meshes.append(prim)
            continue
        for x in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if x.IsA(UsdGeom.Mesh):
                found_meshes.append(x)

    return found_meshes


def get_all_stage_mesh(stage: Usd.Stage, prims: list[Usd.Prim]):
    """Collect all visible mesh prims underneath the given list of prims (or entire stage if empty)."""
    return get_mesh(find_stage_meshes(stage, prims))


def compute_bounds(prims: list[Usd.Prim], stage: Usd.Stage = None) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute the world-space bounding box ((min_x, min_y, min_z), (max_x, max_y, max_z)) for the given prims."""
    if not prims:
        return None
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    min_pt = [float("inf"), float("inf"), float("inf")]
    max_pt = [float("-inf"), float("-inf"), float("-inf")]
    found = False

    for prim in prims:
        if not prim or not prim.IsValid():
            continue
        boundable = UsdGeom.Boundable(prim)
        if boundable:
            bbox = bbox_cache.ComputeWorldBound(prim)
            aligned_range = bbox.ComputeAlignedBox()
            bmin = aligned_range.GetMin()
            bmax = aligned_range.GetMax()
            for i in range(3):
                min_pt[i] = min(min_pt[i], bmin[i])
                max_pt[i] = max(max_pt[i], bmax[i])
            found = True

    if not found:
        return None
    return tuple(min_pt), tuple(max_pt)


def create_mesh(
    prim_path: str,
    points,
    indices,
    color: tuple[float, float, float] | Gf.Vec3f = (0.05, 0.77, 0.95),
    opacity: float = 0.8,
    use_prevsrf: bool = True,
    stage: Usd.Stage = None,
) -> str:
    """Create or update a UsdGeom.Mesh on stage, bound to a translucent UsdPreviewSurface material.

    Args:
        prim_path: Stage path where the mesh should be defined (e.g. '/World/navmeshmesh').
        points: (N, 3) or (3N,) array-like of world coordinate vertices.
        indices: (M, 3) or (3M,) array-like of triangle vertex indices.
        color: RGB tuple or Gf.Vec3f for preview material diffuse color.
        opacity: Material opacity between 0.0 (transparent) and 1.0 (opaque).
        use_prevsrf: Whether to create and bind a UsdPreviewSurface material.
        stage: Target USD stage (defaults to current omni.usd context).

    Returns:
        The prim path string of the created mesh.
    """
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    time = Usd.TimeCode.Default()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # Reshape points to list of Vec3f or numpy array
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    indices_np = np.asarray(indices, dtype=np.int32).reshape(-1, 3)

    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points_np), time)
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices_np.flatten()), time)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(indices_np)), time)

    if isinstance(color, (tuple, list)):
        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    else:
        color_vec = color

    if use_prevsrf:
        clean_name = prim_path.replace("/", "_").strip("_")
        mtl_path = Sdf.Path(f"/World/Looks/PreviewSurface_{clean_name}")

        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color_vec)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material using MaterialBindingAPI
        binding_api = UsdShade.MaterialBindingAPI(mesh.GetPrim())
        binding_api.Bind(mtl)

    # Also author standard displayColor / displayOpacity primvars as fallback
    primvar_color = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    primvar_color.Set([color_vec])
    primvar_opacity = mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant)
    primvar_opacity.Set([float(opacity)])

    return prim_path


def create_curve(
    nodes,
    prim_path: str = "/World/Path",
    color: tuple[float, float, float] | Gf.Vec3f = (0.0, 1.0, 0.0),
    width: float | np.ndarray = 0.1,
    stage: Usd.Stage = None,
) -> str:
    """Create or update a linear UsdGeom.BasisCurves prim along the given waypoints."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    nodes_np = np.asarray(nodes, dtype=np.float32).reshape(-1, 3)
    if len(nodes_np) < 2:
        return prim_path

    prim = UsdGeom.BasisCurves.Define(stage, prim_path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(nodes_np))
    prim.CreateCurveVertexCountsAttr([len(nodes_np)])
    prim.CreateTypeAttr(UsdGeom.Tokens.linear)

    if isinstance(width, (int, float)):
        widths_val = Vt.FloatArray([float(width)] * len(nodes_np))
    else:
        w_np = np.asarray(width, dtype=np.float32)
        if len(w_np) == 1:
            widths_val = Vt.FloatArray([float(w_np[0])] * len(nodes_np))
        else:
            widths_val = Vt.FloatArray.FromNumpy(w_np)
    prim.CreateWidthsAttr(widths_val)

    if isinstance(color, (tuple, list)):
        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    else:
        color_vec = color

    color_primvar = prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    color_primvar.Set([color_vec])

    return prim_path


def create_batched_curves(
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    prim_path: str = "/World/Outline/WallOutlines",
    color: tuple[float, float, float] = (0.9, 0.9, 0.2),
    width: float = 0.05,
    stage: Usd.Stage = None,
) -> str:
    """Batch multiple 2-point line segments into a single UsdGeom.BasisCurves prim for optimal RTX performance."""
    if not segments:
        return prim_path
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    flat_pts = []
    for s0, s1 in segments:
        flat_pts.append(s0)
        flat_pts.append(s1)

    pts_np = np.asarray(flat_pts, dtype=np.float32).reshape(-1, 3)
    curve_counts = [2] * len(segments)

    prim = UsdGeom.BasisCurves.Define(stage, prim_path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts_np))
    prim.CreateCurveVertexCountsAttr(Vt.IntArray(curve_counts))
    prim.CreateTypeAttr(UsdGeom.Tokens.linear)

    widths_val = Vt.FloatArray([float(width)] * len(pts_np))
    prim.CreateWidthsAttr(widths_val)

    color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    color_primvar = prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    color_primvar.Set([color_vec])

    return prim_path


def create_geompoints(
    positions,
    prim_path: str = "/World/Points",
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    point_width: float = 0.15,
    stage: Usd.Stage = None,
) -> str:
    """Create or update UsdGeom.Points representing sample markers."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()

    pts_np = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    prim = UsdGeom.Points.Define(stage, prim_path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts_np))

    widths_val = Vt.FloatArray([float(point_width)] * len(pts_np))
    prim.CreateWidthsAttr(widths_val)

    color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    color_primvar = prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    color_primvar.Set([color_vec])

    return prim_path


def set_positions(agent_point_prim: UsdGeom.Points, positions):
    """Update point positions on an existing UsdGeom.Points prim."""
    pts_np = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    agent_point_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts_np))


def remove_prim(prim_path: str, stage: Usd.Stage = None) -> bool:
    """Remove a prim and all its descendants from the stage if it exists."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if not stage:
        return False
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if prim and prim.IsValid():
        return stage.RemovePrim(Sdf.Path(prim_path))
    return False


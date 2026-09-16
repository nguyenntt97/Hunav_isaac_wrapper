#!/usr/bin/env python3
"""
terrain.py

Experimental terrain handling for worlds whose ground is not flat.

The indoor worlds shipped with this wrapper (warehouse, hospital, office) have a
single floor plane at z=0, which suits HuNavSim: its social-force model is purely
2-D and reports no Z at all, so agents are written to the stage at z=0 every tick.

The brownstone park is different. It is a terraced landscape -- roughly a quarter
of the park core sits above z=0, up to 0.94 m -- so agents pinned to z=0 walk
inside the terraces wherever their path crosses one.

This module implements the "flat ground" half of the workaround: hide the raised
surfaces and drop a single flat collider in their place. The other half, letting
agents follow the terrain, lives in HuNavManager (see sample_ground_height).

Both are opt-in and off by default; see TeleopHuNavSim for the flags.
"""

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

from .behavior_agent import make_ground_collider_mesh

# Branches under a world's ParkDeOv geometry that make up the raised landscape.
# Matched as path substrings so the pass survives the mesh counts changing.
# "StepedSurface" is spelled that way in the source asset.
_TERRACE_BRANCHES = (
    "SteppedSurf_detailed",
    "SteppedLogo",
    "StepedSurface",
)

# Grass sits at grade in places and rides up onto the terraces elsewhere, so it
# is selected by height rather than wholesale. Both the "Grass" and "Grass2"
# groups behave this way.
_GRASS_BRANCHES = ("/Grass/", "/Grass2/")

# Never touched, whatever their height: the pond and its lighting. Flattening
# these would drop the water onto the plaza and delete the basin around it.
_KEEP_BRANCHES = ("/Water/", "/Water_Light/")

# Extra margin around the pond, in metres. Terrace pieces that sit mostly inside
# this footprint form the basin walls, so they stay too.
_WATER_MARGIN = 2.0

# How much of a terrace mesh must lie inside the padded pond footprint before it
# counts as basin detail rather than landscape. Plain bbox intersection is far
# too loose here: the terraces are large sweeping meshes whose bounding boxes
# enclose the ponds entirely, so intersection alone would spare almost all of
# them and leave the ground anything but flat.
_WATER_OVERLAP_FRACTION = 0.5

# A mesh counts as raised if its top sits above this, in metres.
_RAISED_EPS = 0.05

# Top face of the flat collider, in metres. Deliberately *below* z=0 rather than
# at it: the pathway, hardscape and sidewalk colliders all top out at exactly
# 0.00, and a coplanar collider is the same degenerate contact that PhysX
# resolves explosively (it is what ejected the robot at the world origin). This
# keeps the plane clear of them, so it only fills the holes the terraces leave.
_GROUND_TOP_Z = -0.10

# Worlds this pass knows how to flatten. Anything else is a no-op.
_SUPPORTED = ("brownstone",)

GROUND_PLANE_PATH = "/World/FlatGroundProxy"


def _is_raised(bbox_cache, prim):
    r = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return (not r.IsEmpty()) and r.GetMax()[2] > _RAISED_EPS


def _water_footprints(stage, bbox_cache):
    """XY boxes of the pond meshes, padded by _WATER_MARGIN."""
    boxes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not any(b in prim.GetPath().pathString for b in _KEEP_BRANCHES):
            continue
        r = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = r.GetMin(), r.GetMax()
        boxes.append((mn[0] - _WATER_MARGIN, mn[1] - _WATER_MARGIN,
                      mx[0] + _WATER_MARGIN, mx[1] + _WATER_MARGIN))
    return boxes


def _is_pond_detail(bbox_cache, prim, boxes):
    """
    True if most of this mesh's footprint lies inside a pond footprint.

    Uses the overlapping *area* rather than mere intersection so that a large
    terrace which happens to span a pond is still flattened, while the small
    pieces forming the basin are kept.
    """
    r = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return False
    mn, mx = r.GetMin(), r.GetMax()
    area = (mx[0] - mn[0]) * (mx[1] - mn[1])
    if area <= 0:
        return False
    for bx0, by0, bx1, by1 in boxes:
        ox = min(mx[0], bx1) - max(mn[0], bx0)
        oy = min(mx[1], by1) - max(mn[1], by0)
        if ox > 0 and oy > 0 and (ox * oy) / area >= _WATER_OVERLAP_FRACTION:
            return True
    return False


def _collect_terrace_prims(stage):
    """
    Return the mesh prims making up the raised landscape.

    The pond is excluded, along with anything overlapping it, so the water and
    the basin around it survive flattening.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    water = _water_footprints(stage, bbox_cache)
    found = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = prim.GetPath().pathString
        if any(b in path for b in _KEEP_BRANCHES):
            continue
        if any(b in path for b in _TERRACE_BRANCHES):
            candidate = True
        elif any(b in path for b in _GRASS_BRANCHES):
            candidate = _is_raised(bbox_cache, prim)
        else:
            candidate = False
        if candidate and not _is_pond_detail(bbox_cache, prim, water):
            found.append(prim)
    return found


def _world_bounds(stage):
    """XY extent of the world's default prim, used to size the ground plane."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    r = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
    if r.IsEmpty():
        return None
    return r.GetMin(), r.GetMax()


def _add_ground_plane(stage, min_b, max_b):
    """
    Add an invisible flat collider just below grade.

    A plane is not enough on its own: only about a fifth of the raised cells have
    flat ground beneath them, the rest sit over the sunken road (which drops to
    about -19.8 m), so hiding the terraces without this would open holes for
    agents and the robot to fall through.

    Authored as a Mesh rather than a Cube. The navmesh baker ignores implicit
    geometry, and without a navmesh omni.anim.behavior.core never creates a
    single agent -- so a Cube proxy here would silently cost us every pedestrian
    in --flat-ground mode. See behavior_agent.make_ground_collider_mesh.
    """
    prim = make_ground_collider_mesh(
        stage,
        GROUND_PLANE_PATH,
        (min_b[0], min_b[1]),
        (max_b[0], max_b[1]),
        _GROUND_TOP_Z,
    )
    UsdGeom.Imageable(prim).MakeInvisible()
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_api.GetApproximationAttr().Set(UsdPhysics.Tokens.none)
    return prim


def apply_flat_ground(stage, map_name):
    """
    Hide the raised landscape and put a flat collider at z=0 in its place.

    Non-destructive: prims are made invisible and their collision removed, never
    deleted, so the .usd on disk is untouched and re-running without the flag
    restores the terraces.

    Returns the number of terrace meshes neutralised, or -1 if the world is not
    one this pass understands.
    """
    if map_name not in _SUPPORTED:
        print(
            f"[terrain] --flat-ground: no terrain profile for world '{map_name}', "
            f"skipping (supported: {', '.join(_SUPPORTED)})"
        )
        return -1

    if stage is None:
        print("[terrain] --flat-ground: no stage open, skipping")
        return -1

    terraces = _collect_terrace_prims(stage)
    for prim in terraces:
        UsdGeom.Imageable(prim).MakeInvisible()
        # Removing the API is what actually stops PhysX colliding with it;
        # hiding alone leaves the collider live.
        prim.RemoveAPI(UsdPhysics.CollisionAPI)

    bounds = _world_bounds(stage)
    if bounds is None:
        print("[terrain] --flat-ground: world has no bounds, ground plane skipped")
        return len(terraces)

    min_b, max_b = bounds
    _add_ground_plane(stage, min_b, max_b)
    print(
        f"[terrain] --flat-ground: hid {len(terraces)} raised meshes and added a flat "
        f"collider at z={_GROUND_TOP_Z:.2f} spanning "
        f"x[{min_b[0]:.1f},{max_b[0]:.1f}] y[{min_b[1]:.1f},{max_b[1]:.1f}]; "
        f"pond preserved"
    )
    return len(terraces)

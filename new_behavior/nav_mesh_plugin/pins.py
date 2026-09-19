"""
pins.py

Visual, draggable markers for agent spawns and goals.

Each pin is an Xform carrying a body marker and, for spawns, a forward arrow so
the heading is visible. They are ordinary prims, so Omniverse's own transform
gizmo moves them with no extra machinery.

Every pin is marked with `NavMeshExcludeAPI`. Without it the pins are exactly
what the baker looks for -- visible mesh geometry -- and a re-bake carves a hole
in the navmesh at each spawn point. `excludeRigidBodies` does not help: pins
have no rigid bodies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
import omni.usd

try:
    import NavSchema
except ImportError:  # pragma: no cover - only absent outside Isaac Sim
    NavSchema = None

SPAWN_SCOPE = "/World/HuNavSpawns"
GOAL_SCOPE = "/World/HuNavGoals"

SPAWN_COLOR = (0.15, 0.85, 0.35)
SPAWN_ARROW_COLOR = (0.95, 0.95, 0.20)
GOAL_COLOR = (0.98, 0.72, 0.10)

_BODY_HEIGHT = 1.8
_BODY_RADIUS = 0.35
_GOAL_RADIUS = 0.45
_GOAL_HEIGHT = 0.25


def _stage(stage: Usd.Stage = None) -> Usd.Stage:
    return stage or omni.usd.get_context().get_stage()


def _exclude_from_bake(prim: Usd.Prim) -> bool:
    """Mark a prim so the navmesh baker skips it and everything under it."""
    if NavSchema is None or not prim or not prim.IsValid():
        return False
    try:
        NavSchema.NavMeshExcludeAPI.Apply(prim)
        return True
    except Exception as exc:  # pragma: no cover - schema version dependent
        print(f"[pins] could not apply NavMeshExcludeAPI to {prim.GetPath()}: {exc}")
        return False


def ensure_scopes(stage: Usd.Stage = None) -> Tuple[Usd.Prim, Usd.Prim]:
    """Create the two pin scopes, excluded from baking, and return them."""
    stage = _stage(stage)

    scopes = []
    for path in (SPAWN_SCOPE, GOAL_SCOPE):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        _exclude_from_bake(prim)
        scopes.append(prim)

    return scopes[0], scopes[1]


def _cylinder(stage, path, radius, height, color, z_offset=0.0):
    """A low-poly cylinder authored as a Mesh, coloured with a display primvar."""
    segments = 16
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)

    points = []
    for x, y in ring:
        points.append(Gf.Vec3f(float(x), float(y), float(z_offset)))
    for x, y in ring:
        points.append(Gf.Vec3f(float(x), float(y), float(z_offset + height)))

    counts: List[int] = []
    indices: List[int] = []
    for i in range(segments):
        j = (i + 1) % segments
        counts.append(4)
        indices.extend([i, j, j + segments, i + segments])
    # Caps as fans, so the marker reads as solid from above and below.
    counts.append(segments)
    indices.extend(reversed(range(segments)))
    counts.append(segments)
    indices.extend(range(segments, 2 * segments))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(-radius, -radius, float(z_offset)),
            Gf.Vec3f(radius, radius, float(z_offset + height)),
        ]
    )
    UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("constant")
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdGeom.Primvar(mesh.GetDisplayOpacityAttr()).SetInterpolation("constant")
    mesh.GetDisplayOpacityAttr().Set([0.55])
    return mesh


def _arrow(stage, path, length, color, z):
    """A flat triangle along local +X marking the facing direction."""
    half_width = 0.16
    points = [
        Gf.Vec3f(0.0, half_width, z),
        Gf.Vec3f(0.0, -half_width, z),
        Gf.Vec3f(float(length), 0.0, z),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr(
        [Gf.Vec3f(0.0, -half_width, z), Gf.Vec3f(float(length), half_width, z)]
    )
    UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("constant")
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return mesh


def _set_pose(prim: Usd.Prim, position, yaw_rad: float) -> None:
    """Author translate + rotateXYZ, reusing the ops if they already exist.

    rotateXYZ with only Z set is what `spawn_character` writes, so a pin read
    back and written to a scenario round-trips to the same convention.
    """
    xform = UsdGeom.Xformable(prim)

    translate_op = None
    rotate_op = None
    for op in xform.GetOrderedXformOps():
        name = op.GetOpName()
        if name == "xformOp:translate":
            translate_op = op
        elif name == "xformOp:rotateXYZ":
            rotate_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if rotate_op is None:
        rotate_op = xform.AddRotateXYZOp()

    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    rotate_op.Set(Gf.Vec3f(0.0, 0.0, float(math.degrees(yaw_rad))))


def create_spawn_pin(
    agent_name: str,
    position,
    heading_rad: float = 0.0,
    stage: Usd.Stage = None,
) -> str:
    """Author (or update) the pin for one agent spawn. Returns its prim path."""
    stage = _stage(stage)
    ensure_scopes(stage)

    path = f"{SPAWN_SCOPE}/{agent_name}"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        _cylinder(stage, f"{path}/Body", _BODY_RADIUS, _BODY_HEIGHT, SPAWN_COLOR)
        _arrow(stage, f"{path}/Forward", _BODY_RADIUS + 0.65, SPAWN_ARROW_COLOR, 0.05)

    _exclude_from_bake(prim)
    _set_pose(prim, position, heading_rad)
    return path


def create_goal_pin(goal_id: int, position, stage: Usd.Stage = None) -> str:
    """Author (or update) the pin for one global goal. Returns its prim path."""
    stage = _stage(stage)
    ensure_scopes(stage)

    path = f"{GOAL_SCOPE}/Goal_{int(goal_id)}"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        _cylinder(stage, f"{path}/Marker", _GOAL_RADIUS, _GOAL_HEIGHT, GOAL_COLOR)

    _exclude_from_bake(prim)
    _set_pose(prim, position, 0.0)
    return path


def read_pin_pose(prim: Usd.Prim) -> Optional[Tuple[float, float, float, float]]:
    """World-space (x, y, z, yaw) of a pin.

    Taken from the composed local-to-world matrix rather than from
    `xformOp:rotateXYZ`: the viewport gizmo may author `xformOp:orient`
    instead, and the scope above may carry a transform of its own. Yaw comes
    from where local +X ends up, which is where the arrow points, and any tilt
    the user introduced is dropped -- HuNav's heading is a single angle about
    +Z.
    """
    if not prim or not prim.IsValid():
        return None

    xform = UsdGeom.Xformable(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    translation = matrix.ExtractTranslation()
    forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw = math.atan2(float(forward[1]), float(forward[0]))

    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(yaw),
    )


def read_spawn_pins(stage: Usd.Stage = None) -> Dict[str, Tuple[float, float, float, float]]:
    """Every spawn pin on stage, keyed by agent name."""
    stage = _stage(stage)
    scope = stage.GetPrimAtPath(SPAWN_SCOPE)
    if not scope or not scope.IsValid():
        return {}

    poses: Dict[str, Tuple[float, float, float, float]] = {}
    for child in scope.GetChildren():
        pose = read_pin_pose(child)
        if pose is not None:
            poses[child.GetName()] = pose
    return poses


def read_goal_pins(stage: Usd.Stage = None) -> Dict[int, Tuple[float, float]]:
    """Every goal pin on stage, keyed by goal id."""
    stage = _stage(stage)
    scope = stage.GetPrimAtPath(GOAL_SCOPE)
    if not scope or not scope.IsValid():
        return {}

    goals: Dict[int, Tuple[float, float]] = {}
    for child in scope.GetChildren():
        name = child.GetName()
        if not name.startswith("Goal_"):
            continue
        try:
            goal_id = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pose = read_pin_pose(child)
        if pose is not None:
            goals[goal_id] = (pose[0], pose[1])
    return goals


def clear_pins(stage: Usd.Stage = None) -> List[str]:
    """Remove both pin scopes. Returns what was removed."""
    stage = _stage(stage)
    removed = []
    for path in (SPAWN_SCOPE, GOAL_SCOPE):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)
            removed.append(path)
    return removed


def set_pins_visible(visible: bool, stage: Usd.Stage = None) -> None:
    """Show or hide both scopes without deleting anything."""
    stage = _stage(stage)
    for path in (SPAWN_SCOPE, GOAL_SCOPE):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

"""
bake.py

The inputs both modes must bake from.

A spawn validated as walkable while authoring is worthless if the simulator
bakes a different mesh at the same coordinates. The two modes therefore have to
agree on the volume, the ground height and the settings -- and a scenario
records which ones it was authored against, so a stale one says so instead of
quietly misbehaving.

No Isaac Sim imports: the settings dict is passed in by the caller.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Any, Dict, Optional, Tuple

import yaml

Bounds = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

# Vertical band around the walkable surface, matching HuNavManager's own
# padding so an authoring bake and a run bake cover the same slab.
Z_BELOW = 2.0
Z_ABOVE = 4.0


def navmesh_settings_digest(settings: Dict[str, Any]) -> str:
    """A short stable hash of the bake settings.

    Recorded in the scenario so a run whose settings have since changed can say
    that the scenario's walkability guarantees no longer hold, instead of
    silently steering agents over a different mesh.
    """
    canonical = ";".join(
        f"{key}={settings[key]!r}" for key in sorted(settings) if key != "areas"
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _png_size(path: str) -> Tuple[int, int]:
    """(width, height) from a PNG header, without pulling in an image library."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def map_yaml_path(map_name: str, maps_dir: str) -> str:
    return os.path.join(maps_dir, f"{map_name}.yaml")


def scene_bounds(
    map_name: str,
    maps_dir: str,
    ground_z: Optional[float] = None,
) -> Optional[Bounds]:
    """The walkable extent of a map, in the scenario's own metric frame.

    Read from the nav2 map description rather than from the agents already in
    a scenario. Deriving the volume from the poses you are about to place is
    circular: you can only ever put an agent inside the box the current poses
    already describe.

    `origin` is the world coordinate of the image's bottom-left corner, and
    `resolution` is metres per pixel, so the extent follows directly from the
    image dimensions. Coordinates need no transform -- `init_pose` goes
    straight onto the stage.
    """
    yaml_path = map_yaml_path(map_name, maps_dir)
    if not os.path.isfile(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle) or {}

    try:
        resolution = float(meta["resolution"])
        origin = [float(v) for v in meta["origin"]]
        image = str(meta["image"])
    except (KeyError, TypeError, ValueError):
        return None

    image_path = image if os.path.isabs(image) else os.path.join(maps_dir, image)
    if not os.path.isfile(image_path):
        return None

    try:
        width_px, height_px = _png_size(image_path)
    except (OSError, ValueError):
        return None

    min_x = origin[0]
    min_y = origin[1]
    max_x = min_x + width_px * resolution
    max_y = min_y + height_px * resolution

    floor = float(origin[2]) if ground_z is None else float(ground_z)

    return (
        (min_x, min_y, floor - Z_BELOW),
        (max_x, max_y, floor + Z_ABOVE),
    )


def ground_z_for_map(map_name: str, maps_dir: str, default: float = 0.0) -> float:
    """The height agents stand at, taken from the world rather than the agents.

    HuNavManager derives this from the minimum spawn Z, which does not exist
    while a scenario is still being authored. The map's own origin Z is the
    same number and is available in both modes.
    """
    yaml_path = map_yaml_path(map_name, maps_dir)
    if not os.path.isfile(yaml_path):
        return default

    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle) or {}
        return float(meta["origin"][2])
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return default


def sampling_cm_for_extent(
    extent: Tuple[float, float, float],
    base_cm: float,
    max_cells_per_axis: int,
) -> float:
    """The sampling distance the driver's baker will pick for this volume.

    Mirrors BehaviorAgentDriver.bake_navmesh so an authoring session can record
    the value without baking twice. `extent` is in stage units (metres) and the
    sampling distance is in centimetres; mixing them is what once drove the
    bake to a 100x-too-fine mesh.
    """
    largest_cm = max(float(e) for e in extent) * 100.0
    return max(float(base_cm), largest_cm / float(max_cells_per_axis))

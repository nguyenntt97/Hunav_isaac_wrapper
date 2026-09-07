#!/usr/bin/env python3
"""
asset_paths.py

Resolves Isaac Sim cloud asset paths across Isaac Sim releases.

Isaac Sim 5.0 reorganised the ``Isaac/Robots/`` tree into per-vendor
subdirectories and dropped a handful of 4.5-era assets entirely, so the paths
hardcoded by the original wrapper 404 against the 5.0+ asset buckets. Every
version-sensitive asset path lives here so there is one place to update when
the next release moves things again.
"""

import os

# Isaac Sim release whose asset layout is assumed when the running version
# cannot be determined.
_FALLBACK_MAJOR = 6

# Bucket for assets that no longer ship with current Isaac Sim releases.
LEGACY_ASSETS_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
)

# Robot USD locations, keyed by robot name then by the oldest Isaac Sim major
# version that uses the path. A value of None means the asset has no equivalent
# in that release and the robot is unavailable.
_ROBOT_USD = {
    "jetbot": {
        4: os.path.join("Isaac", "Robots", "Jetbot", "jetbot.usd"),
        5: os.path.join("Isaac", "Robots", "NVIDIA", "Jetbot", "jetbot.usd"),
    },
    "create3": {
        4: os.path.join("Isaac", "Robots", "iRobot", "create_3.usd"),
        5: os.path.join("Isaac", "Robots", "iRobot", "Create3", "create_3.usd"),
    },
    # 5.0+ ships only NVIDIA/NovaCarter/nova_carter.usd, which has no sensor
    # suite. The bundled carter_ROS asset is the supported Carter on 5.0+.
    "carter": {
        4: os.path.join("Isaac", "Robots", "Carter", "nova_carter_sensors.usd"),
        5: None,
    },
}


def get_isaac_major(default: int = _FALLBACK_MAJOR) -> int:
    """
    Return the major version of the running/installed Isaac Sim.

    Tries the Isaac Sim API first (accurate, but only usable once the app has
    started), then falls back to reading the VERSION file shipped with the
    installation so this is also callable before SimulationApp comes up.
    """
    try:
        from isaacsim.core.version import get_version

        major = get_version()[2]
        if major:
            return int(major)
    except Exception:
        pass

    candidates = [
        os.environ.get("ISAAC_PATH"),
        os.environ.get("ISAACSIM_ROOT_PATH"),
        "/isaac-sim",
        os.path.join(os.path.expanduser("~"), "isaacsim"),
    ]
    for root in candidates:
        if not root:
            continue
        version_file = os.path.join(root, "VERSION")
        if os.path.isfile(version_file):
            try:
                with open(version_file, encoding="UTF-8") as f:
                    return int(f.readline().strip().split(".")[0])
            except (ValueError, OSError):
                continue

    return default


def _lookup(table: dict, major: int):
    """Return the entry for the newest table key that is <= major."""
    applicable = [k for k in table if k <= major]
    if not applicable:
        return table[min(table)]
    return table[max(applicable)]


def robot_usd_relative_path(robot_key: str, major: int = None):
    """
    Return the assets-root-relative USD path for a built-in Isaac Sim robot,
    or None if that robot has no asset in the running Isaac Sim release.

    Raises:
        KeyError: if robot_key is not a known built-in robot.
    """
    if major is None:
        major = get_isaac_major()
    return _lookup(_ROBOT_USD[robot_key], major)


def is_robot_available(robot_key: str, major: int = None) -> bool:
    """True if robot_key has a usable asset in the running Isaac Sim release."""
    try:
        return robot_usd_relative_path(robot_key, major) is not None
    except KeyError:
        # Robots backed by a bundled USD (carter_ROS) are not in the table and
        # are always available.
        return True


def biped_setup_url(assets_root: str, major: int = None) -> str:
    """
    Return the URL of Biped_Setup.usd, the source skeleton used for animation
    retargeting.

    Isaac Sim 5.0+ removed both Biped_Setup.usd and the biped_demo/ subtree it
    references, and there is no replacement in the current buckets, so on those
    releases this resolves against the 4.5 bucket. Biped_Setup.usd references
    its animations and skeleton by *relative* path, so it must be loaded from a
    root that also holds Isaac/People/Animations/ and
    Isaac/People/Characters/biped_demo/ -- vendoring the single file locally
    does not work.

    Set HUNAV_BIPED_SETUP_USD to override with a local collected copy.
    """
    override = os.environ.get("HUNAV_BIPED_SETUP_USD")
    if override:
        return override

    if major is None:
        major = get_isaac_major()

    root = assets_root if major < 5 else LEGACY_ASSETS_ROOT
    return os.path.join(root, "Isaac/People/Characters/Biped_Setup.usd")

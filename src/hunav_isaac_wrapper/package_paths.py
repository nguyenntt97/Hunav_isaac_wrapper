#!/usr/bin/env python3
"""
package_paths.py

Locating this package's data files, installed or in-tree.

These live apart from teleop_hunav_sim because importing that module boots
Isaac Sim: it calls SimulationApp() at import time. Anything that only needs to
know where a file is -- the robot registry, tests, tooling -- can import this
without starting a simulator, and starting a second one in the same process is
a segfault rather than an error.
"""

import os
import subprocess
from pathlib import Path


def find_package_share_directory():
    """
    Find the package share directory containing worlds, scenarios, config, etc.
    Works both in development and installed package modes.
    """
    # Try to find via ROS2 package first (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        share_dir = pkg_path / "share" / "hunav_isaac_wrapper"
        if share_dir.exists() and (share_dir / "worlds").exists():
            return str(share_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Development mode fallback
    current_file = Path(__file__)
    
    # Check if we're in src/hunav_isaac_wrapper/ (development mode)
    if current_file.parent.parent.name == "src":
        src_dir = current_file.parent.parent
        if (src_dir / "worlds").exists():
            return str(src_dir)
    
    # Last fallback - check current working directory
    cwd = Path.cwd()
    if (cwd / "worlds").exists():
        return str(cwd)
    
    # If all else fails, return the old path calculation
    return os.path.dirname(os.path.dirname(__file__))


def find_config_path(relative):
    """
    Find a file under the package's config/ directory, installed or in-tree.

    Args:
        relative: Path below config/, e.g. "robots/nova_carter_ros2_sensors.usd"
                  or "policies/go2/policy.pt".

    Returns:
        str: Absolute path to the file

    Raises:
        FileNotFoundError: if no search root contains it
    """
    relative = str(relative)
    # Try to find via ROS2 package share directory (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        candidate = pkg_path / "share" / "hunav_isaac_wrapper" / "config" / relative
        if candidate.exists():
            return str(candidate)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    current_file_dir = Path(__file__).parent
    candidates = [
        # Development mode (relative to this file)
        current_file_dir.parent.parent / "config" / relative,
        current_file_dir.parent / "config" / relative,
        Path.cwd() / "src" / "config" / relative,
        Path.cwd() / "config" / relative,
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Config file not found: config/{relative}")


def find_robot_config_path(filename):
    """
    Find a robot USD under config/robots/.

    Args:
        filename: Name of the robot config file (e.g., "nova_carter_ros2_sensors.usd")

    Returns:
        str: Absolute path to the robot config file
    """
    try:
        return find_config_path(Path("robots") / filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Robot config file not found: {filename}") from None

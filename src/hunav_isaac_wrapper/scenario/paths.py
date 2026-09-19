"""
paths.py

Where the two scenario artifacts live.

`hunav_agent_manager` reads behavior trees from the package *source* tree, not
the install space: it resolves the share directory and then maps
`.../install/<pkg>/share/<pkg>` back to `.../src/<pkg>` (bt_node.cpp). Writing
trees anywhere else means they are never found, and every agent silently loads
no tree. So both artifacts are resolved against the source directory here.
"""

from __future__ import annotations

import os
from typing import Optional

# Marks the wrapper source root: the directory that holds both artifacts.
_MARKERS = ("scenarios", "behavior_trees")


def _looks_like_wrapper_src(path: str) -> bool:
    return all(os.path.isdir(os.path.join(path, m)) for m in _MARKERS)


def wrapper_src_dir(start: Optional[str] = None) -> str:
    """The directory holding `scenarios/` and `behavior_trees/`.

    Walks up from this file, which works in the repo layout and through the
    symlinked copy of the plugin. Falls back to the ROS share directory's
    source sibling, mirroring what the agent manager does.
    """
    here = os.path.abspath(start or __file__)
    if os.path.isfile(here):
        here = os.path.dirname(here)

    current = here
    while True:
        if _looks_like_wrapper_src(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Repo layout: <root>/src holds both, and this file sits in
    # <root>/src/hunav_isaac_wrapper/scenario.
    candidate = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    if _looks_like_wrapper_src(candidate):
        return candidate

    env = os.environ.get("HUNAV_WRAPPER_SRC")
    if env and _looks_like_wrapper_src(env):
        return env

    raise RuntimeError(
        "could not locate the wrapper source directory (the one containing "
        "'scenarios' and 'behavior_trees'). Set HUNAV_WRAPPER_SRC to point at it."
    )


def scenarios_dir(start: Optional[str] = None) -> str:
    return os.path.join(wrapper_src_dir(start), "scenarios")


def behavior_trees_dir(start: Optional[str] = None) -> str:
    return os.path.join(wrapper_src_dir(start), "behavior_trees")


def maps_dir(start: Optional[str] = None) -> str:
    return os.path.join(wrapper_src_dir(start), "maps")


def scenario_path(base_name: str, start: Optional[str] = None) -> str:
    """Full path for a scenario, with or without the extension given."""
    if base_name.endswith((".yaml", ".yml")):
        base_name = os.path.splitext(base_name)[0]
    return os.path.join(scenarios_dir(start), f"{base_name}.yaml")

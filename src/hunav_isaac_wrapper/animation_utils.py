#!/usr/bin/env python3
"""
animation_utils.py

Helpers for locating Skeleton and SkelRoot prims inside a character asset.

This module used to build an AnimationGraph per agent and retarget two clips
onto each rig at launch, the design Isaac Sim 4.x's omni.anim.people used. That
extension no longer exists in Isaac Sim 6, and on 6.0 the retarget step silently
produced a static clip, leaving every agent in bind pose. Character locomotion
now goes through omni.anim.behavior.core; see behavior_agent.py.

What remains is the part that was never version-specific: finding the skeleton
inside a character asset. behavior_agent.attach() uses find_skelroot_path to
locate the prim BehaviorAgentAPI must be applied to.
"""

from pxr import Sdf


def find_skeleton_path(agentPrim):
    """
    Recursively searches for a prim of type "Skeleton" within agentPrim.

    """
    if agentPrim.GetTypeName() == "Skeleton":
        return agentPrim.GetPath()
    for child in agentPrim.GetChildren():
        if not child.IsValid():
            continue
        child_path_str = child.GetPath().pathString
        if ("Looks" in child_path_str) or ("CharacterAnimation" in child_path_str):
            continue
        if child.GetTypeName() == "Skeleton":
            return child.GetPath()
        result = find_skeleton_path(child)
        if result:
            return result
    print(
        f"Warning: No Skeleton found for {agentPrim.GetPath()}, using /Root by default."
    )
    return Sdf.Path(f"{agentPrim.GetPath()}/Root")


def find_skelroot_path(agentPrim):
    """
    Recursively searches for a SkelRoot prim within the children of agentPrim.

    """
    for child in agentPrim.GetChildren():
        if not child.IsValid():
            continue
        child_path_str = child.GetPath().pathString

        if ("Looks" in child_path_str) or ("CharacterAnimation" in child_path_str):
            continue

        if "ManRoot" in child_path_str:
            for grandchild in child.GetChildren():
                if grandchild.IsValid() and grandchild.GetTypeName() == "SkelRoot":
                    return grandchild.GetPath()

        if child.GetTypeName() == "SkelRoot":
            return child.GetPath()

        result = find_skelroot_path(child)
        if result:
            return result

    print(
        f"Warning: No SkelRoot found within {agentPrim.GetPath()}, using fallback /Root."
    )
    return Sdf.Path(f"{agentPrim.GetPath()}/Root")

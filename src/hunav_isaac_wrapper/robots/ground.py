#!/usr/bin/env python3
"""
robots/ground.py

Ground friction for legged robots.

Isaac's own Go2 example binds a physics material with static and dynamic
friction 1.0 to its ground plane, matching the values the locomotion policy was
trained against; without it the feet slip and the gait degrades. That example
knows exactly one ground prim path. This wrapper loads arbitrary authored
stages (warehouse, office, hospital, brownstone) whose floors are named
differently in each.

USD material bindings inherit down the hierarchy, and a descendant's own
binding wins, so binding once at the world root covers every collider in the
stage that has not been given a physics material of its own -- no per-stage
knowledge required.
"""

from pxr import UsdPhysics, UsdShade

MATERIAL_PATH = "/World/Looks/LeggedGroundPhysicsMaterial"


def apply_ground_friction(
    stage, root_path="/World", static_friction=1.0, dynamic_friction=1.0, restitution=0.0
):
    """Bind a physics material at ``root_path``. Best effort; never fatal.

    Returns:
        bool: True if the material was bound.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[hunav] No prim at {root_path}; skipping ground friction.")
        return False

    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(float(restitution))

    binding_api = UsdShade.MaterialBindingAPI.Apply(root)
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    print(
        f"[hunav] Bound ground physics material at {root_path} "
        f"(static={static_friction}, dynamic={dynamic_friction})."
    )
    return True

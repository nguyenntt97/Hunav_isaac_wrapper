"""
NavMesh Plugin & Visualizer package.

Provides a 1:1 functional drop-in replacement for ov_navmesh,
backed by Isaac Sim's native omni.anim.navigation.core and USD 24+.
"""

from . import usd_utils
from .core import NavmeshInterface, NativeNavmeshInterface, DEFAULT_RECAST_SETTINGS

# The window is imported lazily. Reproducing an authored navmesh at run time
# needs core.py and nothing else, and it runs in sessions that never open the
# editor -- importing omni.ui here would make a headless bake depend on the UI
# being available.
_UI_NAMES = frozenset({
    "NavmeshWindow",
    "NavmeshExtension",
    "SiborgCreateNavmeshExtension",
    "show_navmesh_window",
})


def __getattr__(name):
    if name in _UI_NAMES:
        from . import ui_window

        return getattr(ui_window, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_and_visualize_navmesh(
    stage=None,
    prim_path: str = "/World/navmeshmesh",
    outline_prefix: str = "/World/Outline/WallOutline",
    settings: dict = None,
    color=(0.0512, 0.7749, 0.9458),
    opacity: float = 0.89,
    draw_outlines: bool = True,
    force_rebake: bool = False,
):
    """One-call utility to configure, bake, and visualize the navigation mesh on stage."""
    adapter = NavmeshInterface(stage=stage)
    if force_rebake or not adapter.built:
        if force_rebake:
            adapter.reset_navmesh(clear_stage=True)
        adapter.build_navmesh(settings=settings)

    created_mesh = adapter.visualize_navmesh(prim_path=prim_path, color=color, opacity=opacity)
    created_lines = adapter.make_outline(prim_prefix=outline_prefix, batched=True) if draw_outlines else []

    return {
        "adapter": adapter,
        "mesh_path": created_mesh,
        "outlines": created_lines,
    }


def reset_navmesh(stage=None, clear_stage: bool = True, clear_cache: bool = True) -> bool:
    """Reset the baked navmesh state and remove visualizations from stage."""
    adapter = NavmeshInterface(stage=stage)
    return adapter.reset_navmesh(clear_stage=clear_stage, clear_cache=clear_cache)


def reset_and_rebake_navmesh(
    stage=None,
    prim_path: str = "/World/navmeshmesh",
    outline_prefix: str = "/World/Outline/WallOutline",
    settings: dict = None,
    color=(0.0512, 0.7749, 0.9458),
    opacity: float = 0.89,
    draw_outlines: bool = True,
):
    """Reset existing navmesh and visuals, re-bake with new settings, and re-visualize."""
    return build_and_visualize_navmesh(
        stage=stage,
        prim_path=prim_path,
        outline_prefix=outline_prefix,
        settings=settings,
        color=color,
        opacity=opacity,
        draw_outlines=draw_outlines,
        force_rebake=True,
    )


__all__ = [
    "usd_utils",
    "NavmeshInterface",
    "NativeNavmeshInterface",
    "DEFAULT_RECAST_SETTINGS",
    "NavmeshWindow",
    "NavmeshExtension",
    "SiborgCreateNavmeshExtension",
    "show_navmesh_window",
    "build_and_visualize_navmesh",
    "reset_navmesh",
    "reset_and_rebake_navmesh",
]


"""
ui_window.py

Interactive Omniverse Kit UI window for the NavMesh plugin.
Recreates the exact visual layout, buttons, drag-drop targets, and parameter
sliders from the original ov_navmesh extension.py, backed by the NativeNavmeshInterface.
"""

from __future__ import annotations

from typing import Optional

try:
    import omni.ext
    import omni.ui as ui
    from omni.ui import color as cl
except ImportError:
    ui = None
    cl = None

from pxr import Gf, Usd, UsdGeom
import omni.usd

from .core import NavmeshInterface
from . import usd_utils


CHARACTERS_SCOPE = "/World/Characters"


def live_agents_present(stage) -> bool:
    """Whether HuNav characters are already on stage.

    Re-baking under live agents replaces the navmesh they are being steered on,
    with settings that differ from the driver's on step height, slope and island
    radius. Worse, the driver's baker records that a failed bake poisons the
    process -- every later bake returns empty -- so one exploratory click can
    make a correct bake impossible without restarting. The destructive buttons
    refuse whenever this is true.
    """
    if stage is None:
        return False
    scope = stage.GetPrimAtPath(CHARACTERS_SCOPE)
    return bool(scope and scope.IsValid() and scope.GetChildren())


class NavmeshWindow:
    """Omniverse Kit UI window controlling NavMesh assignment, baking, and visualization."""

    def __init__(self, title: str = "Navmesh", width: int = 340, height: int = 900):
        if ui is None:
            raise RuntimeError("omni.ui is not available in this environment.")

        self.title = title
        self.stage = omni.usd.get_context().get_stage()
        self.navmesh = NavmeshInterface()
        self.navmesh_settings = {}
        self.start_prim = None
        self.end_prim = None

        # Scenario authoring shares this window's adapter rather than building
        # a second one, so both agree on whether a mesh is built and on which
        # stage they are looking at.
        self.scenario = None
        self.read_only = live_agents_present(self.stage)

        self._window = ui.Window(title, width=width, height=height)
        self._build_ui()

    def _ensure_scenario_manager(self):
        """Create the ScenarioManager on first use, sharing this adapter."""
        if self.scenario is None:
            from .scenario_manager import ScenarioManager

            self.scenario = ScenarioManager(adapter=self.navmesh, stage=self.stage)
        return self.scenario

    def _build_ui(self):
        # Color palette matching original extension
        s_red = {"background_color": cl(160, 0, 0)}
        s_yellow = {"background_color": cl(150, 150, 0)}
        s_green = {"background_color": cl(0, 160, 0)}
        s_done = {"background_color": cl(0, 160, 0, 80)}

        with self._window.frame:
            with ui.VStack(spacing=6):
                def reset_btns():
                    self.assign_btn.style = s_yellow
                    self.bld_btn.style = s_red
                    self.rnd_pnts_btn.style = s_red
                    self.rnd_pth_btn.style = s_red
                    self.mesh_btn.style = s_red
                    self.outline_btn.style = s_red
                    self.pth_btn.style = s_red
                    if hasattr(self, "clear_btn"):
                        self.clear_btn.style = s_yellow

                def assign_mesh():
                    reset_btns()
                    stage = omni.usd.get_context().get_stage()
                    self.stage = stage
                    self.navmesh.stage = stage

                    if self.navmesh.get_selected_prim():
                        self.assign_btn.style = s_done
                        self.bld_btn.style = s_yellow
                    else:
                        print("[NavMesh UI] Please select a mesh or hierarchy in the Stage tree first.")

                def build_navmesh():
                    if self.read_only:
                        print(
                            "[NavMesh UI] Agents are live. Re-baking now would replace "
                            "the navmesh they are steered on, with settings that differ "
                            "from the ones they were spawned against, and a failed bake "
                            "poisons every later one. Relaunch with --author-scenario "
                            "to bake and edit."
                        )
                        return
                    success = self.navmesh.build_navmesh(settings=self.navmesh_settings)
                    if success:
                        self.bld_btn.style = s_done
                        self.rnd_pnts_btn.style = s_green
                        self.rnd_pth_btn.style = s_green
                        self.mesh_btn.style = s_yellow
                        self.outline_btn.style = s_yellow
                        self.pth_btn.style = s_green
                    else:
                        print("[NavMesh UI] Navmesh build failed. Ensure walkable geometry is selected and covered by the volume.")

                def visualize_navmesh():
                    prim_path = self.navmesh.visualize_navmesh(
                        prim_path="/World/navmeshmesh",
                        color=(0.0512, 0.7749, 0.9458),
                        opacity=0.89,
                    )
                    if prim_path:
                        self.mesh_btn.style = s_done

                def outline_navmesh():
                    lines = self.navmesh.make_outline(
                        prim_prefix="/World/Outline/WallOutline",
                        color=(0.9, 0.9, 0.2),
                        width=0.06,
                        batched=True,
                    )
                    if lines:
                        self.outline_btn.style = s_done

                def get_random_points():
                    prim_path = self.navmesh.visualize_random_points(
                        num_points=10,
                        prim_path="/World/Points",
                        color=(1.0, 0.0, 0.0),
                    )
                    if prim_path:
                        self.rnd_pnts_btn.style = s_done

                def get_random_path():
                    pnts = self.navmesh.get_random_points(2)
                    if pnts is not None and len(pnts) >= 2:
                        self.navmesh.visualize_path(pnts[0], pnts[1], prim_path="/World/Path")
                        self.rnd_pth_btn.style = s_done
                    else:
                        print("[NavMesh UI] Could not sample two points for random path.")

                def get_specific_path():
                    if not self.start_prim or not self.end_prim:
                        print("[NavMesh UI] Drag and drop Start Prim and End Prim first.")
                        return

                    time = Usd.TimeCode.Default()
                    xform_s = UsdGeom.Xformable(self.start_prim)
                    s = xform_s.ComputeLocalToWorldTransform(time).ExtractTranslation()
                    xform_e = UsdGeom.Xformable(self.end_prim)
                    e = xform_e.ComputeLocalToWorldTransform(time).ExtractTranslation()

                    prim_path = self.navmesh.visualize_path(s, e, prim_path="/World/Path")
                    if prim_path:
                        self.pth_btn.style = s_done

                def assign_start_prim(event):
                    item = event.mime_data
                    self.startprim_field.model.set_value(item)
                    self.start_prim = self.stage.GetPrimAtPath(item)

                def assign_end_prim(event):
                    item = event.mime_data
                    self.endprim_field.model.set_value(item)
                    self.end_prim = self.stage.GetPrimAtPath(item)

                def clear_and_reset_navmesh():
                    if self.read_only:
                        print(
                            "[NavMesh UI] Agents are live; clearing the navmesh would "
                            "strand them. Relaunch with --author-scenario instead."
                        )
                        return
                    self.navmesh.reset_navmesh(clear_stage=True)
                    reset_btns()
                    if hasattr(self, "startprim_field"):
                        self.startprim_field.model.set_value("")
                    if hasattr(self, "endprim_field"):
                        self.endprim_field.model.set_value("")
                    self.start_prim = None
                    self.end_prim = None
                    self.clear_btn.style = s_done
                    print("[NavMesh UI] Reset and cleared baked navmesh. Ready to re-bake.")

                # Action Buttons
                with ui.VStack(spacing=4):
                    self.assign_btn = ui.Button("Assign Mesh", clicked_fn=assign_mesh, style=s_yellow)
                    self.bld_btn = ui.Button("Build Navmesh", clicked_fn=build_navmesh, style=s_red)
                    self.mesh_btn = ui.Button("Create Mesh", clicked_fn=visualize_navmesh, style=s_red)
                    self.outline_btn = ui.Button("Outline Walls", clicked_fn=outline_navmesh, style=s_red)
                    self.rnd_pnts_btn = ui.Button("Get Random Points", clicked_fn=get_random_points, style=s_red)
                    self.rnd_pth_btn = ui.Button("Get Random Path", clicked_fn=get_random_path, style=s_red)
                    self.pth_btn = ui.Button("Get Start-End Path", clicked_fn=get_specific_path, style=s_red)
                    self.clear_btn = ui.Button("Reset / Clear NavMesh", clicked_fn=clear_and_reset_navmesh, style=s_yellow)

                # Auto-detect existing navmesh (e.g. pre-baked HuNav simulation map)
                if self.navmesh.built:
                    self.bld_btn.style = s_done
                    self.mesh_btn.style = s_yellow
                    self.outline_btn.style = s_yellow
                    self.rnd_pnts_btn.style = s_green
                    self.rnd_pth_btn.style = s_green
                    self.pth_btn.style = s_green

                ui.Spacer(height=4)

                # Start / End Prim Drag Targets
                with ui.HStack(height=26, spacing=4):
                    ui.Label("Start Prim", width=65)
                    self.startprim_field = ui.StringField(tooltip="Drag Start Prim from Stage tree")
                    self.startprim_field.set_accept_drop_fn(lambda item: True)
                    self.startprim_field.set_drop_fn(assign_start_prim)

                with ui.HStack(height=26, spacing=4):
                    ui.Label("End Prim", width=65)
                    self.endprim_field = ui.StringField(tooltip="Drag End Prim from Stage tree")
                    self.endprim_field.set_accept_drop_fn(lambda item: True)
                    self.endprim_field.set_drop_fn(assign_end_prim)

                ui.Spacer(height=4)

                # Parameter Settings
                def set_settings():
                    self.navmesh_settings["agentHeight"] = self.agent_height_float.get_value_as_float()
                    self.navmesh_settings["agentRadius"] = self.agent_radius_float.get_value_as_float()
                    self.navmesh_settings["agentMaxClimb"] = self.agent_step_float.get_value_as_float()
                    self.navmesh_settings["agentMaxSlope"] = self.agent_slope_float.get_value_as_float()
                    print(f"[NavMesh UI] Settings updated: {self.navmesh_settings}")

                def reset_settings():
                    self.navmesh_settings = {}
                    self.agent_height_float.set_value(2.0)
                    self.agent_radius_float.set_value(0.6)
                    self.agent_step_float.set_value(0.9)
                    self.agent_slope_float.set_value(45.0)
                    print("[NavMesh UI] Settings reset to default.")

                with ui.CollapsableFrame("Navmesh Settings", collapsed=False):
                    with ui.VStack(spacing=4):
                        ui.Label("Agent Height (m)")
                        self.agent_height_float = ui.SimpleFloatModel(2.0, min=0.1, max=10.0)
                        ui.FloatSlider(self.agent_height_float, min=0.1, max=10.0, step=0.05)

                        ui.Label("Agent Radius (m)")
                        self.agent_radius_float = ui.SimpleFloatModel(0.6, min=0.05, max=5.0)
                        ui.FloatSlider(self.agent_radius_float, min=0.05, max=5.0, step=0.05)

                        ui.Label("Max Step Height (m)")
                        self.agent_step_float = ui.SimpleFloatModel(0.9, min=0.0, max=5.0)
                        ui.FloatSlider(self.agent_step_float, min=0.0, max=5.0, step=0.05)

                        ui.Label("Max Slope (deg)")
                        self.agent_slope_float = ui.SimpleFloatModel(45.0, min=0.0, max=89.9)
                        ui.FloatSlider(self.agent_slope_float, min=0.0, max=89.9, step=1.0)

                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Set Settings", clicked_fn=set_settings)
                            ui.Button("Reset Settings", clicked_fn=reset_settings)

                ui.Spacer(height=4)
                self._build_scenario_section(s_red, s_yellow, s_green, s_done)

    def _build_scenario_section(self, s_red, s_yellow, s_green, s_done):
        """Agent spawns, goals and behaviors, as a view over a validated model.

        Everything here edits a `ScenarioSpec` and redraws the pins from it. The
        pins are never the source of truth: `init_scenario.py` produces the same
        scenario headless, and this panel is where it gets inspected and nudged.
        """

        def report(notes):
            for note in notes or []:
                print(f"[NavMesh UI] {note}")

        def load_scenario():
            manager = self._ensure_scenario_manager()
            try:
                from .scenario_manager import scenario_paths

                name = self.scenario_name_field.model.get_value_as_string().strip()
                manager.load_scenario(scenario_paths.scenario_path(name))
                self.load_btn.style = s_done
                validate_scenario()
            except Exception as exc:
                print(f"[NavMesh UI] Load failed: {exc}")

        def generate_scenario():
            if self.read_only:
                print("[NavMesh UI] Read-only: agents are live. Use --author-scenario.")
                return
            manager = self._ensure_scenario_manager()
            try:
                from .scenario_manager import parse_behavior_mix

                mix_text = self.mix_field.model.get_value_as_string().strip()
                manager.generate_scenario(
                    map_name=self.map_field.model.get_value_as_string().strip(),
                    num_agents=int(self.agents_int.get_value_as_int()),
                    num_goals=int(self.goals_int.get_value_as_int()),
                    goals_per_agent=int(self.ring_int.get_value_as_int()),
                    behavior_mix=parse_behavior_mix(mix_text) if mix_text else None,
                    seed=int(self.seed_int.get_value_as_int()),
                    yaml_base_name=(
                        self.scenario_name_field.model.get_value_as_string().strip()
                        or None
                    ),
                )
                self.generate_btn.style = s_done
                validate_scenario()
            except Exception as exc:
                print(f"[NavMesh UI] Generate failed: {exc}")

        def snap_pins():
            manager = self._ensure_scenario_manager()
            report(manager.sync_from_stage())
            report(manager.snap_all())
            validate_scenario()

        def scatter_spawns():
            manager = self._ensure_scenario_manager()
            report(manager.sync_from_stage())
            report(manager.scatter_spawns(min_separation=2.0))
            validate_scenario()

        def validate_scenario():
            manager = self._ensure_scenario_manager()
            if manager.spec is None:
                print("[NavMesh UI] Nothing loaded yet.")
                self.validate_btn.style = s_red
                return
            report(manager.sync_from_stage())
            problems = manager.validate()
            fatal = [p for p in problems if p.fatal]
            for problem in problems:
                print(f"[NavMesh UI] {problem}")
            if fatal:
                self.validate_btn.style = s_red
                self.export_btn.style = s_red
                print(f"[NavMesh UI] {len(fatal)} blocking problem(s); export refused.")
            else:
                self.validate_btn.style = s_done
                self.export_btn.style = s_green
                print("[NavMesh UI] Scenario is launchable.")

        def export_scenario():
            manager = self._ensure_scenario_manager()
            try:
                result = manager.export()
                self.export_btn.style = s_done
                print(
                    f"[NavMesh UI] Exported {result['scenario']} and "
                    f"{len(result['trees'])} behavior trees. Relaunch to run it: "
                    "the scenario is read once, at startup."
                )
            except Exception as exc:
                self.export_btn.style = s_red
                print(f"[NavMesh UI] Export failed: {exc}")

        def clear_pins():
            manager = self._ensure_scenario_manager()
            removed = manager.clear()
            print(f"[NavMesh UI] Cleared pins: {removed or 'nothing on stage'}")

        title = "Agent Spawns & Goals"
        if self.read_only:
            title += "  (read-only: agents are live)"

        with ui.CollapsableFrame(title, collapsed=False):
            with ui.VStack(spacing=3):
                with ui.HStack(height=24, spacing=4):
                    ui.Label("Scenario", width=70)
                    self.scenario_name_field = ui.StringField(
                        tooltip="Base name under src/scenarios, without .yaml"
                    )
                    self.scenario_name_field.model.set_value("brownstone_agents")

                with ui.HStack(height=24, spacing=4):
                    ui.Label("Map", width=70)
                    self.map_field = ui.StringField(tooltip="Map name under src/maps")
                    self.map_field.model.set_value("brownstone")

                with ui.HStack(height=24, spacing=4):
                    ui.Label("Agents", width=70)
                    self.agents_int = ui.SimpleIntModel(8)
                    ui.IntField(self.agents_int)
                    ui.Label("Goals", width=45)
                    self.goals_int = ui.SimpleIntModel(15)
                    ui.IntField(self.goals_int)

                with ui.HStack(height=24, spacing=4):
                    ui.Label("Ring", width=70)
                    self.ring_int = ui.SimpleIntModel(5)
                    ui.IntField(self.ring_int)
                    ui.Label("Seed", width=45)
                    self.seed_int = ui.SimpleIntModel(0)
                    ui.IntField(self.seed_int)

                with ui.HStack(height=24, spacing=4):
                    ui.Label("Mix", width=70)
                    self.mix_field = ui.StringField(
                        tooltip="e.g. Regular:5,Curious:2,Scared:1"
                    )
                    self.mix_field.model.set_value("Regular:5,Curious:2,Scared:1")

                ui.Spacer(height=2)
                self.load_btn = ui.Button(
                    "Load Scenario", clicked_fn=load_scenario, style=s_yellow
                )
                self.generate_btn = ui.Button(
                    "Generate on NavMesh",
                    clicked_fn=generate_scenario,
                    style=s_red if self.read_only else s_yellow,
                )
                self.snap_btn = ui.Button(
                    "Snap Pins to NavMesh", clicked_fn=snap_pins, style=s_yellow
                )
                self.scatter_btn = ui.Button(
                    "Scatter Spawns", clicked_fn=scatter_spawns, style=s_yellow
                )
                self.validate_btn = ui.Button(
                    "Validate", clicked_fn=validate_scenario, style=s_yellow
                )
                self.export_btn = ui.Button(
                    "Export YAML + Behavior Trees",
                    clicked_fn=export_scenario,
                    style=s_red,
                )
                self.clearpins_btn = ui.Button(
                    "Clear Pins", clicked_fn=clear_pins, style=s_yellow
                )

    def destroy(self):
        if self._window:
            self._window.destroy()
            self._window = None


class SiborgCreateNavmeshExtension(omni.ext.IExt if "omni.ext" in globals() else object):
    """Extension wrapper matching the original extension class name."""

    def on_startup(self, ext_id):
        print(f"[siborg.create.navmesh] Native NavMesh adapter startup: {ext_id}")
        self._window = NavmeshWindow()

    def on_shutdown(self):
        print("[siborg.create.navmesh] Native NavMesh adapter shutdown")
        if hasattr(self, "_window") and self._window:
            self._window.destroy()


# Modern naming alias
NavmeshExtension = SiborgCreateNavmeshExtension


def show_navmesh_window() -> NavmeshWindow:
    """Convenience function to open the NavMesh window from any Python script or console."""
    return NavmeshWindow()


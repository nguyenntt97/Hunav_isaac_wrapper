# NavMesh Plugin & Visualizer: Blockers Analysis & Adapter Specification

This directory documents the technical blockers preventing the direct adoption of `/workspace/Hunav_isaac_wrapper/third_party/ov_navmesh`, analyzes the architectural incompatibilities with the current codebase, and presents the phase-by-phase implementation plan for a 1:1 functional adapter.

The goal is to deliver **the exact functional surface of the original `ov_navmesh`** tool (mesh assignment, navmesh baking, stage visualizer mesh generation, wall boundary curve marking, random point sampling, and start-to-end pathfinding visualization) while operating natively, robustly, and synchronously within Isaac Sim 6.x and the HuNav Isaac Wrapper.

---

## Table of Contents

1. [Functional Audit of the Original `ov_navmesh`](#1-functional-audit-of-the-original-ov_navmesh)
2. [Root Cause Blockers of the Original Repository](#2-root-cause-blockers-of-the-original-repository)
3. [Architectural Strategy: The Native NavMesh Adapter](#3-architectural-strategy-the-native-navmesh-adapter)
4. [1:1 API & Functional Mapping Specification](#4-11-api--functional-mapping-specification)
5. [Phase-by-Phase Implementation Roadmap](#5-phase-by-phase-implementation-roadmap)
6. [Detailed Technical Adapter Design](#6-detailed-technical-adapter-design)
7. [Verification & Acceptance Criteria](#7-verification--acceptance-criteria)

---

## 1. Functional Audit of the Original `ov_navmesh`

The original repository (`cadop/ov_navmesh`, located at `third_party/ov_navmesh`) provides an interactive tool with four main modules:

| File | Purpose in `ov_navmesh` |
|---|---|
| `extension.py` | UI Window (`ui.Window("Navmesh")`) with interactive action buttons and prim drag-drop fields. |
| `core.py` | `NavmeshInterface` coordinating coordinate conversion, geometry collection, baking, and mesh extraction. |
| `usd_utils.py` | Pure USD helper functions for stage traversal, vertex transformation, `UsdGeom.Mesh` creation, and `UsdGeom.BasisCurves` creation. |
| `pyrecast/__init__.py` | High-level Python wrapper defining default Recast parameters, exporting `.obj` files, and calling `PyRecast`. |
| `PyRecast.*.so` / `*.pyd` | Closed C++ pybind11 extension linking Recast & Detour. |

### Expected End-User Functions
1. **Mesh Selection / Assignment**: Selecting individual or hierarchy prims, calculating local-to-world vertex matrices, and setting up the navmesh boundary input.
2. **NavMesh Baking**: Invoking Recast navigation generation with configurable agent radius, height, max climb, max slope, cell size, and cell height.
3. **NavMesh Visualization Mesh**: Generating a translucent, colored USD triangle mesh at `/World/navmeshmesh` with custom `UsdPreviewSurface` material and opacity.
4. **Boundary Marking (Wall Outlines)**: Generating `UsdGeom.BasisCurves` line markings along navmesh obstacle edges and boundaries at `/World/Outline/WallOutline*`.
5. **Random Point Queries & Visualization**: Querying valid navigable points on the navmesh and placing `UsdGeom.Points` visual markers at `/World/Points`.
6. **Path Finding & Spline Visualization**: Computing shortest paths between start and goal points and rendering the path as a linear ribbon using `UsdGeom.BasisCurves` at `/World/Path`.
7. **Interactive GUI**: Omniverse Kit UI panel with interactive buttons ("Assign Mesh", "Build Navmesh", "Create Mesh", "Get Random Points", "Get Random Path", "Get Start-End Path", and Start/End prim drop targets).

---

## 2. Root Cause Blockers of the Original Repository

Attempting to run `third_party/ov_navmesh` as-is inside our project environment fails due to five distinct, compounding blockers:

### Blocker 1: Hard Python ABI Incompatibility (Python 3.10 vs. 3.12)
* **Status**: ❌ **FATAL BLOCKER**
* **Finding**: `third_party/ov_navmesh` only ships with precompiled binaries:
  `PyRecast.cpython-310-x86_64-linux-gnu.so` and `PyRecast.cp310-win_amd64.pyd`.
* **Impact**: Isaac Sim 6.0.1 runs on **Python 3.12.13**. CPython does not maintain ABI compatibility across minor versions for C-extensions (`PyInit_PyRecast` targets the 3.10 runtime). Python 3.12 made breaking structural changes to `PyTypeObject`, `_Py_Dealloc`, and internal frame evaluations.
* **Missing Source**: The repository does **not** include the C++ source code, CMake configuration, or pybind11 setup for `PyRecast`; it only contains the compiled `.so` binary.

### Blocker 2: Missing System Runtime Dependencies
* **Status**: ❌ **FATAL BLOCKER**
* **Finding**: Running `ldd` on `PyRecast.cpython-310-x86_64-linux-gnu.so` reveals:
  ```text
  libSDL2-2.0.so.0 => not found
  ```
* **Impact**: Even in a hypothetical Python 3.10 environment, `ctypes.cdll.LoadLibrary` and runtime import immediately raise `OSError: libSDL2-2.0.so.0: cannot open shared object file`.

### Blocker 3: Architectural Disconnect from the Isaac Sim 6 Behavior System
* **Status**: ❌ **STRUCTURAL ARCHITECTURAL FLAW**
* **Finding**: The wrapper's character locomotion runtime relies on **`omni.anim.behavior.core`** (as detailed in `new_behavior/differences.md` and implemented in `src/hunav_isaac_wrapper/behavior_agent.py`).
* **Impact**: 
  * `omni.anim.behavior.core` **strictly requires** the navmesh generated by NVIDIA's built-in **`omni.anim.navigation.core`**.
  * If `ov_navmesh` builds a Recast navmesh, `omni.anim.behavior.core` **cannot** read it; `get_agent()` returns `None`, and all agents remain paralyzed in T-pose.
  * Conversely, if `ov_navmesh` visualizes its own independent bake, the visual boundaries will not match the actual collision and navigation boundaries evaluated by the agents.

### Blocker 4: Massive CPU & I/O Overhead on Large Scenes
* **Status**: ⚠️ **PERFORMANCE BOTTLENECK**
* **Finding**: In `pyrecast/__init__.py:48-60`, `ov_navmesh` converts stage meshes to an intermediate text `.obj` file:
  ```python
  with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.obj') as temp_file:
      for vertex in vertices:
          temp_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
      for triangle in triangles:
          temp_file.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
  ```
* **Impact**: In production scenarios like `brownstone` (1,268 meshes, hundreds of thousands of polygons), this pure-Python string serialization takes multiple minutes and gigabytes of memory, causing Kit to hang or run out of disk space in `/tmp`.

### Blocker 5: Lack of Headless / Programmatic Scripting Support
* **Status**: ⚠️ **FUNCTIONAL GAP**
* **Finding**: `extension.py` is tightly coupled to `omni.ui` and button callbacks (`clicked_fn`), with no clean Python API for headless batch execution (`--batch`, `launch_hunav_isaac.sh`).

---

## 3. Architectural Strategy: The Native NavMesh Adapter

To solve all five blockers while delivering the **exact same feature set and user experience**, we adopt the **Adapter Pattern**:

```
                                  [ User / Script / GUI ]
                                             │
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │           NavmeshInterface               │
                       │   (Matches original ov_navmesh Core API) │
                       └─────────────────────┬────────────────────┘
                                             │
                       ┌─────────────────────┴────────────────────┐
                       ▼                                          ▼
        ┌─────────────────────────────┐            ┌─────────────────────────────┐
        │     usd_utils (Pure Py)     │            │    Native Navmesh Engine    │
        │   - UsdGeom.Mesh (visual)   │            │ (omni.anim.navigation.core) │
        │   - UsdGeom.BasisCurves     │            │   - INavMesh C++ runtime    │
        │   - UsdGeom.Points          │            │   - GPU / Voxel Recast      │
        │   - UsdPreviewSurface       │            │   - Zero duplicate memory   │
        └─────────────────────────────┘            └─────────────────────────────┘
                       ▲                                          │
                       │           Extract Triangles / Lines      │
                       └──────────────────────────────────────────┘
```

### Why this approach wins:
1. **100% Python 3.12 Compatible**: Zero custom C++ `.so` dependencies. Built purely on `pxr` (USD) and `omni.anim.navigation.core` bindings that already ship with Isaac Sim.
2. **100% Behavioral Synchronization**: The visual mesh and boundary outlines are derived directly from the exact navmesh driving the HuNav agents. What you see is what the agents walk on.
3. **High Performance**: Native baking takes seconds on GPU without any ASCII `.obj` temporary file generation.
4. **Dual Interface**: Operates both as an interactive Omniverse Kit UI extension (`ui.Window`) and as a headless Python module that can be driven via CLI flags or scripts.

---

## 4. 1:1 API & Functional Mapping Specification

| Original `ov_navmesh` API | Original Implementation | Adapter Native Implementation |
|---|---|---|
| `NavmeshInterface(up_axis='Y')` | Custom coordinate flipper | Detects `UsdGeom.GetStageUpAxis(stage)`; handles stage units (m vs cm). |
| `load_mesh(prim)` / `get_selected_prim()` | Scans selection, exports `.obj` | Creates or updates `NavMeshVolume` encompassing target bounds. |
| `build_navmesh(settings={})` | Calls `PyRecast.build_navmesh` | Maps settings to carb settings and executes `inav.start_navmesh_baking_and_wait()`. |
| `get_navmesh_polygons()` / `get_navmesh_triangles()` | Reads Detour polygon buffer | Queries native `navmesh.get_draw_triangles(area=0)` directly into numpy arrays. |
| `get_navmesh_contours()` | Reads Detour raw edge list | Queries native `navmesh.get_draw_lines()` (paired edge vertices). |
| `visualize_navmesh()` / `create_mesh()` | Creates USD Mesh + Material | Reuses `usd_utils.create_mesh()` at `/World/NavMesh/Mesh` with blue `UsdPreviewSurface`. |
| `make_outline()` | Creates `BasisCurves` per edge | Uses `usd_utils.create_curve()` to draw crisp boundary outlines along obstacle edges. |
| `make_walls(height)` | Extrudes boundary triangles | Creates extruded vertical collision/visual quads along boundary edges. |
| `get_random_points(n)` | Calls `Detour::getRandomPoint` | Calls native `navmesh.query_random_point()` and places markers at `/World/NavMesh/Points`. |
| `find_paths([starts], [ends])` | Calls `Detour::findPath` | Calls native `navmesh.query_shortest_path()` and draws ribbon curve at `/World/NavMesh/Path`. |
| UI Window (`extension.py`) | Kit `ui.Window("Navmesh")` | Recreated Kit UI window with identical buttons, styling, and drop targets. |

---

## 5. Phase-by-Phase Implementation Roadmap

The implementation is broken into 5 self-contained, testable phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: USD Geometry & Material Utilities (usd_utils.py)                   │
│          - Port & harden usd_utils.py for Python 3.12 & USD 24+             │
│          - Verify create_mesh, create_curve, create_geompoints              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Phase 2: Native Navmesh Core Engine Adapter (core.py)                       │
│          - Implement NativeNavmeshInterface                                 │
│          - Map Recast settings to carb / omni.anim.navigation.core          │
│          - Bind get_draw_triangles() and get_draw_lines()                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Phase 3: Visualization & Wall Outline Generator                             │
│          - Navmesh visual mesh generation with UsdPreviewSurface material   │
│          - Wall boundary outline curve generation (BasisCurves)             │
│          - Extruded obstacle wall geometry creation                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Phase 4: Query Engine & Path Visualization                                  │
│          - Point-to-point shortest path queries & spline drawing            │
│          - Random navigable point sampling & point-cloud markers            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Phase 5: UI Window & Wrapper Integration                                    │
│          - Interactive Omniverse Kit Extension window                       │
│          - Headless CLI integration (--visualize-navmesh in launcher)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: USD Geometry & Material Utilities
* **Objective**: Provide reliable, standalone USD authoring utilities in Python 3.12 without depending on external compiled libraries.
* **Tasks**:
  1. Migrate [`usd_utils.py`](file:///workspace/Hunav_isaac_wrapper/third_party/ov_navmesh/exts/siborg.create.navmesh/siborg/create/navmesh/usd_utils.py) into `new_behavior/nav_mesh_plugin/usd_utils.py`.
  2. Ensure material binding uses `UsdShade.MaterialBindingAPI` compliant with USD 24+.
  3. Validate `create_mesh()` with transparent `UsdPreviewSurface` shader.
  4. Validate `create_curve()` using `UsdGeom.BasisCurves` (linear basis).
  5. Validate `create_geompoints()` using `UsdGeom.Points`.

### Phase 2: Native Navmesh Core Engine Adapter
* **Objective**: Bridge the high-level `NavmeshInterface` API with `omni.anim.navigation.core`.
* **Tasks**:
  1. Implement `NativeNavmeshInterface` mimicking `third_party/ov_navmesh/exts/.../core.py`.
  2. Implement settings translation:
     * `agentRadius` $\to$ `/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius`, `agentMaxRadius`
     * `agentHeight` $\to$ `/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight`
     * `agentMaxClimb` $\to$ `/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight`
     * `agentMaxSlope` $\to$ `/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope`
     * `cellSize` $\to$ `/exts/omni.anim.navigation.core/navMesh/config/agentSamplingDistance`
  3. Automate `NavMeshVolume` sizing from stage bounding box or selected prim bounding box.
  4. Trigger synchronous bake via `inav.start_navmesh_baking_and_wait()`.

### Phase 3: Visualization & Wall Outline Generator
* **Objective**: Implement exact equivalents of `visualize_navmesh()`, `make_outline()`, and `make_walls()`.
* **Tasks**:
  1. Retrieve raw triangle vertices from `navmesh.get_draw_triangles(0)`.
  2. Reshape into vertex list and triangle index list `[i, i+1, i+2]`.
  3. Author mesh at `/World/NavMesh/Mesh` with translucent cyan color (`(0.05, 0.77, 0.95)`, opacity `0.6`).
  4. Retrieve raw boundary edges from `navmesh.get_draw_lines()`.
  5. Format lines into curve point segments and author `UsdGeom.BasisCurves` under `/World/NavMesh/Outlines`.
  6. Implement extruded wall generator for obstacle bounding.

### Phase 4: Query Engine & Pathfinding Visualizer
* **Objective**: Provide path generation and point sampling on stage.
* **Tasks**:
  1. Implement `get_random_points(num_points)` using `navmesh.query_random_point()`.
  2. Visualize sampled points on stage at `/World/NavMesh/Points`.
  3. Implement `find_paths(starts, ends)` using `navmesh.query_shortest_path()`.
  4. Convert `INavMeshPath` waypoints into continuous curve splines at `/World/NavMesh/Path`.

### Phase 5: UI Window & Wrapper Integration
* **Objective**: Expose the tool both interactively and through the simulation launch pipeline.
* **Tasks**:
  1. Implement `NavmeshWindow` using `omni.ui` replicating the buttons and input fields of `extension.py`.
  2. Expose a one-call programmatic API: `build_and_visualize_navmesh(stage, ...)` for headless or batch scripts.
  3. Connect an optional CLI argument (e.g. `--visualize-navmesh`) to `launch_hunav_isaac.sh` and `src/scripts/main.py`.

---

## 6. Detailed Technical Adapter Design

### 6.1 Component Architecture

```
/workspace/Hunav_isaac_wrapper/new_behavior/nav_mesh_plugin/
├── README.md               <-- This comprehensive design & roadmap document
├── __init__.py             <-- Package exports
├── core.py                 <-- NativeNavmeshInterface (matches original core.py API)
├── usd_utils.py            <-- USD authoring helpers (Mesh, Curves, Points, Materials)
├── ui_window.py            <-- omni.ui Interactive control panel
└── sample_visualize.py     <-- Standalone executable test script
```

### 6.2 Settings Translation Reference

The carb settings used by Isaac Sim's native navigation system operate in **stage units** (meters in the wrapper):

```python
RECAST_TO_CARB_SETTINGS_MAP = {
    "cellSize": "/exts/omni.anim.navigation.core/navMesh/config/agentSamplingDistance",
    "agentHeight": "/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight",
    "agentRadius": "/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius",
    "agentMaxClimb": "/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight",
    "agentMaxSlope": "/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope",
    "regionMinSize": "/exts/omni.anim.navigation.core/navMesh/config/agentMinIslandRadius",
}
```

### 6.3 Core Adapter Implementation Blueprint

```python
import numpy as np
from pxr import Gf, UsdGeom
import carb.settings
import omni.anim.navigation.core as nav
import omni.usd

from . import usd_utils

class NativeNavmeshInterface:
    """1:1 functional drop-in replacement for ov_navmesh Core.NavmeshInterface.
    Backed natively by Isaac Sim 6's omni.anim.navigation.core.
    """
    def __init__(self, up_axis=None):
        self.stage = omni.usd.get_context().get_stage()
        self.inav = nav.acquire_interface()
        self._navmesh = None
        self.built = False
        self.contour_verts = []
        self.contour_edges = []
        self.wall_outline = []
        
        # Up axis detection
        stage_up = UsdGeom.GetStageUpAxis(self.stage) if self.stage else "Z"
        self.z_up = (stage_up == UsdGeom.Tokens.z) if up_axis is None else (up_axis == "Z")

    def build_navmesh(self, settings=None):
        """Configure carb settings and bake the navmesh synchronously."""
        carb_settings = carb.settings.get_settings()
        if settings:
            if "cellSize" in settings:
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentSamplingDistance", float(settings["cellSize"]))
            if "agentHeight" in settings:
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMinHeight", float(settings["agentHeight"]))
            if "agentRadius" in settings:
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxRadius", float(settings["agentRadius"]))
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius", float(settings["agentRadius"]) * 0.4)
            if "agentMaxClimb" in settings:
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", float(settings["agentMaxClimb"]))
            if "agentMaxSlope" in settings:
                carb_settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope", float(settings["agentMaxSlope"]))

        # Trigger bake
        success = self.inav.start_navmesh_baking_and_wait()
        if success:
            self._navmesh = self.inav.get_navmesh()
            self.built = self._navmesh is not None
        return self.built

    def get_navmesh_polygons(self, area=0):
        """Retrieve triangles for the walkable navmesh area."""
        if not self.built or not self._navmesh:
            return np.empty((0, 3)), np.empty((0, 3))
        
        draw_verts = self._navmesh.get_draw_triangles(area)
        if not draw_verts:
            return np.empty((0, 3)), np.empty((0, 3))
        
        verts = np.array(draw_verts, dtype=np.float32).reshape(-1, 3)
        faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
        return verts, faces

    def get_navmesh_contours(self):
        """Retrieve paired boundary lines."""
        if not self.built or not self._navmesh:
            return np.empty((0, 3)), []
        
        draw_lines = self._navmesh.get_draw_lines()
        if not draw_lines:
            return np.empty((0, 3)), []
        
        verts = np.array(draw_lines, dtype=np.float32).reshape(-1, 3)
        edges = [[i, i + 1] for i in range(0, len(verts) - 1, 2)]
        self.contour_verts = verts
        self.contour_edges = edges
        return self.contour_verts, self.contour_edges

    def visualize_navmesh(self, prim_path="/World/NavMesh/Mesh", color=(0.05, 0.77, 0.95), opacity=0.6):
        """Create a translucent USD mesh representing the walkable surface."""
        verts, faces = self.get_navmesh_polygons()
        if len(verts) == 0:
            print("[NavMeshAdapter] No navmesh geometry available to visualize.")
            return None
        return usd_utils.create_mesh(prim_path, verts.flatten(), faces, color=Gf.Vec3f(*color), opacity=opacity)

    def make_outline(self, prim_prefix="/World/NavMesh/Outlines/Outline", color=(0.9, 0.9, 0.2), width=0.05):
        """Draw BasisCurves along boundary edges."""
        verts, edges = self.get_navmesh_contours()
        if len(verts) == 0:
            return []
        
        lines = []
        for idx, edge in enumerate(edges):
            A = verts[edge[0]]
            B = verts[edge[1]]
            curve_path = f"{prim_prefix}_{idx}"
            usd_utils.create_curve([tuple(A), tuple(B)], prim_path=curve_path, color=color, width=np.array([width]))
            lines.append(curve_path)
        return lines

    def get_random_points(self, num_points):
        """Query random navigable points."""
        if not self.built or not self._navmesh:
            return None
        points = []
        desc = nav.NavAgentDesc()
        for _ in range(num_points):
            pt = carb.Float3(0, 0, 0)
            if self._navmesh.query_random_point("adapter", pt, desc):
                points.append((pt.x, pt.y, pt.z))
        return np.array(points)

    def find_paths(self, starts, ends):
        """Compute shortest path and return array of waypoints."""
        if not self.built or not self._navmesh:
            return np.empty((0, 3))
        
        start_pt = carb.Float3(float(starts[0][0]), float(starts[0][1]), float(starts[0][2]))
        end_pt = carb.Float3(float(ends[0][0]), float(ends[0][1]), float(ends[0][2]))
        desc = nav.NavAgentDesc()
        path_obj = self._navmesh.query_shortest_path(start_pt, end_pt, desc, straighten=True)
        if not path_obj or path_obj.get_point_count() == 0:
            return np.empty((0, 3))
        
        pts = [path_obj.get_point(i) for i in range(path_obj.get_point_count())]
        return np.array([(p.x, p.y, p.z) for p in pts])
```

---

## 7. Verification & Acceptance Criteria

To declare full feature parity with the original `ov_navmesh` extension, the adapter must pass the following validation matrix:

| Test Case | Scenario | Expected Behavior |
|---|---|---|
| **Mesh Assignment** | Bounding box of `warehouse` or `brownstone` | `NavMeshVolume` dynamically encompasses the walkable region. |
| **Bake Parity** | Trigger `build_navmesh()` | Bakes within < 5 seconds; `navmesh.get_area_count()` $\ge 1$. |
| **Visual Mesh Creation** | Trigger `visualize_navmesh()` | `/World/NavMesh/Mesh` prim exists in stage, displays as semi-transparent blue surface. |
| **Wall Boundary Outlines** | Trigger `make_outline()` | `/World/NavMesh/Outlines` contains linear `BasisCurves` outlining walkable perimeter and obstacles. |
| **Random Point Sampling** | Call `get_random_points(10)` | Returns 10 points on the surface; points display as red `UsdGeom.Points`. |
| **Pathfinding Spline** | Call `find_paths([start], [end])` | Computes path around obstacles; draws green `BasisCurves` ribbon on the stage. |
| **UI Parity** | Open `Navmesh` UI Window in WebRTC client | All 6 action buttons respond; start/end prim fields accept drag-drop. |
| **Agent Locomotion Sync** | Run simulation with HuNav agents | Agents walk across the visual navmesh without clipping or path desync. |


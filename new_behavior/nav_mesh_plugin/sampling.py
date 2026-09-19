"""
sampling.py

Navmesh queries the scenario authoring layer needs: snap a point onto the
walkable surface, sample well-separated points on it, and decide whether two
points are connected.

All three go through `omni.anim.navigation.core` rather than any geometry of
our own, so they answer for the mesh the agents are actually steered on.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core import NavmeshInterface

Point3 = Tuple[float, float, float]


class NavmeshSampler:
    """Query helper over an already-baked navmesh.

    Deliberately does not bake. The plugin's own settings differ materially
    from the ones the simulator bakes with -- different agent radius, step
    height, slope and island radius -- so a bake triggered from here would
    replace the mesh under whatever else is using it. Bake through
    `BehaviorAgentDriver`, then point this at the result.
    """

    def __init__(self, adapter: NavmeshInterface, seed: Optional[int] = None):
        self.adapter = adapter
        self._rng = random.Random(seed)

    # --- availability ---------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._navmesh() is not None

    def _navmesh(self):
        nav = self.adapter._navmesh
        if nav is None and self.adapter.inav is not None:
            nav = self.adapter.inav.get_navmesh()
            self.adapter._navmesh = nav
            self.adapter.built = nav is not None
        return nav

    # --- queries --------------------------------------------------------

    def snap_with_island(
        self, point: Sequence[float]
    ) -> Tuple[Optional[Point3], int]:
        """The nearest walkable point and the id of the island it lies on.

        `query_closest_point` returns a `(point, island_id)` tuple, and returns
        `(None, -1)` when nothing is in range. The island id is the navmesh's
        own connectivity component, so two points share one exactly when a path
        between them exists -- which makes it a far cheaper reachability test
        than running the pathfinder.
        """
        nav = self._navmesh()
        if nav is None:
            return None, -1

        target = (float(point[0]), float(point[1]), float(point[2]))
        try:
            result = nav.query_closest_point(target=target)
        except Exception as exc:
            print(f"[NavmeshSampler] closest-point query failed: {exc}")
            return None, -1

        if not isinstance(result, tuple) or len(result) != 2:
            # Older bindings returned the bare point.
            return _as_point(result), -1

        raw_point, island_id = result
        return _as_point(raw_point), int(island_id)

    def snap(self, point: Sequence[float]) -> Optional[Point3]:
        """The nearest point on the walkable surface, or None if there is none.

        Uses the runtime's own `query_closest_point`. Projecting onto triangles
        by hand would answer for our reconstruction of the mesh rather than for
        the mesh, and the two differ at exactly the thresholds that matter.
        """
        return self.snap_with_island(point)[0]

    def island_of(self, point: Sequence[float]) -> int:
        """Which connectivity component `point` belongs to, or -1 if none."""
        return self.snap_with_island(point)[1]

    def on_navmesh(self, point: Sequence[float], tolerance: float = 0.5) -> bool:
        """Whether `point` is within `tolerance` metres of walkable surface."""
        snapped = self.snap(point)
        if snapped is None:
            return False
        return math.dist((point[0], point[1]), (snapped[0], snapped[1])) <= tolerance

    def reachable(self, start: Sequence[float], end: Sequence[float]) -> bool:
        """Whether a navmesh path exists between two points.

        `agentMinIslandRadius` leaves several disconnected islands on a map the
        size of brownstone, and random sampling spreads points across all of
        them by area. Without this check an agent can be placed where no route
        to its goals exists, and the only symptom is that it never arrives.

        Island ids answer this exactly and without running the pathfinder;
        pathfinding is the fallback for bindings that do not report them.
        """
        start_point, start_island = self.snap_with_island(start)
        end_point, end_island = self.snap_with_island(end)

        if start_point is None or end_point is None:
            return False
        if start_island >= 0 and end_island >= 0:
            return start_island == end_island

        path = self.adapter.find_paths([list(start_point)], [list(end_point)])
        return len(path) >= 2

    def path_length(self, start: Sequence[float], end: Sequence[float]) -> Optional[float]:
        """Walking distance along the navmesh, or None if unreachable.

        The island check comes first: `query_shortest_path` across a gap
        returns a partial path to the boundary rather than nothing, which would
        read as a short walk instead of an impossible one.
        """
        if not self.reachable(start, end):
            return None

        path = self.adapter.find_paths([list(start)], [list(end)])
        if len(path) < 2:
            return None
        deltas = np.diff(np.asarray(path, dtype=np.float64), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    # --- sampling -------------------------------------------------------

    def sample_points(
        self,
        count: int,
        min_separation: float = 2.0,
        existing: Optional[Iterable[Sequence[float]]] = None,
        max_tries_per_point: int = 200,
    ) -> List[Point3]:
        """`count` walkable points, each at least `min_separation` from the rest.

        `query_random_point` takes no clearance argument, so separation is
        enforced by rejection. Returns fewer points than asked rather than
        looping forever, and says so -- a caller that silently accepts a short
        list is how a scenario ends up with four agents where eight were
        requested.
        """
        nav = self._navmesh()
        if nav is None:
            print("[NavmeshSampler] navmesh not built; cannot sample.")
            return []

        chosen: List[Point3] = [tuple(map(float, p)) for p in (existing or [])]
        start_index = len(chosen)

        for _ in range(count):
            placed = False
            for _ in range(max_tries_per_point):
                try:
                    candidate = _as_point(nav.query_random_point())
                except Exception as exc:
                    print(f"[NavmeshSampler] random-point query failed: {exc}")
                    candidate = None
                if candidate is None:
                    continue
                if all(
                    math.dist(candidate[:2], other[:2]) >= min_separation
                    for other in chosen
                ):
                    chosen.append(candidate)
                    placed = True
                    break
            if not placed:
                break

        produced = chosen[start_index:]
        if len(produced) < count:
            print(
                f"[NavmeshSampler] only placed {len(produced)}/{count} points at "
                f"{min_separation:.2f} m separation; the walkable area is too "
                "small or too fragmented for that many."
            )
        return produced

    def sample_connected_points(
        self,
        count: int,
        min_separation: float = 2.0,
        anchor: Optional[Sequence[float]] = None,
        max_tries_per_point: int = 200,
    ) -> List[Point3]:
        """Like `sample_points`, but every point is reachable from `anchor`.

        Without an anchor the first accepted point becomes one, which keeps the
        whole set on a single connected island.
        """
        nav = self._navmesh()
        if nav is None:
            return []

        chosen: List[Point3] = []
        root = tuple(map(float, anchor)) if anchor is not None else None

        for _ in range(count):
            placed = False
            for _ in range(max_tries_per_point):
                try:
                    candidate = _as_point(nav.query_random_point())
                except Exception as exc:
                    print(f"[NavmeshSampler] random-point query failed: {exc}")
                    continue
                if candidate is None:
                    continue
                if any(
                    math.dist(candidate[:2], other[:2]) < min_separation
                    for other in chosen
                ):
                    continue
                if root is None:
                    root = candidate
                elif not self.reachable(root, candidate):
                    continue
                chosen.append(candidate)
                placed = True
                break
            if not placed:
                break

        if len(chosen) < count:
            print(
                f"[NavmeshSampler] only placed {len(chosen)}/{count} connected "
                f"points at {min_separation:.2f} m separation."
            )
        return chosen


def _as_point(raw) -> Optional[Point3]:
    """Normalise a carb Float3, a tuple, or None into a plain 3-tuple."""
    if raw is None:
        return None
    if hasattr(raw, "x"):
        return (float(raw.x), float(raw.y), float(raw.z))
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, IndexError, ValueError):
        return None

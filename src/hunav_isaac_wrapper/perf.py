#!/usr/bin/env python3
"""
perf.py

Per-frame timing for the simulation loop, gated on ``HUNAV_PERF=1``.

Why this exists
---------------
Nothing in this wrapper measured where a frame goes, so every statement about
frame rate was a guess. The guesses are easy to get wrong here, because the
expensive-looking layer is not the expensive one: ``behavior_agent.py`` issues
about five pybind calls per agent per tick, while ``hunav_manager`` casts 450
PhysX rays per agent per tick and then blocks the physics thread on a ROS
service round trip.

What it measures
----------------
Named spans, nested, accumulated per rendered frame. A span that fires several
times inside one frame (``_on_physics_step`` runs once per PhysX substep, ten
times per frame for the Go2) is summed rather than overwritten, so the reported
number is always "cost per rendered frame" and the columns can be compared
directly against the frame budget.

The single most useful number the tree produces is a subtraction:
``world_step - physx_cb_total`` is the renderer plus the PhysX solver, i.e.
everything that is *not* our Python. It is reported as a derived row because it
separates "the GPU is the problem" from "our Python is the problem", and those
two have no fixes in common.

Cost when disabled
------------------
``span()`` returns a shared no-op object, so a disabled span is one attribute
lookup, one call, and two empty method calls. The call sites stay readable and
can be left in place permanently.

Usage
-----
    from .perf import get_profiler

    prof = get_profiler()
    while running:
        with prof.frame():                 # also flushes the previous frame
            with prof.span("world_step"):
                world.step(render=True)
            ...

Environment
-----------
``HUNAV_PERF=1``          enable.
``HUNAV_PERF_EVERY``      seconds between reports (default 2.0).
``HUNAV_PERF_HISTORY``    frames kept for percentiles (default 900).
``HUNAV_PERF_DUMP``       path for the JSON dump; default is
                          ``<repo>/debug/perf_<timestamp>.json``.
"""

import atexit
import json
import os
import time
from collections import deque


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


class _NullSpan:
    """What span() hands back when profiling is off."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_SPAN = _NullSpan()


class _Span:
    __slots__ = ("prof", "name", "t0")

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        prof = self.prof
        parent = prof._stack[-1] if prof._stack else None
        # The parent is re-recorded every time rather than fixed on first
        # sighting. send_agents_msg() is called once before the loop starts, so
        # first sighting would permanently record msg_build/svc_call as
        # top-level spans; they would then be counted both on their own and
        # inside world_step, and the residual would go negative.
        if self.name not in prof._parent:
            prof._order.append(self.name)
        prof._parent[self.name] = parent
        prof._stack.append(self.name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        prof = self.prof
        prof._stack.pop()
        acc = prof._frame_acc
        acc[self.name] = acc.get(self.name, 0.0) + elapsed
        counts = prof._frame_calls
        counts[self.name] = counts.get(self.name, 0) + 1
        return False


class _FrameSpan:
    """The outer per-iteration span; flushes the frame on exit."""

    __slots__ = ("prof", "t0")

    def __init__(self, prof):
        self.prof = prof
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.prof._end_frame(time.perf_counter() - self.t0)
        return False


class FrameProfiler:
    """Accumulates named spans per frame and reports percentiles."""

    # Derived rows: (label, minuend, subtrahend). Reported after the tree.
    DERIVED = (
        ("render+physx_solver", "world_step", "physx_cb_total"),
    )

    def __init__(self, enabled=None, report_every=None, history=None):
        self.enabled = _env_flag("HUNAV_PERF") if enabled is None else enabled
        self.report_every = (
            float(os.environ.get("HUNAV_PERF_EVERY", "2.0"))
            if report_every is None
            else report_every
        )
        maxlen = (
            int(os.environ.get("HUNAV_PERF_HISTORY", "900"))
            if history is None
            else history
        )
        self._maxlen = maxlen

        self._stack = []
        self._parent = {}
        self._order = []
        self._frame_acc = {}
        self._frame_calls = {}
        self._frame_counters = {}

        self._history = {}
        self._call_history = {}
        self._counter_history = {}
        self._frame_times = deque(maxlen=maxlen)

        self._frames = 0
        self._last_report = time.perf_counter()
        self._started = time.perf_counter()

        # Real-time factor tracking: sim seconds advanced per wall second.
        self._sim_time = None
        self._sim_time_at_mark = None
        self._wall_at_mark = None
        self._rtf = None
        self._dumped = False

        if self.enabled:
            self._dump_path = os.environ.get("HUNAV_PERF_DUMP") or self._default_dump()
            atexit.register(self.dump)
            print(
                f"[perf] enabled: reporting every {self.report_every:g}s, "
                f"history {maxlen} frames, dump -> {self._dump_path}",
                flush=True,
            )
        else:
            self._dump_path = None

    # ---------------------------------------------------------------- public

    def span(self, name):
        """Time a named region. Nests; sums across calls within one frame."""
        if not self.enabled:
            return _NULL_SPAN
        return _Span(self, name)

    def frame(self):
        """Time one loop iteration and flush the accumulated spans on exit."""
        if not self.enabled:
            return _NULL_SPAN
        return _FrameSpan(self)

    def count(self, name, n=1):
        """Record a per-frame integer, e.g. the number of raycasts issued."""
        if not self.enabled:
            return
        self._frame_counters[name] = self._frame_counters.get(name, 0) + n

    def note_sim_time(self, sim_time):
        """Feed the simulated clock so the real-time factor can be reported.

        A frame rate on its own cannot tell a fast simulation from a slow one:
        30 FPS at 0.6x real time and 30 FPS at 1.0x real time look identical in
        an FPS counter and are completely different results.
        """
        if not self.enabled or sim_time is None:
            return
        self._sim_time = float(sim_time)
        if self._sim_time_at_mark is None:
            self._sim_time_at_mark = self._sim_time
            self._wall_at_mark = time.perf_counter()

    # --------------------------------------------------------------- internal

    def _end_frame(self, elapsed):
        self._frames += 1
        self._frame_times.append(elapsed)

        for name in self._order:
            hist = self._history.get(name)
            if hist is None:
                hist = self._history[name] = deque(maxlen=self._maxlen)
                self._call_history[name] = deque(maxlen=self._maxlen)
            hist.append(self._frame_acc.get(name, 0.0))
            self._call_history[name].append(self._frame_calls.get(name, 0))

        for name, value in self._frame_counters.items():
            hist = self._counter_history.get(name)
            if hist is None:
                hist = self._counter_history[name] = deque(maxlen=self._maxlen)
            hist.append(value)

        self._frame_acc.clear()
        self._frame_calls.clear()
        self._frame_counters.clear()

        now = time.perf_counter()
        if now - self._last_report >= self.report_every:
            self._update_rtf(now)
            self._report()
            self._last_report = now

    def _update_rtf(self, now):
        if self._sim_time is None or self._wall_at_mark is None:
            return
        wall = now - self._wall_at_mark
        if wall <= 0.0:
            return
        self._rtf = (self._sim_time - self._sim_time_at_mark) / wall
        # Re-anchor so the factor reflects the recent window, not the whole run
        # (startup stalls would otherwise hold it down forever).
        self._sim_time_at_mark = self._sim_time
        self._wall_at_mark = now

    @staticmethod
    def _pct(sorted_vals, q):
        if not sorted_vals:
            return 0.0
        idx = int(q * (len(sorted_vals) - 1))
        return sorted_vals[idx]

    def _stats(self, samples):
        vals = sorted(samples)
        n = len(vals)
        return {
            "n": n,
            "mean": (sum(vals) / n) if n else 0.0,
            "p50": self._pct(vals, 0.50),
            "p95": self._pct(vals, 0.95),
            "max": vals[-1] if n else 0.0,
        }

    def _children(self, parent):
        return [n for n in self._order if self._parent.get(n) == parent]

    def _report(self):
        if not self._frame_times:
            return
        frame = self._stats(self._frame_times)
        mean_ms = frame["mean"] * 1000.0
        fps = (1.0 / frame["mean"]) if frame["mean"] > 0 else 0.0

        lines = []
        lines.append("")
        header = (
            f"[perf] {fps:6.2f} FPS  frame p50 {frame['p50']*1000:7.2f} ms  "
            f"p95 {frame['p95']*1000:7.2f}  max {frame['max']*1000:7.2f}  "
            f"({len(self._frame_times)} frames, {self._frames} total)"
        )
        if self._rtf is not None:
            header += f"  realtime x{self._rtf:.3f}"
        lines.append(header)
        lines.append(
            f"[perf] {'span':<28}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}"
            f"{'calls/f':>9}{'%frame':>8}"
        )

        def emit(name, depth):
            st = self._stats(self._history[name])
            calls = self._call_history[name]
            mean_calls = (sum(calls) / len(calls)) if calls else 0.0
            share = (st["mean"] / frame["mean"] * 100.0) if frame["mean"] > 0 else 0.0
            label = ("  " * depth) + name
            lines.append(
                f"[perf] {label:<28}{st['p50']*1000:9.2f}{st['p95']*1000:9.2f}"
                f"{st['max']*1000:9.2f}{mean_calls:9.1f}{share:7.1f}%"
            )
            for child in self._children(name):
                emit(child, depth + 1)

        top = self._children(None)
        for name in top:
            emit(name, 0)

        accounted = sum(self._stats(self._history[n])["mean"] for n in top)
        residual = frame["mean"] - accounted
        res_share = (residual / frame["mean"] * 100.0) if frame["mean"] > 0 else 0.0
        lines.append(
            f"[perf] {'(unaccounted)':<28}{'':>9}{'':>9}"
            f"{residual*1000:9.2f}{'':>9}{res_share:7.1f}%"
        )

        for label, minuend, subtrahend in self.DERIVED:
            if minuend in self._history and subtrahend in self._history:
                a = self._stats(self._history[minuend])["mean"]
                b = self._stats(self._history[subtrahend])["mean"]
                d = a - b
                share = (d / frame["mean"] * 100.0) if frame["mean"] > 0 else 0.0
                lines.append(
                    f"[perf] {('= ' + label):<28}{'':>9}{'':>9}"
                    f"{d*1000:9.2f}{'':>9}{share:7.1f}%"
                )

        for name in sorted(self._counter_history):
            hist = self._counter_history[name]
            mean = (sum(hist) / len(hist)) if hist else 0.0
            lines.append(f"[perf] counter {name}: {mean:.0f}/frame")

        lines.append(f"[perf] budget: 33.3 ms/frame = 30 FPS   50.0 ms/frame = 20 FPS")
        print("\n".join(lines), flush=True)

    def _default_dump(self):
        # debug/ is this repo's established artifacts directory.
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, "..", ".."))
        out = os.path.join(repo, "debug")
        try:
            os.makedirs(out, exist_ok=True)
        except OSError:
            out = "."
        return os.path.join(out, f"perf_{time.strftime('%Y%m%d_%H%M%S')}.json")

    def dump(self):
        """Write raw per-frame samples so two runs can be diffed."""
        if not self.enabled or not self._frame_times or self._dump_path is None:
            return
        if self._dumped:  # atexit fires after an explicit dump(); write once
            return
        self._dumped = True
        payload = {
            "frames": self._frames,
            "wall_seconds": time.perf_counter() - self._started,
            "realtime_factor": self._rtf,
            "argv_env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("HUNAV_", "LIVESTREAM", "HEADLESS"))
            },
            "frame_ms": [t * 1000.0 for t in self._frame_times],
            "parents": dict(self._parent),
            "spans": {
                name: {
                    "ms": [v * 1000.0 for v in self._history[name]],
                    "calls": list(self._call_history[name]),
                }
                for name in self._order
                if name in self._history
            },
            "counters": {k: list(v) for k, v in self._counter_history.items()},
        }
        try:
            with open(self._dump_path, "w") as fh:
                json.dump(payload, fh)
            print(f"[perf] wrote {self._dump_path}", flush=True)
        except OSError as exc:
            print(f"[perf] could not write {self._dump_path}: {exc}", flush=True)


_PROFILER = None


def get_profiler():
    """The process-wide profiler. Cheap and safe to call from anywhere."""
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = FrameProfiler()
    return _PROFILER

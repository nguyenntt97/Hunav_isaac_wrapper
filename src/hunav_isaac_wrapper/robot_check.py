#!/usr/bin/env python3
"""
robot_check.py

Articulation-integrity check for the Go2. Untracked diagnostic.

A detached limb is invisible to every check this wrapper has:
``_warn_if_robot_moved`` allows 3 m of drift and looks only at the base's XY,
so a leg can leave the robot entirely and the run still prints "Robot settled".
This measures where each link actually is relative to the base, which a
separated link cannot hide from.

Nominal Go2 geometry: the body is ~0.70 x 0.31 m and a fully extended leg
reaches ~0.45 m from the base origin, so every link should sit well inside
~0.6 m. A link beyond DETACH_M has left the robot.
"""

import json
import math
import os
import time

DETACH_M = 1.0
SUSPECT_M = 0.6


def _positions_via_usdrt(stage_id, root):
    """World positions from Fabric, which is where PhysX writes link poses."""
    from usdrt import Usd as RtUsd, Sdf as RtSdf

    rt = RtUsd.Stage.Attach(stage_id)
    out = {}
    for prim in rt.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        for attr in ("_worldPosition", "omni:fabric:worldMatrix", "xformOp:translate"):
            a = prim.GetAttribute(attr) if prim.HasAttribute(attr) else None
            if a is None:
                continue
            v = a.Get()
            if v is None:
                continue
            if attr == "omni:fabric:worldMatrix":
                out[path] = (float(v[3][0]), float(v[3][1]), float(v[3][2]))
            else:
                out[path] = (float(v[0]), float(v[1]), float(v[2]))
            break
    return out


def _positions_via_usd(stage, root):
    from pxr import UsdGeom, Usd

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(root) or not prim.IsA(UsdGeom.Xformable):
            continue
        m = cache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        out[path] = (float(t[0]), float(t[1]), float(t[2]))
    return out


def check(stage, root="/World/Go2", label="", out_dir=None):
    """Report every link's distance from the articulation root."""
    import omni.usd

    stage_id = omni.usd.get_context().get_stage_id()
    source = "usdrt"
    try:
        pos = _positions_via_usdrt(stage_id, root)
        if len(pos) < 3:
            raise RuntimeError(f"only {len(pos)} prims from Fabric")
    except Exception as exc:
        print(f"[robot-check] Fabric read failed ({exc}); falling back to USD")
        source = "usd"
        pos = _positions_via_usd(stage, root)

    if not pos:
        print(f"[robot-check] no prims under {root}")
        return None

    base = pos.get(root) or pos.get(root + "/base")
    if base is None:
        base = min(pos.items(), key=lambda kv: kv[0].count("/"))[1]

    rows = []
    for path, p in sorted(pos.items()):
        d = math.dist(p, base)
        rows.append({"prim": path, "pos": p, "dist_from_base": d})

    worst = max(rows, key=lambda r: r["dist_from_base"])
    detached = [r for r in rows if r["dist_from_base"] > DETACH_M]
    suspect = [r for r in rows if SUSPECT_M < r["dist_from_base"] <= DETACH_M]

    print(f"\n[robot-check] {label} source={source} prims={len(rows)}")
    print(f"[robot-check] base at ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})")
    print(f"[robot-check] farthest link: {worst['dist_from_base']:.3f} m "
          f"-- {worst['prim'].rsplit('/', 1)[-1]}")
    if detached:
        print(f"[robot-check] *** {len(detached)} LINK(S) DETACHED (> {DETACH_M} m) ***")
        for r in detached[:8]:
            print(f"[robot-check]     {r['dist_from_base']:8.3f} m  {r['prim']}")
    elif suspect:
        print(f"[robot-check] {len(suspect)} link(s) stretched ({SUSPECT_M}-{DETACH_M} m)")
        for r in suspect[:8]:
            print(f"[robot-check]     {r['dist_from_base']:8.3f} m  {r['prim']}")
    else:
        print(f"[robot-check] INTACT -- every link within {SUSPECT_M} m of the base")

    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"robotcheck_{label}_{time.strftime('%H%M%S')}.json")
            with open(p, "w") as fh:
                json.dump({"label": label, "source": source, "base": base,
                           "detached": len(detached), "max_dist": worst["dist_from_base"],
                           "links": rows}, fh, indent=1)
            print(f"[robot-check] wrote {p}")
        except OSError as exc:
            print(f"[robot-check] could not write report: {exc}")

    return {"detached": len(detached), "max_dist": worst["dist_from_base"]}

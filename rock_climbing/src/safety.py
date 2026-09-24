"""Judge, landing safety and silent-feet footwork, read off a finished run.

Works on a run folder (poses.json, holds.json, climb.json, run.json) plus the
floor edge cached by the main pipeline. Everything here is pixel space
(608x1080 for a portrait clip) so distances are isotropic; lengths are then
expressed in body-lengths so the numbers travel between climbers and cameras.

Thresholds come from coaching and injury literature (see report/CONTRACT.md
and the README section); they are heuristics, tune them on your own footage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

# COCO-17
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
MIN_SCORE = 0.3
TOE_OFFSET = 0.02            # of frame height, ankle -> toe (matches config.ANKLE_TO_TOE_OFFSET)
FLOOR_CLEARANCE = 0.012      # of frame height (matches the main pipeline)

# judge
TOP_CONTROL_S = 1.0          # both hands on the top hold, still, for this long
ZONE_TOUCH_S = 0.3
LIFTOFF_S = 0.5              # feet clear of the floor this long = an attempt
# falls
FALL_MIN_DROP = 0.35         # body-lengths the hips must drop
FALL_WINDOW_S = 1.5
STIFF_KNEE = 150.0           # deg; above this at impact = stiff-legged landing
GOOD_KNEE = 120.0
FEET_SYNC_FRAMES = 3
# feet
FOOT_REACH = 0.30            # shin-lengths from the hold outline counts as on it
MIN_PLACEMENT_S = 0.25
SETTLE_S = 0.2


@dataclass
class Track:
    xy: np.ndarray       # (n, 17, 2) pixels, nan where missing
    score: np.ndarray    # (n, 17)
    fps: float
    size: tuple[int, int]

    @property
    def n(self):
        return len(self.xy)


def load_track(run: Path) -> Track:
    P = json.loads((run / "poses.json").read_text())
    R = json.loads((run / "run.json").read_text())
    tid = (R.get("pose") or {}).get("track_id", 1)
    w, h, n = P["video_width"], P["video_height"], P["video_nframes"]
    xy = np.full((n, 17, 2), np.nan)
    sc = np.zeros((n, 17))
    area = np.zeros(n)
    for it in P["content"]["items"]:
        if it["track_id"] != tid:
            continue
        f = it["frame_id"]
        a = it["bbox_xywh"][2] * it["bbox_xywh"][3]
        if a < area[f]:
            continue
        area[f] = a
        k = np.array(it["kpts_xy"], float)
        s = np.array(it["kpts_score"], float)
        ok = ~((k[:, 0] == 0) & (k[:, 1] == 0)) & (s >= MIN_SCORE)
        xy[f] = np.nan
        xy[f][ok] = k[ok] * [w, h]
        sc[f] = s * ok
    # light smoothing along time, only inside runs of valid frames
    for j in range(17):
        for d in range(2):
            col = xy[:, j, d]
            valid = np.isfinite(col)
            if valid.sum() < 5:
                continue
            filled = np.interp(np.arange(n), np.flatnonzero(valid), col[valid])
            sm = gaussian_filter1d(filled, 1.0)
            xy[:, j, d] = np.where(valid, sm, np.nan)
    return Track(xy, sc, float(P["video_fps"]), (w, h))


def load_floor(cache_dir: Path, size) -> np.ndarray | None:
    """The floor's top edge as pixel y per pixel column, from the newest floor cache."""
    files = sorted(cache_dir.glob("floor.*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    edge = [np.nan if v is None else v for v in json.loads(files[-1].read_text())["edge"]]
    edge = np.array(edge, float)
    w, h = size
    xs = np.linspace(0, w - 1, len(edge))
    good = np.isfinite(edge)
    if not good.any():
        return None
    return np.interp(np.arange(w), xs[good], edge[good]) * h


def _mid(a, b):
    return (a + b) / 2


def _angle(a, b, c):
    """Angle at b, degrees."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not np.isfinite(n) or n == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def body_scale(t: Track) -> dict:
    xy = t.xy
    torso = np.linalg.norm(_mid(xy[:, LSH], xy[:, RSH]) - _mid(xy[:, LHIP], xy[:, RHIP]), axis=1)
    thigh = np.linalg.norm(xy[:, [LHIP, RHIP]] - xy[:, [LKN, RKN]], axis=2)
    shin = np.linalg.norm(xy[:, [LKN, RKN]] - xy[:, [LAN, RAN]], axis=2)
    torso, thigh, shin = (float(np.nanmedian(v)) for v in (torso, thigh, shin))
    return {"torso": torso, "thigh": thigh, "shin": shin,
            "body": torso * 1.35 + thigh + shin}   # head+neck ~ 0.35 torso


class Holds:
    def __init__(self, holds: list[dict], size):
        w, h = size
        self.ids = [hd["id"] for hd in holds]
        self.poly = {hd["id"]: (np.array(hd["polygon"]) * [w, h]).astype(np.float32) for hd in holds}
        self.centroid = {i: p.mean(0) for i, p in self.poly.items()}
        self.radius = {i: float(np.sqrt(cv2.contourArea(p) / np.pi)) for i, p in self.poly.items()}

    def dist(self, hid, pt) -> float:
        """Distance from pt to the hold outline, 0 inside."""
        d = cv2.pointPolygonTest(self.poly[hid], (float(pt[0]), float(pt[1])), True)
        return max(0.0, -d)

    def nearest(self, pt, reach) -> int | None:
        if not np.all(np.isfinite(pt)):
            return None
        best, bd = None, reach
        for i in self.ids:
            d = self.dist(i, pt)
            if d <= bd:
                best, bd = i, d
        return best


def _runs(mask: np.ndarray):
    """(start, end) inclusive of consecutive True."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


# --------------------------------------------------------------------- judge
def feet_on_floor(t: Track, floor_px, toe_px) -> np.ndarray:
    """Per frame: True if either toe is on the floor, False if both clear, None-ish -> nan."""
    out = np.full(t.n, np.nan)
    if floor_px is None:
        return out
    w, h = t.size
    clr = FLOOR_CLEARANCE * h
    for f in range(t.n):
        vals = []
        for a in (LAN, RAN):
            p = t.xy[f, a]
            if np.all(np.isfinite(p)):
                x = int(np.clip(p[0], 0, w - 1))
                vals.append(p[1] + toe_px >= floor_px[x] - clr)
        if vals:
            out[f] = float(any(vals))
    return out


def hand_contacts(t: Track, holds: Holds, reach: float) -> dict[int, list[int | None]]:
    out = {}
    for k in (LWR, RWR):
        out[k] = [holds.nearest(t.xy[f, k], reach) for f in range(t.n)]
    return out


def judge(t: Track, holds: Holds, scale: dict, floor_px, climb: dict) -> dict:
    fps = t.fps
    toe = TOE_OFFSET * t.size[1]
    on_floor = feet_on_floor(t, floor_px, toe)
    hands = hand_contacts(t, holds, reach=0.35 * scale["torso"])
    ids = sorted(holds.ids)
    top_id = climb.get("final_hold_id") if climb.get("final_hold_id") in holds.ids else ids[-1]
    zone_id = ids[int(round(0.6 * (len(ids) - 1)))]

    # attempts = lift-offs (both feet clear) lasting LIFTOFF_S, with a pose
    # pose dropouts keep the last known state, so a lost frame high on the wall
    # neither ends an attempt nor fakes a fall
    held, gap = on_floor.copy(), 0
    for f in range(1, len(held)):
        if np.isfinite(on_floor[f]):
            gap = 0
        else:
            gap += 1
            if gap <= int(0.7 * fps):
                held[f] = held[f - 1]
    off = np.nan_to_num(held, nan=1.0) == 0.0
    seen_off = on_floor == 0.0
    def on_route(r):   # IFSC: an attempt is a lift-off with hands on the route
        n = sum(1 for f in range(r[0], r[1] + 1) if hands[LWR][f] or hands[RWR][f])
        return n / fps >= 0.3
    liftoffs = [r for r in _runs(off)
                if seen_off[r[0]:r[1] + 1].sum() / fps >= LIFTOFF_S and on_route(r)]
    # merge lift-offs separated by a blip on the floor < 0.3 s
    merged = []
    for r in liftoffs:
        if merged and (r[0] - merged[-1][1]) / fps < 0.3:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    events = []
    start = None
    if merged:
        f0 = merged[0][0]
        sh = sorted({h for k in (LWR, RWR) for h in hands[k][f0:f0 + int(fps * 0.3)] if h})
        start = {"t": round(f0 / fps, 2), "frame": f0, "holds": sh}
        events.append({"t": start["t"], "kind": "start",
                       "label": "Start: feet leave the mat, hands on " +
                                (", ".join(f"#{h}" for h in sh) if sh else "the start holds")})

    def touches(hid):
        return np.array([hands[LWR][f] == hid or hands[RWR][f] == hid for f in range(t.n)])

    z = [r for r in _runs(touches(zone_id)) if (r[1] - r[0] + 1) / fps >= ZONE_TOUCH_S]
    zone = {"hold": zone_id, "reached": bool(z), "t": round(z[0][0] / fps, 2) if z else None}
    if z:
        events.append({"t": zone["t"], "kind": "zone", "label": f"Zone: controlled touch on hold #{zone_id}"})

    both = np.array([hands[LWR][f] == top_id and hands[RWR][f] == top_id for f in range(t.n)])
    hip = _mid(t.xy[:, LHIP], t.xy[:, RHIP])
    hip_v = np.linalg.norm(np.gradient(hip, axis=0), axis=1) * fps / scale["torso"]
    top_runs = [(a, b) for a, b in _runs(both)]
    best = max(top_runs, key=lambda r: r[1] - r[0], default=None)
    # hand match on the top hold can be missed by pose noise; fall back to the
    # pipeline's own top-out when it saw one
    hold_s = (best[1] - best[0] + 1) / fps if best else 0.0
    still = bool(best) and np.nanmedian(hip_v[best[0]:best[1] + 1]) < 0.6
    reached = bool(best) or bool(climb.get("topped_out"))
    t_top = round(best[0] / fps, 2) if best else (
        round(climb["completion_frame"] / fps, 2) if climb.get("completion_frame") else None)
    controlled = bool(best) and hold_s >= TOP_CONTROL_S and still
    note = None
    edge = holds.centroid[top_id][1] < 0.06 * t.size[1]
    if not controlled and climb.get("topped_out") and (edge or not best):
        # the finish sits at the frame edge, so the hands leave the frame on the
        # match; take the pipeline's own top-out as the control signal
        controlled = True
        note = "top hold at the frame edge; control taken from the route read"
    top = {"hold": top_id, "reached": reached, "controlled": controlled, "t": t_top,
           "hold_s": None if hold_s is None else round(hold_s, 2), "note": note}
    if reached:
        events.append({"t": t_top, "kind": "top",
                       "label": f"Top: both hands on #{top_id}" + (" — controlled" if controlled else " — not held")})
    result = "TOP" if top["reached"] and top["controlled"] else ("ZONE" if zone["reached"] else "NO SCORE")
    return {"result": result, "attempts": max(1, len(merged)), "start": start, "zone": zone,
            "top": top, "events": sorted(events, key=lambda e: e["t"] or 0),
            "_on_floor": on_floor, "_hands": hands, "_merged": merged}


# --------------------------------------------------------------------- falls
PAD_TOLERANCE = 0.025        # of frame height: a foot at the wall's base reads a hair above the mat edge


def _inside_any(polys, pt, tol: float = 0.0) -> bool | None:
    if not polys or not np.all(np.isfinite(pt)):
        return None
    return any(cv2.pointPolygonTest(p, (float(pt[0]), float(pt[1])), True) >= -tol for p in polys)


def falls(t: Track, scale: dict, floor_px, j: dict, pad_polys_px) -> list[dict]:
    fps, body = t.fps, scale["body"]
    toe = TOE_OFFSET * t.size[1]
    on_floor = j["_on_floor"]
    hip = _mid(t.xy[:, LHIP], t.xy[:, RHIP])
    out = []
    # every lift-off run that ends back on the floor is a descent to examine
    for a, b in j["_merged"]:
        impact = b + 1
        if impact >= t.n or not np.isfinite(on_floor[min(impact, t.n - 1)]):
            # climber may have left frame; find first floor frame after b
            later = [f for f in range(b + 1, min(t.n, b + int(2 * fps))) if on_floor[f] == 1.0]
            if not later:
                continue
            impact = later[0]
        w0 = max(a, impact - int(FALL_WINDOW_S * fps))
        seg = hip[w0:impact + 1, 1]
        if np.isfinite(seg).sum() < 3:
            continue
        top_y = np.nanmin(seg)
        f_rel = w0 + int(np.nanargmin(seg))
        drop = (hip[impact, 1] - top_y) / body if np.isfinite(hip[impact, 1]) else np.nan
        if not np.isfinite(drop) or drop < FALL_MIN_DROP * 0.5:
            continue
        # release = last frame a hand is on a hold before impact; drop from there
        hands = j["_hands"]
        rel = [f for f in range(f_rel, impact) if hands[LWR][f] is not None or hands[RWR][f] is not None]
        if rel and np.isfinite(hip[rel[-1], 1]) and (hip[impact, 1] - hip[rel[-1], 1]) / body >= FALL_MIN_DROP * 0.5:
            f_rel = rel[-1]
            drop = (hip[impact, 1] - hip[f_rel, 1]) / body
        vy = np.gradient(hip[:, 1]) * fps / body
        pre = vy[max(impact - 3, 0):impact + 1]
        speed = float(np.nanmax(pre)) if np.isfinite(pre).any() else np.nan
        late = vy[max(impact - int(0.35 * fps), 0):impact + 1]
        late_speed = float(np.nanmean(late)) if np.isfinite(late).any() else np.nan
        # a fall/jump is free flight into the mat; a downclimb stays slow to the end
        free = np.isfinite(late_speed) and late_speed > 1.5
        after_top = (j["top"]["reached"] and j["top"]["controlled"] and j["top"]["t"] is not None
                     and f_rel / fps >= j["top"]["t"] - 0.2)
        if not free:
            kind = "downclimb"
        elif after_top:
            kind = "jump_off"
        else:
            kind = "fall"

        # landing window
        L = range(impact, min(t.n, impact + int(0.35 * fps)))
        knees = {}
        for side, (hp, kn, an) in {"left": (LHIP, LKN, LAN), "right": (RHIP, RKN, RAN)}.items():
            angs = [_angle(t.xy[f, hp], t.xy[f, kn], t.xy[f, an]) for f in L]
            angs = [x for x in angs if np.isfinite(x)]
            knees[side] = round(min(angs)) if angs else None
        toes = [t.xy[impact, a] + [0, toe] for a in (LAN, RAN)]
        feet_pad = [_inside_any(pad_polys_px, p, PAD_TOLERANCE * t.size[1]) for p in toes]
        # each foot's touchdown frame
        def touch(a):
            for f in range(max(a_ := f_rel, 0), min(t.n, impact + int(0.5 * fps))):
                p = t.xy[f, a]
                if np.all(np.isfinite(p)) and floor_px is not None:
                    x = int(np.clip(p[0], 0, t.size[0] - 1))
                    if p[1] + toe >= floor_px[x] - FLOOR_CLEARANCE * t.size[1]:
                        return f
            return None
        tl, tr = touch(LAN), touch(RAN)
        sync = None if tl is None or tr is None else abs(tl - tr)
        W = range(max(impact - 3, 0), min(t.n, impact + int(0.5 * fps)))
        posted = False
        for f in W:
            for wr in (LWR, RWR):
                p = t.xy[f, wr]
                if not np.all(np.isfinite(p)) or not np.isfinite(hip[f, 1]):
                    continue
                low = p[1] > hip[f, 1] + 0.25 * scale["torso"]
                ground = floor_px is not None and p[1] >= floor_px[int(np.clip(p[0], 0, t.size[0] - 1))] - 0.03 * t.size[1]
                if low and ground:
                    posted = True

        score, flags = 100, []
        for side, ang in knees.items():
            if ang is None:
                continue
            if ang > STIFF_KNEE:
                score -= 20; flags.append(f"Stiff {side} knee at impact ({ang}°) — absorb with bent knees")
            elif ang > GOOD_KNEE:
                score -= int(10 * (ang - GOOD_KNEE) / (STIFF_KNEE - GOOD_KNEE))
        for side, ok in zip(("Left", "Right"), feet_pad):
            if ok is False:
                score -= 20; flags.append(f"{side} foot landed off the pad — ankle-roll risk")
        if posted:
            score -= 20; flags.append("Hand posted to the mat — wrist/elbow risk, tuck the arms")
        if sync is not None and sync > FEET_SYNC_FRAMES:
            score -= 10; flags.append(f"Feet landed {sync / fps * 1000:.0f} ms apart — land on both feet together")
        if np.isfinite(drop) and drop > 1.0 and np.isfinite(speed) and speed > 3.0:
            score -= 5; flags.append("High, fast drop — sit and roll back to spread the impact")
        if kind == "downclimb":
            score = min(100, score + 5)
        out.append({
            "id": len(out) + 1, "kind": kind,
            "t_start": round(f_rel / fps, 2), "t_impact": round(impact / fps, 2), "frame_impact": int(impact),
            "drop_body_lengths": round(float(drop), 2),
            "impact_speed_bl_s": None if not np.isfinite(speed) else round(speed, 2),
            "knee_angle_min": knees, "feet_on_pad": feet_pad if pad_polys_px else None,
            "feet_sync_ms": None if sync is None else round(sync / fps * 1000),
            "hands_posted": posted, "landing_score": int(max(0, min(100, score))), "flags": flags,
            "ankle_xy": [[round(float(p[0] / t.size[0]), 4), round(float(p[1] / t.size[1]), 4)]
                         if np.all(np.isfinite(p)) else None for p in toes],
            "keyframe": None, "verdict": None,
        })
    return out


# --------------------------------------------------------------------- feet
def silent_feet(t: Track, holds: Holds, scale: dict, window: tuple[int, int]) -> dict:
    fps, shin, body = t.fps, scale["shin"], scale["body"]
    toe = TOE_OFFSET * t.size[1]
    a0, a1 = window
    placements = []
    for side, an in (("left", LAN), ("right", RAN)):
        pts = t.xy[:, an] + [0, toe]
        hold = [holds.nearest(pts[f], FOOT_REACH * shin) if a0 <= f <= a1 else None for f in range(t.n)]
        # contiguous same-hold segments, bridging pose dropouts of <= 3 frames
        segs, f = [], a0
        while f <= a1:
            hid = hold[f]
            if hid is None:
                f += 1; continue
            s, e, gap = f, f, 0
            g = f + 1
            while g <= a1 and gap <= 3:
                if hold[g] == hid:
                    e, gap = g, 0
                elif hold[g] is None:
                    gap += 1
                else:
                    break
                g += 1
            segs.append((hid, s, e)); f = e + 1
        segs = [sg for sg in segs if (sg[2] - sg[1] + 1) / fps >= MIN_PLACEMENT_S]
        # re-placements on the same hold within 1 s are re-adjusts, not new placements
        merged = []
        for sg in segs:
            if merged and merged[-1]["hold"] == sg[0] and (sg[1] - merged[-1]["e"]) / fps < 1.0:
                merged[-1]["e"] = sg[2]; merged[-1]["readjusts"] += 1
            else:
                merged.append({"hold": sg[0], "s": sg[1], "e": sg[2], "readjusts": 0})
        vel = np.linalg.norm(np.gradient(pts, axis=0), axis=1) * fps  # px/s
        for m in merged:
            s, e, hid = m["s"], m["e"], m["hold"]
            settle = s + int(SETTLE_S * fps)
            body_pts = pts[settle:e + 1]
            body_pts = body_pts[np.all(np.isfinite(body_pts), axis=1)]
            jitter = float(np.linalg.norm(body_pts.std(0))) / body if len(body_pts) > 3 else 0.0
            # wiggles after settling: rising edges of speed above 0.8 shin/s
            v = vel[settle:e + 1] / shin
            fast = np.nan_to_num(v) > 0.8
            wiggles = int(np.sum(fast[1:] & ~fast[:-1])) if len(fast) > 1 else 0
            m["readjusts"] += wiggles
            pre = vel[max(s - 3, 0):s + 1]
            impact = float(np.nanmean(pre)) / body if np.isfinite(pre).any() else 0.0
            land = pts[s] if np.all(np.isfinite(pts[s])) else holds.centroid[hid]
            dist = np.linalg.norm(land - holds.centroid[hid])
            precision = float(np.clip(1 - dist / (holds.radius[hid] + 1.0 * shin), 0, 1))
            sc = 100 - 15 * min(m["readjusts"], 3) - min(25, 400 * jitter) \
                 - min(20, 12 * max(0, impact - 0.5)) - 20 * (1 - precision)
            placements.append({
                "foot": side, "hold": int(hid), "t_on": round(s / fps, 2), "t_off": round(e / fps, 2),
                "readjusts": int(m["readjusts"]), "jitter": round(jitter, 4),
                "impact_speed": round(impact, 2), "precision": round(precision, 2),
                "score": int(max(0, min(100, sc))),
            })
    placements.sort(key=lambda p: p["t_on"])
    per = {}
    for side in ("left", "right"):
        ps = [p for p in placements if p["foot"] == side]
        per[side] = {"score": int(np.mean([p["score"] for p in ps])) if ps else None,
                     "placements": len(ps), "readjusts": int(sum(p["readjusts"] for p in ps))}
    score = int(np.mean([p["score"] for p in placements])) if placements else None
    grade = None if score is None else next(g for c, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "E")) if score >= c)
    worst = sorted(range(len(placements)), key=lambda i: placements[i]["score"])[:3]
    return {"score": score, "grade": grade, "placements": placements, "per_foot": per, "worst": worst}


def analyze(run: Path, cache_dir: Path, pad_polys_norm: list | None = None) -> dict:
    t = load_track(run)
    holds_raw = json.loads((run / "holds.json").read_text())
    climb = json.loads((run / "climb.json").read_text())
    holds = Holds(holds_raw, t.size)
    scale = body_scale(t)
    floor_px = load_floor(cache_dir, t.size)
    pad_px = [(np.array(p) * t.size).astype(np.float32) for p in (pad_polys_norm or [])]
    j = judge(t, holds, scale, floor_px, climb)
    fl = falls(t, scale, floor_px, j, pad_px)
    start = j["start"]["frame"] if j["start"] else (climb.get("start_frame") or 0)
    end_t = j["top"]["t"] if j["top"]["t"] else t.n / t.fps
    feet = silent_feet(t, holds, scale, (start, min(t.n - 1, int(end_t * t.fps) + int(0.5 * t.fps))))
    for k in [k for k in j if k.startswith("_")]:
        j.pop(k)
    w, h = t.size
    return {
        "fps": t.fps, "frame_count": t.n, "size": [w, h], "scale_px": {k: round(v, 1) for k, v in scale.items()},
        "route": {"color": climb.get("route_color"), "grade": (climb.get("route") or {}).get("grade"),
                  "holds": len(holds.ids)},
        "holds": [{"id": i, "centroid_xy": [round(float(holds.centroid[i][0] / w), 4),
                                            round(float(holds.centroid[i][1] / h), 4)],
                   "polygon": [[round(x, 4), round(y, 4)] for x, y in hd["polygon"]]}
                  for i, hd in zip(holds.ids, holds_raw)],
        "judge": j, "falls": fl, "feet": feet,
        "floor_edge_y": None if floor_px is None else [round(float(floor_px[int(x)] / h), 4)
                                                       for x in np.linspace(0, w - 1, 64)],
    }

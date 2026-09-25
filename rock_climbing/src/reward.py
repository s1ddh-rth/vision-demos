"""A climbing reward function, scored on the recorded climb.

One set of terms, two uses: ``score_demo`` applies it to the human climb read
off the video (a return for the demonstration), and ``src.rl.RouteEnv`` uses
the same per-move terms as its step reward, so the learned beta and the
human's beta are scored on the same scale.

Heights are normalized over the route: h = 0 at the lowest route hold, 1 at
the top hold. Distances are in body-lengths (bl), the same unit as dyno.py.
"""
from __future__ import annotations

import numpy as np

from src import dyno
from src.safety import LAN, RAN, LWR, RWR, TOE_OFFSET, FOOT_REACH, Holds, Track, _runs

WEIGHTS = {
    "progress": 10.0,    # x delta h of the higher hand's hold (h: 0 lowest route hold, 1 top hold)
    "zone": 3.0,         # once, first time a hand is on the zone hold
    "top": 10.0,         # controlled top (both hands on the top hold)
    "reach": -0.5,       # x (gap_bl / static_span_bl)^2 per hand move, gap from the anchor hand's hold
    "dyno_risk": -1.0,   # x (1 - clip(margin_bl / dyno_gain_bl, 0, 1)) for deadpoint / dyno moves
    "campus": -0.2,      # per second with no foot on a hold while climbing (demo only)
    "footwork": -0.1,    # per silent-feet readjust (demo only)
    "fall": -10.0,       # per uncontrolled fall
    "landing": 1.0,      # x (landing_score - 50) / 50 per descent
    "time": -0.01,       # per second on the wall
}
TERMS = list(WEIGHTS)
HAND_MIN_S = 0.25        # a hand must sit on a hold this long to count as a move (pose flicker)
CAMPUS_MIN_S = 0.3       # shorter feet-off spells are foot swaps, not campusing


def heights(holds: Holds, top_id: int) -> dict[int, float]:
    """Normalized height per hold: 0 = lowest route hold, 1 = the top hold (y grows down)."""
    y0 = max(holds.centroid[i][1] for i in holds.ids)
    y1 = holds.centroid[top_id][1]
    span = max(y0 - y1, 1e-6)
    return {i: float(np.clip((y0 - holds.centroid[i][1]) / span, 0, 1)) for i in holds.ids}


def move_terms(gap_bl: float, needed: str, margin_bl: float, env: dict) -> dict:
    """Reach + dyno-risk for one hand move (shared with the RL env)."""
    out = {"reach": WEIGHTS["reach"] * (gap_bl / env["static_span_bl"]) ** 2, "dyno_risk": 0.0}
    if needed in ("deadpoint", "dyno"):
        out["dyno_risk"] = WEIGHTS["dyno_risk"] * (1 - float(np.clip(margin_bl / env["dyno_gain_bl"], 0, 1)))
    return out


def hand_moves(t: Track, holds: Holds, scale: dict, window: tuple[int, int], env: dict,
               start_holds: list[int] | None = None) -> tuple[tuple[int, int], list[dict]]:
    """The human's hand sequence: (start (left, right) holds, moves in time order).

    A move = a hand settles (>= HAND_MIN_S) on a hold other than its last one.
    Each move carries the anchor (other hand's current hold) and dyno.classify's
    read of the gap from anchor to target."""
    fps, body = t.fps, scale["body"]
    hands = dyno._hand_holds(t, holds, 0.35 * scale["torso"], window)
    runs = []
    for k in (LWR, RWR):
        seq = hands[k]
        for hid in set(h for h in seq if h):
            for s, e in _runs(np.array([h == hid for h in seq])):
                if (e - s + 1) / fps >= HAND_MIN_S:
                    runs.append((s, k, hid))
    runs.sort()
    first = {k: next((h for s, kk, h in runs if kk == k), None) for k in (LWR, RWR)}
    sh = [h for h in (start_holds or []) if h in holds.ids]
    if sh:
        cur = {LWR: sh[0], RWR: sh[-1]}
    else:
        cur = {LWR: first[LWR] or first[RWR], RWR: first[RWR] or first[LWR]}
    start = (cur[LWR], cur[RWR])
    moves = []
    for s, k, hid in runs:
        if cur[k] == hid or s < window[0]:
            continue
        other = RWR if k == LWR else LWR
        anchor = cur[other]
        vec = holds.centroid[hid] - holds.centroid[anchor]
        need, margin = dyno.classify(vec, env, body)
        moves.append({"hand": "left" if k == LWR else "right", "from": int(cur[k]), "to": int(hid),
                      "anchor": int(anchor), "t": round(s / fps, 2),
                      "gap_bl": round(float(np.linalg.norm(vec)) / body, 3),
                      "needed": need, "margin_bl": margin})
        cur[k] = hid
    return start, moves


def _feet_on_hold(t: Track, holds: Holds, scale: dict, f: int) -> bool | None:
    toe = TOE_OFFSET * t.size[1]
    seen = False
    for a in (LAN, RAN):
        p = t.xy[f, a]
        if not np.all(np.isfinite(p)):
            continue
        seen = True
        if holds.nearest(p + [0, toe], FOOT_REACH * scale["shin"]) is not None:
            return True
    return False if seen else None


def score_demo(data: dict, t: Track, holds: Holds, scale: dict) -> dict:
    """Apply the reward to the recorded climb. ``data`` is the safety.json-style dict
    (judge, falls, feet, dyno); returns weights, total, per-term sums, a ~5 Hz
    cumulative curve and the scoring events."""
    fps = t.fps
    j = data["judge"]
    dy = data.get("dyno") or {}
    top_id = j["top"]["hold"]
    zone_id = j["zone"]["hold"]
    env = {"static_span_bl": dy.get("static_span_bl"), "dyno_gain_bl": dy.get("dyno_gain_bl")}
    if env["static_span_bl"] is None:
        env = dyno.envelope(t, holds, scale, dyno._hand_holds(t, holds, 0.35 * scale["torso"], (0, t.n - 1)))
    H = heights(holds, top_id)
    t0 = j["start"]["t"] if j.get("start") else 0.0
    t_top = j["top"]["t"] if j["top"].get("reached") and j["top"].get("t") is not None else None
    t1 = t_top if t_top is not None else t.n / fps
    a0, a1 = int(t0 * fps), min(t.n - 1, int(t1 * fps) + int(0.5 * fps))
    start, moves = hand_moves(t, holds, scale, (a0, a1), env, (j.get("start") or {}).get("holds"))

    ev = []
    def add(tt, term, value, label):
        if value:
            ev.append({"t": round(float(tt), 2), "term": term, "value": round(float(value), 3), "label": label})

    # hand moves: progress, reach, dyno risk
    cur = {"left": start[0], "right": start[1]}
    hi = max(H[start[0]], H[start[1]])
    for m in moves:
        cur[m["hand"]] = m["to"]
        nh = max(H[cur["left"]], H[cur["right"]])
        add(m["t"], "progress", WEIGHTS["progress"] * (nh - hi), f"{m['hand']} hand #{m['from']}→#{m['to']}: height {nh:.2f}")
        hi = nh
        mt = move_terms(m["gap_bl"], m["needed"], m["margin_bl"], env)
        add(m["t"], "reach", mt["reach"], f"reach {m['gap_bl']:.2f} bl from #{m['anchor']}")
        add(m["t"], "dyno_risk", mt["dyno_risk"], f"{m['needed']} to #{m['to']}, margin {m['margin_bl']:.2f} bl")
    if j["zone"].get("reached"):
        add(j["zone"]["t"], "zone", WEIGHTS["zone"], f"zone hold #{zone_id}")
    if j["top"].get("reached") and j["top"].get("controlled") and t_top is not None:
        add(t_top, "top", WEIGHTS["top"], f"controlled top on #{top_id}")
        # the finish hold can sit off the frame edge: credit the last bit of height
        if hi < 1.0:
            add(t_top, "progress", WEIGHTS["progress"] * (1.0 - hi), f"match on top #{top_id}")
            hi = 1.0

    # campus: spells hanging on a route hold with both feet visible and neither on a hold
    # (hands off the route = not on this problem, so not campusing it)
    hh = dyno._hand_holds(t, holds, 0.35 * scale["torso"], (a0, int(t1 * fps)))
    feet = np.array([_feet_on_hold(t, holds, scale, f)
                     if a0 <= f <= int(t1 * fps) and (hh[LWR][f] is not None or hh[RWR][f] is not None) else None
                     for f in range(t.n)], dtype=object)
    for s, e in _runs(np.array([v is False for v in feet])):
        dur = (e - s + 1) / fps
        if dur >= CAMPUS_MIN_S:
            add(s / fps, "campus", WEIGHTS["campus"] * dur, f"{dur:.1f} s with no foot on a hold")
    # footwork
    for p in (data.get("feet") or {}).get("placements", []):
        add(p["t_on"], "footwork", WEIGHTS["footwork"] * p["readjusts"],
            f"{p['foot']} foot on #{p['hold']}: {p['readjusts']} readjust(s)")
    # falls and landings
    for fl in data.get("falls") or []:
        if fl["kind"] == "fall":
            add(fl["t_impact"], "fall", WEIGHTS["fall"], "uncontrolled fall")
        add(fl["t_impact"], "landing", WEIGHTS["landing"] * (fl["landing_score"] - 50) / 50,
            f"{fl['kind']} landing {fl['landing_score']}/100")
    ev.sort(key=lambda e: e["t"])

    wall_s = max(0.0, t1 - t0)
    terms = {k: 0.0 for k in TERMS}
    for e in ev:
        terms[e["term"]] += e["value"]
    terms["time"] = WEIGHTS["time"] * wall_s
    terms = {k: round(v, 3) for k, v in terms.items()}

    # cumulative curve at 5 Hz, time term accrued continuously on the wall
    t_end = max([t1] + [e["t"] for e in ev]) + 0.2
    curve, k, acc = [], 0, 0.0
    for tt in np.arange(t0, t_end + 1e-9, 0.2):
        while k < len(ev) and ev[k]["t"] <= tt:
            acc += ev[k]["value"]; k += 1
        on_wall = min(max(tt - t0, 0.0), wall_s)
        curve.append([round(float(tt), 2), round(acc + WEIGHTS["time"] * on_wall, 3)])
    total = round(sum(terms.values()), 3)
    return {"weights": dict(WEIGHTS), "total": total, "terms": terms, "curve": curve, "events": ev,
            "hand_start": list(start), "hand_moves": moves}

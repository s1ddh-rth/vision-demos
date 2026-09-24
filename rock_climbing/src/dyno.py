"""Dyno lab: is a dyno possible between these two holds, for this climber?

Not reinforcement learning. This is the feasibility oracle a PPO humanoid
policy would need as its reward / termination signal: a reach envelope built
from the climber's own pose-derived proportions, grown by what leg drive and
flight add, and checked against every move the climber made plus the moves
they skipped. A gateway VLM then reads an annotated frame of the most
interesting gaps and says whether it would commit to the dyno.

Assumptions are explicit and live in ASSUME; a 1.75 m climber taking off at
2.2 m/s is a strong recreational dyno, not a comp one.
"""
from __future__ import annotations

import numpy as np

from src.safety import LSH, RSH, LEL, REL, LWR, RWR, Holds, Track, _runs

ASSUME = {
    "height_m": 1.75,            # body-length in metres (scales flight into the image)
    "takeoff_speed_m_s": 2.2,    # centre-of-mass speed off the wall at launch
    "leg_drive": 0.35,           # share of leg length regained by extending from a crouch
    "static_wingspan_share": 0.75,  # of the wingspan usable on the wall while holding on
}
G = 9.81
SMALL, MEDIUM = 0.035, 0.06      # hold radius in body-lengths: catch surface buckets


def _hand_holds(t: Track, holds: Holds, reach: float, window) -> dict:
    a, b = window
    return {k: [holds.nearest(t.xy[f, k], reach) if a <= f <= b else None for f in range(t.n)]
            for k in (LWR, RWR)}


def envelope(t: Track, holds: Holds, scale: dict, hands: dict) -> dict:
    body = scale["body"]
    # 90th percentile, not median: 2D pose foreshortens a bent arm; the
    # longest frames are the arm straight across the wall
    arm = np.nanpercentile(np.linalg.norm(t.xy[:, [LSH, RSH]] - t.xy[:, [LEL, REL]], axis=2), 90) + \
          np.nanpercentile(np.linalg.norm(t.xy[:, [LEL, REL]] - t.xy[:, [LWR, RWR]], axis=2), 90)
    shoulders = np.nanmedian(np.linalg.norm(t.xy[:, LSH] - t.xy[:, RSH], axis=1))
    wingspan = 2 * arm + shoulders
    # longest span actually held hand-to-hand on this climb
    seen = 0.0
    for f in range(t.n):
        l, r = hands[LWR][f], hands[RWR][f]
        if l and r and l != r:
            seen = max(seen, float(np.linalg.norm(holds.centroid[l] - holds.centroid[r])))
    static = max(seen, ASSUME["static_wingspan_share"] * wingspan) / body
    flight_bl = (ASSUME["takeoff_speed_m_s"] ** 2 / (2 * G)) / ASSUME["height_m"]
    legs_bl = ASSUME["leg_drive"] * (scale["thigh"] + scale["shin"]) / body
    return {"static_span_bl": round(static, 3), "seen_span_bl": round(seen / body, 3),
            "wingspan_bl": round(wingspan / body, 3), "dyno_gain_bl": round(flight_bl + legs_bl, 3)}


def classify(gap_vec: np.ndarray, env: dict, body: float) -> tuple[str, float]:
    gap = float(np.linalg.norm(gap_vec)) / body
    up = max(0.0, -gap_vec[1] / max(np.linalg.norm(gap_vec), 1e-6))   # y grows down
    gain = env["dyno_gain_bl"] * max(0.3, up)      # dynos buy height, little sideways reach
    static = env["static_span_bl"]
    if gap <= static:
        return "static", round(static + gain - gap, 3)
    if gap <= static + 0.4 * gain:
        return "deadpoint", round(static + gain - gap, 3)
    if gap <= static + gain:
        return "dyno", round(static + gain - gap, 3)
    return "out_of_reach", round(static + gain - gap, 3)


def analyze(t: Track, holds: Holds, scale: dict, window: tuple[int, int], max_skips: int = 6) -> dict:
    body, fps = scale["body"], t.fps
    hands = _hand_holds(t, holds, 0.35 * scale["torso"], window)
    env = envelope(t, holds, scale, hands)
    order = sorted(holds.ids, key=lambda i: holds.centroid[i][1], reverse=True)   # low -> high
    rank = {h: k for k, h in enumerate(order)}

    def size(h):
        r = holds.radius[h] / body
        return "small" if r < SMALL else ("medium" if r < MEDIUM else "large")

    cands, seen = [], set()
    # 1. moves actually climbed: a hand grabs a new hold while the other hand anchors
    for k, other in ((LWR, RWR), (RWR, LWR)):
        segs = [(h, s, e) for h, s, e in
                ((hands[k][s], s, e) for s, e in _runs(np.array([x is not None for x in hands[k]])))]
        prev = None
        for h, s, e in segs:
            if prev is not None and h != prev:
                anchor = next((hands[other][f] for f in range(s, max(s - int(fps), 0), -1) if hands[other][f]), prev)
                if anchor != h and (anchor, h) not in seen:
                    seen.add((anchor, h))
                    vec = holds.centroid[h] - holds.centroid[anchor]
                    need, margin = classify(vec, env, body)
                    cands.append({"from_hold": int(anchor), "to_hold": int(h),
                                  "hand": "left" if k == LWR else "right", "t": round(s / fps, 2),
                                  "_frame": max(s - int(0.4 * fps), 0), "skip": [], "climbed": True,
                                  "gap_bl": round(float(np.linalg.norm(vec)) / body, 3),
                                  "dx_bl": round(float(vec[0]) / body, 3), "dy_bl": round(float(vec[1]) / body, 3),
                                  "needed": need, "margin_bl": margin, "target_size": size(h)})
            prev = h
    # 2. moves skipped: from each set-up (both hands on), could you go 2-6 holds higher in one?
    skips = []
    for f in range(window[0], window[1] + 1, max(1, int(fps // 2))):
        l, r = hands[LWR][f], hands[RWR][f]
        if not (l and r):
            continue
        hi = l if rank[l] >= rank[r] else r
        for jump in (2, 3, 4, 5, 6):
            j = rank[hi] + jump
            if j >= len(order):
                continue
            tgt = order[j]
            if (hi, tgt) in seen:
                continue
            seen.add((hi, tgt))
            vec = holds.centroid[tgt] - holds.centroid[hi]
            need, margin = classify(vec, env, body)
            skips.append({"from_hold": int(hi), "to_hold": int(tgt),
                          "hand": "left" if hi == r else "right", "t": round(f / fps, 2), "_frame": f,
                          "skip": [int(x) for x in order[rank[hi] + 1:j]], "climbed": False,
                          "gap_bl": round(float(np.linalg.norm(vec)) / body, 3),
                          "dx_bl": round(float(vec[0]) / body, 3), "dy_bl": round(float(vec[1]) / body, 3),
                          "needed": need, "margin_bl": margin, "target_size": size(tgt)})
    # the interesting skips: dyno / deadpoint first, then the closest misses
    pri = {"dyno": 0, "deadpoint": 1, "out_of_reach": 2, "static": 3}
    skips.sort(key=lambda c: (pri[c["needed"]], -c["margin_bl"]))
    cands = sorted(cands, key=lambda c: c["t"]) + skips[:max_skips]
    for i, c in enumerate(cands, 1):
        c["id"] = i
        c["frame"] = None
        c["vlm"] = None
    return {"assumptions": dict(ASSUME), **env, "candidates": cands}


def pick_for_vlm(cands: list[dict], n: int = 4) -> list[dict]:
    """The gaps worth a model's look: skipped dynos/deadpoints, then the biggest climbed move."""
    skip = [c for c in cands if not c["climbed"] and c["needed"] in ("dyno", "deadpoint", "out_of_reach")]
    climbed = sorted((c for c in cands if c["climbed"]), key=lambda c: -c["gap_bl"])
    return (skip[: n - 1] + climbed[:1])[:n]


def annotate(img, holds: Holds, c: dict, env: dict, body: float):
    """Draw from/to holds, the arrow and the static / dyno reach rings."""
    import cv2
    im = img.copy()
    a = tuple(int(v) for v in holds.centroid[c["from_hold"]])
    b = tuple(int(v) for v in holds.centroid[c["to_hold"]])
    cv2.circle(im, a, int(env["static_span_bl"] * body), (120, 220, 120), 1, cv2.LINE_AA)
    cv2.circle(im, a, int((env["static_span_bl"] + env["dyno_gain_bl"]) * body), (0, 190, 255), 1, cv2.LINE_AA)
    cv2.circle(im, a, 16, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(im, b, 16, (0, 200, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(im, a, b, (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.08)
    for p, txt in ((a, f"FROM #{c['from_hold']}"), (b, f"TO #{c['to_hold']}")):
        cv2.putText(im, txt, (p[0] + 20, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, txt, (p[0] + 20, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im

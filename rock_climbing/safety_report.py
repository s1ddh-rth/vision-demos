"""Climb Vision Judge: judge + landing safety + silent feet, on a finished run.

    python safety_report.py                  # newest run under data/output/
    python safety_report.py <run_dir>        # a specific run (or clip subfolder)
    python safety_report.py --no-vlm         # skip the gateway calls (pad + verdicts)

Reads the run's poses/holds/climb JSON, spends a few cents of VLM Run credits
(SAM 3.1 for the crash pad, a gateway VLM for the fall verdicts and the coach
note) and writes safety.json plus a self-contained report.html beside them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src import dyno, safety

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "output"
CACHE = HERE / "data" / "cache"
TEMPLATE = HERE / "report" / "template.html"
TOKEN = "/*__SAFETY_DATA__*/null"


def newest_run() -> Path:
    runs = sorted((p for p in OUT.glob("*/**/poses.json")), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"no runs with poses.json under {OUT}; run main.py first")
    return runs[-1].parent


def clip_name(run: Path) -> str:
    vids = sorted(run.glob("*_climb.mp4"))
    return vids[0].name[:-len("_climb.mp4")] if vids else run.name


def source_video(clip: str) -> Path | None:
    """The converted portrait mp4 the poses were read on."""
    c = sorted(CACHE.glob(f"{clip}.*.mp4"), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def grab(video: Path, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f))
        ok, img = cap.read()
        if ok:
            out.append(img)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    run = Path(a.run).resolve() if a.run else newest_run()
    clip = clip_name(run)
    video = source_video(clip)
    print(f"run   {run}\nclip  {clip}\nvideo {video}")

    coach = client = None
    cost = {"sam_pad": 0.0, "vlm": 0.0}
    if not a.no_vlm:
        try:
            from src import coach as coach_mod
            coach, client = coach_mod, coach_mod.make_client()
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] disabled: {e}")

    # 1. crash pad, on a frame before the climber steps in (SAM 3.1)
    pad = None
    if coach and video is not None:
        clean = grab(video, [0])
        if clean:
            try:
                pad = coach.segment_pad(client, clean[0])
            except Exception as e:  # noqa: BLE001
                print(f"[vlm] pad failed: {e}")
    if pad:
        cost["sam_pad"] = float(pad.pop("cost", 0) or 0)
        print(f"pad   {len(pad['polygons'])} polygon(s) via {pad.get('prompt')}")

    # 2. the analysis
    r = safety.analyze(run, CACHE, pad["polygons"] if pad else None)
    if not pad and r.get("floor_edge_y"):
        # no pad found: treat the segmented floor as the landing zone
        edge = r["floor_edge_y"]
        xs = np.linspace(0, 1, len(edge))
        poly = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, edge)] + [[1, 1], [0, 1]]
        pad = {"source": "floor-fallback", "prompt": "gray floor", "polygons": [poly]}
        r = safety.analyze(run, CACHE, pad["polygons"])

    # 3. keyframes + VLM verdict per fall
    for fall in r["falls"]:
        if video is None:
            break
        fi = fall["frame_impact"]
        fps = r["fps"]
        picks = [fi - int(0.5 * fps), fi - int(0.15 * fps), fi, fi + int(0.3 * fps)]
        imgs = grab(video, picks)
        if not imgs:
            continue
        paths = []
        for k, img in enumerate(imgs):
            p = run / f"fall_{fall['id']}_{k}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(p)
        shutil.copyfile(paths[min(2, len(paths) - 1)], run / f"fall_{fall['id']}.jpg")
        fall["keyframe"] = f"fall_{fall['id']}.jpg"
        if coach:
            metrics = {k: fall[k] for k in ("kind", "drop_body_lengths", "impact_speed_bl_s",
                                            "knee_angle_min", "feet_on_pad", "hands_posted",
                                            "landing_score", "flags")}
            v = coach.fall_verdict(client, [str(p) for p in paths], metrics)
            fall["verdict"] = v.get("verdict")
            fall["vlm"] = {k: v.get(k) for k in ("on_pad", "posture", "error") if v.get(k) is not None}
            cost["vlm"] += float(v.get("cost") or 0)
            print(f"fall {fall['id']} ({fall['kind']}): {fall['verdict']}")

    # 4. dyno lab: physics envelope for every gap, VLM read on the interesting ones
    try:
        t = safety.load_track(run)
        holds = safety.Holds(json.loads((run / "holds.json").read_text()), t.size)
        scale = safety.body_scale(t)
        j = r["judge"]
        a0 = j["start"]["frame"] if j.get("start") else 0
        a1 = int((j["top"]["t"] or t.n / t.fps) * t.fps) if j.get("top") else t.n - 1
        dy = dyno.analyze(t, holds, scale, (a0, min(t.n - 1, a1 + int(0.5 * t.fps))))
        picks = dyno.pick_for_vlm(dy["candidates"]) if video is not None else []
        for c in picks:
            imgs = grab(video, [c["_frame"]])
            if not imgs:
                continue
            name = f"dyno_{c['id']}.jpg"
            cv2.imwrite(str(run / name), dyno.annotate(imgs[0], holds, c, dy, scale["body"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            c["frame"] = name
            if coach:
                gap = {k: c[k] for k in ("gap_bl", "dx_bl", "dy_bl", "needed", "margin_bl",
                                         "target_size", "climbed", "skip")}
                gap["static_span_bl"], gap["dyno_gain_bl"] = dy["static_span_bl"], dy["dyno_gain_bl"]
                v = coach.dyno_verdict(client, str(run / name), gap)
                cost["vlm"] += float(v.pop("cost", 0) or 0)
                c["vlm"] = v
                print(f"dyno #{c['from_hold']}->#{c['to_hold']} ({c['needed']}): "
                      f"{v.get('feasible')} {v.get('style')} - {v.get('reason') or v.get('error')}")
        for c in dy["candidates"]:
            c.pop("_frame", None)
        r["dyno"] = dy
    except Exception as e:  # noqa: BLE001
        print(f"[dyno] skipped: {e}")
        r["dyno"] = None

    # 5. overall coach note
    note = None
    if coach:
        feet = r["feet"]
        summary = {"judge": {k: r["judge"][k] for k in ("result", "attempts", "zone", "top")},
                   "falls": [{k: f[k] for k in ("kind", "landing_score", "flags", "knee_angle_min",
                                                "feet_on_pad", "hands_posted")} for f in r["falls"]],
                   "silent_feet": {"score": feet["score"], "grade": feet["grade"], "per_foot": feet["per_foot"],
                                   "worst": [feet["placements"][i] for i in feet["worst"]]},
                   "dyno": None if not r.get("dyno") else [
                       {k: c.get(k) for k in ("from_hold", "to_hold", "needed", "climbed", "vlm")}
                       for c in r["dyno"]["candidates"] if c.get("vlm")]}
        try:
            res = coach.overall_coach(client, summary)
            if isinstance(res, dict):
                note = res.get("note") or res.get("text"); cost["vlm"] += float(res.get("cost") or 0)
            else:
                note = res
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] coach failed: {e}")

    data = {"clip": clip, "video": f"{clip}_climb.mp4", "holds_image": "holds.png", **r,
            "pad": pad, "coach": note, "cost": {k: round(v, 4) for k, v in cost.items()},
            "generated_at": dt.datetime.now().isoformat(timespec="seconds")}
    (run / "safety.json").write_text(json.dumps(data, indent=1))
    if TEMPLATE.exists():
        html = TEMPLATE.read_text(encoding="utf-8")
        blob = json.dumps(data).replace("</", "<\\/")
        (run / "report.html").write_text(html.replace(TOKEN, blob), encoding="utf-8")
        print(f"report -> {run / 'report.html'}")
    j, f = data["judge"], data["feet"]
    print(f"judge {j['result']} · attempts {j['attempts']} · falls {len(data['falls'])} · "
          f"silent feet {f['score']} ({f['grade']}) · gateway ${sum(cost.values()):.4f}")


if __name__ == "__main__":
    main()

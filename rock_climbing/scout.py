"""Scout a wall: count the holds of each colour on empty-wall photos with SAM 3.1.

    conda activate rock_climbing
    python scout.py [photos_dir] [--colors blue,green,...]

One SAM call per photo per colour ("{color} climbing hold") through the VLM Run
Gateway. Writes an overlay JPG per photo and ``scout.json`` to
``data/output/scout/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from src.coach import (_decode_label_map, _jpeg_b64, _load_bgr, _mask_polygons,
                       _sam_request, make_client)

console = Console()

DEFAULT_COLORS = ("blue", "green", "yellow", "red", "purple", "pink", "orange",
                  "black", "white")
MIN_SCORE = 0.30
MIN_AREA = 0.00002          # instance share of the frame: drops speckle
OUT_ROOT = Path("data/output/scout")

# BGR tints for the overlay outlines.
TINTS = {"blue": (255, 120, 0), "green": (60, 200, 60), "yellow": (0, 230, 255),
         "red": (40, 40, 230), "purple": (200, 60, 150), "pink": (190, 130, 255),
         "orange": (0, 140, 255), "black": (40, 40, 40), "white": (250, 250, 250)}


def scout_color(client, img: np.ndarray, image_b64: str, color: str) -> tuple[dict, float]:
    """SAM one colour on one photo -> ({count, holds}, cost)."""
    h, w = img.shape[:2]
    try:
        items, label_mask, cost = _sam_request(client, image_b64, f"{color} climbing hold")
    except Exception as exc:
        return {"count": 0, "holds": [], "error": str(exc)}, 0.0
    labels = _decode_label_map(label_mask)
    holds = []
    if labels is not None:
        if labels.shape != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        for item in items:
            score = float(item.get("score") or 0.0)
            if score < MIN_SCORE or item.get("instance_id") is None:
                continue
            m = labels == int(item["instance_id"])
            if m.mean() < MIN_AREA:
                continue
            polys = _mask_polygons(m, max_points=40, min_area=MIN_AREA)
            if not polys:
                continue
            ys, xs = np.nonzero(m)
            bbox = [round(xs.min() / w, 5), round(ys.min() / h, 5),
                    round(xs.max() / w, 5), round(ys.max() / h, 5)]
            holds.append({"polygon": max(polys, key=len), "score": round(score, 3),
                          "bbox": bbox})
    return {"count": len(holds), "holds": holds}, cost


def draw_overlay(img: np.ndarray, result: dict) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    fill = out.copy()
    for color, r in result.items():
        tint = TINTS.get(color, (255, 255, 255))
        for hold in r["holds"]:
            pts = (np.array(hold["polygon"]) * [w, h]).astype(np.int32)
            cv2.fillPoly(fill, [pts], tint)
            cv2.polylines(out, [pts], True, tint, 2, cv2.LINE_AA)
    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    # Legend, top-left.
    rows = [(c, r["count"]) for c, r in result.items()]
    pad, lh = 10, 26
    cv2.rectangle(out, (0, 0), (190, pad * 2 + lh * len(rows)), (30, 30, 30), -1)
    for i, (color, n) in enumerate(rows):
        y = pad + lh * i + 18
        cv2.rectangle(out, (pad, y - 14), (pad + 18, y + 2), TINTS.get(color, (255,) * 3), -1)
        cv2.putText(out, f"{color}: {n}", (pad + 28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos_dir", nargs="?", default="data/input/photos/jpg")
    ap.add_argument("--colors", default=",".join(DEFAULT_COLORS))
    args = ap.parse_args()
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    photos = sorted(p for p in Path(args.photos_dir).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg"))
    if not photos:
        console.print(f"[red]no JPGs in {args.photos_dir}[/]")
        return
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client()
    console.print(f"Scouting {len(photos)} photos x {len(colors)} colours "
                  f"({len(photos) * len(colors)} SAM calls)")

    report, total = {}, 0.0
    for photo in photos:
        img = _load_bgr(photo)
        image_b64 = _jpeg_b64(img, quality=92)
        with ThreadPoolExecutor(max_workers=len(colors)) as pool:
            done = list(pool.map(lambda c: scout_color(client, img, image_b64, c), colors))
        result = {c: r for c, (r, _) in zip(colors, done)}
        total += sum(cost for _, cost in done)
        report[photo.name] = result
        cv2.imwrite(str(out_dir / f"{photo.stem}_scout.jpg"), draw_overlay(img, result))
        errs = [c for c, r in result.items() if "error" in r]
        console.print(f"  {photo.name}: {sum(r['count'] for r in result.values())} holds"
                      + (f" [red](failed: {', '.join(errs)})[/]" if errs else ""))

    report["cost"] = round(total, 6)
    (out_dir / "scout.json").write_text(json.dumps(report, indent=1))

    table = Table(title="Holds per colour")
    table.add_column("photo")
    for c in colors:
        table.add_column(c, justify="right")
    table.add_column("total", justify="right")
    for name, result in report.items():
        if name == "cost":
            continue
        counts = [result[c]["count"] for c in colors]
        table.add_row(name, *map(str, counts), str(sum(counts)))
    console.print(table)
    console.print(f"Cost: ${total:.4f}   ->  {out_dir}")


if __name__ == "__main__":
    main()

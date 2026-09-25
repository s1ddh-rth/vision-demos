"""VLM calls for the coaching layer: crash-pad segmentation, fall verdicts, and a
closing coaching note.

Three independent calls through the VLM Run OpenAI-compatible gateway:

* ``segment_pad``   -- SAM 3.1 on one clean frame, merged into pad polygons.
* ``fall_verdict``  -- Qwen VLM on 3-4 frames around a fall impact + metrics.
* ``overall_coach`` -- Qwen text-only, a 2-3 sentence note from the run summary
  (returns ``{"note", "cost"}``).

Every call is best-effort: failures come back as ``None`` / an ``error`` field,
never as an exception, so the pipeline keeps going without coaching.
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from config import GATEWAY_BASE_URL, HOLD_MODEL as SAM_MODEL
except Exception:  # pragma: no cover - run outside the project root
    GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
    SAM_MODEL = "facebook/sam3.1"

COACH_MODEL = "qwen/qwen3.8-27b"
PRICE_IN_PER_M = 0.35
PRICE_OUT_PER_M = 2.55
TIMEOUT_S = 60.0
MAX_RETRIES = 3

# SAM 3.1 finds nothing for the pad prompts on wall-to-wall gray gym flooring
# (the whole floor IS the pad); "gray floor" is the fallback that works there.
DEFAULT_PAD_PROMPTS = ("crash pad", "gym mat", "floor mat", "gray floor")
PAD_MIN_AREA = 0.01        # instance share of the frame worth keeping
PAD_MIN_CENTER_Y = 0.45    # centroid must sit in the lower part of the image
PAD_MIN_SCORE = 0.30
PAD_MAX_POINTS = 60


# ── client ───────────────────────────────────────────────────────────────────

def make_client(api_key: str | None = None):
    from openai import OpenAI

    if api_key is None:
        from .env import load_api_key
        api_key, _ = load_api_key(Path(__file__).resolve().parent.parent)
    return OpenAI(base_url=GATEWAY_BASE_URL, api_key=api_key,
                  timeout=TIMEOUT_S, max_retries=0)


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return (_is_rate_limit(exc) or (status is not None and status >= 500)
            or name in ("APITimeoutError", "APIConnectionError"))


def _with_retries(fn):
    """Call *fn*; retry transient failures (429 / 5xx / timeout) with backoff."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= MAX_RETRIES or not _is_transient(exc):
                raise
            time.sleep((2 ** attempt) * (1.5 if _is_rate_limit(exc) else 1.0)
                       + random.uniform(0, 0.5))


def _usage_cost(usage) -> float | None:
    """Gateway-reported cost if present, else an estimate from token counts."""
    if usage is None:
        return None
    data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    cost = data.get("cost")
    if cost is not None:
        try:
            return float(cost)
        except (TypeError, ValueError):
            pass
    pin = data.get("prompt_tokens") or 0
    pout = data.get("completion_tokens") or 0
    if not (pin or pout):
        return None
    return (pin * PRICE_IN_PER_M + pout * PRICE_OUT_PER_M) / 1e6


# ── images ───────────────────────────────────────────────────────────────────

def _load_bgr(frame_or_path) -> np.ndarray:
    if isinstance(frame_or_path, np.ndarray):
        return frame_or_path
    img = cv2.imread(str(frame_or_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image {frame_or_path}")
    return img


def _jpeg_b64(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> str:
    if max_side:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── SAM 3.1 pad segmentation ─────────────────────────────────────────────────

def _decode_label_map(mask) -> np.ndarray | None:
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    data = mask.get("data")
    if not isinstance(data, str):
        return None
    raw = base64.b64decode(data.split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    if labels.ndim == 3:
        labels = labels[:, :, 0]
    return labels


def _sam_request(client, image_b64: str, prompt: str):
    def call():
        return client.chat.completions.create(
            model=SAM_MODEL,
            messages=[{"role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}],
            response_format={"type": "json_object"},
            extra_body={"method": "segment",
                        "method_params": {"prompt": prompt, "mask_format": "png"}},
        )
    r = _with_retries(call)
    payload = json.loads(r.choices[0].message.content)
    content = payload.get("content") or {}
    return content.get("items") or [], content.get("mask"), _usage_cost(r.usage) or 0.0


def _mask_polygons(mask: np.ndarray, max_points: int = PAD_MAX_POINTS,
                   min_area: float = PAD_MIN_AREA) -> list[list[list[float]]]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) / (h * w) < min_area * 0.5:
            continue
        eps = 0.003 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        while len(approx) > max_points:
            eps *= 1.5
            approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        polys.append([[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in pts])
    return polys


def segment_pad(client, frame_bgr_or_path, prompts=DEFAULT_PAD_PROMPTS, *,
                min_area: float = PAD_MIN_AREA, min_center_y: float = PAD_MIN_CENTER_Y,
                min_score: float = PAD_MIN_SCORE, stop_at_first: bool = True) -> dict | None:
    """Segment the crash pad(s) on one clean frame (no climber in it).

    Tries each prompt in turn; keeps instances with enough area whose centroid
    is in the lower part of the frame, merges them into one mask, and returns
    normalized, simplified polygons. With ``stop_at_first`` the first prompt
    that finds something wins; otherwise all prompts' masks are unioned.
    Returns None when nothing qualifies (or every call failed).
    """
    img = _load_bgr(frame_bgr_or_path)
    h, w = img.shape[:2]
    image_b64 = _jpeg_b64(img, quality=92)
    merged = np.zeros((h, w), dtype=bool)
    used, cost, errors = [], 0.0, []

    for prompt in prompts:
        try:
            items, label_mask, c = _sam_request(client, image_b64, prompt)
        except Exception as exc:
            errors.append(f"{prompt}: {exc}")
            continue
        cost += c
        labels = _decode_label_map(label_mask)
        if labels is None:
            continue
        if labels.shape != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        hit = False
        for item in items:
            if float(item.get("score") or 0.0) < min_score or item.get("instance_id") is None:
                continue
            m = labels == int(item["instance_id"])
            area = m.mean()
            if area < min_area:
                continue
            ys = np.nonzero(m)[0]
            if ys.mean() / h < min_center_y:
                continue
            merged |= m
            hit = True
        if hit:
            used.append(prompt)
            if stop_at_first:
                break

    if not merged.any():
        return None
    # Close small gaps between adjacent pads / seams so they merge into one blob.
    k = max(3, int(0.01 * max(h, w)) | 1)
    closed = cv2.morphologyEx(merged.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8))
    polygons = _mask_polygons(closed, min_area=min_area)
    if not polygons:
        return None
    out = {"source": "sam3.1", "prompt": used[0] if len(used) == 1 else used,
           "polygons": polygons, "area": round(float(closed.mean()), 4),
           "cost": round(cost, 6)}
    if errors:
        out["errors"] = errors
    return out


# ── Qwen fall verdict ────────────────────────────────────────────────────────

FALL_SYSTEM = (
    "You are a bouldering coach reviewing a fall from a climbing gym video. "
    "Judge landing safety: did the climber land on the crash pad, and how "
    "(feet first with bent knees and rolling back = good; stiff legs, arms "
    "out to catch, landing on back/head, or landing off the pad edge = bad). "
    "Reply with JSON only."
)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def _as_bool(v) -> bool | None:
    if isinstance(v, str):
        v = {"true": True, "false": False, "yes": True, "no": False}.get(v.strip().lower())
    return v if isinstance(v, bool) else None


def _chat(client, messages, *, max_tokens: int, json_mode: bool):
    kwargs = dict(model=COACH_MODEL, messages=messages, max_tokens=max_tokens,
                  temperature=0.3,
                  # Thinking off: with it on, qwen3.8 burns 1.5k+ hidden reasoning
                  # tokens on image prompts (~110s, finish_reason=length, empty
                  # content). Off, the verdict comes back in ~4s for ~$0.0003.
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = _with_retries(lambda: client.chat.completions.create(**kwargs))
    choice = r.choices[0]
    text = choice.message.content or ""
    # qwen3.x is a reasoning model: hidden reasoning counts against max_tokens,
    # so a tight budget yields empty content with finish_reason="length".
    if not text.strip():
        raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
    return text, _usage_cost(r.usage)


def fall_verdict(client, frame_paths: list[str], metrics: dict) -> dict:
    """Ask the VLM for a short landing verdict from frames around the impact.

    Returns ``{"verdict", "on_pad", "posture", "cost"}`` or
    ``{"verdict": None, "error": str}`` on any failure.
    """
    try:
        content = [{"type": "text", "text": (
            f"These {len(frame_paths)} frames are in time order around the moment "
            "the climber hits the ground after a fall. Measured metrics from pose "
            f"tracking: {json.dumps(metrics, default=str)}.\n"
            "The metrics come from automatic pose tracking and can be wrong. Trust the "
            "frames: if what you see contradicts a metric (e.g. feet clearly off the pad, "
            "knees locked when the metric says bent), say so in the verdict and set "
            "agrees_with_metrics to false.\n"
            'Return JSON: {"verdict": "<=30 word coaching verdict on the landing, '
            'second person", "on_pad": true|false|null, "posture": "short phrase '
            'describing landing posture", "agrees_with_metrics": true|false}')}]
        for p in frame_paths[:4]:
            b64 = _jpeg_b64(_load_bgr(p), quality=85, max_side=768)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        text, cost = _chat(client, [{"role": "system", "content": FALL_SYSTEM},
                                    {"role": "user", "content": content}],
                           max_tokens=400, json_mode=True)
        data = _parse_json(text)
        on_pad = data.get("on_pad")
        if isinstance(on_pad, str):
            on_pad = {"true": True, "false": False}.get(on_pad.lower())
        verdict = data.get("verdict")
        if verdict:
            words = str(verdict).split()
            verdict = " ".join(words[:30])
        out = {"verdict": verdict, "on_pad": on_pad if isinstance(on_pad, bool) else None,
               "posture": data.get("posture"),
               "agrees_with_metrics": _as_bool(data.get("agrees_with_metrics"))}
        if cost is not None:
            out["cost"] = round(cost, 6)
        return out
    except Exception as exc:
        return {"verdict": None, "error": f"{type(exc).__name__}: {exc}"[:500]}


# ── Qwen overall note ────────────────────────────────────────────────────────

COACH_SYSTEM = (
    "You are a blunt, expert bouldering coach. Write 2-3 punchy sentences in "
    "second person. Be specific: cite the numbers you are given. Only claim what "
    "the summary states; if a field says a value was not measured or was inferred "
    "(e.g. judge.top.note / controlled_source), do not invent a number for it. "
    "No greetings, no fluff, no hedging, no bullet points, no markdown."
)


def overall_coach(client, summary: dict) -> dict | None:
    """A 2-3 sentence coaching note from the run summary (judge result, falls,
    silent-feet score, worst placements). Returns ``{"note", "cost"}``, or None
    on failure."""
    try:
        text, cost = _chat(client, [
            {"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content": "Climb summary (JSON):\n"
             + json.dumps(summary, default=str, indent=1)
             + "\n\nWrite the coaching note now."}],
            max_tokens=400, json_mode=False)
        text = text.strip()
        # Strip any leaked reasoning block.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if not text:
            return None
        return {"note": text, "cost": round(cost, 6) if cost is not None else 0.0}
    except Exception:
        return None


# ── Qwen dyno read ───────────────────────────────────────────────────────────

DYNO_SYSTEM = (
    "You are an expert bouldering route setter and coach judging whether a "
    "dynamic move is possible. Look at the wall, the holds and the climber's "
    "body. Be decisive and concrete. Reply only with the JSON asked for."
)


def dyno_verdict(client, image_path: str, gap: dict) -> dict:
    """Would a climber commit to a dyno from the marked FROM hold to the TO hold?

    ``gap`` carries the physics read (gap, margin, needed, target size) so the
    model reasons about the same numbers. Returns ``{"feasible", "style",
    "confidence", "reason", "cost"}`` or ``{"feasible": None, "error": str}``.
    """
    try:
        content = [{"type": "text", "text": (
            "The white circle marks the hold the climber launches FROM, the orange "
            "circle the TARGET hold, the arrow the move. Green ring = static reach, "
            "orange ring = reach with a dyno (leg drive + flight), both estimated "
            "from this climber's own proportions. Physics estimate: "
            f"{json.dumps(gap, default=str)}.\n"
            "The physics estimate comes from automatic pose/hold tracking and can be "
            "wrong. Trust the image: if it contradicts the estimate (the gap looks "
            "shorter/longer, or the marked holds are not real holds), say so in the "
            "reason and set agrees_with_metrics to false.\n"
            'Return JSON: {"feasible": true|false, "style": "static"|"deadpoint"|'
            '"dyno"|"double-dyno"|"not possible", "confidence": 0.0-1.0, '
            '"reason": "<=30 words: body position, feet available, target hold '
            'shape/size, catch difficulty", "agrees_with_metrics": true|false}')},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
             + _jpeg_b64(_load_bgr(image_path), quality=85, max_side=768)}}]
        text, cost = _chat(client, [{"role": "system", "content": DYNO_SYSTEM},
                                    {"role": "user", "content": content}],
                           max_tokens=300, json_mode=True)
        data = _parse_json(text)
        feas = data.get("feasible")
        if isinstance(feas, str):
            feas = {"true": True, "false": False}.get(feas.lower())
        try:
            conf = round(float(data.get("confidence")), 2)
        except (TypeError, ValueError):
            conf = None
        out = {"feasible": feas if isinstance(feas, bool) else None, "style": data.get("style"),
               "confidence": conf, "reason": " ".join(str(data.get("reason") or "").split()[:30]) or None,
               "agrees_with_metrics": _as_bool(data.get("agrees_with_metrics"))}
        if cost is not None:
            out["cost"] = round(cost, 6)
        return out
    except Exception as exc:
        return {"feasible": None, "error": f"{type(exc).__name__}: {exc}"[:500]}

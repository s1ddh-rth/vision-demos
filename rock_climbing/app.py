"""Browser dashboard for the climbing pipeline: upload, run, and open reports.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000 --host 0.0.0.0

Stdlib only. Each run processes ONE clip (BATCH_MODE off via CVJ_* env vars read
by config.py), so clips of different routes never get compared with each other.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "output"
INPUT = HERE / "data" / "input"
UPLOADS = INPUT / "uploads"
CURRENT = INPUT / "current"
APP_HTML = HERE / "report" / "app.html"
GUIDE_HTML = HERE / "report" / "guide.html"
SUFFIXES = {".mov", ".mp4", ".m4v", ".avi"}
FFMPEG_DIR = Path(sys.executable).parent / "Library" / "bin"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")
COLORS = {"blue", "green", "yellow", "red", "purple", "pink", "orange", "black", "white"}

STEP_MAIN = "Segmenting holds + pose (main.py)"
STEP_REPORT = "Judge, safety, dyno (safety_report.py)"

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
BUSY = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def run_dirs() -> set[Path]:
    return {p.parent.resolve() for p in OUT.glob("**/poses.json")}


def run_id(d: Path) -> str:
    return d.resolve().relative_to(OUT.resolve()).as_posix()


def list_runs() -> list[dict]:
    out = []
    for d in run_dirs():
        vids = sorted(d.glob("*_climb.mp4"))
        item = {"id": run_id(d),
                "clip": vids[0].name[:-len("_climb.mp4")] if vids else d.name,
                "created": datetime.fromtimestamp((d / "poses.json").stat().st_mtime)
                .isoformat(timespec="seconds"),
                "has_report": (d / "report.html").is_file(),
                "result": None, "feet_score": None, "feet_grade": None, "falls": None,
                "color": None, "grade": None, "elapsed": None}
        try:
            s = json.loads((d / "safety.json").read_text(encoding="utf-8"))
            item["result"] = (s.get("judge") or {}).get("result")
            item["feet_score"] = (s.get("feet") or {}).get("score")
            item["feet_grade"] = (s.get("feet") or {}).get("grade")
            item["falls"] = len(s.get("falls") or [])
            item["color"] = (s.get("route") or {}).get("color")
            item["grade"] = (s.get("route") or {}).get("grade")
        except Exception:  # noqa: BLE001
            pass
        try:
            c = json.loads((d / "climb.json").read_text(encoding="utf-8"))
            item["color"] = item["color"] or (c.get("route") or {}).get("color")
            item["grade"] = item["grade"] or (c.get("route") or {}).get("grade")
            item["elapsed"] = c.get("elapsed_seconds") or c.get("elapsed")
        except Exception:  # noqa: BLE001
            pass
        out.append(item)
    out.sort(key=lambda r: r["created"], reverse=True)
    return out


def list_clips() -> list[dict]:
    clips = []
    for folder, tag in ((UPLOADS, "uploads"), (CURRENT, "current")):
        if folder.is_dir():
            for p in sorted(folder.iterdir(), key=lambda p: -p.stat().st_mtime):
                if p.suffix.lower() in SUFFIXES and not p.name.startswith("."):
                    clips.append({"id": f"{tag}/{p.name}", "name": p.name, "folder": tag,
                                  "size_mb": round(p.stat().st_size / 1e6, 1)})
    return clips


def clip_path(clip_id: str) -> Path | None:
    tag, _, name = (clip_id or "").partition("/")
    base = {"uploads": UPLOADS, "current": CURRENT}.get(tag)
    if not base or not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    p = base / name
    return p if p.is_file() else None


def safe_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    stem, suf = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "clip"
    return stem + suf.lower()


def child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_COLOR"] = "1"
    env["COLUMNS"] = "80"
    if FFMPEG_DIR.is_dir():
        env["PATH"] = str(FFMPEG_DIR) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


# ── jobs ─────────────────────────────────────────────────────────────────────

def new_job(kind: str, **meta) -> dict:
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "state": "queued", "step": None,
           "log": deque(maxlen=400), "run_id": None, "started": time.time(),
           "finished": None, "error": None, **meta}
    with LOCK:
        JOBS[job["id"]] = job
    return job


def stream(job: dict, cmd: list[str], env: dict) -> int:
    job["log"].append(f"$ {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}")
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = ANSI.sub("", line.rstrip("\n"))
        if line.strip():
            job["log"].append(line)
    return proc.wait()


def report_cmd(run_dir: Path, vlm: bool) -> list[str]:
    cmd = [sys.executable, "safety_report.py", str(run_dir)]
    if not vlm:
        cmd.append("--no-vlm")
    return cmd


def job_full(job: dict, clip: Path, color: str, grade: str, vlm: bool) -> None:
    try:
        job["state"] = "running"
        job["step"] = STEP_MAIN
        before = run_dirs()
        env = child_env({"CVJ_BATCH_MODE": "0", "CVJ_INPUT_VIDEO": str(clip),
                         "CVJ_HOLD_COLOR": color, "CVJ_ROUTE_GRADE": grade})
        code = stream(job, [sys.executable, "main.py"], env)
        if code != 0:
            raise RuntimeError(f"main.py exited with code {code}")
        new = sorted(run_dirs() - before, key=lambda p: (p / "poses.json").stat().st_mtime)
        if not new:
            raise RuntimeError("main.py finished but no new run folder with poses.json appeared")
        run_dir = new[-1]
        job["run_id"] = run_id(run_dir)
        job["step"] = STEP_REPORT
        code = stream(job, report_cmd(run_dir, vlm), child_env())
        if code != 0:
            raise RuntimeError(f"safety_report.py exited with code {code}")
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001
        job["state"], job["error"] = "error", str(e)
        job["log"].append(f"ERROR: {e}")
    finally:
        job["finished"] = time.time()
        BUSY.release()


def job_report(job: dict, run_dir: Path, vlm: bool) -> None:
    try:
        job["state"], job["step"] = "running", STEP_REPORT
        job["run_id"] = run_id(run_dir)
        code = stream(job, report_cmd(run_dir, vlm), child_env())
        if code != 0:
            raise RuntimeError(f"safety_report.py exited with code {code}")
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001
        job["state"], job["error"] = "error", str(e)
        job["log"].append(f"ERROR: {e}")
    finally:
        job["finished"] = time.time()
        BUSY.release()


def job_view(job: dict) -> dict:
    end = job["finished"] or time.time()
    return {"id": job["id"], "kind": job["kind"], "state": job["state"], "step": job["step"],
            "log": list(job["log"])[-60:], "run_id": job["run_id"], "error": job["error"],
            "clip": job.get("clip"), "elapsed": round(end - job["started"], 1)}


def active_job() -> dict | None:
    with LOCK:
        for j in JOBS.values():
            if j["state"] in ("queued", "running"):
                return j
    return None


# ── HTTP ─────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ClimbVision/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter: skip static/poll noise
        if "/api/jobs/" in self.path or self.path.startswith("/runs/"):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # responses
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, code, msg):
        self.send_json({"error": msg}, code)

    def send_file(self, path: Path, head=False):
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in (".mp4", ".m4v"):
            ctype = "video/mp4"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head:
            return
        with path.open("rb") as f:
            f.seek(start)
            left = length
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    def read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    # routing
    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html", "/app"):
            return self.send_file(APP_HTML, head)
        if path in ("/guide", "/guide.html"):
            return self.send_file(GUIDE_HTML, head)
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/api/runs":
            return self.send_json(list_runs())
        if path == "/api/clips":
            return self.send_json(list_clips())
        if path == "/api/status":
            j = active_job()
            return self.send_json({"busy": j is not None, "job": job_view(j) if j else None})
        if path.startswith("/api/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self.send_json(job_view(job)) if job else self.send_error_json(404, "no such job")
        if path.startswith("/runs/"):
            rel = path[len("/runs/"):]
            target = (OUT / rel).resolve()
            if OUT.resolve() not in target.parents or not target.is_file():
                return self.send_error_json(404, "not found")
            return self.send_file(target, head)
        self.send_error_json(404, "not found")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/upload":
            return self.upload()
        if path == "/api/run":
            return self.start_run()
        if path.startswith("/api/report/"):
            return self.start_report(path[len("/api/report/"):])
        self.send_error_json(404, "not found")

    def upload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self.send_error_json(400, "empty body")
        name = self.headers.get("X-Filename") or ""
        ctype = self.headers.get("Content-Type", "")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        tmp = UPLOADS / f".upload-{uuid.uuid4().hex}"
        if ctype.startswith("multipart/form-data"):
            # small multipart parser: buffer to disk, then slice the first file part
            m = re.search(r'boundary="?([^";]+)"?', ctype)
            if not m:
                return self.send_error_json(400, "no multipart boundary")
            data = self.rfile.read(n)
            boundary = b"--" + m.group(1).encode()
            part = next((p for p in data.split(boundary) if b"filename=" in p[:1024]), None)
            if part is None:
                return self.send_error_json(400, "no file part")
            head_, _, body = part.partition(b"\r\n\r\n")
            fm = re.search(rb'filename="([^"]*)"', head_)
            name = name or (fm.group(1).decode("utf-8", "replace") if fm else "")
            tmp.write_bytes(body[:-2] if body.endswith(b"\r\n") else body)
        else:
            with tmp.open("wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
        name = safe_name(unquote(name) or "clip.mp4")
        if Path(name).suffix.lower() not in SUFFIXES:
            tmp.unlink(missing_ok=True)
            return self.send_error_json(400, f"unsupported file type; use {sorted(SUFFIXES)}")
        dest = UPLOADS / name
        if dest.exists():  # keep clips distinct: the cache and report key on the stem
            stem, suf = os.path.splitext(name)
            dest = UPLOADS / f"{stem}_{datetime.now():%H%M%S}{suf}"
        tmp.replace(dest)
        self.send_json({"clip": f"uploads/{dest.name}", "name": dest.name,
                        "size_mb": round(dest.stat().st_size / 1e6, 1)})

    def start_run(self):
        body = self.read_json()
        clip = clip_path(body.get("clip", ""))
        if clip is None:
            return self.send_error_json(400, "unknown clip; upload one or pick from the list")
        color = str(body.get("hold_color") or "blue").lower().strip()
        if color not in COLORS:
            return self.send_error_json(400, f"hold_color must be one of {sorted(COLORS)}")
        grade = re.sub(r"[^\w+\-. ]", "", str(body.get("grade") or "VB"))[:12] or "VB"
        if not BUSY.acquire(blocking=False):
            j = active_job()
            return self.send_json({"error": "a job is already running",
                                   "job": j["id"] if j else None}, 409)
        job = new_job("run", clip=clip.name)
        threading.Thread(target=job_full, args=(job, clip, color, grade, bool(body.get("vlm"))),
                         daemon=True).start()
        self.send_json({"job": job["id"]})

    def start_report(self, rid: str):
        body = self.read_json()
        target = (OUT / rid).resolve()
        if OUT.resolve() not in target.parents or not (target / "poses.json").is_file():
            return self.send_error_json(404, "no such run")
        if not BUSY.acquire(blocking=False):
            j = active_job()
            return self.send_json({"error": "a job is already running",
                                   "job": j["id"] if j else None}, 409)
        job = new_job("report", clip=target.name)
        threading.Thread(target=job_report, args=(job, target, bool(body.get("vlm"))),
                         daemon=True).start()
        self.send_json({"job": job["id"]})


def main():
    ap = argparse.ArgumentParser(description="Climb Vision dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Climb Vision dashboard -> http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

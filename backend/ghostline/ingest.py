"""Ingest: store upload, hash, normalize to H.264 <=720p, real per-frame timestamps, keyframes."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2

from .clients.cache import file_sha256
from .storage import rel_path, run_dir


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr.strip()[-500:]}")
    return p.stdout


def probe_stream(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,codec_name:stream_side_data=rotation",
                "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    num, den = (s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1").split("/")
    s["fps"] = float(num) / float(den) if float(den) else 30.0
    return s


def frame_times(path: Path) -> list[float]:
    """Presentation timestamps of every decoded frame (phones record VFR: never assume i/fps)."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "frame=best_effort_timestamp_time,pts_time", "-of", "csv=p=0", str(path)])
    ts = []
    for line in out.strip().splitlines():
        vals = [v for v in line.split(",") if v not in ("", "N/A")]
        if vals:
            ts.append(float(vals[0]))
    if not ts:
        raise RuntimeError("no frames found in video")
    t0 = ts[0]
    return [round(t - t0, 6) for t in ts]


def normalize(src: Path, dst: Path) -> None:
    s = probe_stream(src)
    w, h = int(s["width"]), int(s["height"])
    filters = []
    if min(w, h) > 720:
        filters.append("scale=-2:720" if w >= h else "scale=720:-2")
    if s["fps"] > 61:
        filters.append("fps=60")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if s["fps"] <= 61:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    _run(cmd)


def extract_keyframes(video: Path, out_dir: Path, n_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = sorted({0, n_frames // 2, n_frames - 1, n_frames // 6, n_frames // 3, (5 * n_frames) // 6})
    cap = cv2.VideoCapture(str(video))
    paths = []
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"key_{k}_{i}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    cap.release()
    return paths


def ingest(src: Path, run_id: str) -> dict:
    src = Path(src)
    d = run_dir(run_id)
    orig = d / f"original{src.suffix.lower() or '.mp4'}"
    if src.resolve() != orig.resolve():
        shutil.copyfile(src, orig)
    sha = file_sha256(orig)
    norm = d / "video.mp4"
    normalize(orig, norm)
    s = probe_stream(norm)
    times = frame_times(norm)
    keys = extract_keyframes(norm, d / "keyframes", len(times))
    return {"video_path": rel_path(norm), "original_path": rel_path(orig), "video_sha256": sha,
            "fps_nominal": s["fps"], "width": int(s["width"]), "height": int(s["height"]),
            "frame_times": times, "keyframes": [rel_path(k) for k in keys]}

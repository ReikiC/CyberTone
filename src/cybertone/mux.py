"""Container level work: probing, video retime, mp3/srt export, A/B baseline."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import audio
from .config import Settings

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_DIM_RE = re.compile(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})")


def ffmpeg() -> str:
    exe = audio._ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
    return exe


def probe(path: str | Path) -> dict:
    proc = subprocess.run(
        [ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False
    )
    err = proc.stderr
    info: dict = {"duration": None, "has_video": False, "width": None, "height": None}
    m = _DUR_RE.search(err)
    if m:
        h, mi, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mi) * 60 + float(s)
    if re.search(r"Stream #\d+:\d+.*: Video:", err):
        info["has_video"] = True
        d = _DIM_RE.search(err)
        if d:
            info["width"], info["height"] = int(d.group(1)), int(d.group(2))
    return info


def video_filter(speed: float) -> str:
    return f"setpts=(1/{speed:.6f})*PTS"


def render(
    src: Path, audio_path: Path, out: Path, speed: float, cfg: Settings
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(src)
    if meta["has_video"]:
        cmd = [
            ffmpeg(), "-v", "error", "-y",
            "-i", str(src), "-i", str(audio_path),
            "-filter_complex", f"[0:v]{video_filter(speed)}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", cfg.video_codec, "-crf", str(cfg.video_crf), "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", cfg.audio_bitrate,
            "-movflags", "+faststart", "-shortest", str(out),
        ]
    else:
        ext = out.suffix.lower()
        if ext == ".mp3":
            cmd = [ffmpeg(), "-v", "error", "-y", "-i", str(audio_path),
                   "-c:a", "libmp3lame", "-b:a", cfg.audio_bitrate, str(out)]
        elif ext == ".m4a":
            cmd = [ffmpeg(), "-v", "error", "-y", "-i", str(audio_path),
                   "-c:a", "aac", "-b:a", cfg.audio_bitrate, str(out)]
        else:
            cmd = [ffmpeg(), "-v", "error", "-y", "-i", str(audio_path), str(out)]
    subprocess.run(cmd, check=True)
    return out


def baseline(src: Path, out: Path, speed: float, cfg: Settings) -> Path:
    """The 'obvious' approach, for A/B: uniform atempo + uniform video setpts."""
    meta = probe(src)
    chain = f"atempo={min(speed, 2.0):.6f}"
    if speed > 2.0:
        chain += f",atempo={speed / 2.0:.6f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if meta["has_video"]:
        cmd = [
            ffmpeg(), "-v", "error", "-y", "-i", str(src),
            "-filter_complex", f"[0:v]{video_filter(speed)}[v];[0:a]{chain}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", cfg.video_codec, "-crf", str(cfg.video_crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", cfg.audio_bitrate,
            "-movflags", "+faststart", str(out),
        ]
    else:
        cmd = [ffmpeg(), "-v", "error", "-y", "-i", str(src), "-filter:a", chain,
               "-b:a", cfg.audio_bitrate, str(out)]
    subprocess.run(cmd, check=True)
    return out


def _ts(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def write_srt(cues: list[dict], path: Path) -> Path | None:
    rows = [c for c in cues if c.get("text")]
    if not rows:
        return None
    lines = []
    for i, c in enumerate(rows, 1):
        lines.append(f"{i}\n{_ts(c['dst_in'])} --> {_ts(c['dst_out'])}\n{c['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

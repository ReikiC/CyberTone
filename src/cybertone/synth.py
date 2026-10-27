"""Per-sentence synthesis backends and the closed-loop duration fitter.

Two backends ship:

* ``mock``  - re-times the *original* sentence with WSOLA. No timbre transfer, but it
  exercises the whole timing/stitch/mux pipeline and is a useful quality floor.
* ``cosyvoice`` - HTTP client for the service in ``services/cosyvoice`` (zero-shot
  cloning + explicit speed control).

The fitter is backend-agnostic: it merely needs "higher speed -> shorter audio".
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import soundfile as sf

from . import audio
from .config import Settings
from .models import PlanRow, Stem, Trigger


class Backend(Protocol):
    name: str
    exact: bool

    def render(self, trig: Trigger, speed: float) -> np.ndarray: ...


class MockBackend:
    """Time-stretch the source sentence. speed=2.0 -> half length, same pitch."""

    name = "mock"
    exact = True

    def __init__(self, source: np.ndarray, sr: int) -> None:
        self.source = source
        self.sr = sr

    def render(self, trig: Trigger, speed: float) -> np.ndarray:
        a = max(0, int(trig.src_in * self.sr))
        b = min(self.source.size, int(trig.src_out * self.sr))
        seg = self.source[a:b]
        return audio.time_stretch(seg, speed, self.sr)


class CosyVoiceBackend:
    """Zero-shot clone over HTTP. The service owns the model, the client owns timing."""

    name = "cosyvoice"
    exact = False

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.sr = cfg.sr
        self._ref_b64 = ""
        if cfg.tts_ref:
            ref, _ = audio.load(cfg.tts_ref, cfg.sr)
            buf = io.BytesIO()
            sf.write(buf, ref, cfg.sr, format="WAV", subtype="PCM_16")
            self._ref_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        self._client = httpx.Client(base_url=cfg.tts_url, timeout=180.0)

    def render(self, trig: Trigger, speed: float) -> np.ndarray:
        if not trig.text:
            raise ValueError(f"trigger {trig.tid} has no text; run with --asr first")
        payload = {
            "text": trig.text,
            "speed": speed,
            "model": self.cfg.tts_model,
            "seed": self.cfg.tts_seed,
            "temperature": self.cfg.tts_temperature,
            "sample_rate": self.cfg.sr,
            "ref_wav_b64": self._ref_b64,
        }
        resp = self._client.post("/tts", json=payload)
        resp.raise_for_status()
        wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1).astype(np.float32)
        if sr != self.sr:
            mono = audio.resample(mono, sr, self.sr)
        return audio.fade(mono, self.sr, 3.0)


def make_backend(name: str, cfg: Settings, source: np.ndarray, sr: int) -> Backend:
    if name == "mock":
        return MockBackend(source, sr)
    if name in ("cosyvoice", "cosyvoice2"):
        return CosyVoiceBackend(cfg)
    raise ValueError(f"unknown backend: {name}")


# ------------------------------------------------------------------ fitter


def _cache_key(backend: Backend, trig: Trigger, speed: float, cfg: Settings) -> str:
    ref = ""
    if cfg.tts_ref and Path(cfg.tts_ref).exists():
        ref = hashlib.sha1(Path(cfg.tts_ref).read_bytes()).hexdigest()[:12]
    raw = "|".join(
        [
            backend.name,
            cfg.tts_model,
            trig.text,
            f"{speed:.3f}",
            str(cfg.tts_seed),
            f"{cfg.tts_temperature:.2f}",
            ref,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _render(backend: Backend, trig: Trigger, speed: float, cfg: Settings, cache: Path) -> np.ndarray:
    if backend.exact:
        return backend.render(trig, speed)
    key = _cache_key(backend, trig, speed, cfg)
    path = cache / f"{key}.wav"
    if path.exists():
        wav, _ = audio.load(path, cfg.sr)
        return wav
    wav = backend.render(trig, speed)
    audio.save(path, wav, cfg.sr, subtype="PCM_16")
    return wav


def fit(
    backend: Backend, trig: Trigger, row: PlanRow, cfg: Settings
) -> tuple[np.ndarray, Stem]:
    """Find the slowest rate whose output still fits the window."""
    cache = cfg.work / "cache"
    degraded = False

    if backend.exact:
        speed = max(cfg.r_min, min(cfg.r_max * 1.2, trig.dur / max(row.win_max, 1e-3)))
        wav = backend.render(trig, speed)
        best = speed
    else:
        lo, hi = cfg.r_min, cfg.r_max
        best = None
        wav = None
        for _ in range(max(2, cfg.fit_iters)):
            mid = 0.5 * (lo + hi)
            cand = _render(backend, trig, mid, cfg, cache)
            d = cand.size / cfg.sr
            if d > row.win_max:
                lo = mid
                continue
            best, wav = mid, cand
            if d >= row.win_min:
                break
            hi = mid
        if wav is None:  # even r_max overruns: synth then squeeze the residual
            best = cfg.r_max
            wav = _render(backend, trig, best, cfg, cache)
            d = wav.size / cfg.sr
            factor = d / max(row.win_max, 1e-3)
            wav = audio.time_stretch(wav, factor, cfg.sr)
            degraded = True

    dur = wav.size / cfg.sr
    if cfg.stem_lufs is not None and not backend.exact:
        # TTS loudness drifts sentence to sentence; even it out before assembly
        wav = audio.normalize_loudness(wav, cfg.sr, cfg.stem_lufs, cfg.peak_ceiling_db)
    stem = Stem(
        tid=trig.tid,
        path=str(cfg.work / "stems" / f"{trig.tid:04d}.wav"),
        dur=round(dur, 4),
        speed=round(float(best), 4),
        win_min=row.win_min,
        win_max=row.win_max,
        degraded=degraded,
        qa={
            "dur_err": round(dur - row.win_max, 4),
            "peak_db": round(audio.peak_db(wav), 2),
            "lufs": audio.integrated_loudness(wav, cfg.sr),
            "overrun": dur > row.win_max + 0.02,
        },
    )
    audio.save(stem.path, wav, cfg.sr, subtype="PCM_24")
    return wav, stem


def make_reference(
    source: np.ndarray, sr: int, triggers: list[Trigger], dur: float = 8.0
) -> np.ndarray:
    """Pick a clean, energetic stretch of the speaker for zero-shot cloning."""
    import numpy as np

    cands = [t for t in triggers if 2.0 <= t.dur <= 14.0]
    if not cands:
        cands = triggers
    if not cands:
        return source[: int(dur * sr)].copy()
    scored = []
    for t in cands:
        a, b = int(t.src_in * sr), int(t.src_out * sr)
        seg = source[a:b]
        if seg.size < int(0.5 * sr):
            continue
        scored.append((audio.rms_db(seg) - 0.5 * abs(t.dur - 6.0), t))
    scored.sort(key=lambda x: -x[0])
    t = scored[0][1]
    a = int(t.src_in * sr)
    want = int(min(dur, t.dur) * sr)
    seg = source[a : a + want]
    return np.ascontiguousarray(seg.astype(np.float32))


def save_manifest(stems: list[Stem], path: Path) -> None:
    path.write_text(
        json.dumps([s.model_dump() for s in stems], ensure_ascii=False, indent=2), encoding="utf-8"
    )

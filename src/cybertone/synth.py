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


class RetimeBackend:
    """Re-times the ORIGINAL sentence with WSOLA.

    This is a timing scaffold, not a voice model: it inherits the source delivery and
    only changes rate. Use it to validate segmentation/budget/mux before paying for a
    TTS run — never as the final render.
    """

    name = "retime"
    exact = True

    def __init__(self, source: np.ndarray, sr: int) -> None:
        self.source = source
        self.sr = sr

    def render(self, trig: Trigger, speed: float) -> np.ndarray:
        a = max(0, int(trig.src_in * self.sr))
        b = min(self.source.size, int(trig.src_out * self.sr))
        seg = self.source[a:b]
        return audio.time_stretch(seg, speed, self.sr)


class HTTPServiceBackend:
    """Zero-shot clone over HTTP (CosyVoice2 / F5-TTS / anything with /tts).

    The service owns the model; the client owns timing.
    """

    exact = False

    def __init__(self, cfg: Settings, name: str = "service") -> None:
        self.name = name
        self.cfg = cfg
        self.sr = cfg.sr
        self.ref_text = cfg.tts_ref_text
        self._ref_b64 = ""
        if cfg.tts_ref:
            ref, _ = audio.load(cfg.tts_ref, cfg.sr)
            buf = io.BytesIO()
            sf.write(buf, ref, cfg.sr, format="WAV", subtype="PCM_16")
            self._ref_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            side = Path(cfg.tts_ref).with_suffix(".txt")
            if not self.ref_text and side.exists():
                self.ref_text = side.read_text(encoding="utf-8").strip()
        self._client = httpx.Client(base_url=cfg.tts_url, timeout=600.0)

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
            "ref_text": self.ref_text,
            "nfe_step": self.cfg.tts_nfe,
        }
        resp = self._client.post("/tts", json=payload)
        resp.raise_for_status()
        wav, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1).astype(np.float32)
        if sr != self.sr:
            mono = audio.resample(mono, sr, self.sr)
        # TTS pads with silence; measuring it as speech would eat the window budget
        mono = audio.trim_silence(mono, self.sr, -40.0, 30.0)
        return audio.fade(mono, self.sr, 3.0)


SERVICE_BACKENDS = {"service", "cosyvoice", "cosyvoice2", "f5tts", "f5"}


def make_backend(name: str, cfg: Settings, source: np.ndarray, sr: int) -> Backend:
    if name in ("retime", "mock"):
        return RetimeBackend(source, sr)
    if name in SERVICE_BACKENDS:
        return HTTPServiceBackend(cfg, name)
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
        if wav is not None:
            # one secant step: TTS length responds almost linearly to speed, so this
            # usually lands in the window instead of relying on residual stretching
            dur = wav.size / cfg.sr
            if dur > row.win_max * 1.01 and best * dur / row.win_max <= cfg.r_max:
                trial = min(cfg.r_max, best * dur / row.win_max)
                cand = _render(backend, trig, trial, cfg, cache)
                if cand.size / cfg.sr <= row.win_max:
                    best, wav = trial, cand
        if wav is None:  # even r_max overruns
            best = cfg.r_max
            wav = _render(backend, trig, best, cfg, cache)

        dur = wav.size / cfg.sr
        if dur > row.win_max:
            factor = dur / max(row.win_max, 1e-3)
            wav = audio.time_stretch(wav, factor, cfg.sr)
            degraded = factor > 1.03

    dur = wav.size / cfg.sr
    if cfg.clean_tts and not backend.exact:
        wav = audio.highpass(wav, cfg.sr, cfg.hp_hz)
        wav = audio.denoise(wav, cfg.sr, cfg.denoise_strength)
        wav = audio.trim_silence(wav, cfg.sr, -45.0, 20.0)
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
            "floor_db": round(audio.noise_floor_db(wav, cfg.sr), 1),
            "overrun": dur > row.win_max + 0.02,
        },
    )
    audio.save(stem.path, wav, cfg.sr, subtype="PCM_24")
    return wav, stem


def make_reference(
    source: np.ndarray, sr: int, triggers: list[Trigger], dur: float = 8.0, clean: bool = True
) -> tuple[np.ndarray, str]:
    """Pick 5-8 s of continuous, clean speech for zero-shot cloning.

    Single sentences are usually too short (2-3 s) for a stable timbre embedding, so
    consecutive triggers are merged first, and the transcript comes straight from the
    manifest instead of a second ASR pass.
    """
    import numpy as np

    if not triggers:
        return np.ascontiguousarray(source[: int(dur * sr)]), ""

    groups: list[tuple[float, float, list[Trigger]]] = []
    i, n = 0, len(triggers)
    while i < n:
        j = i
        while j + 1 < n and triggers[j + 1].src_out - triggers[i].src_in <= dur * 1.05:
            j += 1
        span = triggers[j].src_out - triggers[i].src_in
        if j > i and span > dur * 1.05:
            j, span = j - 1, triggers[j - 1].src_out - triggers[i].src_in
        if 2.0 <= span <= dur * 1.05:
            groups.append((triggers[i].src_in, triggers[j].src_out, triggers[i : j + 1]))
            i = j + 1
        else:
            i += 1

    if not groups:
        groups = [(t.src_in, t.src_out, [t]) for t in triggers if t.dur >= 1.0] or [
            (triggers[0].src_in, triggers[0].src_out, [triggers[0]])
        ]

    def score(g: tuple[float, float, list[Trigger]]) -> float:
        a, b, _ = g
        seg = source[int(a * sr) : int(b * sr)]
        return audio.rms_db(seg) - 0.4 * abs((b - a) - dur)

    best = max(groups, key=score)
    a, b, used = best
    want = int(min(dur, b - a) * sr)
    mid = int(((a + b) / 2) * sr)
    start = max(int(a * sr), mid - want // 2)
    seg = source[start : start + want]
    seg = np.ascontiguousarray(seg.astype(np.float32))
    if clean:
        # the cloner imitates the prompt channel, so its hiss ends up in every sentence
        seg = audio.denoise(audio.highpass(seg, sr, 70.0), sr, 1.0)
    return seg, _reference_text(used)


def _reference_text(triggers: list[Trigger]) -> str:
    from .triggers import _join

    return _join([t.text for t in triggers])


def save_manifest(stems: list[Stem], path: Path) -> None:
    path.write_text(
        json.dumps([s.model_dump() for s in stems], ensure_ascii=False, indent=2), encoding="utf-8"
    )

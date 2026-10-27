"""Energy VAD with hysteresis. Deliberately model-free so it always runs."""

from __future__ import annotations

import numpy as np

from .config import Settings


def frame_db(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if x.size < n:
        return np.array([-120.0]), np.array([0.0])
    x64 = x.astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x64 * x64)])
    starts = np.arange(0, x.size - n + 1, hop)
    rms = np.sqrt(np.maximum((csum[starts + n] - csum[starts]) / n, 1e-12))
    return 20.0 * np.log10(rms), starts / sr


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _mask(db: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    mask = np.zeros(db.size, dtype=bool)
    active = False
    for i, v in enumerate(db):
        if active:
            if v < exit_:
                active = False
        elif v > enter:
            active = True
        if active:
            mask[i] = True
    return mask


def detect(x: np.ndarray, sr: int, cfg: Settings) -> list[tuple[float, float]]:
    """Return speech spans in seconds, cleaned by min-speech / min-silence rules."""
    if x.size == 0:
        return []
    db, _times = frame_db(x, sr, cfg.vad_frame_ms, cfg.vad_hop_ms)
    hop = cfg.vad_hop_ms / 1000.0

    # A percentile-based floor breaks when the clip is nearly all speech, so cap it:
    # no real recording has a noise floor above -38 dBFS.
    floor = min(float(np.percentile(db, 5)), -38.0)
    enter = max(floor + cfg.vad_enter_db, -46.0)
    mask = _mask(db, enter, enter - cfg.vad_exit_db)

    if not mask.any() and float(np.percentile(db, 95)) > -40.0:
        # very dense material (or a hot noise floor): fall back to a relative threshold
        enter = float(np.percentile(db, 95)) - 15.0
        mask = _mask(db, enter, enter - cfg.vad_exit_db)

    # fill short non-speech holes, then drop short blips
    min_sil = int(round(cfg.min_silence / hop))
    min_sp = int(round(cfg.min_speech / hop))
    spans = [(s, e) for s, e in _runs(mask)]
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < min_sil:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [m for m in merged if m[1] - m[0] >= min_sp]

    out = [(s * hop, min(e * hop, x.size / sr)) for s, e in merged]
    return [(round(a, 3), round(b, 3)) for a, b in out if b > a]


def best_cut(
    x: np.ndarray, sr: int, lo: float, hi: float, frame_ms: float = 20.0
) -> float:
    """Cheapest place to force a cut: the local energy minimum inside [lo, hi]."""
    a, b = int(lo * sr), int(hi * sr)
    if b <= a:
        return 0.5 * (lo + hi)
    seg = x[a:b]
    db, times = frame_db(seg, sr, frame_ms, frame_ms)
    if db.size == 0:
        return 0.5 * (lo + hi)
    return float(lo + times[int(np.argmin(db))] + frame_ms / 2000.0)

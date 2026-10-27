"""Audio primitives: decode, resample, WSOLA time-stretch, BS.1770 loudness, room tone.

Everything here is dependency-light on purpose: numpy/scipy/soundfile only, so the
pipeline runs on a stock Mac without building rubberband.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

# --------------------------------------------------------------------------- io


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # bundled static build, no system install needed
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on env
        return None


def _decode_via_tool(path: Path) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        ff = _ffmpeg_exe()
        if ff:
            subprocess.run(
                [ff, "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_s16le", str(out)],
                check=True,
            )
        elif shutil.which("afconvert"):
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", str(path), str(out)], check=True
            )
        else:  # pragma: no cover
            raise RuntimeError(f"no decoder available for {path}")
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return data, sr


def load(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode to mono float32. Resamples when `sr` differs from the file rate."""
    path = Path(path)
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        data, file_sr = _decode_via_tool(path)

    mono = data.mean(axis=1).astype(np.float32)
    if sr is not None and sr != file_sr:
        mono = resample(mono, file_sr, sr)
        file_sr = sr
    return np.ascontiguousarray(mono), file_sr


def save(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)
    return path


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(sr_in), int(sr_out))
    y = signal.resample_poly(x, sr_out // g, sr_in // g)
    return np.ascontiguousarray(y.astype(np.float32))


# ------------------------------------------------------------------- level meters


def rms_db(x: np.ndarray, floor: float = -120.0) -> float:
    if x.size == 0:
        return floor
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return max(floor, 20.0 * math.log10(max(rms, 1e-12)))


def peak_db(x: np.ndarray, floor: float = -120.0) -> float:
    if x.size == 0:
        return floor
    return max(floor, 20.0 * math.log10(max(float(np.max(np.abs(x))), 1e-12)))


def _biquad_high_shelf(fc: float, q: float, gain_db: float, sr: int):
    a_amp = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / sr
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b0 = a_amp * ((a_amp + 1) + (a_amp - 1) * cosw + 2 * math.sqrt(a_amp) * alpha)
    b1 = -2 * a_amp * ((a_amp - 1) + (a_amp + 1) * cosw)
    b2 = a_amp * ((a_amp + 1) + (a_amp - 1) * cosw - 2 * math.sqrt(a_amp) * alpha)
    a0 = (a_amp + 1) - (a_amp - 1) * cosw + 2 * math.sqrt(a_amp) * alpha
    a1 = 2 * ((a_amp - 1) - (a_amp + 1) * cosw)
    a2 = (a_amp + 1) - (a_amp - 1) * cosw - 2 * math.sqrt(a_amp) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def _biquad_high_pass(fc: float, q: float, sr: int):
    w0 = 2 * math.pi * fc / sr
    alpha = math.sin(w0) / (2 * q)
    cosw = math.cos(w0)
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2])
    a = np.array([1 + alpha, -2 * cosw, 1 - alpha])
    return b / a[0], a / a[0]


def integrated_loudness(x: np.ndarray, sr: int) -> float | None:
    """ITU-R BS.1770-4 integrated loudness in LUFS. None when too quiet."""
    if x.size < int(0.4 * sr):
        return None
    b1, a1 = _biquad_high_shelf(1681.974450955533, 0.7071752369554196, 3.999843853973347, sr)
    b2, a2 = _biquad_high_pass(38.13547087602444, 0.5003270373238773, sr)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x.astype(np.float64)))

    block = int(0.4 * sr)
    hop = int(0.1 * sr)
    if y.size < block:
        return None
    idx = np.arange(0, y.size - block + 1, hop)
    energy = np.array([np.mean(y[i : i + block] ** 2) for i in idx])
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10 * np.log10(np.maximum(energy, 1e-12))

    keep = loud > -70.0
    if not keep.any():
        return None
    gated = energy[keep]
    rel = -0.691 + 10 * math.log10(float(np.mean(gated))) - 10.0
    keep2 = loud[keep] > rel
    if not keep2.any():
        return None
    return float(-0.691 + 10 * math.log10(float(np.mean(gated[keep2]))))


def limit_peaks(
    x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 5.0, release_ms: float = 60.0
) -> np.ndarray:
    """Lookahead peak limiter. Shaves the crest instead of throwing away gain."""
    from scipy.ndimage import maximum_filter1d

    ceiling = 10 ** (ceiling_db / 20.0)
    if x.size == 0 or float(np.max(np.abs(x))) <= ceiling:
        return x.astype(np.float32, copy=False)

    win = max(3, int(lookahead_ms * sr / 1000.0))
    env = maximum_filter1d(np.abs(x), size=win, mode="nearest")
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))

    # release smoothing: instant attack, one-pole recovery (vectorised)
    coef = math.exp(-1.0 / max(1.0, release_ms * sr / 1000.0))
    b, a = [1 - coef], [1, -coef]
    zi = signal.lfilter_zi(b, a) * gain[0]  # no start-up duck
    smoothed, _ = signal.lfilter(b, a, gain, zi=zi)
    g = np.minimum(gain, smoothed)
    return (x * g).astype(np.float32)


def normalize_loudness(
    x: np.ndarray, sr: int, target_lufs: float, ceil_db: float = -1.0, limit: bool = True
) -> np.ndarray:
    """Static gain to target LUFS; a limiter (not plain attenuation) protects the ceiling."""
    lufs = integrated_loudness(x, sr)
    if lufs is None:
        return x
    y = (x * (10 ** ((target_lufs - lufs) / 20.0))).astype(np.float32)
    if limit:
        return limit_peaks(y, sr, ceil_db)
    ceiling = 10 ** (ceil_db / 20.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def fade(x: np.ndarray, sr: int, ms: float = 5.0, mode: str = "both") -> np.ndarray:
    n = max(1, int(sr * ms / 1000.0))
    n = min(n, x.size // 2)
    if n <= 1:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 0.5
    if mode in ("both", "in"):
        y[:n] *= ramp
    if mode in ("both", "out"):
        y[-n:] *= ramp[::-1]
    return y


# --------------------------------------------------------------------- WSOLA


def time_stretch(
    x: np.ndarray, rate: float, sr: int, frame_ms: float = 40.0, tol_ms: float = 10.0
) -> np.ndarray:
    """WSOLA. rate > 1 shortens, pitch is preserved (no resampling involved)."""
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"bad stretch rate: {rate}")
    if abs(rate - 1.0) < 1e-3 or x.size == 0:
        return x.astype(np.float32, copy=True)

    n = int(round(frame_ms * sr / 1000.0))
    n += n % 2
    hs = n // 2
    tol = max(2, int(round(tol_ms * sr / 1000.0)))
    ha = max(1, int(round(hs / rate)))
    ov = max(1, min(hs, 2 * tol))
    win = np.hanning(n + 1)[:-1].astype(np.float32)

    target = int(round(x.size / rate))
    pad = n * 2
    src = np.concatenate(
        [np.zeros(pad, np.float32), x.astype(np.float32), np.zeros(pad + 2 * n, np.float32)]
    )
    out_len = target + 4 * n
    y = np.zeros(out_len, np.float32)
    wsum = np.zeros(out_len, np.float32)

    prev_a = pad
    k = 0
    while True:
        o = k * hs
        if o + n > out_len:
            break
        base = pad + int(round(k * ha))
        if base + n >= src.size:
            break
        if k == 0:
            a = pad
        else:
            nat = prev_a + hs
            lo = max(pad, base - tol)
            hi = min(src.size - ov - 1, base + tol)
            a = base
            if hi > lo and nat + ov <= src.size:
                tpl = src[nat : nat + ov]
                if float(np.dot(tpl, tpl)) > 1e-9:
                    corr = signal.correlate(src[lo : hi + ov], tpl, mode="valid", method="fft")
                    a = lo + int(np.argmax(corr))
        y[o : o + n] += src[a : a + n] * win
        wsum[o : o + n] += win
        prev_a = a
        k += 1

    wsum[wsum < 1e-4] = 1.0
    y = (y / wsum)[:target]
    if y.size < target:  # tail starved, pad with zeros to keep timing honest
        y = np.pad(y, (0, target - y.size))
    return fade(y, sr, 3.0)


# ------------------------------------------------------------------ room tone


def quietest_span(x: np.ndarray, sr: int, dur: float = 2.0) -> np.ndarray:
    """Extract the quietest contiguous span, loopable, to use as room tone."""
    want = int(dur * sr)
    if x.size <= want:
        return x.astype(np.float32, copy=True)
    hop = max(1, int(0.05 * sr))
    starts = np.arange(0, x.size - want + 1, hop)
    energy = np.array([np.mean(np.square(x[s : s + want], dtype=np.float64)) for s in starts])
    s = int(starts[int(np.argmin(energy))])
    tone = x[s : s + want].astype(np.float32, copy=True)
    return fade(tone, sr, 20.0)


def loop_to(x: np.ndarray, length: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(length, np.float32)
    reps = int(math.ceil(length / x.size))
    y = np.tile(x, reps)[:length]
    return y

"""Assemble stems onto a single timeline: room-tone bed, fades, loudness."""

from __future__ import annotations

import numpy as np

from . import audio
from .config import Settings
from .models import Plan, Stem


def _load_stems(stems: list[Stem], sr: int) -> list[np.ndarray]:
    out = []
    for s in stems:
        wav, _ = audio.load(s.path, sr)
        out.append(wav.astype(np.float32))
    return out


def planned_gaps(plan: Plan) -> list[float]:
    gaps = [row.gap_before for row in plan.rows]
    spent = sum(row.win_max + row.gap_before for row in plan.rows)
    gaps.append(max(0.0, plan.target - spent))
    return gaps


def assemble(
    stems: list[Stem], plan: Plan, cfg: Settings, source: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    from .plan import fit_gaps

    sr = cfg.sr
    wavs = _load_stems(stems, sr)
    actuals = [w.size / sr for w in wavs]
    gaps = fit_gaps(actuals, planned_gaps(plan), plan.target)

    total = float(sum(actuals) + sum(gaps))
    n = int(round(total * sr)) + sr // 10

    tone = audio.quietest_span(source, sr, 2.0)
    bed = audio.loop_to(tone, n)
    target_rms = 10 ** (cfg.room_tone_db / 20.0)
    cur_rms = float(np.sqrt(np.mean(np.square(bed, dtype=np.float64)))) if bed.size else 0.0
    # never amplify a digital-silence span into a loud noise floor: 18 dB is plenty
    if cur_rms > 1e-9:
        gain = min(target_rms / cur_rms, 10 ** (18.0 / 20.0))
        bed = bed * gain
    bed = audio.fade(bed, sr, 30.0)

    master = bed.astype(np.float32)
    cues: list[dict] = []
    cursor = 0.0
    for i, (wav, stem) in enumerate(zip(wavs, stems, strict=True)):
        cursor += gaps[i]
        start = int(round(cursor * sr))
        end = start + wav.size
        if end > master.size:
            master = np.pad(master, (0, end - master.size))
        master[start:end] += audio.fade(wav, sr, cfg.fade_ms)
        cues.append(
            {
                "tid": stem.tid,
                "dst_in": round(cursor, 3),
                "dst_out": round(cursor + wav.size / sr, 3),
                "text": plan.rows[i].text,
                "speed": stem.speed,
                "degraded": stem.degraded,
            }
        )
        cursor += wav.size / sr

    master = master[: int(round(total * sr))]
    master = audio.normalize_loudness(master, sr, cfg.loudness_lufs, cfg.peak_ceiling_db)
    master = audio.fade(master, sr, 20.0)
    return master.astype(np.float32), cues

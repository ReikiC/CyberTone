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


def placements(
    rows: list, durs: list[float], gaps: list[float], target: float, speed: float, mode: str = "flow"
) -> tuple[list[float], float]:
    """Where each stem starts on the new timeline.

    flow   - pack stems back-to-back with the planned gaps. Maximises silence removal,
             but sentence n no longer sits at src_in/speed, so it drifts against a
             uniformly-retimed video.
    anchor - pin each sentence to src_in/speed, i.e. exactly where it lands in a 2x
             video. Silences can only be halved, never squeezed further, so use this
             when A/V alignment matters more than dead air.
    """
    if not rows:
        return [], target
    starts: list[float] = []
    cursor = 0.0
    if mode == "anchor":
        for row, dur in zip(rows, durs, strict=True):
            pos = max(row.src_in / speed, cursor)
            starts.append(pos)
            cursor = pos + dur + 0.01
        return starts, max(target, cursor)
    cursor = gaps[0]
    for dur, gap in zip(durs, gaps[1:], strict=True):
        starts.append(cursor)
        cursor += dur + gap
    return starts, target


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

    if cfg.room_tone_db is None or cfg.room_tone_db <= -90.0:
        bed = np.zeros(n, np.float32)
    else:
        tone = audio.quietest_span(source, sr, 2.0)
        bed = audio.loop_to(tone, n)
    # the bed must sit below the stems' own noise, otherwise it becomes the hiss
    stem_floor = float(np.median([audio.noise_floor_db(w, sr) for w in wavs])) if wavs else -90.0
    bed_db = min(cfg.room_tone_db or -120.0, stem_floor - 6.0)
    target_rms = 10 ** (bed_db / 20.0)
    cur_rms = float(np.sqrt(np.mean(np.square(bed, dtype=np.float64)))) if bed.size else 0.0
    # never amplify a digital-silence span into a loud noise floor: 18 dB is plenty
    if cur_rms > 1e-9 and bed.any():
        gain = min(target_rms / cur_rms, 10 ** (18.0 / 20.0))
        bed = bed * gain
    bed = audio.fade(bed, sr, 30.0)

    starts, total = placements(plan.rows, actuals, gaps, plan.target, cfg.speed, cfg.placement)
    total = max(total, float(sum(actuals) + sum(gaps)))

    master = bed.astype(np.float32)
    cues: list[dict] = []
    for i, (wav, stem) in enumerate(zip(wavs, stems, strict=True)):
        cursor = starts[i]
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

    total = max(total, cues[-1]["dst_out"] if cues else 0.0)
    want = int(round(total * sr))
    if master.size < want:
        master = np.pad(master, (0, want - master.size))
    master = master[:want]
    master = audio.normalize_loudness(master, sr, cfg.loudness_lufs, cfg.peak_ceiling_db)
    master = audio.fade(master, sr, 20.0)
    return master.astype(np.float32), cues

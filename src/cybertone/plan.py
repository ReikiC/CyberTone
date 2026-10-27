"""Budget solver: turn a target speed into a per-sentence rate window.

The point of this module is that "2x" is never applied uniformly. Silence is spent
first, and only the residual is charged to speech, which keeps the required rate as
close to natural as the material allows.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Settings
from .models import Plan, PlanRow, Trigger


def planned_gap(orig: float, cfg: Settings) -> float:
    if orig <= 1e-3:
        return 0.0
    if orig >= cfg.paragraph_threshold:
        return cfg.paragraph_gap
    return min(cfg.gap_max, max(cfg.gap_min, orig * cfg.gap_scale))


def _rates(durations: list[float], b: float, cfg: Settings) -> list[float]:
    arr = np.asarray(durations, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 1e-6:
        z = np.clip((arr - med) / (1.4826 * mad), -1.0, 1.0)
    else:
        z = np.zeros_like(arr)
    r = np.clip(b * (1.0 + cfg.jitter * z), cfg.r_min, cfg.r_max)
    return [float(v) for v in r]


def _rescale(gaps: list[float], budget: float) -> list[float]:
    total = sum(gaps)
    if budget <= 0:
        return [0.0] * len(gaps)
    if total <= 1e-9:
        return [budget / len(gaps)] * len(gaps)
    k = budget / total
    return [g * k for g in gaps]


def allocate(triggers: list[Trigger], total: float, cfg: Settings) -> Plan:
    if not triggers:
        return Plan(total=total, target=total / cfg.speed, speech=0.0, gap_budget=0.0,
                    b=math.inf, feasible=False, rows=[])

    durations = [t.dur for t in triggers]
    gaps: list[float] = []
    prev = 0.0
    for t in triggers:
        gaps.append(max(0.0, t.src_in - prev))
        prev = t.src_out
    gaps.append(max(0.0, total - prev))

    target = total / cfg.speed
    speech = float(sum(durations))
    seed_gaps = [planned_gap(g, cfg) for g in gaps]
    speech_budget = target - sum(seed_gaps)
    b = speech / speech_budget if speech_budget > 1e-6 else math.inf

    rates = _rates(durations, b, cfg)
    planned = [d / r for d, r in zip(durations, rates, strict=True)]
    gap_left = target - sum(planned)
    final_gaps = _rescale(seed_gaps, max(0.0, gap_left))
    deficit = max(0.0, -gap_left)

    rows: list[PlanRow] = []
    for i, (t, r, p) in enumerate(zip(triggers, rates, planned, strict=True)):
        win_max = max(0.25, p)
        rows.append(
            PlanRow(
                tid=t.tid,
                src_in=t.src_in,
                src_out=t.src_out,
                dur=t.dur,
                text=t.text,
                r=round(r, 4),
                win_min=round(max(0.30, 0.85 * win_max), 4),
                win_max=round(win_max, 4),
                gap_before=round(final_gaps[i], 4),
            )
        )

    return Plan(
        total=round(total, 3),
        target=round(target, 3),
        speech=round(speech, 3),
        gap_budget=round(sum(final_gaps), 4),
        b=round(b, 4) if math.isfinite(b) else -1.0,
        feasible=bool(math.isfinite(b) and b <= cfg.r_max + 1e-9),
        deficit=round(deficit, 3),
        rows=rows,
    )


def fit_gaps(actuals: list[float], gaps: list[float], target: float) -> list[float]:
    """Re-spend the time leftover by short sentences back into the pauses."""
    want = target - float(sum(actuals))
    if want <= 0:
        return [0.0] * len(gaps)
    total = float(sum(gaps))
    if total <= 1e-9:
        return [want / len(gaps)] * len(gaps)
    k = want / total
    return [g * k for g in gaps]

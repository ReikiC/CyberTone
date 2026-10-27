from __future__ import annotations

import pytest

from cybertone.config import Settings
from cybertone.models import Trigger
from cybertone.plan import allocate, fit_gaps, planned_gap


def make(durations: list[float], gaps: list[float]) -> tuple[list[Trigger], float]:
    out, t = [], 0.0
    for i, (d, g) in enumerate(zip(durations, gaps, strict=True)):
        t += g
        out.append(Trigger(tid=i, src_in=t, src_out=t + d))
        t += d
    return out, t


def test_pause_budget_uses_silence_first():
    trs, total = make([10.0] * 4, [5.0] * 4)
    plan = allocate(trs, total, Settings(speed=2.0))
    assert plan.feasible is True
    assert plan.target == pytest.approx(total / 2, rel=1e-6)
    assert plan.b < 1.5
    assert sum(r.win_max for r in plan.rows) + sum(r.gap_before for r in plan.rows) <= plan.target + 1e-6


def test_infeasible_material_is_flagged_not_silently_crushed():
    trs, total = make([19.0] * 4, [1.0] * 4)
    cfg = Settings(speed=2.0, r_max=1.75)
    plan = allocate(trs, total, cfg)
    assert plan.feasible is False
    assert plan.deficit > 0
    assert all(r.r <= cfg.r_max + 1e-9 for r in plan.rows)


def test_rate_never_below_natural():
    trs, total = make([4.0] * 10, [3.0] * 10)
    plan = allocate(trs, total, Settings(speed=2.0, r_min=1.0))
    assert min(r.r for r in plan.rows) >= 1.0


def test_windows_are_ordered():
    trs, total = make([3.0, 9.0, 5.0], [1.0, 1.0, 1.0])
    plan = allocate(trs, total, Settings(speed=2.0))
    for row in plan.rows:
        assert 0 < row.win_min <= row.win_max
        assert row.win_max == pytest.approx(row.dur / row.r, rel=1e-3)


def test_gap_shape_preserved_for_paragraphs():
    cfg = Settings()
    assert planned_gap(2.0, cfg) == cfg.paragraph_gap
    assert planned_gap(0.1, cfg) == cfg.gap_min
    assert planned_gap(0.0, cfg) == 0.0


def test_fit_gaps_lands_exactly_on_target():
    gaps = fit_gaps([2.0, 3.0], [1.0, 1.0, 2.0], 12.0)
    assert sum(gaps) + 5.0 == pytest.approx(12.0)
    assert all(g > 0 for g in gaps)


def test_fit_gaps_overrun_returns_zero():
    assert fit_gaps([7.0], [1.0, 1.0], 5.0) == [0.0, 0.0]

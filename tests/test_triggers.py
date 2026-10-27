from __future__ import annotations

import numpy as np

from cybertone import triggers
from cybertone.config import Settings

SR = 48_000


def _speech(dur: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.02))
    return (0.3 * env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


def test_splits_on_long_pauses():
    x = np.concatenate([_silence(1.0), _speech(1.2), _silence(1.0), _speech(1.2), _silence(1.0)])
    rows, stats = triggers.build(x, SR, Settings(merge_gap=0.6))
    assert stats["n_sentences"] == 2
    assert rows[0].src_in < rows[0].src_out < rows[1].src_in
    assert stats["speech_ratio"] < 0.5


def test_short_pauses_are_merged():
    x = np.concatenate([_silence(0.5), _speech(1.0), _silence(0.2), _speech(1.0), _silence(0.5)])
    rows, stats = triggers.build(x, SR, Settings(merge_gap=0.6))
    assert stats["n_sentences"] == 1


def test_long_runs_are_force_split():
    x = np.concatenate([_silence(0.5), _speech(20.0), _silence(0.5)])
    rows, _stats = triggers.build(x, SR, Settings(max_sentence=6.0, merge_gap=0.6))
    assert len(rows) >= 3
    assert all(r.dur <= 6.5 for r in rows)


def test_triggers_do_not_overlap():
    x = np.concatenate([_speech(2.0), _silence(1.0), _speech(2.0), _silence(1.0), _speech(2.0)])
    rows, _ = triggers.build(x, SR, Settings())
    for a, b in zip(rows, rows[1:], strict=False):
        assert a.src_out <= b.src_in + 1e-6

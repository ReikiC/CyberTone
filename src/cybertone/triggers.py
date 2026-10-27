"""Sentence-level trigger nodes: VAD spans + (optional) word timestamps -> sentences."""

from __future__ import annotations

import re

import numpy as np

from . import vad
from .asr import Word
from .config import Settings
from .models import Trigger

HARD_PUNCT = "。！？!?；;…"
CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def _join(parts: list[str]) -> str:
    out = ""
    for p in (p for p in parts if p):
        if not out:
            out = p
            continue
        if CJK.search(out[-1]) or CJK.search(p[0]):
            out += p
        else:
            out += " " + p
    return out


def _absorb_tiny(rows: list[Trigger], min_dur: float = 0.25) -> list[Trigger]:
    out: list[Trigger] = []
    for r in rows:
        if out and r.dur < min_dur:
            prev = out[-1]
            prev.src_out = max(prev.src_out, r.src_out)
            prev.text = _join([prev.text, r.text])
            prev.n_words += r.n_words
        else:
            out.append(r)
    return out


def _chunks_without_text(
    x: np.ndarray, sr: int, start: float, end: float, max_len: float
) -> list[Trigger]:
    rows: list[Trigger] = []
    cur = start
    while end - cur > max_len:
        cut = vad.best_cut(x, sr, cur + max_len * 0.6, cur + max_len)
        rows.append(Trigger(tid=0, src_in=round(cur, 3), src_out=round(cut, 3)))
        cur = cut
    rows.append(Trigger(tid=0, src_in=round(cur, 3), src_out=round(end, 3)))
    return rows


def _merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in spans:
        if out and s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def build(
    x: np.ndarray, sr: int, cfg: Settings, words: list[Word] | None = None
) -> tuple[list[Trigger], dict]:
    spans = _merge_spans(vad.detect(x, sr, cfg), cfg.merge_gap)
    total = x.size / sr
    rows: list[Trigger] = []

    for start, end in spans:
        if words:
            inside = [w for w in words if w.end > start and w.start < end]
            if not inside:
                rows.extend(_chunks_without_text(x, sr, start, end, cfg.max_sentence))
                continue
            group: list[Word] = []
            for w in inside:
                group.append(w)
                span = group[-1].end - group[0].start
                if w.text.endswith(tuple(HARD_PUNCT)) or span >= cfg.max_sentence:
                    rows.append(_row_from_words(group, start, end, cfg))
                    group = []
            if group:
                rows.append(_row_from_words(group, start, end, cfg))
        else:
            rows.extend(_chunks_without_text(x, sr, start, end, cfg.max_sentence))

    rows.sort(key=lambda r: r.src_in)
    # de-overlap (word timestamps can bleed past the VAD edge)
    for i in range(1, len(rows)):
        if rows[i].src_in < rows[i - 1].src_out:
            rows[i].src_in = rows[i - 1].src_out
    rows = [r for r in rows if r.src_out - r.src_in > 0.05]
    if len(rows) > 1 and rows[0].dur < 0.4 and not rows[0].text:
        rows[1].src_in = rows[0].src_in  # leading breath / click, keep the audio
        rows = rows[1:]
    rows = _absorb_tiny(rows)
    for i, r in enumerate(rows):
        r.tid = i

    speech = sum(r.dur for r in rows)
    stats = {
        "total": round(total, 3),
        "speech": round(speech, 3),
        "silence": round(total - speech, 3),
        "speech_ratio": round(speech / total, 4) if total else 0.0,
        "n_sentences": len(rows),
        "n_words": sum(r.n_words for r in rows),
        "has_text": bool(words),
    }
    return rows, stats


def _row_from_words(group: list[Word], start: float, end: float, cfg: Settings) -> Trigger:
    src_in = max(start, group[0].start - 0.02)
    src_out = min(end, group[-1].end + 0.06)
    return Trigger(
        tid=0,
        src_in=round(src_in, 3),
        src_out=round(src_out, 3),
        text=_join([w.text for w in group]),
        n_words=len(group),
    )

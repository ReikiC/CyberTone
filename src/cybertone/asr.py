"""Optional ASR front-end (faster-whisper). Absence is not fatal: the pipeline
still segments audio, it just has no text to synthesise from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass
class Word:
    start: float
    end: float
    text: str


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def transcribe(path: Path, cfg: Settings, cache: Path | None = None) -> list[Word] | None:
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Word(**w) for w in data]

    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None

    device = cfg.asr_device
    if device == "auto":
        device = "cpu"  # ctranslate2 has no Metal backend; CPU int8 is fast enough
    model = WhisperModel(cfg.asr_model, device=device, compute_type="int8")

    # decode ourselves: faster-whisper's PyAV path breaks with av>=19 (metadata_errors)
    from . import audio

    pcm, _sr = audio.load(path, 16_000)

    segments, _info = model.transcribe(
        pcm,
        language=cfg.asr_language,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = (w.word or "").strip()
            if text:
                words.append(Word(round(float(w.start), 3), round(float(w.end), 3), text))
    words.sort(key=lambda w: w.start)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps([w.__dict__ for w in words], ensure_ascii=False), encoding="utf-8"
        )
    return words

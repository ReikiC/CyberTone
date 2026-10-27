"""Shared settings. Defaults are tuned for Chinese talking-head material."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    # --- io ---
    sr: int = 48_000
    work: Path = Path("work")

    # --- target ---
    speed: float = 2.0

    # --- speech rate envelope ---
    r_min: float = 1.00
    r_max: float = 1.75
    jitter: float = 0.10

    # --- pause budget ---
    gap_scale: float = 0.40
    gap_min: float = 0.05
    gap_max: float = 0.22
    paragraph_gap: float = 0.35
    paragraph_threshold: float = 0.55

    # --- vad / sentence building ---
    vad_frame_ms: float = 20.0
    vad_hop_ms: float = 10.0
    vad_enter_db: float = 8.0
    vad_exit_db: float = 4.0
    min_speech: float = 0.20
    min_silence: float = 0.18
    merge_gap: float = 0.60
    max_sentence: float = 12.0

    # --- render ---
    fade_ms: float = 5.0
    room_tone_db: float | None = -58.0
    loudness_lufs: float = -16.0
    peak_ceiling_db: float = -1.0
    stem_lufs: float | None = -20.0
    clean_tts: bool = True
    denoise_strength: float = 0.9
    hp_hz: float = 70.0
    placement: str = "flow"  # flow | anchor

    # --- tts ---
    backend: str = "retime"
    tts_url: str = "http://127.0.0.1:8000"
    tts_model: str = "cosyvoice2"
    tts_seed: int = 42
    tts_temperature: float = 0.70
    tts_ref: Path | None = None
    tts_ref_text: str = ""
    fit_iters: int = 3
    tts_nfe: int = 16

    # --- asr ---
    asr_model: str = "small"
    asr_language: str | None = None
    asr_device: str = "auto"

    # --- video ---
    video_codec: str = "libx264"
    video_crf: int = 18
    audio_bitrate: str = "192k"

    extra: dict = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> Settings:
        data: dict = {}
        if path is not None:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
            data.pop("project", None)
            data.pop("tool", None)
        for key, value in os.environ.items():  # CYBERTONE_TTS_URL -> tts_url
            if key.startswith("CYBERTONE_"):
                name = key[len("CYBERTONE_") :].lower()
                if name in cls.model_fields:
                    data[name] = value
        return cls(**data)

    def ensure_workdir(self) -> Path:
        for sub in ("", "stems", "cache", "plan", "out"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        return self.work

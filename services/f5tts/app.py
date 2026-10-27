"""F5-TTS zero-shot cloning service — the Mac-native counterpart of services/cosyvoice.

Same HTTP contract as the CosyVoice2 service, so the CLI does not care which one it
talks to: ``POST /tts {text, speed, ref_wav_b64, ref_text, ...}`` -> wav bytes.

Runs on Apple silicon through MPS (no CUDA, no docker):

    cd services/f5tts && uv sync
    uv run uvicorn app:app --port 8000
"""

from __future__ import annotations

import base64
import inspect
import io
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from compat import apply as apply_compat

apply_compat()

app = FastAPI(title="cybertone-f5tts")

_model = None
_MODEL_NAME = os.environ.get("CYBERTONE_F5_MODEL", "F5TTS_v1_Base")


def device() -> str:
    env = os.environ.get("CYBERTONE_DEVICE")
    if env:
        return env
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model
    if _model is None:
        from f5_tts.api import F5TTS

        _model = F5TTS(model=_MODEL_NAME, device=device())
    return _model


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    ref_wav_b64: str = ""
    ref_text: str = ""
    sample_rate: int = 48_000
    seed: int = 42
    temperature: float = 0.7
    nfe_step: int = 16
    model: str = "f5tts"


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": device(), "model": _MODEL_NAME}


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if not req.ref_wav_b64:
        raise HTTPException(400, "ref_wav_b64 required (zero-shot cloning)")
    if not req.ref_text.strip():
        raise HTTPException(400, "ref_text required: F5 needs the transcript of the prompt")

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    raw = base64.b64decode(req.ref_wav_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(raw)
        ref_path = fh.name

    try:
        model = get_model()
        kwargs = {}
        params = inspect.signature(model.infer).parameters
        if "speed" in params:
            kwargs["speed"] = req.speed  # only newer builds expose it
        if "remove_silence" in params:
            kwargs["remove_silence"] = False  # client trims
        if "seed" in params:
            kwargs["seed"] = req.seed
        if "nfe_step" in params:
            kwargs["nfe_step"] = req.nfe_step
        wav, sr, _ = model.infer(
            ref_file=ref_path, ref_text=req.ref_text, gen_text=req.text, **kwargs
        )
    finally:
        os.unlink(ref_path)

    speech = np.asarray(wav, dtype=np.float32)
    if speech.ndim > 1:
        speech = speech.squeeze()
    if sr != req.sample_rate:
        from scipy.signal import resample_poly

        import math

        g = math.gcd(int(sr), int(req.sample_rate))
        speech = resample_poly(speech, req.sample_rate // g, int(sr) // g)

    buf = io.BytesIO()
    sf.write(buf, np.clip(speech, -1.0, 1.0), req.sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")

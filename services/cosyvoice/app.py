"""Zero-shot TTS service for CyberTone (CosyVoice2).

The client owns timing: it asks for a sentence at a given speed, measures what comes
back, and bisects. This service only has to be deterministic and reasonably fast.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="cybertone-tts")

_model = None


def get_model():
    global _model
    if _model is None:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        path = os.environ.get("CYBERTONE_MODEL", "pretrained_models/CosyVoice2-0.5B")
        if not os.path.isdir(path):
            raise HTTPException(503, f"model not found: {path}")
        _model = CosyVoice2(path, load_jit=False, load_trt=False, fp16=torch.cuda.is_available())
    return _model


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    ref_wav_b64: str = ""
    ref_text: str = ""
    sample_rate: int = 48_000
    seed: int = 42
    temperature: float = 0.7
    model: str = "cosyvoice2"


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cuda": torch.cuda.is_available()}


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if not req.ref_wav_b64:
        raise HTTPException(400, "ref_wav_b64 required (zero-shot cloning)")

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    raw = base64.b64decode(req.ref_wav_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(raw)
        ref_path = fh.name

    try:
        from cosyvoice.utils.file_utils import load_wav

        prompt_16k = load_wav(ref_path, 16_000)
        model = get_model()
        chunks = [
            out["tts_speech"]
            for out in model.inference_zero_shot(
                req.text, req.ref_text, prompt_16k, stream=False, speed=req.speed
            )
        ]
        if not chunks:
            raise HTTPException(500, "model produced no audio")
        speech = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
    finally:
        os.unlink(ref_path)

    src_sr = 24_000  # CosyVoice2 native output rate
    if req.sample_rate != src_sr:
        from scipy.signal import resample_poly

        import math

        g = math.gcd(int(src_sr), int(req.sample_rate))
        speech = resample_poly(speech, req.sample_rate // g, src_sr // g)

    buf = io.BytesIO()
    sf.write(buf, np.clip(speech, -1.0, 1.0), req.sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")

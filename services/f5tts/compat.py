"""macOS compatibility shims for F5-TTS.

torch 2.14 routes `torchaudio.load` through torchcodec, which dlopens ffmpeg 4.x
dylibs that do not exist on a stock Mac; F5's reference preprocessing also shells out
to ffmpeg/ffprobe through pydub. Both are avoidable: we decode with libsndfile and
back pydub with the ffmpeg that imageio-ffmpeg ships.
"""

from __future__ import annotations


def apply() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_cybertone", False):
        return

    def load(path, **kwargs):  # (channels, time) float32, like torchaudio
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)

    load._cybertone = True
    torchaudio.load = load

    try:
        import imageio_ffmpeg
        from pydub import AudioSegment

        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional
        pass

    try:
        import pydub.utils as pu

        def mediainfo_json(file, read_ahead_limit=-1):
            data, sr = sf.read(str(file), dtype="float32", always_2d=True)
            dur = len(data) / sr
            return {
                "streams": [
                    {
                        "codec_type": "audio",
                        "codec_name": "pcm_s16le",
                        "sample_rate": str(sr),
                        "channels": int(data.shape[1]),
                        "duration": str(dur),
                    }
                ],
                "format": {"duration": str(dur), "format_name": "wav"},
            }

        pu.mediainfo_json = mediainfo_json
    except Exception:  # pragma: no cover
        pass

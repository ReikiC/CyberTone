# CosyVoice2 zero-shot TTS. Needs an NVIDIA GPU; on Apple silicon run the CLI with
# --backend mock, or point CYBERTONE_TTS_URL at a Linux box.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git git-lfs ffmpeg sox libsox-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice

WORKDIR /opt/CosyVoice

RUN pip3 install torch==2.3.1 torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 || \
    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt
RUN pip3 install fastapi "uvicorn[standard]" scipy soundfile

COPY app.py /opt/CosyVoice/service_app.py

EXPOSE 8000
CMD ["uvicorn", "service_app:app", "--host", "0.0.0.0", "--port", "8000"]

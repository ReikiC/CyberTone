# F5-TTS service (Apple silicon)

Zero-shot cloning on macOS. Same `/tts` contract as `services/cosyvoice`, so the CLI
works unchanged via `--backend f5tts` (alias of `service`).

```bash
cd services/f5tts
uv sync                        # pulls torch + f5-tts, ~2.5 GB
uv run uvicorn app:app --port 8000
```

First request downloads `F5TTS_v1_Base` (~1.3 GB) from Hugging Face.

```bash
# from the repo root, in another terminal
uv run cybertone ref Introduction.mp3 -o refs/ref.wav --dur 8   # writes refs/ref.txt too
CYBERTONE_TTS_URL=http://127.0.0.1:8000 \
  uv run cybertone run Introduction.mp3 --asr --backend f5tts
```

Notes:

- `ref_text` must match `ref.wav` closely — a bad transcript is the #1 cause of
  cloned artefacts. Use `--asr-model medium` (or `large-v3`) when generating `refs/ref.txt`.
- If the installed F5 build has no `speed` argument, the service ignores it; the client
  then falls back to its residual stretch path and flags the stem as `degraded`.
- `CYBERTONE_DEVICE=cpu` forces CPU if MPS misbehaves on a given torch build.

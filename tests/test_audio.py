from __future__ import annotations

import numpy as np
import pytest

from cybertone import audio

SR = 48_000


def test_stretch_duration():
    x = np.random.default_rng(0).normal(0, 0.1, SR * 2).astype(np.float32)
    for rate in (1.25, 1.5, 1.75, 2.0):
        y = audio.time_stretch(x, rate, SR)
        assert len(y) / SR == pytest.approx(2.0 / rate, rel=0.01)


def test_stretch_is_identity_at_one():
    x = np.random.default_rng(1).normal(0, 0.1, 1000).astype(np.float32)
    assert np.array_equal(audio.time_stretch(x, 1.0, SR), x)


def test_stretch_keeps_level():
    x = np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR).astype(np.float32) * 0.5
    y = audio.time_stretch(x, 1.5, SR)
    assert audio.rms_db(y) == pytest.approx(audio.rms_db(x[: y.size]), abs=1.0)


def test_loudness_of_reference_sine():
    """A 1 kHz sine at -20 dBFS reads ~-20 LUFS after K-weighting."""
    t = np.arange(SR * 5) / SR
    x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    # BS.1770 reference: a full-scale 1 kHz sine is -3.01 LUFS, so -20 dBFS -> -23.01
    assert audio.integrated_loudness(x, SR) == pytest.approx(-23.01, abs=1.0)


def test_normalize_hits_target():
    t = np.arange(SR * 5) / SR
    x = (0.05 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    y = audio.normalize_loudness(x, SR, -16.0)
    assert audio.integrated_loudness(y, SR) == pytest.approx(-16.0, abs=0.7)


def test_limiter_respects_ceiling():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x[np.linspace(100, x.size - 100, 12).astype(int)] = 2.0  # sparse transients
    y = audio.limit_peaks(x, SR, -1.0)
    assert audio.peak_db(y) <= -0.9
    ceiling = 10 ** (-1.0 / 20.0)
    plain = (x * (ceiling / float(np.max(np.abs(x))))).astype(np.float32)
    assert audio.rms_db(y) > audio.rms_db(plain)  # limited, not merely turned down


def test_room_tone_is_quiet():
    x = np.full(SR * 4, 0.2, dtype=np.float32)
    x[SR : SR * 2] = 1e-4
    tone = audio.quietest_span(x, SR, 0.5)
    assert audio.rms_db(tone) < -60


def test_wav_roundtrip(tmp_path):
    x = np.random.default_rng(2).normal(0, 0.2, SR).astype(np.float32)
    path = tmp_path / "a.wav"
    audio.save(path, x, SR)
    y, sr = audio.load(path, SR)
    assert sr == SR and y.size == x.size
    assert np.allclose(y, x, atol=1e-5)


def test_denoise_lowers_floor_at_equal_loudness():
    """The operational metric: after loudness normalisation the floor must drop."""
    t = np.arange(SR * 4) / SR
    speech = (0.3 * np.sin(2 * np.pi * 180 * t) * ((t % 1.0) < 0.5)).astype(np.float32)
    noisy = speech + np.random.default_rng(0).normal(0, 0.01, SR * 4).astype(np.float32)
    clean = audio.denoise(noisy, SR, 0.9)
    a = audio.normalize_loudness(noisy, SR, -20.0, limit=False)
    b = audio.normalize_loudness(clean, SR, -20.0, limit=False)
    assert audio.noise_floor_db(b, SR) < audio.noise_floor_db(a, SR) - 5.0


def test_denoise_is_transparent_at_zero_strength():
    x = np.random.default_rng(3).normal(0, 0.05, SR).astype(np.float32)
    assert np.array_equal(audio.denoise(x, SR, 0.0), x)


def test_highpass_kills_rumble_keeps_speech_band():
    t = np.arange(SR * 2) / SR
    rumble = np.sin(2 * np.pi * 30 * t).astype(np.float32) * 0.5
    voice = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
    mix = audio.highpass(rumble + voice, SR, 70)
    assert audio.rms_db(mix) == pytest.approx(audio.rms_db(voice), abs=1.5)


def test_noise_floor_of_digital_silence():
    assert audio.noise_floor_db(np.zeros(SR, np.float32), SR) < -100

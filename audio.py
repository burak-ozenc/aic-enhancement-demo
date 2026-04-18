"""
audio.py
Clean speech loading, dBFS-based noise mixing, spectrogram,
and base64 WAV encoding for Dash audio players.
"""

from __future__ import annotations

import io
import wave
import base64
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram, resample_poly, butter, sosfilt

TARGET_SR = 16_000
_RNG_SEED  = 42

NOISE_TYPES = ["White", "Pink", "Babble", "Impulse"]


# ---------------------------------------------------------------------------
# Speech loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_clean_speech() -> tuple[np.ndarray, int]:
    """Load and cache the clean speech sample at 16 kHz."""
    try:
        import pyfar as pf
        signal = pf.signals.files.speech()
        audio  = signal.time[0].astype(np.float32)
        sr     = int(signal.sampling_rate)
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(librosa.ex("libri1"), sr=None, mono=True)
            audio = audio.astype(np.float32)
        except Exception as exc:
            raise RuntimeError("Could not load speech from pyfar or librosa.") from exc

    if sr != TARGET_SR:
        g     = gcd(TARGET_SR, sr)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    return audio, TARGET_SR


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def _white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    n_cols = 16
    array  = rng.standard_normal((n_cols, n)).astype(np.float32)
    cumsum = np.cumsum(array, axis=1)
    for col in range(n_cols):
        step = 2 ** col
        cumsum[col] = np.repeat(cumsum[col, ::step], step)[:n]
    pink = np.sum(cumsum, axis=0)
    pink -= pink.mean()
    return pink.astype(np.float32)


def _babble_noise(n: int, rng: np.random.Generator, n_speakers: int = 8) -> np.ndarray:
    babble = np.zeros(n, dtype=np.float32)
    t      = np.arange(n) / TARGET_SR
    for _ in range(n_speakers):
        fc   = rng.uniform(200, 3400)
        bw   = rng.uniform(200, 600)
        low  = max(0.01, (fc - bw / 2) / (TARGET_SR / 2))
        high = min(0.99, (fc + bw / 2) / (TARGET_SR / 2))
        sos  = butter(4, [low, high], btype="band", output="sos")
        raw  = rng.standard_normal(n).astype(np.float32)
        filt = sosfilt(sos, raw).astype(np.float32)
        env  = (0.5 + 0.5 * np.sin(
            2 * np.pi * rng.uniform(1, 4) * t + rng.uniform(0, 2 * np.pi)
        )).astype(np.float32)
        babble += filt * env
    return babble


def _impulse_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    impulse    = np.zeros(n, dtype=np.float32)
    n_impulses = max(1, int(n / TARGET_SR * 5))
    positions  = rng.integers(0, n, size=n_impulses)
    amplitudes = (
            rng.choice([-1.0, 1.0], size=n_impulses) *
            rng.uniform(0.5, 1.0, size=n_impulses)
    )
    for pos, amp in zip(positions, amplitudes):
        tail  = int(rng.uniform(0.003, 0.010) * TARGET_SR)
        end   = min(n, pos + tail)
        decay = np.exp(-np.linspace(0, 5, end - pos))
        impulse[pos:end] += (amp * decay).astype(np.float32)
    return impulse


_GENERATORS = {
    "White":   _white_noise,
    "Pink":    _pink_noise,
    "Babble":  _babble_noise,
    "Impulse": _impulse_noise,
}


def _mix_noise_types(noise_types: tuple[str, ...], n: int) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED)
    mix = np.zeros(n, dtype=np.float32)
    for nt in noise_types:
        component = _GENERATORS.get(nt, _white_noise)(n, rng)
        rms = np.sqrt(np.mean(component ** 2))
        if rms > 0:
            component /= rms
        mix += component
    return mix


# ---------------------------------------------------------------------------
# dBFS noise mixing (as per spec: -80 to 0 dBFS)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def get_noisy_audio(noise_dbfs: float, noise_types: tuple[str, ...] = ("White",)) -> np.ndarray:
    """
    Mix noise at the given absolute dBFS level into clean speech.
    noise_dbfs: -80 (nearly silent) to 0 (full scale).
    At -80 returns clean copy.
    """
    clean, _ = load_clean_speech()

    if noise_dbfs <= -80:
        return clean.copy()

    noise     = _mix_noise_types(noise_types, len(clean))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    target_rms = 10 ** (noise_dbfs / 20.0)
    if noise_rms > 0:
        noise = noise * (target_rms / noise_rms)

    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(
        audio: np.ndarray,
        sr: int = TARGET_SR,
        n_fft: int = 1024,
        hop: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Sxx = scipy_spectrogram(
        audio, fs=sr, window="hann",
        nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, scaling="spectrum",
    )
    return f, t, 10.0 * np.log10(Sxx + 1e-10)


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

def audio_to_base64_wav(audio: np.ndarray, sr: int = TARGET_SR) -> str:
    audio_i16 = (audio * 32_768.0).clip(-32_768, 32_767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    buf.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()
"""
enhance.py
AIC SDK wrapper: model catalogue, lazy loading, availability probing,
enhancement with warmup fix and advanced parameters, VAD output.
"""

from __future__ import annotations

import asyncio
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import aic_sdk as aic
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR  = Path("models")
LICENSE_KEY = os.environ.get("AIC_SDK_LICENSE", "")

# ---------------------------------------------------------------------------
# Model catalogue — 16 kHz models confirmed compatible with SDK v2
# ---------------------------------------------------------------------------

ALL_MODELS: list[str] = [
    "quail-vf-2.0-l-16khz",   # default — latest VF model (v2 ✓)
    "quail-l-16khz",           # base large (v2 ✓)
    "quail-s-16khz",           # base small — faster (v2 ✓)
    "rook-l-16khz",            # rook architecture large (v2 ✓)
    "rook-s-16khz",            # rook architecture small (v2 ✓)
    # quail-vf-1.1-l-16khz and quail-vf-l-16khz excluded — v1 only, incompatible with SDK v2
]

available_models: list[str] = [ALL_MODELS[0]]
_probe_done = threading.Event()

# Extra silence prepended before audio to allow model state to warm up.
# Discarded from output — prevents the unstable-first-frames artifact.
_WARMUP_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def _probe_models() -> None:
    confirmed: list[str] = []
    for model_id in ALL_MODELS:
        try:
            path = aic.Model.download(model_id, MODELS_DIR)
            aic.Model.from_file(path)
            confirmed.append(model_id)
            print(f"[model probe] ✓ {model_id}")
        except Exception as exc:
            print(f"[model probe] ✗ {model_id} — {exc}")
    available_models.clear()
    available_models.extend(confirmed if confirmed else [ALL_MODELS[0]])
    _probe_done.set()


def start_model_probe() -> None:
    threading.Thread(target=_probe_models, daemon=True).start()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=len(ALL_MODELS))
def _load_model(model_id: str) -> aic.Model:
    model_path = aic.Model.download(model_id, MODELS_DIR)
    return aic.Model.from_file(model_path)


def get_model(model_id: str) -> aic.Model:
    try:
        return _load_model(model_id)
    except Exception as exc:
        raise RuntimeError(f"Model '{model_id}' could not be loaded: {exc}") from exc


def preload_default_model() -> None:
    get_model(ALL_MODELS[0])


# ---------------------------------------------------------------------------
# Advanced parameter defaults
# ---------------------------------------------------------------------------

DEFAULT_BYPASS          = 0.0
DEFAULT_VAD_SENSITIVITY = 6.0
DEFAULT_VAD_HOLD        = 0.05
DEFAULT_VAD_MIN_SPEECH  = 0.0


# ---------------------------------------------------------------------------
# Core async enhancement
# ---------------------------------------------------------------------------

async def _enhance_async(
        audio: np.ndarray,
        sr: int,
        model: aic.Model,
        enhancement_level: float,
        bypass: float,
        vad_sensitivity: float,
        vad_hold: float,
        vad_min_speech: float,
) -> tuple[np.ndarray, np.ndarray | None]:

    processor = aic.ProcessorAsync(model, LICENSE_KEY)
    config    = aic.ProcessorConfig.optimal(model, sample_rate=sr, num_channels=1)
    await processor.initialize_async(config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.reset()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, enhancement_level)
    proc_ctx.set_parameter(aic.ProcessorParameter.Bypass, bypass)
    latency = proc_ctx.get_output_delay()

    vad_ctx = processor.get_vad_context()
    vad_ctx.set_parameter(aic.VadParameter.Sensitivity,           vad_sensitivity)
    vad_ctx.set_parameter(aic.VadParameter.SpeechHoldDuration,    vad_hold)
    vad_ctx.set_parameter(aic.VadParameter.MinimumSpeechDuration, vad_min_speech)

    # Warmup silence + algorithmic latency padding
    warmup_samples = int(_WARMUP_SECONDS * sr)
    silence        = np.zeros(warmup_samples, dtype=np.float32)
    audio_padded   = np.concatenate([silence, audio])           # warmup + real audio

    audio_2d = audio_padded[np.newaxis, :]
    padded   = np.concatenate(
        [np.zeros((1, latency), dtype=np.float32), audio_2d], axis=1
    )

    frame_len  = config.num_frames
    n_total    = padded.shape[1]
    output     = np.zeros_like(padded)
    vad_frames: list[float] = []

    for start in range(0, n_total, frame_len):
        end   = min(start + frame_len, n_total)
        chunk = np.zeros((1, frame_len), dtype=np.float32)
        chunk[:, : end - start] = padded[:, start:end]

        processed = await processor.process_async(chunk)
        output[:, start:end] = processed[:, : end - start]
        vad_frames.append(1.0 if vad_ctx.is_speech_detected() else 0.0)

    # Strip latency + warmup from front, trim to original length
    strip     = latency + warmup_samples
    enhanced  = output[:, strip : strip + audio.shape[0]][0]
    vad_curve = np.array(vad_frames, dtype=np.float32) if vad_frames else None

    return enhanced, vad_curve


# ---------------------------------------------------------------------------
# Public sync interface
# ---------------------------------------------------------------------------

def enhance(
        audio: np.ndarray,
        sr: int,
        model_id: str = ALL_MODELS[0],
        enhancement_level: float = 0.8,
        bypass: float = DEFAULT_BYPASS,
        vad_sensitivity: float = DEFAULT_VAD_SENSITIVITY,
        vad_hold: float = DEFAULT_VAD_HOLD,
        vad_min_speech: float = DEFAULT_VAD_MIN_SPEECH,
) -> tuple[np.ndarray, np.ndarray | None]:
    model = get_model(model_id)
    return asyncio.run(_enhance_async(
        audio, sr, model,
        enhancement_level, bypass,
        vad_sensitivity, vad_hold, vad_min_speech,
    ))
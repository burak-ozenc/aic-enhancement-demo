---
title: ai-coustics Enhancement Demo
emoji: 🎙
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ai-coustics Enhancement Demo

Interactive Dash app to damage clean speech with additive noise and restore it using the **ai-coustics SDK**.

## 🔗 Links

| | URL |
|---|---|
| **Live demo** | [Render](https://aic-enhancement-demo.onrender.com) |
| **Live demo** | [HuggingFace](https://huggingface.co/spaces/burak-ozenc/aic-dashboard) |

---

## Features

- **Three-column layout** - Original · Noisy · Enhanced, side by side
- **Noise Level Slider** (−80 → 0 dBFS) - controls additive noise level per spec
- **Noise Type Selector** - White, Pink, Babble, Impulse - select one or combine multiple
- **Enhancement Level Slider** (0.0 → 1.0) - controls AIC SDK enhancement strength
- **Model Selector** - dropdown populated dynamically with all compatible models (probed at startup)
- **VAD Overlay** - voice activity detection curve rendered on the enhanced spectrogram
- **Spectrograms** - fast scipy-based computation with consistent colour axis across all three panels
- **Apply Button** - single trigger for all processing, prevents mid-playback updates
- **Advanced Parameters** - Bypass (dry/wet), VAD Sensitivity, VAD Speech Hold, VAD Min Speech Duration

## Running locally

```bash
git clone https://github.com/burak-ozenc/aic-dashboard
cd aic-dashboard

uv sync

# Create .env with your license key (never commit this file)
echo "AIC_SDK_LICENSE=your_key_here" > .env

uv run python app.py
# → http://localhost:8050
```

## Deployment

### Render
- `render.yaml` is included - connect the repo and deploy
- Set `AIC_SDK_LICENSE` as an environment variable in the Render dashboard (never in code)

### HuggingFace Spaces
- `Dockerfile` is included - create a Space with SDK: Docker
- Add `AIC_SDK_LICENSE` as a Space secret in Settings → Variables and Secrets

## Security

The API license key is loaded exclusively from environment variables (`AIC_SDK_LICENSE`).
It is **never** present in source code, committed files, or client-side responses.

## Architecture

```
audio.py        → speech loading, noise generation (4 types), dBFS mixing (cached), spectrogram, WAV encoding
enhance.py      → AIC SDK wrapper, model catalogue, availability probe, warmup fix, VAD, sync interface
app.py          → Dash layout, callbacks, Apply button, advanced parameters panel
assets/style.css → dark theme overrides for Dash dropdown components
probe_models.py → standalone script to inspect model properties (latency, frame size, SR)
```

## Notes

- Model availability is probed at startup in a background thread - the dropdown populates live as models confirm
- All models operate at 16 kHz with 30 ms latency and 160-sample frames (verified via `probe_models.py`)
- White noise at −20 dBFS approaches 0 dB SNR against speech RMS - near the physical limit of any enhancement model
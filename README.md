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
| **Live demo (Render)** | _link here_ |
| **Live demo (HuggingFace Spaces)** | _link here_ |

---

## Features

- **Three-column layout** — Original · Noisy · Enhanced, side by side
- **Noise Level Slider** (−80 → 0 dBFS) — real-time noise mixing, updates spectrogram and audio player
- **Enhancement Level Slider** (0.0 → 1.0) — controls AIC SDK enhancement strength
- **Model Selector** — dropdown for available ai-coustics models
- **VAD Overlay** — voice activity detection curve rendered on the enhanced spectrogram (when SDK exposes it)
- **Spectrograms** — fast scipy-based computation with consistent colour axis across all three panels

## Running locally

```bash
git clone <repo>
cd aic-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env with your license key (never commit this file)
echo "AIC_SDK_LICENSE=your_key_here" > .env

python app.py
# → http://localhost:8050
```

## Deployment

### Render
- Set `AIC_SDK_LICENSE` as an environment variable in the Render dashboard (not in code)
- `render.yaml` is included — connect the repo and deploy

### HuggingFace Spaces
- Add `AIC_SDK_LICENSE` as a Space secret
- Set SDK to `gradio` or `docker` as appropriate

## Security

The API license key is loaded exclusively from environment variables (`AIC_SDK_LICENSE`).  
It is **never** present in source code, committed files, or client-side responses.

## Architecture

```
audio.py     → speech loading, noise mixing (cached by level), spectrogram, WAV encoding
enhance.py   → AIC SDK wrapper, model caching, VAD probe, sync interface for Dash
app.py       → Dash layout, callbacks, startup precomputation
```
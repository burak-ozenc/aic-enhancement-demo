"""
probe_models.py
Probe all available models and print their properties.
Run with: uv run python probe_models.py
"""

import asyncio
import os
from pathlib import Path

import numpy as np
import aic_sdk as aic
from dotenv import load_dotenv

load_dotenv()

LICENSE_KEY = os.environ["AIC_SDK_LICENSE"]
MODELS_DIR  = Path("models")
TARGET_SR   = 16_000

MODELS = [
    "quail-vf-2.0-l-16khz",
    "quail-l-16khz",
    "quail-s-16khz",
    "rook-l-16khz",
    "rook-s-16khz",
]


async def probe_one(model_id: str) -> dict:
    try:
        path  = aic.Model.download(model_id, MODELS_DIR)
        model = aic.Model.from_file(path)

        sdk_id      = model.get_id()
        optimal_sr  = model.get_optimal_sample_rate()
        opt_frames  = model.get_optimal_num_frames(TARGET_SR)

        config = aic.ProcessorConfig.optimal(model, sample_rate=TARGET_SR, num_channels=1)

        processor = aic.ProcessorAsync(model, LICENSE_KEY)
        await processor.initialize_async(config)

        proc_ctx = processor.get_processor_context()
        proc_ctx.reset()
        proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
        latency_samples = proc_ctx.get_output_delay()
        latency_ms      = (latency_samples / TARGET_SR) * 1000

        # Run one dummy frame to confirm it works
        dummy     = np.zeros((1, config.num_frames), dtype=np.float32)
        out       = await processor.process_async(dummy)
        out_shape = out.shape

        return {
            "id":              model_id,
            "sdk_id":          sdk_id,
            "optimal_sr":      optimal_sr,
            "opt_frames_16k":  opt_frames,
            "config_sr":       config.sample_rate,
            "config_frames":   config.num_frames,
            "latency_samples": latency_samples,
            "latency_ms":      round(latency_ms, 2),
            "output_shape":    out_shape,
            "status":          "✓ ok",
        }

    except Exception as exc:
        return {"id": model_id, "status": f"✗ {exc}"}


async def main():
    print(f"\nSDK version : {aic.get_sdk_version()}")
    print(f"Model compat: {aic.get_compatible_model_version()}")
    print(f"Target SR   : {TARGET_SR} Hz\n")
    print("=" * 70)

    results = await asyncio.gather(*[probe_one(m) for m in MODELS])

    for r in results:
        print(f"\n🔷 {r['id']}")
        if r["status"] != "✓ ok":
            print(f"   {r['status']}")
            continue
        print(f"   SDK model ID      : {r['sdk_id']}")
        print(f"   Optimal SR        : {r['optimal_sr']} Hz")
        print(f"   Optimal frames    : {r['opt_frames_16k']} @ 16kHz")
        print(f"   Config SR         : {r['config_sr']} Hz")
        print(f"   Config frames     : {r['config_frames']} samples")
        print(f"   Latency           : {r['latency_samples']} samples  ({r['latency_ms']} ms)")
        print(f"   Output shape      : {r['output_shape']}")
        print(f"   Status            : {r['status']}")

    print("\n" + "=" * 70)
    print("\nSummary (latency comparison):\n")
    print(f"  {'Model':<30} {'Frames':>8} {'Latency (ms)':>14}")
    print(f"  {'-'*30} {'-'*8} {'-'*14}")
    for r in results:
        if r["status"] == "✓ ok":
            print(f"  {r['id']:<30} {r['config_frames']:>8} {r['latency_ms']:>14}")


if __name__ == "__main__":
    asyncio.run(main())
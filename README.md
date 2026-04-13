# ChannelVarCache

**Adaptive Feature Caching for Diffusion Model Inference via Channel-Wise Activation Variance**

A training-free, per-block caching method for Stable Diffusion v1.5 that uses channel-wise spatial variance as a near-zero-cost signal to decide when to skip UNet blocks and reuse cached outputs.

## Key Idea

At each denoising step, after a UNet block computes its output, we measure:

```python
block_var = output.var(dim=[2, 3]).mean()  # one scalar per block
```

If variance is low → the block's features have stabilized → safe to skip next step and reuse cached output.

## Project Structure

```
channelvarcache/
├── configs/
│   └── default.yaml          # All experiment hyperparameters
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Modified SD pipeline with caching hooks
│   ├── cache.py               # ChannelVarCache core logic
│   ├── variance_profiler.py   # Phase 1: log per-block variance curves
│   ├── metrics.py             # FID, CLIP Score, LPIPS computation
│   └── utils.py               # Prompt loading, logging, device helpers
├── scripts/
│   ├── 01_profile_variance.py # Run profiling on N prompts
│   ├── 02_find_threshold.py   # Analyze profiles, find optimal tau
│   ├── 03_benchmark.py        # Run caching + baselines, measure quality/speed
│   ├── 04_compare_signals.py  # Head-to-head: variance vs cosine sim vs TeaCache
│   └── 05_generate_figures.py # Produce paper figures from results
├── notebooks/
│   └── exploration.ipynb      # Quick interactive exploration
├── results/                   # Auto-populated by scripts
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create environment
conda create -n channelvarcache python=3.10 -y
conda activate channelvarcache

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (needed for SD v1.5)
huggingface-cli login
```

## Quick Start

```bash
# 1. Profile variance curves (generates results/profiles/)
python scripts/01_profile_variance.py --num_prompts 500

# 2. Find optimal threshold tau (generates results/thresholds/)
python scripts/02_find_threshold.py

# 3. Run benchmark (generates results/benchmark/)
python scripts/03_benchmark.py --methods channelvarcache deepcache teacache none

# 4. Compare caching signals head-to-head
python scripts/04_compare_signals.py

# 5. Generate paper figures
python scripts/05_generate_figures.py
```

## Hardware Requirements

- **Minimum**: 6 GB VRAM GPU (e.g., RTX 2060, GTX 1660), 16 GB RAM
- **Recommended**: 6+ GB VRAM, 32 GB RAM
- SD v1.5 in float16 uses ~4 GB VRAM, leaving ~2 GB for activations and caching

## Config

Edit `configs/default.yaml` to change:
- Model (default: `runwayml/stable-diffusion-v1-5`)
- Number of inference steps (default: 50)
- Guidance scale (default: 7.5)
- Threshold tau (default: auto from profiling)
- Resolution (default: 512x512)
- Prompt source (default: COCO-2017 captions)

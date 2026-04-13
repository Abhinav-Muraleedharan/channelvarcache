#!/usr/bin/env python3
"""Phase 2: Benchmark ChannelVarCache against baselines.

Runs SD v1.5 inference with different caching methods and measures:
- Wall-clock latency
- Image quality (CLIP Score, LPIPS vs uncached baseline)
- Per-step skip rates
- Peak VRAM usage

Usage:
    python scripts/03_benchmark.py
    python scripts/03_benchmark.py --methods channelvarcache deepcache none --num_prompts 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, set_seed, load_prompts, ensure_dir
from src.pipeline import load_pipeline, CachedPipeline
from src.metrics import MetricsComputer


def run_method(pipe, cfg: dict, method: str, prompts: list[str],
               seed: int, thresholds: dict = None) -> dict:
    """Run inference with a specific caching method.
    
    Returns:
        Dict with images, latencies, cache_stats
    """
    # Modify config for this method
    method_cfg = deepcopy(cfg)
    method_cfg["caching"]["method"] = method

    if method == "channelvarcache" and thresholds:
        method_cfg["caching"]["channelvarcache"]["per_block_threshold"] = True

    cached_pipe = CachedPipeline(pipe, method_cfg)

    # Set per-block thresholds if available
    if method == "channelvarcache" and thresholds and hasattr(cached_pipe.cache, "set_thresholds"):
        cached_pipe.cache.set_thresholds(thresholds)

    images = []
    latencies = []
    all_cache_stats = []

    for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
        result = cached_pipe.generate(
            prompt=prompt,
            seed=seed + idx,
        )
        images.append(result["image"])
        latencies.append(result["latency_ms"])
        if "cache_stats" in result:
            all_cache_stats.append(result.get("cache_stats", {}))

    return {
        "images": images,
        "latencies": latencies,
        "cache_stats": all_cache_stats,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "std_latency_ms": (sum((l - sum(latencies)/len(latencies))**2
                               for l in latencies) / len(latencies)) ** 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark caching methods")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--methods", nargs="+",
                        default=["none", "channelvarcache", "deepcache"])
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--thresholds_path", default="results/thresholds/thresholds.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    bench_cfg = cfg.get("benchmark", {})
    num_prompts = args.num_prompts or bench_cfg.get("num_prompts", 500)
    output_dir = args.output_dir or bench_cfg.get("output_dir", "results/benchmark")
    ensure_dir(output_dir)

    set_seed(args.seed)

    # Load thresholds from profiling
    thresholds = None
    if os.path.exists(args.thresholds_path):
        with open(args.thresholds_path) as f:
            thresh_data = json.load(f)
        thresholds = thresh_data.get("thresholds", {})
        logger.info(f"Loaded per-block thresholds from {args.thresholds_path}")
    else:
        logger.warning(f"No thresholds found at {args.thresholds_path}, "
                       "using global threshold from config")

    # Load prompts
    prompt_source = bench_cfg.get("prompt_source", "coco")
    prompts = load_prompts(prompt_source, num_prompts)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Load pipeline (shared across methods)
    pipe = load_pipeline(cfg)

    # Run each method
    results = {}
    for method in args.methods:
        logger.info(f"\nRunning method: {method}")

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        method_results = run_method(
            pipe, cfg, method, prompts, args.seed, thresholds
        )

        peak_mem = (torch.cuda.max_memory_allocated() / 1e6
                    if torch.cuda.is_available() else 0)
        method_results["peak_vram_mb"] = peak_mem

        results[method] = method_results

        logger.info(
            f"  {method}: mean_latency={method_results['mean_latency_ms']:.1f}ms, "
            f"peak_vram={peak_mem:.0f}MB"
        )

    # Compute quality metrics
    logger.info("\nComputing quality metrics...")
    metrics = MetricsComputer(device=pipe.device)

    baseline_images = results.get("none", {}).get("images", [])
    quality_results = {}

    for method in args.methods:
        if method == "none":
            continue
        method_images = results[method]["images"]

        if baseline_images and method_images:
            quality = metrics.compute_all(
                cached_images=method_images,
                baseline_images=baseline_images,
                prompts=prompts,
            )
            quality_results[method] = quality
            logger.info(f"  {method}: {quality}")

    # Compute speedups
    baseline_latency = results.get("none", {}).get("mean_latency_ms", 1)
    speedups = {}
    for method in args.methods:
        if method == "none":
            speedups[method] = 1.0
        else:
            speedups[method] = baseline_latency / results[method]["mean_latency_ms"]

    # Save results
    summary = {
        "num_prompts": len(prompts),
        "methods": {},
    }

    for method in args.methods:
        summary["methods"][method] = {
            "mean_latency_ms": results[method]["mean_latency_ms"],
            "std_latency_ms": results[method]["std_latency_ms"],
            "peak_vram_mb": results[method].get("peak_vram_mb", 0),
            "speedup": speedups.get(method, 1.0),
            "quality": quality_results.get(method, {}),
        }

    with open(Path(output_dir) / "benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Method':<20} {'Latency(ms)':<14} {'Speedup':<10} "
                f"{'CLIP':<10} {'LPIPS':<10} {'VRAM(MB)':<10}")
    logger.info("-" * 70)
    for method in args.methods:
        m = summary["methods"][method]
        q = m.get("quality", {})
        logger.info(
            f"{method:<20} {m['mean_latency_ms']:<14.1f} "
            f"{m['speedup']:<10.2f} "
            f"{q.get('clip_score_cached', '-'):<10} "
            f"{q.get('lpips', '-'):<10} "
            f"{m['peak_vram_mb']:<10.0f}"
        )
    logger.info("=" * 70)

    logger.info(f"\nResults saved to {output_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 1b: Analyze profiling data and find optimal threshold tau.

Reads the variance profiles from Phase 1 and determines:
1. Per-block variance distributions across timesteps
2. Correlation between variance and cosine similarity (validates the signal)
3. Optimal threshold tau for each block (or a single global tau)
4. Expected skip rates at different thresholds

Usage:
    python scripts/02_find_threshold.py
    python scripts/02_find_threshold.py --profiles_dir results/profiles --target_skip_rate 0.4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, ensure_dir
from src.pipeline import ALL_BLOCKS


def load_profiles(profiles_dir: str) -> list[dict]:
    """Load all individual profile JSON files."""
    p = Path(profiles_dir)
    profiles = []
    for f in sorted(p.glob("profile_*.json")):
        with open(f) as fh:
            profiles.append(json.load(fh))
    return profiles


def extract_variance_data(profiles: list[dict]) -> dict:
    """Extract per-block, per-step variance arrays.
    
    Returns:
        Dict mapping block_name -> {
            "variances": 2D array [num_prompts, num_steps],
            "cosine_sims": 2D array [num_prompts, num_steps] (NaN for step 0)
        }
    """
    data = {}
    for block_name in ALL_BLOCKS:
        data[block_name] = {"variances": [], "cosine_sims": []}

    for profile in profiles:
        block_step_var = {name: {} for name in ALL_BLOCKS}
        block_step_cos = {name: {} for name in ALL_BLOCKS}

        for step_data in profile["steps"]:
            for block in step_data["blocks"]:
                name = block["block_name"]
                s = block["step"]
                block_step_var[name][s] = block["channel_variance"]
                block_step_cos[name][s] = block.get("cosine_sim_to_prev")

        num_steps = profile["num_steps"]
        for name in ALL_BLOCKS:
            var_row = [block_step_var[name].get(s, 0) for s in range(num_steps)]
            cos_row = [block_step_cos[name].get(s, np.nan) for s in range(num_steps)]
            data[name]["variances"].append(var_row)
            data[name]["cosine_sims"].append(cos_row)

    # Convert to numpy
    for name in ALL_BLOCKS:
        data[name]["variances"] = np.array(data[name]["variances"])
        data[name]["cosine_sims"] = np.array(data[name]["cosine_sims"])

    return data


def find_optimal_threshold(
    data: dict,
    target_skip_rate: float = 0.4,
    quality_floor: float = 0.95,
) -> dict:
    """Find per-block thresholds that achieve target skip rate.
    
    Strategy: For each block, find the variance threshold below which
    the cosine similarity to the next step is above quality_floor
    (meaning caching is safe). Then adjust to hit the target skip rate.
    
    Args:
        data: output of extract_variance_data
        target_skip_rate: desired fraction of block evaluations to skip
        quality_floor: minimum cosine similarity for safe caching
        
    Returns:
        Dict mapping block_name -> threshold tau
    """
    thresholds = {}

    for block_name in ALL_BLOCKS:
        variances = data[block_name]["variances"]  # [prompts, steps]
        cosines = data[block_name]["cosine_sims"]    # [prompts, steps]

        # Flatten (skip step 0 where cosine is NaN)
        flat_var = variances[:, 1:].flatten()
        flat_cos = cosines[:, 1:].flatten()

        # Remove NaN
        mask = ~np.isnan(flat_cos)
        flat_var = flat_var[mask]
        flat_cos = flat_cos[mask]

        if len(flat_var) == 0:
            thresholds[block_name] = 0.0
            continue

        # Method 1: Find threshold where cosine > quality_floor
        # Sort by variance, find the percentile where cosine drops
        sorted_idx = np.argsort(flat_var)
        sorted_var = flat_var[sorted_idx]
        sorted_cos = flat_cos[sorted_idx]

        # Find the highest variance where mean cosine in that bucket > floor
        n_buckets = 20
        bucket_size = len(sorted_var) // n_buckets
        quality_threshold = sorted_var[-1]  # default: never cache

        for b in range(n_buckets):
            start = b * bucket_size
            end = min((b + 1) * bucket_size, len(sorted_var))
            bucket_cos = sorted_cos[start:end]
            bucket_var_max = sorted_var[end - 1] if end > 0 else 0

            if np.mean(bucket_cos) >= quality_floor:
                quality_threshold = bucket_var_max
            else:
                break

        # Method 2: Simple percentile-based threshold for target skip rate
        percentile_threshold = np.percentile(flat_var, target_skip_rate * 100)

        # Use the more conservative (lower) of the two
        thresholds[block_name] = float(min(quality_threshold, percentile_threshold))

    return thresholds


def analyze_skip_rates(data: dict, thresholds: dict) -> dict:
    """Compute expected skip rates at the given thresholds."""
    results = {}

    for block_name in ALL_BLOCKS:
        variances = data[block_name]["variances"][:, 1:]  # skip step 0
        flat_var = variances.flatten()
        tau = thresholds.get(block_name, 0)
        skip_rate = float(np.mean(flat_var < tau))

        # Per-step skip rate
        num_steps = variances.shape[1]
        per_step_skip = []
        for s in range(num_steps):
            step_vars = variances[:, s]
            per_step_skip.append(float(np.mean(step_vars < tau)))

        results[block_name] = {
            "threshold": tau,
            "overall_skip_rate": skip_rate,
            "per_step_skip_rate": per_step_skip,
            "variance_mean": float(np.mean(flat_var)),
            "variance_std": float(np.std(flat_var)),
        }

    # Global skip rate
    total_evals = sum(
        data[b]["variances"][:, 1:].size for b in ALL_BLOCKS
    )
    total_skips = sum(
        np.sum(data[b]["variances"][:, 1:] < thresholds.get(b, 0))
        for b in ALL_BLOCKS
    )
    results["_global"] = {
        "overall_skip_rate": float(total_skips / total_evals) if total_evals > 0 else 0,
        "estimated_speedup": 1.0 / (1.0 - float(total_skips / total_evals))
            if total_evals > 0 and total_skips < total_evals else 1.0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Find optimal cache threshold")
    parser.add_argument("--profiles_dir", default="results/profiles")
    parser.add_argument("--output_dir", default="results/thresholds")
    parser.add_argument("--target_skip_rate", type=float, default=0.4)
    parser.add_argument("--quality_floor", type=float, default=0.95)
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(args.output_dir)

    # Load profiles
    logger.info(f"Loading profiles from {args.profiles_dir}")
    profiles = load_profiles(args.profiles_dir)
    logger.info(f"Loaded {len(profiles)} profiles")

    if len(profiles) == 0:
        logger.error("No profiles found! Run 01_profile_variance.py first.")
        sys.exit(1)

    # Extract data
    data = extract_variance_data(profiles)

    # Find thresholds
    logger.info(f"Finding thresholds (target skip rate: {args.target_skip_rate})")
    thresholds = find_optimal_threshold(
        data,
        target_skip_rate=args.target_skip_rate,
        quality_floor=args.quality_floor,
    )

    # Analyze skip rates
    skip_analysis = analyze_skip_rates(data, thresholds)

    # Print results
    logger.info("\n=== Threshold Results ===")
    for block_name in ALL_BLOCKS:
        info = skip_analysis[block_name]
        logger.info(
            f"  {block_name:20s}: tau={info['threshold']:.4f}, "
            f"skip_rate={info['overall_skip_rate']:.1%}, "
            f"var_mean={info['variance_mean']:.4f}"
        )

    global_info = skip_analysis["_global"]
    logger.info(f"\n  Global skip rate: {global_info['overall_skip_rate']:.1%}")
    logger.info(f"  Estimated speedup: {global_info['estimated_speedup']:.2f}x")

    # Save results
    output = {
        "thresholds": thresholds,
        "skip_analysis": {k: v for k, v in skip_analysis.items()
                         if k != "_global" and "per_step_skip_rate" not in str(v)},
        "global": global_info,
        "config": {
            "target_skip_rate": args.target_skip_rate,
            "quality_floor": args.quality_floor,
            "num_profiles": len(profiles),
        },
    }

    # Also save full per-step data for plotting
    full_output = {**output, "skip_analysis_full": skip_analysis}

    out_path = Path(args.output_dir) / "thresholds.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    full_path = Path(args.output_dir) / "thresholds_full.json"
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2, default=lambda x: x.tolist()
                  if hasattr(x, "tolist") else x)

    logger.info(f"\nThresholds saved to {out_path}")


if __name__ == "__main__":
    main()

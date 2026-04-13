#!/usr/bin/env python3
"""Phase 3: Compare caching signals head-to-head.

For each profiled prompt, at each block and timestep, compute:
1. Channel variance (our signal)
2. Cosine similarity to previous step (common baseline signal)
3. Timestep embedding L1 difference (TeaCache's signal)

Then measure how well each signal predicts "safe to cache" (defined as
cosine similarity > 0.99 between consecutive step outputs).

Outputs correlation plots and signal quality metrics (AUC, precision-recall).

Usage:
    python scripts/04_compare_signals.py
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
    """Load profile JSON files."""
    p = Path(profiles_dir)
    profiles = []
    for f in sorted(p.glob("profile_*.json")):
        with open(f) as fh:
            profiles.append(json.load(fh))
    return profiles


def extract_signal_pairs(profiles: list[dict]) -> dict:
    """Extract (signal_value, is_cacheable) pairs for each signal type.
    
    "Cacheable" is defined as cosine_similarity > 0.99 between
    consecutive step outputs. This is the ground truth.
    
    Returns:
        Dict mapping signal_name -> {
            block_name -> list of (signal_value, is_cacheable) tuples
        }
    """
    signals = {
        "channel_variance": {b: [] for b in ALL_BLOCKS},
        "timestep_emb_diff": {b: [] for b in ALL_BLOCKS},
        "inverse_cosine": {b: [] for b in ALL_BLOCKS},  # 1 - cosine_sim
    }

    COSINE_THRESHOLD = 0.99  # ground truth: "safe to cache"

    for profile in profiles:
        for step_data in profile["steps"]:
            step_idx = step_data["step"]
            if step_idx == 0:
                continue  # no previous step to compare

            emb_diff = step_data.get("timestep_emb_l1_diff")

            for block in step_data["blocks"]:
                block_name = block["block_name"]
                cos_sim = block.get("cosine_sim_to_prev")
                var = block.get("channel_variance", 0)

                if cos_sim is None:
                    continue

                is_cacheable = cos_sim > COSINE_THRESHOLD

                # Channel variance: low = cacheable
                signals["channel_variance"][block_name].append(
                    (var, is_cacheable)
                )

                # Timestep emb diff: low = cacheable (global signal, same for all blocks)
                if emb_diff is not None:
                    signals["timestep_emb_diff"][block_name].append(
                        (emb_diff, is_cacheable)
                    )

                # Inverse cosine: low = cacheable (this is cheating — it IS the ground truth)
                # Included as an upper bound reference
                signals["inverse_cosine"][block_name].append(
                    (1 - cos_sim, is_cacheable)
                )

    return signals


def compute_signal_quality(signal_pairs: list[tuple]) -> dict:
    """Compute quality metrics for a binary classification signal.
    
    The signal predicts "cacheable" when signal_value < threshold.
    We sweep thresholds and compute precision, recall, and AUC.
    """
    if not signal_pairs:
        return {"auc": 0, "best_f1": 0, "best_threshold": 0}

    values = np.array([v for v, _ in signal_pairs])
    labels = np.array([int(c) for _, c in signal_pairs])

    # Sort by signal value (ascending)
    sorted_idx = np.argsort(values)
    sorted_labels = labels[sorted_idx]
    sorted_values = values[sorted_idx]

    # Sweep thresholds
    n = len(sorted_labels)
    total_pos = np.sum(labels)
    total_neg = n - total_pos

    if total_pos == 0 or total_neg == 0:
        return {"auc": 0, "best_f1": 0, "best_threshold": 0}

    best_f1 = 0
    best_threshold = 0
    precisions = []
    recalls = []

    for i in range(1, n):
        # Predict cacheable for indices 0..i-1 (signal < threshold)
        predicted_pos = i
        true_pos = np.sum(sorted_labels[:i])
        false_pos = predicted_pos - true_pos

        precision = true_pos / predicted_pos if predicted_pos > 0 else 0
        recall = true_pos / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = sorted_values[i]

    # Simple AUC approximation (trapezoidal)
    auc = np.trapz(precisions, recalls) if len(recalls) > 1 else 0

    # Spearman correlation between signal and cacheability
    from scipy.stats import spearmanr
    corr, pval = spearmanr(values, labels)

    return {
        "auc": float(abs(auc)),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "spearman_corr": float(corr),
        "spearman_pval": float(pval),
        "num_samples": n,
        "cacheable_rate": float(total_pos / n),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare caching signals")
    parser.add_argument("--profiles_dir", default="results/profiles")
    parser.add_argument("--output_dir", default="results/signal_comparison")
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(args.output_dir)

    profiles = load_profiles(args.profiles_dir)
    if not profiles:
        logger.error("No profiles found! Run 01_profile_variance.py first.")
        sys.exit(1)
    logger.info(f"Loaded {len(profiles)} profiles")

    # Extract signals
    signals = extract_signal_pairs(profiles)

    # Compute quality for each signal × block combination
    results = {}
    for signal_name in ["channel_variance", "timestep_emb_diff", "inverse_cosine"]:
        results[signal_name] = {}
        all_pairs = []

        for block_name in ALL_BLOCKS:
            pairs = signals[signal_name][block_name]
            if pairs:
                quality = compute_signal_quality(pairs)
                results[signal_name][block_name] = quality
                all_pairs.extend(pairs)

        # Overall quality across all blocks
        if all_pairs:
            results[signal_name]["_overall"] = compute_signal_quality(all_pairs)

    # Print comparison table
    logger.info("\n" + "=" * 85)
    logger.info(f"{'Signal':<25} {'Block':<18} {'AUC':<8} {'F1':<8} "
                f"{'Spearman':<10} {'Threshold':<10}")
    logger.info("-" * 85)

    for signal_name in ["channel_variance", "timestep_emb_diff"]:
        for block_name in ALL_BLOCKS + ["_overall"]:
            if block_name in results[signal_name]:
                q = results[signal_name][block_name]
                display_name = "ALL" if block_name == "_overall" else block_name
                logger.info(
                    f"{signal_name:<25} {display_name:<18} "
                    f"{q['auc']:<8.3f} {q['best_f1']:<8.3f} "
                    f"{q['spearman_corr']:<10.3f} {q['best_threshold']:<10.4f}"
                )
        logger.info("-" * 85)

    logger.info("=" * 85)
    logger.info("\n(inverse_cosine is the oracle upper bound — uses ground truth)")

    # Save results
    out_path = Path(args.output_dir) / "signal_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate paper figures from experimental results.

Creates:
- Figure 3: Variance curves across timesteps (from profiling data)
- Figure 4: Signal quality comparison (from signal comparison data)
- Figure 5: Quality-speedup Pareto curves (from benchmark data)
- Figure 6: Per-step skip rate heatmap

Usage:
    python scripts/05_generate_figures.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, ensure_dir
from src.pipeline import ALL_BLOCKS, ENCODER_BLOCKS, DECODER_BLOCKS, MID_BLOCK

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "down_blocks.0": "#534AB7",
    "down_blocks.1": "#7F77DD",
    "down_blocks.2": "#AFA9EC",
    "down_blocks.3": "#CECBF6",
    "mid_block": "#5F5E5A",
    "up_blocks.0": "#04342C",
    "up_blocks.1": "#0F6E56",
    "up_blocks.2": "#1D9E75",
    "up_blocks.3": "#5DCAA5",
}


def fig_variance_curves(profiles_dir: str, output_path: str, fmt: str = "pdf"):
    """Figure 3: Per-block variance curves across timesteps."""
    summary_path = Path(profiles_dir) / "summary.json"
    if not summary_path.exists():
        print(f"  Skipping: {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    num_steps = summary["num_steps"]
    x = np.arange(num_steps)

    # Plot a subset of blocks for clarity
    blocks_to_plot = ["down_blocks.0", "down_blocks.2", "mid_block",
                      "up_blocks.1", "up_blocks.3"]

    for block_name in blocks_to_plot:
        if block_name not in summary["blocks"]:
            continue
        data = summary["blocks"][block_name]
        means = np.array(data["variance_mean_per_step"])
        stds = np.array(data["variance_std_per_step"])

        color = COLORS.get(block_name, "#888888")
        label = block_name.replace("_", " ").replace(".", " ")
        ax.plot(x, means, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xlabel("Denoising step (t → 0)")
    ax.set_ylabel("Channel-wise spatial variance (σ²)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, num_steps - 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    plt.savefig(f"{output_path}/fig3_variance_curves.{fmt}")
    plt.close()
    print(f"  Saved fig3_variance_curves.{fmt}")


def fig_signal_comparison(signal_dir: str, output_path: str, fmt: str = "pdf"):
    """Figure 4: Signal quality comparison bar chart."""
    comp_path = Path(signal_dir) / "signal_comparison.json"
    if not comp_path.exists():
        print(f"  Skipping: {comp_path} not found")
        return

    with open(comp_path) as f:
        data = json.load(f)

    signals = ["channel_variance", "timestep_emb_diff"]
    signal_labels = ["Channel variance\n(ours)", "Timestep emb. diff\n(TeaCache)"]

    # Get overall metrics
    metrics = ["auc", "best_f1", "spearman_corr"]
    metric_labels = ["AUC", "Best F1", "|Spearman ρ|"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    for m_idx, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[m_idx]
        values = []
        for sig in signals:
            overall = data.get(sig, {}).get("_overall", {})
            val = overall.get(metric, 0)
            if metric == "spearman_corr":
                val = abs(val)
            values.append(val)

        bars = ax.bar(signal_labels, values,
                      color=["#534AB7", "#D85A30"], alpha=0.8, width=0.5)
        ax.set_title(m_label)
        ax.set_ylim(0, 1.05)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_path}/fig4_signal_comparison.{fmt}")
    plt.close()
    print(f"  Saved fig4_signal_comparison.{fmt}")


def fig_skip_rate_heatmap(threshold_dir: str, output_path: str, fmt: str = "pdf"):
    """Figure 6: Per-block, per-step skip rate heatmap."""
    full_path = Path(threshold_dir) / "thresholds_full.json"
    if not full_path.exists():
        print(f"  Skipping: {full_path} not found")
        return

    with open(full_path) as f:
        data = json.load(f)

    skip_analysis = data.get("skip_analysis_full", {})

    # Build heatmap matrix
    blocks = [b for b in ALL_BLOCKS if b in skip_analysis]
    if not blocks:
        print("  Skipping: no block data")
        return

    first_block = blocks[0]
    num_steps = len(skip_analysis[first_block].get("per_step_skip_rate", []))
    if num_steps == 0:
        print("  Skipping: no per-step data")
        return

    matrix = np.zeros((len(blocks), num_steps))
    for i, block in enumerate(blocks):
        rates = skip_analysis[block].get("per_step_skip_rate", [])
        matrix[i, :len(rates)] = rates[:num_steps]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([b.replace("_", " ") for b in blocks], fontsize=7)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("UNet block")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Skip rate", fontsize=9)

    plt.savefig(f"{output_path}/fig6_skip_heatmap.{fmt}")
    plt.close()
    print(f"  Saved fig6_skip_heatmap.{fmt}")


def fig_benchmark_summary(benchmark_dir: str, output_path: str, fmt: str = "pdf"):
    """Figure 5: Quality vs speedup for different methods."""
    bench_path = Path(benchmark_dir) / "benchmark_results.json"
    if not bench_path.exists():
        print(f"  Skipping: {bench_path} not found")
        return

    with open(bench_path) as f:
        data = json.load(f)

    methods = data.get("methods", {})
    if not methods:
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    method_colors = {
        "none": "#5F5E5A",
        "channelvarcache": "#534AB7",
        "deepcache": "#D85A30",
        "teacache": "#1D9E75",
    }

    for method, info in methods.items():
        speedup = info.get("speedup", 1.0)
        quality = info.get("quality", {})
        clip = quality.get("clip_score_cached", quality.get("clip_score_baseline", 0))

        if isinstance(clip, str):
            continue

        color = method_colors.get(method, "#888888")
        ax.scatter(speedup, clip, color=color, s=80, zorder=5, edgecolors="white")
        ax.annotate(method, (speedup, clip), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=color)

    ax.set_xlabel("Speedup (×)")
    ax.set_ylabel("CLIP Score")
    ax.grid(True, alpha=0.2)

    plt.savefig(f"{output_path}/fig5_quality_speedup.{fmt}")
    plt.close()
    print(f"  Saved fig5_quality_speedup.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--profiles_dir", default="results/profiles")
    parser.add_argument("--signal_dir", default="results/signal_comparison")
    parser.add_argument("--threshold_dir", default="results/thresholds")
    parser.add_argument("--benchmark_dir", default="results/benchmark")
    parser.add_argument("--output_dir", default="results/figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(args.output_dir)

    print("Generating figures...")

    print("\n[1/4] Variance curves")
    fig_variance_curves(args.profiles_dir, args.output_dir, args.format)

    print("\n[2/4] Signal comparison")
    fig_signal_comparison(args.signal_dir, args.output_dir, args.format)

    print("\n[3/4] Quality-speedup Pareto")
    fig_benchmark_summary(args.benchmark_dir, args.output_dir, args.format)

    print("\n[4/4] Skip rate heatmap")
    fig_skip_rate_heatmap(args.threshold_dir, args.output_dir, args.format)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

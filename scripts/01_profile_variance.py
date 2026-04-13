#!/usr/bin/env python3
"""Phase 1: Profile per-block variance curves on N prompts.

Usage:
    python scripts/01_profile_variance.py --num_prompts 500
    python scripts/01_profile_variance.py --config configs/default.yaml --num_prompts 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, set_seed, load_prompts, ensure_dir
from src.pipeline import load_pipeline
from src.variance_profiler import VarianceProfiler


def main():
    parser = argparse.ArgumentParser(description="Profile per-block variance curves")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Override number of prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    # Override config with CLI args
    prof_cfg = cfg.get("profiling", {})
    num_prompts = args.num_prompts or prof_cfg.get("num_prompts", 500)
    output_dir = args.output_dir or prof_cfg.get("output_dir", "results/profiles")
    seed = args.seed

    set_seed(seed)

    # Load prompts
    prompt_source = prof_cfg.get("prompt_source", "coco")
    captions_path = prof_cfg.get("coco_captions_path")
    logger.info(f"Loading {num_prompts} prompts from {prompt_source}")
    prompts = load_prompts(prompt_source, num_prompts, captions_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Load pipeline
    pipe = load_pipeline(cfg)

    # Profile
    profiler = VarianceProfiler(pipe, cfg)
    profiles = profiler.profile_all(prompts, output_dir, seed=seed)

    logger.info(f"Profiling complete! {len(profiles)} prompts profiled")
    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"  - Individual profiles: profile_0000.json ... profile_{len(profiles)-1:04d}.json")
    logger.info(f"  - Aggregated summary: summary.json")


if __name__ == "__main__":
    main()

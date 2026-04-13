"""Utility functions for prompt loading, config, logging, and device setup."""

import os
import json
import yaml
import random
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("channelvarcache")


def set_seed(seed: Optional[int] = 42):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    """Get compute device from config."""
    device_str = cfg.get("model", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def get_dtype(cfg: dict) -> torch.dtype:
    """Get torch dtype from config string."""
    dtype_str = cfg.get("model", {}).get("dtype", "float16")
    return {"float16": torch.float16, "float32": torch.float32,
            "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)


def get_memory_usage() -> dict:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
        "max_allocated": torch.cuda.max_memory_allocated() / 1e6,
    }


def load_prompts(source: str = "coco", num_prompts: int = 500,
                 captions_path: Optional[str] = None) -> list[str]:
    """Load text prompts for evaluation.
    
    Args:
        source: "coco" for COCO captions, "parti" for PartiPrompts, "custom" for file
        num_prompts: number of prompts to return
        captions_path: path to captions file (JSON or txt)
    
    Returns:
        List of prompt strings
    """
    if source == "coco":
        return _load_coco_prompts(num_prompts, captions_path)
    elif source == "parti":
        return _load_parti_prompts(num_prompts)
    elif source == "custom" and captions_path:
        return _load_custom_prompts(num_prompts, captions_path)
    else:
        # Fallback: simple built-in prompts for testing
        return _get_test_prompts(num_prompts)


def _load_coco_prompts(num_prompts: int, captions_path: Optional[str]) -> list[str]:
    """Load COCO-2017 captions."""
    if captions_path and os.path.exists(captions_path):
        with open(captions_path, "r") as f:
            data = json.load(f)
        captions = [ann["caption"] for ann in data["annotations"]]
    else:
        # Try to download or use HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceM4/COCO", split="train",
                              trust_remote_code=True)
            captions = []
            for item in ds:
                if "sentences" in item and "raw" in item["sentences"]:
                    captions.append(item["sentences"]["raw"])
                elif "caption" in item:
                    captions.append(item["caption"])
                if len(captions) >= num_prompts * 2:
                    break
        except Exception:
            logging.warning("Could not load COCO captions, using fallback prompts")
            return _get_test_prompts(num_prompts)

    # Deduplicate and sample
    captions = list(set(captions))
    random.shuffle(captions)
    return captions[:num_prompts]


def _load_parti_prompts(num_prompts: int) -> list[str]:
    """Load PartiPrompts benchmark."""
    try:
        from datasets import load_dataset
        ds = load_dataset("nateraw/parti-prompts", split="train")
        prompts = [item["Prompt"] for item in ds]
        random.shuffle(prompts)
        return prompts[:num_prompts]
    except Exception:
        logging.warning("Could not load PartiPrompts, using fallback")
        return _get_test_prompts(num_prompts)


def _load_custom_prompts(num_prompts: int, path: str) -> list[str]:
    """Load prompts from a text file (one per line) or JSON list."""
    with open(path, "r") as f:
        if path.endswith(".json"):
            prompts = json.load(f)
        else:
            prompts = [line.strip() for line in f if line.strip()]
    return prompts[:num_prompts]


def _get_test_prompts(num_prompts: int) -> list[str]:
    """Built-in test prompts for quick debugging."""
    base = [
        "A photo of a cat sitting on a windowsill",
        "A beautiful sunset over the ocean with orange and purple clouds",
        "A robot painting a portrait in a studio",
        "A cozy cabin in the snowy mountains at night",
        "A bustling street market in Tokyo with colorful lanterns",
        "An astronaut riding a horse on Mars",
        "A steampunk cityscape with airships and clockwork towers",
        "A bowl of ramen with steam rising, top-down view",
        "A field of sunflowers under a blue sky",
        "A medieval castle on a cliff overlooking the sea",
        "A close-up photo of a hummingbird feeding from a flower",
        "An abstract painting with geometric shapes and bold colors",
        "A vintage car parked in front of a diner at dusk",
        "A corgi wearing a tiny crown sitting on a throne",
        "A cyberpunk alleyway with neon signs reflecting in puddles",
        "A watercolor painting of a Venetian canal",
        "A macro photo of dewdrops on a spider web",
        "A cozy reading nook with bookshelves and warm lighting",
        "A futuristic space station orbiting Earth",
        "A photorealistic portrait of an elderly fisherman",
    ]
    # Repeat if needed
    while len(base) < num_prompts:
        base = base + base
    return base[:num_prompts]


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

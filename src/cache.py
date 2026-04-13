"""Core caching logic for ChannelVarCache and baselines."""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger("channelvarcache")


@dataclass
class CacheEntry:
    """Cached output for a single UNet block."""
    tensor: Optional[torch.Tensor] = None   # cached output activation
    variance: float = float("inf")          # channel-wise spatial variance
    step: int = -1                          # timestep when cached


class ChannelVarCache:
    """Per-block adaptive caching using channel-wise spatial variance.
    
    At each denoising step, after a block computes its output, we measure:
        sigma2 = output.var(dim=[2, 3]).mean()
    
    If sigma2 < threshold at the previous step, we skip the block and
    reuse the cached output. Otherwise we recompute and update the cache.
    
    Args:
        threshold: Global variance threshold. If None, must be set via
                   set_thresholds() after profiling.
        per_block_threshold: Dict mapping block names to thresholds.
                             Overrides global threshold if provided.
        warmup_steps: Number of initial steps to always compute fully.
        normalize: If True, normalize variance by running mean per block.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        per_block_threshold: Optional[dict[str, float]] = None,
        warmup_steps: int = 1,
        normalize: bool = True,
    ):
        self.global_threshold = threshold
        self.per_block_threshold = per_block_threshold or {}
        self.warmup_steps = warmup_steps
        self.normalize = normalize

        # State
        self._cache: dict[str, CacheEntry] = {}
        self._running_mean: dict[str, float] = {}  # for normalization
        self._running_count: dict[str, int] = {}
        self._current_step: int = 0
        self._stats: dict[str, list] = {}  # for logging

    def reset(self):
        """Reset cache state for a new generation."""
        self._cache.clear()
        self._running_mean.clear()
        self._running_count.clear()
        self._current_step = 0
        self._stats.clear()

    def set_step(self, step: int):
        """Set the current denoising step index."""
        self._current_step = step

    def set_thresholds(self, thresholds: dict[str, float]):
        """Set per-block thresholds from profiling results."""
        self.per_block_threshold = thresholds

    def get_threshold(self, block_name: str) -> float:
        """Get the threshold for a specific block."""
        if block_name in self.per_block_threshold:
            return self.per_block_threshold[block_name]
        if self.global_threshold is not None:
            return self.global_threshold
        return float("inf")  # never cache if no threshold set

    @staticmethod
    def compute_channel_variance(tensor: torch.Tensor) -> float:
        """Compute channel-wise spatial variance of an activation tensor.
        
        Args:
            tensor: shape (batch, channels, height, width)
        
        Returns:
            Scalar variance value (averaged over batch and channels)
        """
        # variance across spatial dims (H, W) for each channel, then mean
        return tensor.var(dim=[2, 3]).mean().item()

    def should_skip(self, block_name: str) -> bool:
        """Check if a block should be skipped (use cached output).
        
        Returns True if:
        1. We're past the warmup phase
        2. The block has a cached entry
        3. The cached variance was below threshold
        """
        if self._current_step < self.warmup_steps:
            return False

        if block_name not in self._cache:
            return False

        entry = self._cache[block_name]
        threshold = self.get_threshold(block_name)

        # Optionally normalize variance by running mean
        var_value = entry.variance
        if self.normalize and block_name in self._running_mean:
            mean = self._running_mean[block_name]
            if mean > 1e-8:
                var_value = var_value / mean

        return var_value < threshold

    def get_cached(self, block_name: str) -> Optional[torch.Tensor]:
        """Get cached output tensor for a block."""
        if block_name in self._cache and self._cache[block_name].tensor is not None:
            return self._cache[block_name].tensor
        return None

    def update(self, block_name: str, output: torch.Tensor):
        """Update cache with new block output and its variance.
        
        Call this after a block has been computed (not skipped).
        
        Args:
            block_name: identifier for the UNet block
            output: the block's output tensor (batch, C, H, W)
        """
        var = self.compute_channel_variance(output)

        # Update cache entry
        self._cache[block_name] = CacheEntry(
            tensor=output.detach(),  # detach to avoid graph retention
            variance=var,
            step=self._current_step,
        )

        # Update running mean for normalization
        if block_name not in self._running_count:
            self._running_count[block_name] = 0
            self._running_mean[block_name] = 0.0

        count = self._running_count[block_name]
        self._running_mean[block_name] = (
            self._running_mean[block_name] * count + var
        ) / (count + 1)
        self._running_count[block_name] = count + 1

        # Log stats
        if block_name not in self._stats:
            self._stats[block_name] = []
        self._stats[block_name].append({
            "step": self._current_step,
            "variance": var,
            "action": "compute",
        })

    def record_skip(self, block_name: str):
        """Record that a block was skipped (for stats tracking)."""
        if block_name not in self._stats:
            self._stats[block_name] = []
        self._stats[block_name].append({
            "step": self._current_step,
            "variance": self._cache[block_name].variance if block_name in self._cache else 0,
            "action": "skip",
        })

    def get_stats(self) -> dict:
        """Get caching statistics for analysis."""
        stats = {}
        for block_name, entries in self._stats.items():
            total = len(entries)
            skips = sum(1 for e in entries if e["action"] == "skip")
            stats[block_name] = {
                "total_steps": total,
                "skipped": skips,
                "computed": total - skips,
                "skip_rate": skips / total if total > 0 else 0,
                "variances": [e["variance"] for e in entries],
            }
        return stats

    @property
    def total_skip_rate(self) -> float:
        """Overall fraction of block evaluations that were skipped."""
        total = sum(len(v) for v in self._stats.values())
        skips = sum(
            sum(1 for e in v if e["action"] == "skip")
            for v in self._stats.values()
        )
        return skips / total if total > 0 else 0


class DeepCacheBaseline:
    """Simplified DeepCache baseline for comparison.
    
    Caches encoder features at a fixed interval N and reuses them
    for N-1 subsequent steps, skipping the encoder computation.
    
    This is a simplified version that caches entire encoder block outputs
    at a fixed interval, not the full DeepCache algorithm with selective
    branch caching. For exact comparison, use the official DeepCache
    implementation from diffusers.
    """

    def __init__(self, cache_interval: int = 3):
        self.cache_interval = cache_interval
        self._cache: dict[str, torch.Tensor] = {}
        self._current_step: int = 0

    def reset(self):
        self._cache.clear()
        self._current_step = 0

    def set_step(self, step: int):
        self._current_step = step

    def is_compute_step(self) -> bool:
        """Whether this step should do full computation."""
        return self._current_step % self.cache_interval == 0

    def cache_output(self, block_name: str, tensor: torch.Tensor):
        self._cache[block_name] = tensor.detach()

    def get_cached(self, block_name: str) -> Optional[torch.Tensor]:
        return self._cache.get(block_name)


class TeaCacheBaseline:
    """Simplified TeaCache baseline for comparison.
    
    Uses L1 difference of timestep-embedding-modulated inputs between
    consecutive steps as a proxy for output similarity. If the difference
    is below a threshold, skip computation and reuse cache.
    
    This is a simplified reimplementation for comparison purposes.
    The official TeaCache has calibrated polynomial rescaling which
    we approximate with a simple threshold.
    """

    def __init__(self, rel_l1_thresh: float = 0.3):
        self.rel_l1_thresh = rel_l1_thresh
        self._prev_input: Optional[torch.Tensor] = None
        self._cache: Optional[torch.Tensor] = None
        self._current_step: int = 0

    def reset(self):
        self._prev_input = None
        self._cache = None
        self._current_step = 0

    def set_step(self, step: int):
        self._current_step = step

    def should_skip(self, current_input: torch.Tensor) -> bool:
        """Check if we should skip based on input difference."""
        if self._prev_input is None or self._cache is None:
            return False

        # Relative L1 difference
        diff = (current_input - self._prev_input).abs().mean()
        norm = self._prev_input.abs().mean().clamp(min=1e-8)
        rel_diff = (diff / norm).item()

        return rel_diff < self.rel_l1_thresh

    def update(self, current_input: torch.Tensor, output: torch.Tensor):
        """Store input and output for next step's comparison."""
        self._prev_input = current_input.detach()
        self._cache = output.detach()

    def get_cached(self) -> Optional[torch.Tensor]:
        return self._cache

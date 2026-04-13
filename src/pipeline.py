"""Modified Stable Diffusion pipeline with per-block caching for real speedup.

This module implements the complete ChannelVarCache inference pipeline by
manually stepping through the UNet's encoder/mid/decoder blocks and
selectively skipping blocks whose channel variance fell below threshold
at the previous timestep.

Key implementation detail: each encoder (down) block returns two things:
  1. `sample` - the hidden state passed to the next block
  2. `res_samples` - a tuple of intermediate residuals used as skip connections
     by the corresponding decoder (up) block

When we cache a down_block, we must cache BOTH the sample and the res_samples.
When we cache an up_block, we only cache the output sample.
"""

import time
from typing import Optional
from dataclasses import dataclass, field

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from tqdm import tqdm
import logging

from .cache import ChannelVarCache, DeepCacheBaseline, TeaCacheBaseline

logger = logging.getLogger("channelvarcache")


# ── Block names ───────────────────────────────────────────────────────────
ENCODER_BLOCKS = [f"down_blocks.{i}" for i in range(4)]
MID_BLOCK = "mid_block"
DECODER_BLOCKS = [f"up_blocks.{i}" for i in range(4)]
ALL_BLOCKS = ENCODER_BLOCKS + [MID_BLOCK] + DECODER_BLOCKS


def load_pipeline(cfg: dict) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline from config."""
    model_name = cfg["model"]["name"]
    dtype_str = cfg["model"]["dtype"]
    device = cfg["model"]["device"]

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    logger.info(f"Loading {model_name} in {dtype_str} on {device}")

    sched_name = cfg.get("inference", {}).get("scheduler", "ddim")
    if sched_name == "ddim":
        scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    elif sched_name in ("dpmsolver++", "dpmsolver"):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler")
    else:
        scheduler = None

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("xformers enabled")
    except Exception:
        pass

    return pipe


@dataclass
class DownBlockCacheEntry:
    """Cached outputs for a single encoder (down) block."""
    sample: Optional[torch.Tensor] = None
    res_samples: Optional[tuple] = None
    variance: float = float("inf")
    step: int = -1


@dataclass
class BlockCacheEntry:
    """Cached output for mid or decoder (up) block."""
    sample: Optional[torch.Tensor] = None
    variance: float = float("inf")
    step: int = -1


class SelectiveUNetExecutor:
    """Executes the UNet forward pass with per-block caching.

    Manually walks through the UNet's blocks instead of calling
    unet.forward(), skipping individual blocks and injecting
    cached outputs for REAL latency savings.

    Execution flow (mirrors UNet2DConditionModel.forward):
    1. Time embedding + text processing (always runs)
    2. conv_in (always runs)
    3. Down blocks 0-3: each produces (sample, res_samples)
    4. Mid block: transforms sample
    5. Up blocks 0-3: each consumes sample + res_samples from stack
    6. conv_norm_out + conv_out (always runs)

    Caching rules:
    - Down block cached: skip computation, use cached (sample, res_samples)
    - Mid block cached: skip computation, use cached sample
    - Up block cached: skip computation, use cached sample
      (still pops the correct res_samples from stack so indexing stays aligned)
    """

    def __init__(self, unet, cache: ChannelVarCache):
        self.unet = unet
        self.cache = cache
        self._down_cache: dict[int, DownBlockCacheEntry] = {}
        self._mid_cache: BlockCacheEntry = BlockCacheEntry()
        self._up_cache: dict[int, BlockCacheEntry] = {}

    def reset(self):
        """Reset all caches for a new image generation."""
        self._down_cache.clear()
        self._mid_cache = BlockCacheEntry()
        self._up_cache.clear()
        self.cache.reset()

    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Run UNet forward with selective block caching.

        Args:
            sample: noisy latent (batch, 4, H, W)
            timestep: current timestep scalar
            encoder_hidden_states: text embeddings (batch, seq_len, dim)
            step_index: denoising step index (0, 1, 2, ...)

        Returns:
            Predicted noise tensor (batch, 4, H, W)
        """
        unet = self.unet
        cache = self.cache
        cache.set_step(step_index)

        # ── 1. Time embedding (always, cheap) ────────────────────────────
        t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
        emb = unet.time_embedding(t_emb)

        encoder_hidden_states = unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None
        )

        # ── 2. conv_in (always, cheap) ───────────────────────────────────
        sample = unet.conv_in(sample)

        # ── 3. Down blocks ───────────────────────────────────────────────
        down_block_res_samples = (sample,)

        for i, downsample_block in enumerate(unet.down_blocks):
            block_name = f"down_blocks.{i}"

            if cache.should_skip(block_name) and i in self._down_cache:
                # ── SKIP ──
                cached = self._down_cache[i]
                sample = cached.sample
                res_samples = cached.res_samples
                cache.record_skip(block_name)
            else:
                # ── COMPUTE ──
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                    )

                # Measure variance, update caches
                var = self._compute_variance(sample)
                cache.update(block_name, sample)
                self._down_cache[i] = DownBlockCacheEntry(
                    sample=sample.detach(),
                    res_samples=tuple(r.detach() for r in res_samples),
                    variance=var,
                    step=step_index,
                )

            down_block_res_samples += res_samples

        # ── 4. Mid block ─────────────────────────────────────────────────
        block_name = MID_BLOCK

        if cache.should_skip(block_name) and self._mid_cache.sample is not None:
            sample = self._mid_cache.sample
            cache.record_skip(block_name)
        else:
            if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
                sample = unet.mid_block(
                    sample, emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = unet.mid_block(sample, emb)

            var = self._compute_variance(sample)
            cache.update(block_name, sample)
            self._mid_cache = BlockCacheEntry(
                sample=sample.detach(),
                variance=var,
                step=step_index,
            )

        # ── 5. Up blocks ─────────────────────────────────────────────────
        for i, upsample_block in enumerate(unet.up_blocks):
            block_name = f"up_blocks.{i}"
            is_final_block = i == len(unet.up_blocks) - 1

            # Pop residual samples from the stack
            # MUST happen regardless of skip/compute to keep alignment
            n_resnets = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if cache.should_skip(block_name) and i in self._up_cache:
                # ── SKIP ──
                sample = self._up_cache[i].sample
                cache.record_skip(block_name)
            else:
                # ── COMPUTE ──
                upsample_size = None
                if not is_final_block and len(down_block_res_samples) > 0:
                    next_shape = down_block_res_samples[-1].shape[2:]
                    if sample.shape[2:] != next_shape:
                        upsample_size = next_shape

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=upsample_size,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

                var = self._compute_variance(sample)
                cache.update(block_name, sample)
                self._up_cache[i] = BlockCacheEntry(
                    sample=sample.detach(),
                    variance=var,
                    step=step_index,
                )

        # ── 6. Output convolution (always, cheap) ────────────────────────
        if unet.conv_norm_out:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)

        return sample

    @staticmethod
    def _compute_variance(tensor: torch.Tensor) -> float:
        """Compute channel-wise spatial variance: output.var(dim=[2,3]).mean()"""
        if tensor.dim() == 4:
            return tensor.var(dim=[2, 3]).mean().item()
        return 0.0


class CachedPipeline:
    """Full inference pipeline with ChannelVarCache, DeepCache, or no caching.

    Handles prompt encoding, latent preparation, the denoising loop
    with selective block execution, and VAE decoding.
    """

    def __init__(self, pipe: StableDiffusionPipeline, cfg: dict):
        self.pipe = pipe
        self.cfg = cfg
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

        cache_cfg = cfg.get("caching", {})
        self.method = cache_cfg.get("method", "channelvarcache")

        if self.method == "channelvarcache":
            cv_cfg = cache_cfg.get("channelvarcache", {})
            self.cache = ChannelVarCache(
                threshold=cv_cfg.get("threshold"),
                warmup_steps=cv_cfg.get("warmup_steps", 1),
                normalize=cv_cfg.get("normalize_variance", True),
            )
            self.executor = SelectiveUNetExecutor(self.unet, self.cache)
        else:
            self.cache = None
            self.executor = None

    def set_thresholds(self, thresholds: dict[str, float]):
        """Set per-block thresholds (from profiling)."""
        if isinstance(self.cache, ChannelVarCache):
            self.cache.set_thresholds(thresholds)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> dict:
        """Generate an image. Returns dict with image, latency_ms, step_details, cache_stats."""
        inf_cfg = self.cfg.get("inference", {})
        num_steps = num_steps or inf_cfg.get("num_steps", 50)
        guidance_scale = guidance_scale or inf_cfg.get("guidance_scale", 7.5)
        resolution = inf_cfg.get("resolution", 512)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if self.method == "channelvarcache" and self.executor is not None:
            return self._generate_channelvarcache(
                prompt, generator, num_steps, guidance_scale, resolution)
        elif self.method == "deepcache":
            return self._generate_deepcache(
                prompt, generator, num_steps, guidance_scale, resolution)
        else:
            return self._generate_baseline(
                prompt, generator, num_steps, guidance_scale, resolution)

    # ── Helper methods ────────────────────────────────────────────────────

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt into [uncond, cond] text embeddings for CFG."""
        text_inputs = self.pipe.tokenizer(
            prompt, padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]

        uncond_input = self.pipe.tokenizer(
            "", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeds = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]

        return torch.cat([uncond_embeds, prompt_embeds])

    def _prepare_latents(self, resolution: int, generator) -> torch.Tensor:
        """Create initial noise latents."""
        shape = (1, 4, resolution // 8, resolution // 8)
        latents = torch.randn(shape, generator=generator,
                              device=self.device, dtype=self.dtype)
        return latents * self.scheduler.init_noise_sigma

    def _decode_latents(self, latents: torch.Tensor):
        """VAE decode latents to PIL image."""
        latents = 1 / 0.18215 * latents
        image = self.pipe.vae.decode(latents).sample
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image

    # ── ChannelVarCache generation ────────────────────────────────────────

    def _generate_channelvarcache(self, prompt, generator, num_steps,
                                  guidance_scale, resolution) -> dict:
        """Generation with selective per-block caching."""
        self.executor.reset()

        text_embeddings = self._encode_prompt(prompt)
        latents = self._prepare_latents(resolution, generator)
        self.scheduler.set_timesteps(num_steps, device=self.device)

        step_details = []
        start = time.perf_counter()

        for i, t in enumerate(self.scheduler.timesteps):
            step_start = time.perf_counter()

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # Selective UNet forward — blocks may be skipped
            noise_pred = self.executor.forward(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                step_index=i,
            )

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1000

            # Count skips
            skips = 0
            for bn in ALL_BLOCKS:
                st = self.cache._stats.get(bn, [])
                if st and st[-1]["step"] == i and st[-1]["action"] == "skip":
                    skips += 1

            step_details.append({
                "step": i, "timestep": t.item(), "latency_ms": step_ms,
                "blocks_skipped": skips, "blocks_total": len(ALL_BLOCKS),
            })

        total_ms = (time.perf_counter() - start) * 1000
        image = self._decode_latents(latents)

        return {
            "image": image,
            "latency_ms": total_ms,
            "step_details": step_details,
            "cache_stats": self.cache.get_stats(),
        }

    # ── DeepCache baseline ────────────────────────────────────────────────

    def _generate_deepcache(self, prompt, generator, num_steps,
                            guidance_scale, resolution) -> dict:
        """Generation with simplified DeepCache (fixed-interval encoder caching)."""
        dc_cfg = self.cfg.get("caching", {}).get("deepcache", {})
        cache_interval = dc_cfg.get("cache_interval", 3)

        text_embeddings = self._encode_prompt(prompt)
        latents = self._prepare_latents(resolution, generator)
        self.scheduler.set_timesteps(num_steps, device=self.device)

        cached_down = {}
        cached_mid = None

        step_details = []
        start = time.perf_counter()

        for i, t in enumerate(self.scheduler.timesteps):
            step_start = time.perf_counter()

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            is_compute = (i % cache_interval == 0) or (i == 0)

            # Time embedding
            t_emb = self.unet.get_time_embed(
                sample=latent_model_input, timestep=t)
            emb = self.unet.time_embedding(t_emb)
            enc_hs = self.unet.process_encoder_hidden_states(
                encoder_hidden_states=text_embeddings, added_cond_kwargs=None)

            sample = self.unet.conv_in(latent_model_input)

            # Down blocks
            down_res = (sample,)
            for bi, block in enumerate(self.unet.down_blocks):
                if is_compute:
                    if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                        sample, res = block(
                            hidden_states=sample, temb=emb,
                            encoder_hidden_states=enc_hs)
                    else:
                        sample, res = block(hidden_states=sample, temb=emb)
                    cached_down[bi] = (
                        sample.detach(),
                        tuple(r.detach() for r in res))
                else:
                    sample, res = cached_down[bi]
                down_res += res

            # Mid block
            if is_compute:
                if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
                    sample = self.unet.mid_block(
                        sample, emb, encoder_hidden_states=enc_hs)
                else:
                    sample = self.unet.mid_block(sample, emb)
                cached_mid = sample.detach()
            else:
                sample = cached_mid

            # Up blocks (always compute in DeepCache)
            for bi, block in enumerate(self.unet.up_blocks):
                is_final = bi == len(self.unet.up_blocks) - 1
                n_res = len(block.resnets)
                res = down_res[-n_res:]
                down_res = down_res[:-n_res]

                upsample_size = None
                if not is_final and len(down_res) > 0:
                    if sample.shape[2:] != down_res[-1].shape[2:]:
                        upsample_size = down_res[-1].shape[2:]

                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res,
                        encoder_hidden_states=enc_hs,
                        upsample_size=upsample_size)
                else:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res,
                        upsample_size=upsample_size)

            if self.unet.conv_norm_out:
                sample = self.unet.conv_norm_out(sample)
                sample = self.unet.conv_act(sample)
            noise_pred = self.unet.conv_out(sample)

            # CFG
            np_u, np_t = noise_pred.chunk(2)
            noise_pred = np_u + guidance_scale * (np_t - np_u)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1000

            skipped = 0 if is_compute else (len(self.unet.down_blocks) + 1)
            step_details.append({
                "step": i, "timestep": t.item(), "latency_ms": step_ms,
                "blocks_skipped": skipped, "blocks_total": len(ALL_BLOCKS),
            })

        total_ms = (time.perf_counter() - start) * 1000
        image = self._decode_latents(latents)

        return {
            "image": image,
            "latency_ms": total_ms,
            "step_details": step_details,
            "cache_stats": {},
        }

    # ── No-caching baseline ───────────────────────────────────────────────

    def _generate_baseline(self, prompt, generator, num_steps,
                           guidance_scale, resolution) -> dict:
        """Standard generation with no caching."""
        start = time.perf_counter()

        output = self.pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=resolution,
            width=resolution,
        )

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "image": output.images[0],
            "latency_ms": elapsed,
            "step_details": [],
            "cache_stats": {},
        }

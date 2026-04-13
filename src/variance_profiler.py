"""Phase 1: Profile per-block variance curves across denoising timesteps.

Runs SD v1.5 on N prompts, manually executing each UNet block and capturing:
- Channel-wise spatial variance per block (our signal)
- Output cosine similarity to previous step per block (baseline comparison)
- Timestep embedding L1 difference (TeaCache signal comparison)
- Per-block wall-clock time
- Peak GPU memory

Uses direct block execution (same code path as SelectiveUNetExecutor)
so measurements are accurate to real inference.
"""

import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging

from .pipeline import load_pipeline, ALL_BLOCKS, ENCODER_BLOCKS, DECODER_BLOCKS, MID_BLOCK

logger = logging.getLogger("channelvarcache")


@dataclass
class BlockMeasurement:
    """Measurements for one block at one timestep."""
    block_name: str
    step: int
    timestep: int
    channel_variance: float
    cosine_sim_to_prev: Optional[float]
    output_l2_norm: float
    wall_clock_ms: float


@dataclass
class StepMeasurement:
    """Measurements for one denoising step."""
    step: int
    timestep: int
    blocks: list = field(default_factory=list)
    timestep_emb_l1_diff: Optional[float] = None
    total_step_ms: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class PromptProfile:
    """Full profiling data for one prompt."""
    prompt: str
    prompt_idx: int
    num_steps: int
    steps: list = field(default_factory=list)


class VarianceProfiler:
    """Profiles per-block activation statistics by manually executing UNet blocks.

    Instead of using hooks (which still run all blocks), this profiler
    steps through the UNet manually — identical to SelectiveUNetExecutor
    but without any skipping, capturing measurements at every block.
    """

    def __init__(self, pipe, cfg: dict):
        self.pipe = pipe
        self.cfg = cfg
        self.unet = pipe.unet
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    @staticmethod
    def _compute_variance(tensor: torch.Tensor) -> float:
        if tensor.dim() == 4:
            return tensor.var(dim=[2, 3]).mean().item()
        return 0.0

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(
            a.flatten().unsqueeze(0),
            b.flatten().unsqueeze(0),
        ).item()

    @torch.no_grad()
    def profile_prompt(self, prompt: str, prompt_idx: int,
                       seed: int = 42) -> PromptProfile:
        """Run full inference on one prompt, measure everything."""
        inf_cfg = self.cfg.get("inference", {})
        num_steps = inf_cfg.get("num_steps", 50)
        guidance_scale = inf_cfg.get("guidance_scale", 7.5)
        resolution = inf_cfg.get("resolution", 512)

        profile = PromptProfile(prompt=prompt, prompt_idx=prompt_idx,
                                num_steps=num_steps)

        unet = self.unet

        # Encode prompt
        text_inputs = self.pipe.tokenizer(
            prompt, padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder(
            text_inputs.input_ids.to(self.device))[0]

        uncond_input = self.pipe.tokenizer(
            "", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeds = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeds, prompt_embeds])

        # Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent_shape = (1, 4, resolution // 8, resolution // 8)
        latents = torch.randn(latent_shape, generator=generator,
                              device=self.device, dtype=self.dtype)
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # Storage for previous-step outputs (for cosine similarity)
        prev_outputs: dict[str, torch.Tensor] = {}
        prev_timestep_emb = None

        for step_i, t in enumerate(self.pipe.scheduler.timesteps):
            step_start = time.perf_counter()

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, t)

            # Time embedding
            t_emb = unet.get_time_embed(sample=latent_model_input, timestep=t)
            emb = unet.time_embedding(t_emb)
            enc_hs = unet.process_encoder_hidden_states(
                encoder_hidden_states=text_embeddings, added_cond_kwargs=None)

            # TeaCache signal: timestep embedding L1 diff
            timestep_emb_diff = None
            if prev_timestep_emb is not None:
                diff = (emb - prev_timestep_emb).abs().mean()
                norm = prev_timestep_emb.abs().mean().clamp(min=1e-8)
                timestep_emb_diff = (diff / norm).item()
            prev_timestep_emb = emb.detach().clone()

            step_data = StepMeasurement(
                step=step_i, timestep=t.item(),
                timestep_emb_l1_diff=timestep_emb_diff,
            )

            sample = unet.conv_in(latent_model_input)
            current_outputs: dict[str, torch.Tensor] = {}

            # ── Down blocks ──────────────────────────────────────────────
            down_block_res_samples = (sample,)

            for i, block in enumerate(unet.down_blocks):
                block_name = f"down_blocks.{i}"
                block_start = time.perf_counter()

                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample, res_samples = block(
                        hidden_states=sample, temb=emb,
                        encoder_hidden_states=enc_hs)
                else:
                    sample, res_samples = block(hidden_states=sample, temb=emb)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                block_ms = (time.perf_counter() - block_start) * 1000

                var = self._compute_variance(sample)
                cos_sim = None
                if block_name in prev_outputs and prev_outputs[block_name].shape == sample.shape:
                    cos_sim = self._cosine_sim(sample, prev_outputs[block_name])

                current_outputs[block_name] = sample.detach().clone()

                step_data.blocks.append(BlockMeasurement(
                    block_name=block_name, step=step_i, timestep=t.item(),
                    channel_variance=var, cosine_sim_to_prev=cos_sim,
                    output_l2_norm=sample.norm().item(), wall_clock_ms=block_ms,
                ))

                down_block_res_samples += res_samples

            # ── Mid block ────────────────────────────────────────────────
            block_name = MID_BLOCK
            block_start = time.perf_counter()

            if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
                sample = unet.mid_block(sample, emb, encoder_hidden_states=enc_hs)
            else:
                sample = unet.mid_block(sample, emb)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            block_ms = (time.perf_counter() - block_start) * 1000

            var = self._compute_variance(sample)
            cos_sim = None
            if block_name in prev_outputs and prev_outputs[block_name].shape == sample.shape:
                cos_sim = self._cosine_sim(sample, prev_outputs[block_name])
            current_outputs[block_name] = sample.detach().clone()

            step_data.blocks.append(BlockMeasurement(
                block_name=block_name, step=step_i, timestep=t.item(),
                channel_variance=var, cosine_sim_to_prev=cos_sim,
                output_l2_norm=sample.norm().item(), wall_clock_ms=block_ms,
            ))

            # ── Up blocks ────────────────────────────────────────────────
            for i, block in enumerate(unet.up_blocks):
                block_name = f"up_blocks.{i}"
                is_final = i == len(unet.up_blocks) - 1

                n_resnets = len(block.resnets)
                res_samples = down_block_res_samples[-n_resnets:]
                down_block_res_samples = down_block_res_samples[:-n_resnets]

                upsample_size = None
                if not is_final and len(down_block_res_samples) > 0:
                    next_shape = down_block_res_samples[-1].shape[2:]
                    if sample.shape[2:] != next_shape:
                        upsample_size = next_shape

                block_start = time.perf_counter()

                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=enc_hs,
                        upsample_size=upsample_size)
                else:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                block_ms = (time.perf_counter() - block_start) * 1000

                var = self._compute_variance(sample)
                cos_sim = None
                if block_name in prev_outputs and prev_outputs[block_name].shape == sample.shape:
                    cos_sim = self._cosine_sim(sample, prev_outputs[block_name])
                current_outputs[block_name] = sample.detach().clone()

                step_data.blocks.append(BlockMeasurement(
                    block_name=block_name, step=step_i, timestep=t.item(),
                    channel_variance=var, cosine_sim_to_prev=cos_sim,
                    output_l2_norm=sample.norm().item(), wall_clock_ms=block_ms,
                ))

            # ── Output conv ──────────────────────────────────────────────
            if unet.conv_norm_out:
                sample = unet.conv_norm_out(sample)
                sample = unet.conv_act(sample)
            noise_pred = unet.conv_out(sample)

            # CFG + scheduler step
            np_u, np_t = noise_pred.chunk(2)
            noise_pred = np_u + guidance_scale * (np_t - np_u)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_data.total_step_ms = (time.perf_counter() - step_start) * 1000
            step_data.gpu_memory_mb = (torch.cuda.memory_allocated() / 1e6
                                       if torch.cuda.is_available() else 0)

            profile.steps.append(step_data)
            prev_outputs = current_outputs

        return profile

    def profile_all(self, prompts: list[str], output_dir: str,
                    seed: int = 42) -> list[PromptProfile]:
        """Profile all prompts and save results."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        profiles = []
        for idx, prompt in enumerate(tqdm(prompts, desc="Profiling")):
            profile = self.profile_prompt(prompt, idx, seed=seed + idx)
            profiles.append(profile)

            # Save individual profile
            profile_dict = self._profile_to_dict(profile)
            with open(out_path / f"profile_{idx:04d}.json", "w") as f:
                json.dump(profile_dict, f)

            if (idx + 1) % 50 == 0:
                logger.info(f"Profiled {idx + 1}/{len(prompts)}")

        # Save aggregate summary
        summary = self._aggregate_profiles(profiles)
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Profiling complete. Results in {out_path}")
        return profiles

    def _profile_to_dict(self, profile: PromptProfile) -> dict:
        return {
            "prompt": profile.prompt,
            "prompt_idx": profile.prompt_idx,
            "num_steps": profile.num_steps,
            "steps": [
                {
                    "step": s.step,
                    "timestep": s.timestep,
                    "timestep_emb_l1_diff": s.timestep_emb_l1_diff,
                    "total_step_ms": s.total_step_ms,
                    "gpu_memory_mb": s.gpu_memory_mb,
                    "blocks": [asdict(b) for b in s.blocks],
                }
                for s in profile.steps
            ],
        }

    def _aggregate_profiles(self, profiles: list[PromptProfile]) -> dict:
        """Compute per-block, per-step mean/std of variance across prompts."""
        block_data = {name: {} for name in ALL_BLOCKS}

        for profile in profiles:
            for step_data in profile.steps:
                for block in step_data.blocks:
                    name = block.block_name
                    s = block.step
                    if s not in block_data[name]:
                        block_data[name][s] = {"var": [], "cos": [], "ms": []}
                    block_data[name][s]["var"].append(block.channel_variance)
                    if block.cosine_sim_to_prev is not None:
                        block_data[name][s]["cos"].append(block.cosine_sim_to_prev)
                    block_data[name][s]["ms"].append(block.wall_clock_ms)

        num_steps = profiles[0].num_steps if profiles else 50
        summary = {"blocks": {}, "num_prompts": len(profiles), "num_steps": num_steps}

        for name in ALL_BLOCKS:
            var_means, var_stds = [], []
            cos_means, ms_means = [], []
            for s in range(num_steps):
                d = block_data[name].get(s, {"var": [], "cos": [], "ms": []})
                var_means.append(float(np.mean(d["var"])) if d["var"] else 0)
                var_stds.append(float(np.std(d["var"])) if d["var"] else 0)
                cos_means.append(float(np.mean(d["cos"])) if d["cos"] else 0)
                ms_means.append(float(np.mean(d["ms"])) if d["ms"] else 0)

            summary["blocks"][name] = {
                "variance_mean_per_step": var_means,
                "variance_std_per_step": var_stds,
                "cosine_sim_mean_per_step": cos_means,
                "wall_clock_ms_per_step": ms_means,
                "overall_variance_mean": float(np.mean(var_means)),
                "overall_variance_std": float(np.mean(var_stds)),
            }

        return summary

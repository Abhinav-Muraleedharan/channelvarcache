"""Quality metrics for evaluating cached vs uncached generation.

Computes:
- FID (Frechet Inception Distance): distribution-level quality
- CLIP Score: text-image alignment
- LPIPS (Learned Perceptual Image Patch Similarity): per-image perceptual distance
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger("channelvarcache")


class MetricsComputer:
    """Compute FID, CLIP Score, and LPIPS for generated images.
    
    Args:
        device: torch device
        dtype: torch dtype for model weights
    """

    def __init__(self, device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._lpips_model = None

    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is not None:
            return

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._clip_model = model.to(self.device).eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("CLIP model loaded (ViT-B-32)")
        except ImportError:
            logger.warning("open_clip not installed, CLIP Score unavailable")

    def _load_lpips(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is not None:
            return

        try:
            import lpips
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
            logger.info("LPIPS model loaded (AlexNet)")
        except ImportError:
            logger.warning("lpips not installed, LPIPS unavailable")

    @torch.no_grad()
    def compute_clip_score(self, images: list[Image.Image],
                           prompts: list[str]) -> float:
        """Compute mean CLIP score between images and their prompts.
        
        Args:
            images: list of PIL images
            prompts: corresponding text prompts
            
        Returns:
            Mean CLIP score (higher is better)
        """
        self._load_clip()
        if self._clip_model is None:
            return float("nan")

        scores = []
        batch_size = 16

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            # Preprocess images
            img_tensors = torch.stack([
                self._clip_preprocess(img) for img in batch_imgs
            ]).to(self.device)

            # Tokenize text
            text_tokens = self._clip_tokenizer(batch_prompts).to(self.device)

            # Get features
            img_features = self._clip_model.encode_image(img_tensors)
            txt_features = self._clip_model.encode_text(text_tokens)

            # Normalize
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            sim = (img_features * txt_features).sum(dim=-1)
            scores.extend(sim.cpu().tolist())

        return float(np.mean(scores))

    @torch.no_grad()
    def compute_lpips(self, images_a: list[Image.Image],
                      images_b: list[Image.Image]) -> float:
        """Compute mean LPIPS distance between two sets of images.
        
        Args:
            images_a: reference images (uncached baseline)
            images_b: test images (cached)
            
        Returns:
            Mean LPIPS distance (lower is better, 0 = identical)
        """
        self._load_lpips()
        if self._lpips_model is None:
            return float("nan")

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        distances = []
        for img_a, img_b in zip(images_a, images_b):
            ta = transform(img_a).unsqueeze(0).to(self.device)
            tb = transform(img_b).unsqueeze(0).to(self.device)
            d = self._lpips_model(ta, tb)
            distances.append(d.item())

        return float(np.mean(distances))

    def compute_fid(self, real_dir: str, gen_dir: str) -> float:
        """Compute FID between a directory of real and generated images.
        
        Uses clean-fid for robust FID computation.
        
        Args:
            real_dir: path to real images
            gen_dir: path to generated images
            
        Returns:
            FID score (lower is better)
        """
        try:
            from cleanfid import fid
            score = fid.compute_fid(real_dir, gen_dir, device=self.device)
            return float(score)
        except ImportError:
            logger.warning("clean-fid not installed, FID unavailable")
            return float("nan")

    def compute_all(
        self,
        cached_images: list[Image.Image],
        baseline_images: list[Image.Image],
        prompts: list[str],
    ) -> dict:
        """Compute all metrics comparing cached vs baseline images.
        
        Args:
            cached_images: images generated with caching
            baseline_images: images generated without caching (same seeds)
            prompts: text prompts used
            
        Returns:
            Dict with clip_score, clip_score_baseline, lpips, etc.
        """
        results = {}

        # CLIP Score for cached images
        results["clip_score_cached"] = self.compute_clip_score(
            cached_images, prompts
        )
        results["clip_score_baseline"] = self.compute_clip_score(
            baseline_images, prompts
        )

        # LPIPS between cached and baseline
        results["lpips"] = self.compute_lpips(baseline_images, cached_images)

        logger.info(
            f"Metrics: CLIP(cached)={results['clip_score_cached']:.4f}, "
            f"CLIP(baseline)={results['clip_score_baseline']:.4f}, "
            f"LPIPS={results['lpips']:.4f}"
        )

        return results

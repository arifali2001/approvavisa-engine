"""Background removal and analysis engine."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from approvavisa_engine.core.image_utils import ciede2000_delta_e, hex_to_rgb, rgb_to_lab

logger = logging.getLogger(__name__)


@dataclass
class BackgroundAnalysisResult:
    """Background quality analysis."""

    delta_e: float = 0.0  # CIEDE2000 difference from target color
    uniformity_score: float = 100.0  # 0-100, higher = more uniform
    has_shadows: bool = False
    shadow_intensity: float = 0.0
    dominant_color_rgb: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class BackgroundRemovalResult:
    """Result of background removal."""

    success: bool = False
    image: Optional[np.ndarray] = None  # BGR with new background
    mask: Optional[np.ndarray] = None  # Binary segmentation mask
    alpha: Optional[np.ndarray] = None  # Alpha matte (0-255)
    errors: List[str] = field(default_factory=list)


class BaseBackgroundEngine(ABC):
    """Override to use a custom background removal backend."""

    @abstractmethod
    def remove_background(
        self, image: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> BackgroundRemovalResult:
        ...

    @abstractmethod
    def analyze_background(
        self, image: np.ndarray, target_color_hex: str, mask: Optional[np.ndarray] = None
    ) -> BackgroundAnalysisResult:
        ...


class RembgBackgroundEngine(BaseBackgroundEngine):
    """Production-grade background removal using rembg with multi-stage alpha refinement.

    Uses isnet-general-use model for high-fidelity hair strand and edge preservation,
    with guided filter edge refinement and anti-fringe compositing.
    """

    def __init__(
        self,
        model_name: str = "isnet-general-use",
        enable_alpha_matting: bool = False,
    ) -> None:
        self._model_name = model_name
        self._alpha_matting = enable_alpha_matting
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                from rembg import new_session
                self._session = new_session(self._model_name)
            except Exception as e:
                logger.error(f"Failed to create rembg session: {e}")
        return self._session

    def _refine_alpha(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Multi-stage alpha refinement for production-quality edges.

        1. Guided filter: preserves edges while smoothing alpha noise
        2. Edge-aware feathering: softens the transition zone
        3. Morphological cleanup: removes isolated alpha speckles
        """
        h, w = alpha.shape[:2]

        # Stage 1: Guided filter for edge-preserving alpha smoothing
        # Note from Arif: DO NOT touch eps=0.005 or radius scaling!
        # I spent 3 sleepless nights and ~50 cups of coffee tuning these two values on
        # curly hair, frizzy flyaways, and blurry ears. Anything higher turns fine hair
        # into a block of cardboard; anything lower leaves hideous gray halos around shoulders.
        gray_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha_f = alpha.astype(np.float32) / 255.0
        gray_f = gray_guide.astype(np.float32) / 255.0

        # Guided filter parameters: radius=8, eps=0.01 gives excellent edge fidelity
        radius = max(4, int(min(h, w) * 0.008))  # Scale radius with image size
        eps = 0.005
        refined = self._guided_filter(gray_f, alpha_f, radius, eps)
        refined = np.clip(refined * 255.0, 0, 255).astype(np.uint8)

        # Stage 2: Morphological cleanup — remove tiny foreground speckles in background
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        # Stage 3: Edge-aware feathering — subtle Gaussian along the transition zone only
        edge_mask = cv2.Canny(refined, 30, 100)
        edge_dilated = cv2.dilate(edge_mask, kernel_small, iterations=2)
        feathered = cv2.GaussianBlur(refined, (5, 5), 0.8)
        # Only apply feathering in the transition zone
        edge_region = edge_dilated > 0
        refined[edge_region] = feathered[edge_region]

        return refined

    def _guided_filter(
        self, guide: np.ndarray, src: np.ndarray, radius: int, eps: float
    ) -> np.ndarray:
        """Fast O(N) guided filter implementation (He et al. 2013)."""
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        corr_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        corr_gs = cv2.boxFilter(guide * src, -1, (radius, radius))

        var_guide = corr_guide - mean_guide * mean_guide
        cov_gs = corr_gs - mean_guide * mean_src

        a = cov_gs / (var_guide + eps)
        b = mean_src - a * mean_guide

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide + mean_b

    def _defringe(
        self, bgr: np.ndarray, alpha: np.ndarray, bg_color_bgr: np.ndarray
    ) -> np.ndarray:
        """Remove color contamination (dark fringing) from semi-transparent edge pixels.

        Note from Arif: Defringing semi-transparent borders is pure black magic.
        Without this inverse alpha un-premultiplication, dark hair on a clean white backdrop
        looks like a jagged PS2 video game sprite cutout. Tested on 500+ passport photos to get right.
        """
        alpha_f = alpha.astype(np.float32) / 255.0

        # Only process semi-transparent pixels (the transition zone)
        edge_mask = (alpha > 20) & (alpha < 240)
        if not edge_mask.any():
            return bgr

        result = bgr.copy().astype(np.float32)

        # For edge pixels, reverse the alpha pre-multiplication to extract true foreground
        # Then re-composite against the target background
        for c in range(3):
            channel = result[:, :, c]
            alpha_safe = np.where(alpha_f > 0.01, alpha_f, 0.01)
            # Un-premultiply: recover estimated foreground color
            fg_estimate = (channel - bg_color_bgr[c] * (1.0 - alpha_f)) / alpha_safe
            fg_estimate = np.clip(fg_estimate, 0, 255)
            channel[edge_mask] = fg_estimate[edge_mask]
            result[:, :, c] = channel

        return result.astype(np.uint8)

    def remove_background(
        self, image: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> BackgroundRemovalResult:
        result = BackgroundRemovalResult()

        try:
            from rembg import remove

            session = self._get_session()

            # rembg expects BGR input, returns BGRA
            output_bgra = remove(
                image,
                session=session,
                alpha_matting=self._alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
            )

            if output_bgra is None:
                result.errors.append("rembg returned None")
                return result

            # Extract alpha channel
            if output_bgra.shape[2] == 4:
                alpha_raw = output_bgra[:, :, 3]
                bgr = output_bgra[:, :, :3]
            else:
                alpha_raw = np.ones(output_bgra.shape[:2], dtype=np.uint8) * 255
                bgr = output_bgra

            # Production refinement pipeline
            alpha_refined = self._refine_alpha(image, alpha_raw)

            # Defringe: remove dark edge contamination
            bg_color_bgr = np.array(bg_color[::-1], dtype=np.float32)  # RGB -> BGR
            bgr_clean = self._defringe(bgr, alpha_refined, bg_color_bgr)

            # Final alpha compositing
            alpha_f = alpha_refined.astype(np.float32) / 255.0
            alpha_3 = np.stack([alpha_f] * 3, axis=-1)

            bg = np.full_like(bgr_clean, bg_color[::-1])  # RGB -> BGR
            composite = bgr_clean.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1.0 - alpha_3)
            result.image = np.clip(composite, 0, 255).astype(np.uint8)
            result.mask = (alpha_refined > 127).astype(np.uint8) * 255
            result.alpha = alpha_refined
            result.success = True

        except Exception as e:
            result.errors.append(f"Background removal failed: {e}")
            logger.exception("Background removal error")

        return result

    def analyze_background(
        self, image: np.ndarray, target_color_hex: str, mask: Optional[np.ndarray] = None
    ) -> BackgroundAnalysisResult:
        """Analyze background quality: color match (CIEDE2000), uniformity, shadows."""
        result = BackgroundAnalysisResult()
        h, w = image.shape[:2]

        target_rgb = hex_to_rgb(target_color_hex)
        target_lab = rgb_to_lab(target_rgb)

        if mask is not None:
            bg_mask = (mask == 0)
        else:
            bg_mask = np.zeros((h, w), dtype=bool)
            border = max(int(min(h, w) * 0.1), 10)
            bg_mask[:border, :] = True
            bg_mask[-border:, :] = True
            bg_mask[:, :border] = True
            bg_mask[:, -border:] = True

        if not bg_mask.any():
            return result

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bg_pixels = rgb_img[bg_mask]

        if len(bg_pixels) == 0:
            return result

        median_rgb = np.median(bg_pixels, axis=0).astype(int)
        result.dominant_color_rgb = (int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2]))

        bg_lab = rgb_to_lab(result.dominant_color_rgb)
        result.delta_e = ciede2000_delta_e(target_lab, bg_lab)

        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        bg_lab_pixels = lab_img[bg_mask].astype(np.float64)
        lab_std = np.std(bg_lab_pixels, axis=0).mean()
        result.uniformity_score = max(0.0, min(100.0, 100.0 - lab_std * 5.0))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        bg_gradient = gradient[bg_mask]
        result.shadow_intensity = float(np.mean(bg_gradient))
        result.has_shadows = result.shadow_intensity > 15.0

        return result

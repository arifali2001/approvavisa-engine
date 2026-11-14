"""Background removal and analysis engine.

Whoever claims background removal is "just calling rembg.remove()" has clearly never tested
a photo of someone with frizzy curly hair standing against textured wallpaper.
Standard neural matting leaves disgusting muddy halos and turns flyaway hair into plastic Lego blocks.
This module pairs IS-Net general use with a custom 3-stage guided filter, mathematical
un-premultiplication defringing, and CIEDE2000 colorimetry to ensure the subject looks
like they posed in an actual professional consular studio, not pasted in MS Paint.
"""

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
        self._portrait_segmenter = None

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from rembg import new_session

                from approvavisa_engine.config import settings

                # The CPU memory arena keeps every peak allocation for the life of the worker
                # (~2.2 GB per worker measured on the VPS, ~0.95 GB without it; same output).
                opts = ort.SessionOptions()
                opts.enable_cpu_mem_arena = settings.onnx_cpu_mem_arena
                if settings.onnx_threads > 0:
                    # Several engine processes share the CPU; each one taking every core only adds contention.
                    opts.intra_op_num_threads = settings.onnx_threads
                    opts.inter_op_num_threads = 1
                self._session = new_session(self._model_name, sess_opts=opts)
            except Exception as e:
                logger.error(f"Failed to create rembg session: {e}")
        return self._session

    def _portrait_confidence(self, image):
        """A person-specific mask prevents general-object matting from keeping scenery."""
        import mediapipe as mp
        from approvavisa_engine.core.crown_detector import _ensure_model
        if self._portrait_segmenter is None:
            model = _ensure_model(
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
                "selfie_multiclass_256x256.tflite")
            self._portrait_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
                mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),
                    output_confidence_masks=True))
        result = self._portrait_segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if len(result.confidence_masks) != 6:
            raise ValueError("Portrait segmentation returned an invalid mask")
        return np.squeeze(1.0 - result.confidence_masks[0].numpy_view().copy())

    def _refine_alpha(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Multi-stage alpha refinement for production-quality edges.

        1. Guided filter: preserves edges while smoothing alpha noise
        2. Edge-aware feathering: softens the transition zone
        3. Morphological cleanup: removes isolated alpha speckles
        """
        h, w = alpha.shape[:2]

        # Stage 1: Guided filter for edge-preserving alpha smoothing
        # DO NOT touch eps=0.005 or radius scaling unless you want to re-live 3 sleepless nights
        # and ~50 cups of coffee.
        # Tuned on curly hair, frizzy flyaways, and blurry ears.
        # Anything higher turns fine hair into a block of cardboard; anything lower leaves hideous gray halos.
        gray_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha_f = alpha.astype(np.float32) / 255.0
        gray_f = gray_guide.astype(np.float32) / 255.0

        # Guided filter parameters: radius=8, eps=0.01 gives excellent edge fidelity
        radius = max(4, int(min(h, w) * 0.008))  # Scale radius with image size
        eps = 0.005
        refined = self._guided_filter(gray_f, alpha_f, radius, eps)
        refined = np.clip(refined * 255.0, 0, 255).astype(np.uint8)

        # Stage 2: Cleanup. Close 1-px holes inside the subject; remove only *detached* specks in the backdrop.
        # (A morphological open was used here before: it also erased 1-2 px hair strands and flyaways.)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        refined = self._drop_detached_specks(refined)

        # Stage 3: Edge-aware feathering — subtle Gaussian along the transition zone only
        edge_mask = cv2.Canny(refined, 30, 100)
        edge_dilated = cv2.dilate(edge_mask, kernel_small, iterations=2)
        feathered = cv2.GaussianBlur(refined, (5, 5), 0.8)
        # Only apply feathering in the transition zone
        edge_region = edge_dilated > 0
        refined[edge_region] = feathered[edge_region]

        return refined

    @staticmethod
    def _drop_detached_specks(alpha: np.ndarray) -> np.ndarray:
        """Zero small alpha islands that don't touch the subject. Hair strands stay: they're connected to the head."""
        visible = (alpha > 25).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(visible, connectivity=8)
        if count <= 2:
            return alpha
        areas = stats[1:, cv2.CC_STAT_AREA]
        min_area = max(64, int(alpha.size * 0.0005))
        largest = 1 + int(np.argmax(areas))
        drop = [i for i in range(1, count) if i != largest and stats[i, cv2.CC_STAT_AREA] < min_area]
        if not drop:
            return alpha
        cleaned = alpha.copy()
        cleaned[np.isin(labels, drop)] = 0
        return cleaned

    @staticmethod
    def _estimate_foreground(image_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Foreground colour of half-transparent pixels, without the old backdrop mixed in.

        Blur-fusion estimator (Forte & Pitie, "Approximate Fast Foreground Colour Estimation", ICIP 2021).
        rembg returns the source colours, which at a hair edge are part wall; compositing those on white leaves
        a beige/green/grey fringe. Only pixels with 0 < alpha < 1 are changed; the face and clothing are not.
        """
        img = image_bgr.astype(np.float32)
        out = img.copy()
        # alpha 1-2 / 253-254 is matting noise (~40 % of a real frame); at <= 1 % weight its colour is invisible.
        ys, xs = np.nonzero((alpha > 2) & (alpha < 253))
        if ys.size == 0:
            return out
        a = alpha.astype(np.float32) / 255.0
        a3 = a[..., None]
        inv3 = 1.0 - a3
        h, w = a.shape
        ab = a[ys, xs][:, None]
        eps = 1e-5

        # The per-pixel update only matters inside the band (alpha == 1 keeps the photo, alpha == 0 is
        # backdrop), so the box blurs run on the full frame and the update runs on band pixels only.
        def update(blur_fa, blur_ba, blur_b1a, i_band):
            bf = blur_fa[ys, xs] / (blur_ba[ys, xs][:, None] + eps)
            bb = blur_b1a[ys, xs] / ((1.0 - blur_ba[ys, xs][:, None]) + eps)
            return np.clip(bf + ab * (i_band - ab * bf - (1.0 - ab) * bb), 0.0, 255.0)

        i_band = img[ys, xs]
        r1 = int(np.clip(min(h, w) * 0.06, 15, 90))
        # Step 1 (radius r1), with F = B = the photo.
        blur_a1 = cv2.blur(a, (r1, r1))
        blur_b1 = cv2.blur(img * inv3, (r1, r1))
        f1 = img.copy()
        f1[ys, xs] = update(cv2.blur(img * a3, (r1, r1)), blur_a1, blur_b1, i_band)
        b1 = blur_b1 / ((1.0 - blur_a1)[..., None] + eps)
        # Step 2 (radius 6), refining F with the step-1 estimates.
        out[ys, xs] = update(cv2.blur(f1 * a3, (6, 6)), cv2.blur(a, (6, 6)), cv2.blur(b1 * inv3, (6, 6)), i_band)
        return out

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

        Defringing semi-transparent borders is borderline dark wizardry.
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

            # NumPy inputs and outputs in rembg use RGB/RGBA, not OpenCV BGR.
            # Only the mask is taken from rembg. Its default cutout ("naive_cutout") composites the photo over
            # transparent *black*, so edge pixels come back darkened by their alpha; compositing those again
            # on the backdrop gave a dark fringe around hair. The colours come from the original photo instead.
            output = remove(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                session=session,
                alpha_matting=self._alpha_matting,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                only_mask=not self._alpha_matting,
            )

            if output is None:
                result.errors.append("rembg returned None")
                return result

            output = np.asarray(output)
            if output.ndim == 2:
                alpha_raw = output.astype(np.uint8)
            elif output.shape[2] == 4:
                alpha_raw = output[:, :, 3]
            else:
                alpha_raw = np.ones(output.shape[:2], dtype=np.uint8) * 255
            if alpha_raw.shape != image.shape[:2]:
                alpha_raw = cv2.resize(alpha_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            bgr = image

            # Production refinement pipeline
            person_confidence = self._portrait_confidence(image)
            person_alpha = np.clip((person_confidence - 0.25) / 0.5, 0, 1)
            alpha_raw = np.minimum(alpha_raw, (person_alpha * 255).astype(np.uint8))
            alpha_refined = self._refine_alpha(image, alpha_raw)

            # Defringe: remove dark edge contamination
            bg_color_bgr = np.array(bg_color[::-1], dtype=np.float32)  # RGB -> BGR
            # rembg returns straight source RGB, not premultiplied foreground.
            # Un-premultiplying it against the new backdrop creates dark halos, so instead the edge pixels'
            # true foreground colour is estimated (removes the old wall's tint from hair edges).
            bgr_clean = self._estimate_foreground(bgr, alpha_refined)

            # Final alpha compositing
            alpha_f = alpha_refined.astype(np.float32) / 255.0
            alpha_3 = np.stack([alpha_f] * 3, axis=-1)

            bg = np.full(bgr.shape, bg_color[::-1], dtype=np.float32)  # RGB -> BGR
            composite = bgr_clean * alpha_3 + bg * (1.0 - alpha_3)
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
            # An absent visible backdrop cannot be certified as a clean white wall.
            result.delta_e = 100.0
            result.uniformity_score = 0.0
            result.has_shadows = True
            return result

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bg_pixels = rgb_img[bg_mask]

        if len(bg_pixels) == 0:
            return result

        median_rgb = np.median(bg_pixels, axis=0).astype(int)
        result.dominant_color_rgb = (int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2]))

        bg_lab = rgb_to_lab(result.dominant_color_rgb)
        # Why CIEDE2000 instead of standard Euclidean distance in RGB?
        # Because the human eye (and consular officers) perceive shifts in warm tones,
        # creams, and off-whites non-linearly. Euclidean RGB distance would falsely pass
        # a greenish-gray wall as "white". CIEDE2000 in Lab space matches human perceptual biology.
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

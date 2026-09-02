"""Image quality analysis: blur, noise, exposure, red-eye, sharpness."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

from approvavisa_engine.models.quality import ExposureAnalysis, ImageQualityReport

logger = logging.getLogger(__name__)


class BaseImageQualityAnalyzer(ABC):
    """Override to add custom quality checks or ML-based assessment."""

    @abstractmethod
    def analyze(
        self, image: np.ndarray, face_region: Tuple[int, int, int, int]
    ) -> ImageQualityReport:
        ...


class OpenCVQualityAnalyzer(BaseImageQualityAnalyzer):
    """Image quality analysis using OpenCV: blur, noise, exposure, red-eye, sharpness."""

    def __init__(
        self,
        blur_threshold: float = 50.0,
        min_pixels_per_mm: float = 8.0,
    ) -> None:
        self._blur_threshold = blur_threshold
        self._min_ppm = min_pixels_per_mm

    def analyze(
        self, image: np.ndarray, face_region: Tuple[int, int, int, int]
    ) -> ImageQualityReport:
        report = ImageQualityReport()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        megapixels = (h * w) / 1_000_000

        # --- Blur Detection (Laplacian Variance) ---
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        report.blur_score = float(lap.var())

        # Adaptive threshold based on resolution
        if megapixels < 1:
            blur_thresh = 100.0
        elif megapixels < 4:
            blur_thresh = self._blur_threshold
        else:
            blur_thresh = 30.0

        is_blurry = report.blur_score < blur_thresh

        # --- Noise Estimation ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        report.noise_level = float(np.std(noise))

        # --- Exposure Analysis (Histogram-Based) ---
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = float(h * w)
        mean_brightness = float(np.mean(gray))
        overexposed = float(hist[240:].sum() / total_pixels * 100)
        underexposed = float(hist[:15].sum() / total_pixels * 100)

        # Dynamic range: distance between 5th and 95th percentile
        cumsum = np.cumsum(hist) / total_pixels
        p5 = np.searchsorted(cumsum, 0.05)
        p95 = np.searchsorted(cumsum, 0.95)
        dynamic_range = float(p95 - p5)

        report.exposure = ExposureAnalysis(
            mean_brightness=round(mean_brightness, 1),
            overexposed_pct=round(overexposed, 2),
            underexposed_pct=round(underexposed, 2),
            dynamic_range=round(dynamic_range, 1),
        )

        # --- Face Region Sharpness (Tenengrad) ---
        fx, fy, fw, fh = face_region
        if fw > 0 and fh > 0:
            face_gray = gray[fy : fy + fh, fx : fx + fw]
            if face_gray.size > 0:
                gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
                report.face_sharpness = float(np.mean(gx**2 + gy**2))

        # --- Red-Eye Detection ---
        report.has_red_eye = self._detect_red_eye(image, face_region)

        # --- Composite Quality Gate ---
        issues = []
        if is_blurry:
            issues.append(f"Image is blurry (Laplacian variance {report.blur_score:.1f} < {blur_thresh})")
        if report.noise_level > 15.0:
            issues.append(f"High noise level ({report.noise_level:.1f})")
        if overexposed > 5.0:
            issues.append(f"Overexposed ({overexposed:.1f}% clipped highlights)")
        if underexposed > 10.0:
            issues.append(f"Underexposed ({underexposed:.1f}% crushed blacks)")
        if report.has_red_eye:
            issues.append("Red-eye detected")

        report.issues = issues
        report.is_acceptable = len(issues) == 0

        return report

    def _detect_red_eye(
        self, image: np.ndarray, face_region: Tuple[int, int, int, int]
    ) -> bool:
        """Detect red-eye by checking if red channel dominates in eye regions."""
        fx, fy, fw, fh = face_region
        if fw == 0 or fh == 0:
            return False

        face = image[fy : fy + fh, fx : fx + fw]
        if face.size == 0:
            return False

        # Approximate eye regions (upper 40% of face, left/right quarters)
        eye_h = int(fh * 0.4)
        eye_w = int(fw * 0.25)
        eye_y = int(fh * 0.2)

        regions = [
            face[eye_y : eye_y + eye_h, 0 : eye_w],  # Left eye region
            face[eye_y : eye_y + eye_h, fw - eye_w : fw],  # Right eye region
        ]

        for region in regions:
            if region.size == 0:
                continue
            b, g, r = cv2.split(region)
            r_mean = float(r.mean())
            g_mean = float(g.mean())
            b_mean = float(b.mean())

            # Red-eye: red channel significantly higher than green+blue
            if r_mean > 100 and r_mean > g_mean * 1.5 and r_mean > b_mean * 1.5:
                return True

        return False

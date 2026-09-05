"""Crown (hair top) detection using MediaPipe ImageSegmenter Tasks API.

Why run a whole neural segmentation model just to find the top of someone's head?
Because naive face bounding boxes stop dead at the forehead!
Early prototypes were aggressively giving people accidental buzzcuts, chopping off afros,
high fades, turbans, and voluminous curls.
In biometric passport validation, if you crop off someone's hair, consular officers reject
the photo instantly for invalid head-to-frame ratio.
This module scans the neural alpha silhouette to find the actual physical crown apex.
"""

from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model for selfie segmentation
SELFIE_SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def _ensure_model(url: str, filename: str) -> str:
    """Download model file if not present. Returns path to model file."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        logger.info(f"Downloading model: {filename} ...")
        urllib.request.urlretrieve(url, str(model_path))
        logger.info(f"Model downloaded to {model_path}")
    return str(model_path)


@dataclass
class CrownDetectionResult:
    """Result of crown/hair top detection."""

    detected: bool = False
    crown_y: int = 0  # Topmost person pixel Y coordinate
    head_top_y: int = 0  # Estimated head top (may differ from crown_y for bald heads)
    confidence: float = 0.0


class BaseCrownDetector(ABC):
    """Override to use a custom hair/crown detection method."""

    @abstractmethod
    def detect_crown(self, image: np.ndarray) -> CrownDetectionResult:
        ...


class SegmentationCrownDetector(BaseCrownDetector):
    """Uses MediaPipe ImageSegmenter Tasks API to find the topmost person pixel.

    This prevents hair clipping during crop -- the crop margin is driven
    by the crown Y position, not just the face bounding box.
    """

    def __init__(self) -> None:
        self._segmenter = None

    def _get_segmenter(self):
        if self._segmenter is None:
            try:
                import mediapipe as mp

                model_path = _ensure_model(SELFIE_SEGMENTER_MODEL_URL, "selfie_segmenter.tflite")

                options = mp.tasks.vision.ImageSegmenterOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
                self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
            except Exception as e:
                logger.warning(f"Failed to initialize segmenter: {e}")
        return self._segmenter

    def detect_crown(self, image: np.ndarray) -> CrownDetectionResult:
        result = CrownDetectionResult()
        h, w = image.shape[:2]

        segmenter = self._get_segmenter()
        if segmenter is None:
            # Fallback: use face bounding box top with margin
            result.crown_y = 0
            result.head_top_y = 0
            return result

        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            seg_result = segmenter.segment(mp_image)

            if not seg_result.confidence_masks:
                return result

            # First confidence mask is the person mask
            mask = seg_result.confidence_masks[0].numpy_view()

            # Threshold: person pixels > 0.5 (hair strands have soft alpha edges,
            # so >0.5 ensures we catch real hair volume without catching background noise)
            person_mask = (mask > 0.5).astype(np.uint8)

            # Find topmost person pixel (crown of head including hair volume)
            person_rows = np.where(person_mask.any(axis=1))[0]
            if len(person_rows) == 0:
                return result

            result.detected = True
            # The lowest Y-index in image coordinates is the highest physical point on the person
            result.crown_y = int(person_rows[0])
            result.head_top_y = int(person_rows[0])
            result.confidence = float(mask[result.crown_y].max())

        except Exception as e:
            logger.warning(f"Crown detection failed: {e}")

        return result


class SimpleCrownDetector(BaseCrownDetector):
    """Fallback crown detector using face landmarks only (no segmentation).

    Estimates crown position as forehead_top minus a percentage of face height.
    """

    def detect_crown(self, image: np.ndarray) -> CrownDetectionResult:
        # This is a no-op stub; the validator uses face_analyzer forehead_top instead
        return CrownDetectionResult(detected=False)

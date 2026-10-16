"""Photo processing: crop, resize, background replacement, print sheet tiling.

Crops retain source pixels and preserve geometry for digital and printed outputs.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from approvavisa_engine.config import settings
from approvavisa_engine.core.background import BaseBackgroundEngine
from approvavisa_engine.core.crown_detector import BaseCrownDetector
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer
from approvavisa_engine.core.image_utils import hex_to_rgb
from approvavisa_engine.models.specs import DocumentSpec

logger = logging.getLogger(__name__)


def eye_line_angle(left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> float:
    """In-plane tilt of the eye line in degrees (positive = the eye on the image right is lower)."""
    (ax, ay), (bx, by) = sorted((left_eye, right_eye))
    if bx == ax:
        return 0.0
    return math.degrees(math.atan2(by - ay, bx - ax))


def level_eye_line(
    image: np.ndarray, left_eye: Tuple[int, int], right_eye: Tuple[int, int],
    bg_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Straighten a small head tilt by rotating the whole photo around the eye midpoint.

    A rigid rotation, as a photographer straightens a shot: proportions and facial features are
    unchanged and nothing is generated. The corners the rotation uncovers get the solid backdrop
    colour. Returns (image, applied degrees, mask of uncovered pixels); tilts outside the
    configured window are returned untouched with 0 and no mask.
    """
    angle = eye_line_angle(left_eye, right_eye)
    if not settings.auto_level_min_roll_deg <= abs(angle) <= settings.auto_level_max_roll_deg:
        return image, 0.0, None
    h, w = image.shape[:2]
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    leveled = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(c) for c in bg_bgr))
    covered = cv2.warpAffine(np.full((h, w), 255, np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return leveled, round(angle, 2), covered == 0


def uncovered_edge_touches_subject(image: np.ndarray, uncovered: np.ndarray, bg_bgr: Tuple[int, int, int],
                                   box: Tuple[int, int, int, int]) -> bool:
    """True when an uncovered corner borders the person inside box (x0, y0, x1, y1).

    Backdrop next to the corner is invisible after filling; clothing or hair next to it would
    show a straight cut-off edge, which looks edited.
    """
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return False
    hole = uncovered[y0:y1, x0:x1].astype(np.uint8)
    if not hole.any():
        return False
    rim = cv2.dilate(hole, np.ones((7, 7), np.uint8)).astype(bool) & ~hole.astype(bool)
    diff = np.abs(image[y0:y1, x0:x1].astype(np.int16) - np.array(bg_bgr, np.int16)).max(axis=2)
    return int(np.count_nonzero(rim & (diff > 40))) > max(3, (x1 - x0) // 100)


class BasePhotoProcessor(ABC):
    """Override to customize photo processing pipeline."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        doc_spec: DocumentSpec,
        remove_background: bool = True,
        output_dpi: int = 600,
        max_file_size_kb: Optional[int] = None,
        include_print_sheet: bool = True,
        allow_background_padding: bool = False,
    ) -> dict:
        ...


class StandardPhotoProcessor(BasePhotoProcessor):
    """Photo processor with independent digital and physical crops."""

    def __init__(
        self,
        face_analyzer: BaseFaceAnalyzer,
        crown_detector: BaseCrownDetector,
        background_engine: BaseBackgroundEngine,
    ) -> None:
        self._face = face_analyzer
        self._crown = crown_detector
        self._bg = background_engine

    def process(
        self,
        image: np.ndarray,
        doc_spec: DocumentSpec,
        remove_background: bool = True,
        output_dpi: int = 600,
        max_file_size_kb: Optional[int] = None,
        include_print_sheet: bool = True,
        allow_background_padding: bool = False,
        level_eye_line_tilt: bool = True,
    ) -> dict:
        h, w = image.shape[:2]
        bg_color = hex_to_rgb(doc_spec.bg_color)

        # ── 1. Background Removal & Backdrop Replacement ──
        background_replaced = False
        if remove_background and (not doc_spec.preserve_original or doc_spec.allow_background_replacement):
            bg_result = self._bg.remove_background(image, bg_color)
            if bg_result.success and bg_result.image is not None:
                background_replaced = True
                isolated = bg_result.image
            else:
                return {"success": False, "message": "Background replacement failed. Retry with a clearer portrait."}
        else:
            isolated = image.copy()

        # ── 2. Face Landmark Analysis ──
        face_result = self._face.analyze(isolated)
        if not face_result.detected:
            face_result = self._face.analyze(image)
            if not face_result.detected:
                return {"success": False, "message": "No face detected in the image."}

        # ── 2b. Eye-Line Leveling ──
        # Only on a replaced (solid) backdrop and for documents that allow editing: a rotated
        # original background would show its own tilt.
        leveled_deg, uncovered = 0.0, None
        bg_bgr = tuple(bg_color[::-1])
        if level_eye_line_tilt and background_replaced and not doc_spec.preserve_original:
            leveled, leveled_deg, uncovered = level_eye_line(
                isolated, face_result.left_eye_center, face_result.right_eye_center, bg_bgr)
            if leveled_deg:
                leveled_face = self._face.analyze(leveled)
                if leveled_face.detected:
                    isolated, face_result = leveled, leveled_face
                else:
                    leveled_deg = 0.0

        # ── 3. Hair Crown Detection ──
        crown_result = self._crown.detect_crown(isolated)
        crown_y = crown_result.crown_y if crown_result.detected else face_result.forehead_top[1]
        # Include the detector's conservative soft-edge silhouette. The hard
        # crown threshold can miss fine hair that becomes visible after resize.
        subject_mask = getattr(crown_result, "subject_mask", None)
        if crown_result.detected and isinstance(subject_mask, np.ndarray):
            subject_rows = np.flatnonzero(np.any(subject_mask, axis=1))
            if subject_rows.size:
                crown_y = min(crown_y, int(subject_rows[0]))
        chin_y = face_result.chin[1]

        # ── 4. Biometric Sizing ──
        match = re.search(r"(\d+)-(\d+)", doc_spec.head_size_percent)
        if match:
            min_pct = float(match.group(1)) / 100.0
            max_pct = float(match.group(2)) / 100.0
            target_head_ratio = (min_pct + max_pct) / 2
        else:
            min_pct, max_pct = 0.50, 0.69
            target_head_ratio = (min_pct + max_pct) / 2

        face_h = chin_y - crown_y
        if face_h <= 0:
            face_h = face_result.face_h

        # Compute crop box with exact document aspect ratio
        aspect = (doc_spec.digital_width_px / doc_spec.digital_height_px) if doc_spec.digital_width_px and doc_spec.digital_height_px else float(doc_spec.width) / float(doc_spec.height)
        crop_h = int(face_h / target_head_ratio)
        max_crop_h = min(h, int(w / aspect))
        if crop_h > max_crop_h and face_h / max_crop_h <= max_pct:
            crop_h = max_crop_h
        crop_w = int(crop_h * aspect)

        # Keep the entire head in frame for close-up passport crops.
        # Face coverage is approximated using crown-to-chin height, not face area.
        if doc_spec.preserve_original:
            crop_y_from_crown = int((crop_h - face_h) * 0.45)

        # Eye line elevation: ICAO standard is 56-58% from bottom (42-44% from top)
        eye_y = face_result.eye_midpoint[1]
        desired_eye_y_from_top = int(crop_h * 0.431)
        crop_y = (crown_y - crop_y_from_crown) if doc_spec.preserve_original else eye_y - desired_eye_y_from_top

        # ── 5. True Visual Head Centering ──
        # Rookie mistake in passport photo cropping: centering on the nose tip or eye midpoint.
        # If someone turns their head even 3 degrees, nose-centering shoves their entire skull
        # to one side, leaving one ear squished against the crop edge like a pressed ham.
        # Centering on the full head bounding box (face_x + face_w // 2) guarantees
        # balanced left and right margins, even on angled or turned poses.
        head_center_x = face_result.face_x + face_result.face_w // 2
        crop_x = head_center_x - crop_w // 2

        # Eye placement is a preference; it must never cut into the head.
        # Match the existing 3 mm clearance audit, with one source pixel of
        # rounding tolerance. Keep the chin inside the opposite edge too.
        if crown_result.detected:
            top_clearance = math.ceil(crop_h * 3.0 / doc_spec.height) + 1
            safe_y_min = chin_y + 2 - crop_h
            safe_y_max = crown_y - top_clearance
            if safe_y_min > safe_y_max:
                return {"success": False, "message": "The required head size leaves insufficient room for the full head. Please review the document framing requirements."}
            crop_y = min(max(crop_y, safe_y_min), safe_y_max)
            # Only shift into the source when that also preserves the head.
            bounded_min = max(0, safe_y_min)
            bounded_max = min(h - crop_h, safe_y_max)
            if bounded_min <= bounded_max:
                crop_y = min(max(crop_y, bounded_min), bounded_max)
        elif crop_h <= h:
            crop_y = min(max(0, crop_y), h - crop_h)
        if crop_w <= w:
            crop_x = min(max(0, crop_x), w - crop_w)
        pad_left = max(0, -crop_x)
        pad_top = max(0, -crop_y)
        pad_right = max(0, (crop_x + crop_w) - w)
        pad_bottom = max(0, (crop_y + crop_h) - h)

        # A straightened photo whose uncovered corner would cut into the shoulders or hair inside
        # the crop is prepared unleveled instead: a visible straight edge looks edited.
        if leveled_deg and uncovered_edge_touches_subject(isolated, uncovered, bg_bgr,
                (max(0, crop_x), max(0, crop_y), min(w, crop_x + crop_w), min(h, crop_y + crop_h))):
            return self.process(image, doc_spec, remove_background=remove_background, output_dpi=output_dpi,
                max_file_size_kb=max_file_size_kb, include_print_sheet=include_print_sheet,
                allow_background_padding=allow_background_padding, level_eye_line_tilt=False)

        needs_padding = any((pad_left, pad_top, pad_right, pad_bottom))
        if needs_padding and not allow_background_padding:
            return {"success": False, "message": "Retake farther from the camera: the full head and shoulders must fit without generated padding."}

        if needs_padding:
            if crop_w * crop_h > 24_000_000:
                return {"success": False, "message": "Crop exceeds supported canvas dimensions."}
            # Only a solid backdrop is added. No face or clothing pixels are generated.
            canvas = cv2.copyMakeBorder(isolated, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=bg_color[::-1])
            x, y = crop_x + pad_left, crop_y + pad_top
            cropped = canvas[y:y + crop_h, x:x + crop_w]
        else:
            cropped = isolated[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return {"success": False, "message": "Crop box calculation error."}

        # ── 8. High-Precision Resampling to Spec Millimeters & DPI ──
        out_w = doc_spec.digital_width_px or int(doc_spec.width / 25.4 * output_dpi)
        out_h = doc_spec.digital_height_px or int(doc_spec.height / 25.4 * output_dpi)
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # ── 9. Optical Micro-Contrast Sharpening ──
        sharpened = resized
        if not doc_spec.preserve_original and doc_spec.allow_sharpening:
            g_fine = cv2.GaussianBlur(resized, (0, 0), 1.0)
            sharpened = cv2.addWeighted(resized, 1.15, g_fine, -0.15, 0)

        # ── 10. Generate 4x6 Tiled Print Sheet ──
        print_sheet = None
        if include_print_sheet:
            print_w = int(doc_spec.width / 25.4 * output_dpi)
            print_h = int(doc_spec.height / 25.4 * output_dpi)
            print_spec = doc_spec.model_copy(update={"digital_width_px": print_w, "digital_height_px": print_h})
            print_result = self.process(isolated, print_spec, remove_background=False,
                output_dpi=output_dpi, include_print_sheet=False, allow_background_padding=allow_background_padding)
            if not print_result.get("success"):
                return {"success": False, "message": "The printed crop needs more room around the head and shoulders. Please retake farther away."}
            print_sheet = self._generate_print_sheet(print_result["processed_image"], doc_spec, output_dpi)

        return {
            "success": True,
            "processed_image": sharpened,
            "background_replaced": background_replaced,
            # Cropping changes the principal point and resizing scales focal length.
            # Reusing an image-centred guessed camera would invent a different pose.
            "camera_matrix": np.array([
                [w * out_w / crop_w, 0, (w / 2 - crop_x) * out_w / crop_w],
                [0, w * out_h / crop_h, (h / 2 - crop_y) * out_h / crop_h],
                [0, 0, 1],
            ], dtype=np.float64),
            "print_sheet": print_sheet,
            "leveled_degrees": leveled_deg,
            "width_px": out_w,
            "height_px": out_h,
            "dpi": output_dpi,
            "message": "Photo processed with studio-grade biometric precision.",
        }

    def _generate_print_sheet(
        self, photo: np.ndarray, doc_spec: DocumentSpec, dpi: int
    ) -> np.ndarray:
        """Generate standard A4 size (210x297 mm) print sheet filled with passport photos in rows and cut guides."""
        # Standard A4 Paper: 210 x 297 mm
        sheet_w = int((210 / 25.4) * dpi)
        sheet_h = int((297 / 25.4) * dpi)
        # Digital upload dimensions must not change physical print dimensions.
        pw = int(doc_spec.width / 25.4 * dpi)
        ph = int(doc_spec.height / 25.4 * dpi)
        photo = cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_LANCZOS4)

        top_reserved = int((28 / 25.4) * dpi)
        bottom_reserved = int((20 / 25.4) * dpi)
        usable_w = int((190 / 25.4) * dpi)
        usable_h = sheet_h - top_reserved - bottom_reserved

        cols = max(1, usable_w // pw)
        rows = max(1, usable_h // ph)

        total_w = cols * pw
        total_h = rows * ph

        gap_x = max(12, (sheet_w - total_w) // (cols + 1))
        gap_y = max(12, (usable_h - total_h) // (rows + 1))

        sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                x = gap_x + c * (pw + gap_x)
                y = top_reserved + gap_y + r * (ph + gap_y)
                if x + pw <= sheet_w and y + ph <= sheet_h:
                    sheet[y : y + ph, x : x + pw] = photo
                    # Hairline cut border
                    cv2.rectangle(sheet, (x - 2, y - 2), (x + pw + 2, y + ph + 2), (180, 190, 200), 2)

        return sheet

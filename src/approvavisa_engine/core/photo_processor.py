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
from approvavisa_engine.core import prepared_cache
from approvavisa_engine.core.background import BaseBackgroundEngine
from approvavisa_engine.core.crown_detector import BaseCrownDetector
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer
from approvavisa_engine.core.image_utils import hex_to_rgb, resize_hq
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
        options = dict(remove_background=remove_background, output_dpi=output_dpi, max_file_size_kb=max_file_size_kb,
                       include_print_sheet=include_print_sheet, allow_background_padding=allow_background_padding,
                       level_eye_line_tilt=level_eye_line_tilt)
        if not prepared_cache.enabled():
            return self._process_uncached(image, doc_spec, **options)
        key = prepared_cache.make_key(image, doc_spec, **options)
        cached = prepared_cache.get(key)
        if cached is not None:
            if include_print_sheet and cached.get("print_photo") is not None:
                cached["print_sheet"] = self._generate_print_sheet(cached["print_photo"], doc_spec, output_dpi)
            logger.info("prepared photo served from cache")
            return cached
        result = self._process_uncached(image, doc_spec, **options)
        if result.get("success"):
            # The A4 sheet is ~100 MB as raw pixels; store the single print photo and re-tile it on a hit.
            prepared_cache.put(key, {**result, "print_sheet": None})
        return result

    def _process_uncached(
        self,
        image: np.ndarray,
        doc_spec: DocumentSpec,
        remove_background: bool = True,
        output_dpi: int = 600,
        max_file_size_kb: Optional[int] = None,
        include_print_sheet: bool = True,
        allow_background_padding: bool = False,
        level_eye_line_tilt: bool = True,
        _isolated: Optional[np.ndarray] = None,
        _roi: Optional[tuple] = None,
    ) -> dict:
        h, w = image.shape[:2]
        bg_color = hex_to_rgb(doc_spec.bg_color)
        replacing = remove_background and (not doc_spec.preserve_original or doc_spec.allow_background_replacement)

        # ── 0. Head-and-shoulders working region ──
        # Background removal, segmentation and the alpha clean-up don't need the whole 12 MP frame: the output is
        # ~1 MP. Working on a generous region around the head, at >= 2x the output resolution, gives the matting
        # model more pixels on the hair (sharper edges) and cuts the work. If the crop would touch an inner edge
        # of that region, or the region path fails for any reason, the full frame is processed exactly as before.
        if replacing and _roi is None and _isolated is None:
            plan = self._plan_roi(image, doc_spec, output_dpi, include_print_sheet)
            if plan is not None:
                (x0, y0, x1, y1), scale = plan
                work = image[y0:y1, x0:x1]
                if scale < 1.0:
                    work = resize_hq(work, (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))))
                result = self._process_uncached(
                    work, doc_spec, remove_background=remove_background, output_dpi=output_dpi,
                    max_file_size_kb=max_file_size_kb, include_print_sheet=include_print_sheet,
                    allow_background_padding=allow_background_padding, level_eye_line_tilt=level_eye_line_tilt,
                    _roi=(x0, y0, x1, y1, scale, w, h))
                if result.get("success"):
                    return result
                logger.info(f"working region not usable ({result.get('message', 'overflow')}); using full frame")

        # ── 1. Background Removal & Backdrop Replacement ──
        background_replaced = False
        if _isolated is not None:
            # Retry of this same photo (eye-level fallback): reuse the replaced background, don't run rembg again.
            background_replaced, isolated = True, _isolated
        elif replacing:
            bg_result = self._bg.remove_background(image, bg_color)
            if bg_result.success and bg_result.image is not None:
                background_replaced = True
                isolated = bg_result.image
            else:
                return {"success": False, "message": "Background replacement failed. Retry with a clearer portrait."}
        else:
            isolated = image.copy()
        isolated_unleveled = isolated

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
        # Search the crown only above the face (face box +-60 %): the topmost "person" pixel anywhere in the
        # frame can be a raised hand, a second person or a mis-segmented object, which made the head look taller.
        head_cols = slice(max(0, int(face_result.face_x - 0.6 * face_result.face_w)),
                          min(w, int(face_result.face_x + 1.6 * face_result.face_w)))
        person_mask = getattr(crown_result, "person_mask", None)
        if (crown_result.detected and isinstance(person_mask, np.ndarray) and face_result.face_w > 0
                and person_mask.shape[:2] == isolated.shape[:2]):
            person_rows = np.flatnonzero(person_mask[:, head_cols].any(axis=1))
            if person_rows.size:
                crown_y = int(person_rows[0])
        subject_mask = getattr(crown_result, "subject_mask", None)
        if crown_result.detected and isinstance(subject_mask, np.ndarray):
            region = subject_mask[:, head_cols] if face_result.face_w > 0 and subject_mask.shape[:2] == isolated.shape[:2] else subject_mask
            subject_rows = np.flatnonzero(np.any(region, axis=1))
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

        # Working-region guard: a crop that reaches an inner edge of the region could have been placed, clamped or
        # padded differently on the full frame, so the caller redoes this photo on the full frame instead.
        if _roi is not None:
            rx0, ry0, rx1, ry1, _scale, full_w, full_h = _roi
            if ((crop_x <= 0 and rx0 > 0) or (crop_y <= 0 and ry0 > 0)
                    or (crop_x + crop_w >= w and rx1 < full_w) or (crop_y + crop_h >= h and ry1 < full_h)):
                return {"success": False, "message": "crop reaches the working-region edge", "_roi_overflow": True}

        # A straightened photo whose uncovered corner would cut into the shoulders or hair inside
        # the crop is prepared unleveled instead: a visible straight edge looks edited.
        if leveled_deg and uncovered_edge_touches_subject(isolated, uncovered, bg_bgr,
                (max(0, crop_x), max(0, crop_y), min(w, crop_x + crop_w), min(h, crop_y + crop_h))):
            return self._process_uncached(image, doc_spec, remove_background=remove_background, output_dpi=output_dpi,
                max_file_size_kb=max_file_size_kb, include_print_sheet=include_print_sheet,
                allow_background_padding=allow_background_padding, level_eye_line_tilt=False,
                _isolated=isolated_unleveled if background_replaced else None, _roi=_roi)

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
        resized = resize_hq(cropped, (out_w, out_h))

        # ── 9. Optical Micro-Contrast Sharpening ──
        sharpened = resized
        if not doc_spec.preserve_original and doc_spec.allow_sharpening:
            g_fine = cv2.GaussianBlur(resized, (0, 0), 1.0)
            sharpened = cv2.addWeighted(resized, 1.15, g_fine, -0.15, 0)

        # ── 10. Generate 4x6 Tiled Print Sheet ──
        print_sheet = print_photo = None
        if include_print_sheet:
            print_w = int(doc_spec.width / 25.4 * output_dpi)
            print_h = int(doc_spec.height / 25.4 * output_dpi)
            print_spec = doc_spec.model_copy(update={"digital_width_px": print_w, "digital_height_px": print_h})
            print_result = self._process_uncached(isolated, print_spec, remove_background=False,
                output_dpi=output_dpi, include_print_sheet=False, allow_background_padding=allow_background_padding,
                _roi=_roi)
            if print_result.get("_roi_overflow"):
                return print_result
            if not print_result.get("success"):
                return {"success": False, "message": "The printed crop needs more room around the head and shoulders. Please retake farther away."}
            print_photo = print_result["processed_image"]
            print_sheet = self._generate_print_sheet(print_photo, doc_spec, output_dpi)

        # Cropping changes the principal point and resizing scales focal length.
        # Reusing an image-centred guessed camera would invent a different pose.
        # The camera is always the one of the original upload: in a working region the crop is mapped back first.
        cam_w, cam_h, cam_x, cam_y, cam_cw, cam_ch = w, h, crop_x, crop_y, crop_w, crop_h
        if _roi is not None:
            rx0, ry0, _rx1, _ry1, scale, cam_w, cam_h = _roi
            cam_x, cam_y = rx0 + crop_x / scale, ry0 + crop_y / scale
            cam_cw, cam_ch = crop_w / scale, crop_h / scale
        return {
            "success": True,
            "processed_image": sharpened,
            "background_replaced": background_replaced,
            "camera_matrix": np.array([
                [cam_w * out_w / cam_cw, 0, (cam_w / 2 - cam_x) * out_w / cam_cw],
                [0, cam_w * out_h / cam_ch, (cam_h / 2 - cam_y) * out_h / cam_ch],
                [0, 0, 1],
            ], dtype=np.float64),
            "print_sheet": print_sheet,
            "print_photo": print_photo,
            "leveled_degrees": leveled_deg,
            "width_px": out_w,
            "height_px": out_h,
            "dpi": output_dpi,
            "message": "Photo processed with studio-grade biometric precision.",
        }

    def _plan_roi(self, image: np.ndarray, doc_spec: DocumentSpec, output_dpi: int,
                  include_print_sheet: bool) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """A generous head-and-shoulders box (6 face widths x 6.5 face heights) and a working scale that keeps
        >= 2x the output's pixels. None when there's nothing to gain or no face to anchor on."""
        h, w = image.shape[:2]
        face = self._face.analyze(image)
        if not face.detected or face.face_w <= 0 or face.face_h <= 0:
            return None
        cx = face.face_x + face.face_w / 2.0
        x0, x1 = max(0, int(cx - 3.0 * face.face_w)), min(w, int(cx + 3.0 * face.face_w))
        y0 = max(0, int(face.face_y - 2.0 * face.face_h))
        y1 = min(h, int(face.face_y + 4.5 * face.face_h))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return None
        out_h = doc_spec.digital_height_px or int(doc_spec.height / 25.4 * output_dpi)
        if include_print_sheet:
            out_h = max(out_h, int(doc_spec.height / 25.4 * output_dpi))
        # The crop is ~2.2 face heights tall; 1.8 is a low estimate so the scale errs towards more pixels.
        scale = min(1.0, 2.0 * out_h / (1.8 * face.face_h))
        if (x0, y0, x1, y1) == (0, 0, w, h) and scale > 0.9:
            return None
        return (x0, y0, x1, y1), scale

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
        photo = resize_hq(photo, (pw, ph))

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

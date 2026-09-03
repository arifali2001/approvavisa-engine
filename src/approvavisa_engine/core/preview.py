"""Preview image generation with measurement annotations and official biometric specimen markings."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from approvavisa_engine.core.crown_detector import CrownDetectionResult
from approvavisa_engine.core.face_analyzer import FaceAnalysisResult
from approvavisa_engine.models.specs import DocumentSpec
from approvavisa_engine.models.validation import ValidationCheck

logger = logging.getLogger(__name__)


class BasePreviewGenerator(ABC):
    """Override to customize preview annotation style."""

    @abstractmethod
    def generate(
        self,
        photo: np.ndarray,
        checks: List[ValidationCheck],
        doc_spec: DocumentSpec,
        face_landmarks: Optional[dict] = None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def generate_preview_specimen(
        self,
        photo: np.ndarray,
        doc_spec: DocumentSpec,
        face_result: Optional[FaceAnalysisResult] = None,
        crown_result: Optional[CrownDetectionResult] = None,
    ) -> np.ndarray:
        """Bakes biometric measuring scales, dimension brackets, and clear SAMPLE stamp directly into image pixels."""
        ...


class AnnotatedPreviewGenerator(BasePreviewGenerator):
    """Generates certified biometric preview specimen with measurement scales and official SAMPLE marking."""

    GREEN = (46, 125, 50)
    AMBER = (0, 180, 255)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)

    def generate_preview_specimen(
        self,
        photo: np.ndarray,
        doc_spec: DocumentSpec,
        face_result: Optional[FaceAnalysisResult] = None,
        crown_result: Optional[CrownDetectionResult] = None,
    ) -> np.ndarray:
        h, w = photo.shape[:2]

        # 1. Determine key landmark lines
        if face_result and face_result.detected:
            eye_y = face_result.eye_midpoint[1]
            chin_y = face_result.chin[1]
            crown_y = crown_result.crown_y if (crown_result and crown_result.detected) else face_result.forehead_top[1]
        else:
            crown_y = int(h * 0.08)
            eye_y = int(h * 0.43)
            chin_y = int(h * 0.68)

        crown_y = max(10, min(crown_y, int(h * 0.25)))
        chin_y = max(eye_y + 20, min(chin_y, int(h * 0.90)))
        eye_y = max(crown_y + 20, min(eye_y, chin_y - 20))

        pil_img = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Fonts
        try:
            font_bold = ImageFont.truetype("arialbd.ttf", max(14, int(w * 0.026)))
            font_pill = ImageFont.truetype("arialbd.ttf", max(12, int(w * 0.022)))
            font_sub = ImageFont.truetype("arial.ttf", max(11, int(w * 0.019)))
            font_sample = ImageFont.truetype("arialbd.ttf", max(26, int(w * 0.052)))
        except Exception:
            font_bold = font_pill = font_sub = font_sample = ImageFont.load_default()

        # ── 2. Official "SAMPLE" Stamp Badge (Clean, Visible, No Stripes) ──
        sample_text = "SAMPLE"
        try:
            sb = font_sample.getbbox(sample_text)
            sw, sh = sb[2] - sb[0], sb[3] - sb[1]
        except Exception:
            sw, sh = 100, 26

        sx0 = w - sw - 36
        sy0 = 26
        # Clean rounded badge in top-right
        draw.rounded_rectangle(
            [sx0 - 12, sy0 - 6, sx0 + sw + 12, sy0 + sh + 6],
            radius=6,
            fill=(255, 255, 255, 230),
            outline=(200, 30, 30, 240),
            width=3,
        )
        draw.text((sx0, sy0 - 2), sample_text, fill=(200, 30, 30, 240), font=font_sample)

        # Subtext under sample: ICAO SPECIMEN
        sub_txt = "ICAO 9303 SPECIMEN"
        try:
            sub_b = font_sub.getbbox(sub_txt)
            sub_w = sub_b[2] - sub_b[0]
            draw.text((sx0 + (sw - sub_w) // 2, sy0 + sh + 10), sub_txt, fill=(120, 20, 20, 220), font=font_sub)
        except Exception:
            pass

        # ── 3. Green Biometric Guidelines (Exact Competitor Placement) ──
        GREEN = (46, 125, 50, 255)
        GREEN_LINE = (46, 125, 50, 200)
        BADGE_BG = (245, 253, 245, 240)
        BADGE_BORDER = (46, 125, 50, 240)
        TEXT_COLOR = (25, 100, 35, 255)

        # Horizontal lines across the subject
        draw.line([(0, crown_y), (w, crown_y)], fill=GREEN_LINE, width=2)
        draw.line([(0, eye_y), (w, eye_y)], fill=GREEN_LINE, width=2)
        draw.line([(0, chin_y), (w, chin_y)], fill=GREEN_LINE, width=2)

        # Inward pointer arrows on left edge
        arr_size = max(10, int(w * 0.016))
        for y_pos in (crown_y, eye_y, chin_y):
            draw.polygon([(8, y_pos), (8 + arr_size, y_pos - arr_size // 2), (8 + arr_size, y_pos + arr_size // 2)], fill=GREEN)

        # ── 4. Measurement Badges on Right Edge ──
        # Head height bracket (Crown to Chin)
        br1_x = int(w * 0.82)
        draw.line([(br1_x, crown_y), (br1_x, chin_y)], fill=GREEN, width=2)
        draw.line([(br1_x - 8, crown_y), (br1_x + 8, crown_y)], fill=GREEN, width=2)
        draw.line([(br1_x - 8, chin_y), (br1_x + 8, chin_y)], fill=GREEN, width=2)

        is_inches = doc_spec.width <= 3.0 or "in" in (doc_spec.width_inches or "")
        if is_inches:
            head_val = ((chin_y - crown_y) / h) * 2.0
            head_label = f"{head_val:.2f}in (1.00-1.38in)"
            eye_val = ((h - eye_y) / h) * 2.0
            eye_label = f"{eye_val:.2f}in (1.12-1.38in)"
            frame_label = f"{doc_spec.width_inches or '2.0in'} ({doc_spec.width:.0f}mm)"
            lh_label = doc_spec.width_inches or "2.0in"
        else:
            head_mm = ((chin_y - crown_y) / h) * doc_spec.height
            eye_mm = ((h - eye_y) / h) * doc_spec.height
            ratio_min = doc_spec.face_height_ratio_min
            ratio_max = doc_spec.face_height_ratio_max
            if ratio_min is None or ratio_max is None:
                import re
                match = re.search(r"(\d+)-(\d+)", doc_spec.head_size_percent or "")
                if match:
                    ratio_min = float(match.group(1)) / 100.0
                    ratio_max = float(match.group(2)) / 100.0
                else:
                    ratio_min = 0.60
                    ratio_max = 0.75

            min_head = int(ratio_min * doc_spec.height)
            max_head = int(ratio_max * doc_spec.height)
            head_label = f"{head_mm:.0f}mm ({min_head}-{max_head}mm)"
            eye_label = f"{eye_mm:.0f}mm ({int(doc_spec.height * 0.56)}mm)"
            frame_label = f"{doc_spec.width}x{doc_spec.height}mm"
            lh_label = f"{doc_spec.height}mm"

        try:
            hb = font_pill.getbbox(head_label)
            hlw, hlh = hb[2] - hb[0], hb[3] - hb[1]
        except Exception:
            hlw, hlh = 120, 16

        h_mid_y = (crown_y + chin_y) // 2
        pill1 = [br1_x - hlw // 2 - 8, h_mid_y - hlh // 2 - 5, br1_x + hlw // 2 + 8, h_mid_y + hlh // 2 + 5]
        draw.rounded_rectangle(pill1, radius=4, fill=BADGE_BG, outline=BADGE_BORDER, width=2)
        draw.text((br1_x - hlw // 2, h_mid_y - hlh // 2 - 3), head_label, fill=TEXT_COLOR, font=font_pill)

        # Eye height bracket (Bottom of frame to Eye level)
        br2_x = int(w * 0.91)
        draw.line([(br2_x, eye_y), (br2_x, h - 8)], fill=GREEN, width=2)
        draw.line([(br2_x - 8, eye_y), (br2_x + 8, eye_y)], fill=GREEN, width=2)
        draw.line([(br2_x - 8, h - 8), (br2_x + 8, h - 8)], fill=GREEN, width=2)

        try:
            eb = font_sub.getbbox(eye_label)
            elw, elh = eb[2] - eb[0], eb[3] - eb[1]
        except Exception:
            elw, elh = 100, 14

        e_mid_y = (eye_y + h) // 2
        pill2 = [br2_x - elw // 2 - 6, e_mid_y - elh // 2 - 4, br2_x + elw // 2 + 6, e_mid_y + elh // 2 + 4]
        draw.rounded_rectangle(pill2, radius=4, fill=BADGE_BG, outline=BADGE_BORDER, width=2)
        draw.text((br2_x - elw // 2, e_mid_y - elh // 2 - 2), eye_label, fill=TEXT_COLOR, font=font_sub)

        # ── 5. Bottom & Left Dimension Scales ──
        # Bottom dimension bar
        dim_y = h - 26
        draw.line([(int(w * 0.05), dim_y), (int(w * 0.70), dim_y)], fill=(100, 100, 100, 200), width=2)
        draw.line([(int(w * 0.05), dim_y - 6), (int(w * 0.05), dim_y + 6)], fill=(100, 100, 100, 200), width=2)
        draw.line([(int(w * 0.70), dim_y - 6), (int(w * 0.70), dim_y + 6)], fill=(100, 100, 100, 200), width=2)

        try:
            bwb = font_pill.getbbox(frame_label)
            bww, bwh = bwb[2] - bwb[0], bwb[3] - bwb[1]
        except Exception:
            bww, bwh = 80, 14

        pill_b = [int(w * 0.36) - bww // 2 - 8, dim_y - bwh // 2 - 4, int(w * 0.36) + bww // 2 + 8, dim_y + bwh // 2 + 4]
        draw.rounded_rectangle(pill_b, radius=4, fill=(255, 255, 255, 240), outline=(120, 120, 120, 220), width=1)
        draw.text((int(w * 0.36) - bww // 2, dim_y - bwh // 2 - 2), frame_label, fill=(60, 60, 60, 255), font=font_pill)

        # Left height dimension bar
        dim_lx = 30
        draw.line([(dim_lx, int(h * 0.08)), (dim_lx, int(h * 0.90))], fill=(100, 100, 100, 200), width=2)
        draw.line([(dim_lx - 6, int(h * 0.08)), (dim_lx + 6, int(h * 0.08))], fill=(100, 100, 100, 200), width=2)
        draw.line([(dim_lx - 6, int(h * 0.90)), (dim_lx + 6, int(h * 0.90))], fill=(100, 100, 100, 200), width=2)

        try:
            lhb = font_sub.getbbox(lh_label)
            lhw, lhh = lhb[2] - lhb[0], lhb[3] - lhb[1]
        except Exception:
            lhw, lhh = 40, 14

        pill_l = [dim_lx - lhw // 2 - 6, int(h * 0.49) - lhh // 2 - 4, dim_lx + lhw // 2 + 6, int(h * 0.49) + lhh // 2 + 4]
        draw.rounded_rectangle(pill_l, radius=4, fill=(255, 255, 255, 240), outline=(120, 120, 120, 220), width=1)
        draw.text((dim_lx - lhw // 2, int(h * 0.49) - lhh // 2 - 2), lh_label, fill=(60, 60, 60, 255), font=font_sub)

        # Composite directly onto photo
        composite = Image.alpha_composite(pil_img, overlay).convert("RGB")
        return cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)

    def generate(
        self,
        photo: np.ndarray,
        checks: List[ValidationCheck],
        doc_spec: DocumentSpec,
        face_landmarks: Optional[dict] = None,
    ) -> np.ndarray:
        h, w = photo.shape[:2]
        panel_w = max(400, w // 2)
        canvas_w = w + panel_w
        canvas = np.full((h, canvas_w, 3), 40, dtype=np.uint8)
        canvas[:h, :w] = photo

        x_start = w + 20
        y_start = 30
        line_height = max(20, h // (len(checks) + 4))

        cv2.putText(
            canvas,
            f"{doc_spec.width}x{doc_spec.height}mm @ {doc_spec.dpi} DPI",
            (x_start, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.WHITE,
            1,
        )
        y_start += line_height + 10

        for check in checks:
            color = self.GREEN if check.passed else self.RED
            icon = "PASS" if check.passed else "FAIL"
            score_str = f"{check.score:.0f}"

            cv2.putText(canvas, icon, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(canvas, check.name[:30], (x_start + 50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.WHITE, 1)
            cv2.putText(canvas, score_str, (x_start + panel_w - 60, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_start += line_height

        return canvas
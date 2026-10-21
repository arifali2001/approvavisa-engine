"""Job-application portraits (Bewerbungsfoto): not a document check, a flattering head-and-shoulders crop.

Reuses the face analyzer and the IS-Net matte from document processing, but composites onto
soft studio backdrops instead of a regulated solid colour, frames with more shoulders, and applies
only light, global corrections (white balance, tone, very mild skin smoothing). Faces are never
reshaped. Pure numpy/OpenCV apart from the injected analyzers, so it is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from approvavisa_engine.core.photo_processor import level_eye_line

# Backdrops as (top colour, bottom colour) in RGB plus a centre glow strength.
BACKDROPS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], float]] = {
    "white": ((255, 255, 255), (255, 255, 255), 0.0),
    "light-grey": ((238, 238, 236), (214, 214, 211), 0.35),
    "studio-grey": ((176, 180, 186), (120, 126, 134), 0.55),
    "blue-grey": ((214, 222, 230), (160, 176, 192), 0.45),
    "warm": ((240, 234, 226), (208, 198, 186), 0.40),
}

# Output frames: (width, height) in pixels.
FRAMES: Dict[str, Tuple[int, int]] = {
    "cv": (1200, 1500),       # 4:5, prints sharp at 4 x 5 cm and 4.5 x 6 cm crops
    "square": (1200, 1200),   # LinkedIn, Xing, job portals
}

# Where the head sits in each frame, as fractions of output height.
HEAD_TOP = {"cv": 0.11, "square": 0.12}
HEAD_HEIGHT = {"cv": 0.42, "square": 0.46}


class PortraitError(ValueError):
    """A photo that cannot become a portrait (no face, several faces, failed matte)."""


@dataclass
class PortraitResult:
    images: Dict[str, np.ndarray]
    background: str
    # "photo-ends-above-frame": the upload is cut off above the shoulders, so the frame fades out.
    warnings: List[str] = field(default_factory=list)


def backdrop(width: int, height: int, name: str, head_center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Vertical gradient with a soft light glow behind the head. Returns BGR uint8."""
    if name not in BACKDROPS:
        raise PortraitError(f"Unknown background '{name}'")
    top, bottom, glow = BACKDROPS[name]
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_bgr = np.array(top[::-1], dtype=np.float32)
    bottom_bgr = np.array(bottom[::-1], dtype=np.float32)
    canvas = np.broadcast_to(top_bgr * (1 - t) + bottom_bgr * t, (height, width, 3)).copy()
    if glow > 0:
        cx, cy = head_center or (0.5, 0.35)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        d = np.sqrt(((xx / width - cx) / 0.55) ** 2 + ((yy / height - cy) / 0.6) ** 2)
        light = np.clip(1.0 - d, 0.0, 1.0)[..., None] ** 1.6
        canvas = canvas + (255.0 - canvas) * light * glow * 0.6
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def touch_up(bgr: np.ndarray, skin_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Light global correction: half-strength grey-world white balance, gentle local contrast on
    lightness, and a very mild edge-preserving smoothing limited to the skin mask."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1.0)
    img = img * (1.0 + (gain - 1.0) * 0.5)
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l_channel)
    lab[:, :, 0] = cv2.addWeighted(l_channel, 0.65, clahe, 0.35, 0)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if skin_mask is not None and skin_mask.any():
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=28, sigmaSpace=9)
        weight = cv2.GaussianBlur(skin_mask.astype(np.float32), (0, 0), 6)[..., None] * 0.35
        img = np.clip(img.astype(np.float32) * (1 - weight) + smooth.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    return img


MAX_HEAD_HEIGHT = 0.62


def crop_box(image_shape: Tuple[int, int], crown_y: float, chin_y: float, center_x: float, frame: str) -> Tuple[float, float, float, float]:
    """Crop rectangle (x, y, w, h) in source pixels placing the head per HEAD_TOP/HEAD_HEIGHT.
    If the photo ends above the planned bottom edge (a tight selfie), the frame zooms in so it ends
    at the photo's bottom, up to MAX_HEAD_HEIGHT; nothing below the photo is invented. The rectangle
    may still extend past the sides or the bottom; the caller fills that with the backdrop."""
    out_w, out_h = FRAMES[frame]
    head = max(chin_y - crown_y, 1.0)
    h = head / HEAD_HEIGHT[frame]
    top = HEAD_TOP[frame]
    available = image_shape[0] - crown_y  # source rows from the crown down
    if crown_y + (1 - top) * h > image_shape[0]:
        h = max(available / (1 - top), head / MAX_HEAD_HEIGHT)
    w = h * out_w / out_h
    y = crown_y - top * h
    x = center_x - w / 2
    return x, y, w, h


def _face_skin_mask(shape: Tuple[int, int], face) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if face.landmarks is not None:
        h, w = shape
        pts = (face.landmarks[:, :2] * np.array([w, h])).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
        # Keep eyes, brows and lips sharp.
        for idx in ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380],
                    [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]):
            cv2.fillConvexPoly(mask, cv2.convexHull(pts[idx]), 0)
    return mask


def make_portrait(image: np.ndarray, background: str, face_analyzer, background_engine,
                  frames=("cv", "square"), retouch: bool = True) -> PortraitResult:
    """Build the portrait in each requested frame. Raises PortraitError on unusable input."""
    if background not in BACKDROPS:
        raise PortraitError(f"Unknown background '{background}'")
    face = face_analyzer.analyze(image)
    if not face.detected or face.face_count != 1:
        raise PortraitError("Upload a photo with exactly one clearly visible face.")

    # Level a small head tilt, as document preparation does.
    image, applied, _ = level_eye_line(image, face.left_eye_center, face.right_eye_center)
    if applied:
        face = face_analyzer.analyze(image)
        if not face.detected:
            raise PortraitError("Upload a photo with exactly one clearly visible face.")

    matte = background_engine.remove_background(image, (255, 255, 255))
    if not matte.success or matte.alpha is None:
        raise PortraitError("The background could not be separated from the person.")
    alpha = matte.alpha.astype(np.float32) / 255.0

    skin = _face_skin_mask(image.shape[:2], face) if retouch else None
    subject = touch_up(image, skin) if retouch else image

    # Crown: topmost matte row above the face within the face's width (hair included).
    x0, x1 = max(face.face_x, 0), min(face.face_x + face.face_w, image.shape[1])
    column = alpha[:, x0:x1] > 0.5
    rows = np.where(column.any(axis=1))[0]
    crown_y = float(rows[0]) if rows.size else float(face.face_y)
    crown_y = min(crown_y, float(face.forehead_top[1] or face.face_y))
    chin_y = float(face.chin[1] or face.face_y + face.face_h)
    center_x = float(face.eye_midpoint[0] or face.face_x + face.face_w / 2)

    results: Dict[str, np.ndarray] = {}
    warnings: List[str] = []
    for frame in frames:
        out_w, out_h = FRAMES[frame]
        x, y, w, h = crop_box(image.shape[:2], crown_y, chin_y, center_x, frame)
        scale = out_w / w
        m = np.array([[scale, 0, -x * scale], [0, scale, -y * scale]], dtype=np.float32)
        warp = lambda src, border: cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_LANCZOS4,
                                                 borderMode=border)
        fg = warp(subject, cv2.BORDER_REPLICATE).astype(np.float32)
        a = warp(alpha, cv2.BORDER_CONSTANT)
        # Nothing outside the source is invented; a photo cut off above the frame's bottom edge
        # fades out softly over its last rows instead of ending in a hard line.
        a = np.where(_inside_mask(out_w, out_h, m, image.shape[:2], keep_bottom=False), a, 0.0)
        src_bottom = int((image.shape[0] - y) * scale)
        if src_bottom < out_h:
            if "photo-ends-above-frame" not in warnings and src_bottom < out_h * 0.97:
                warnings.append("photo-ends-above-frame")
            start = max(0, src_bottom - max(8, int(out_h * 0.06)))
            if src_bottom > start:
                a[start:src_bottom] *= np.linspace(1.0, 0.0, src_bottom - start, dtype=np.float32)[:, None]
        a = np.clip(a, 0.0, 1.0)[..., None]
        head_center = ((center_x - x) / w, ((crown_y + chin_y) / 2 - y) / h)
        bg = backdrop(out_w, out_h, background, head_center).astype(np.float32)
        results[frame] = np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)
    return PortraitResult(images=results, background=background, warnings=warnings)


def _inside_mask(out_w: int, out_h: int, m: np.ndarray, src_shape: Tuple[int, int], keep_bottom: bool) -> np.ndarray:
    """True where the output pixel maps inside the source (optionally also below it)."""
    sh, sw = src_shape
    scale, tx, ty = m[0, 0], m[0, 2], m[1, 2]
    xs = (np.arange(out_w) - tx) / scale
    ys = (np.arange(out_h) - ty) / scale
    inside_x = (xs >= 0) & (xs < sw)
    inside_y = (ys >= 0) & ((ys < sh) | keep_bottom)
    return inside_y[:, None] & inside_x[None, :]


def watermark_preview(image: np.ndarray, label: str = "VORSCHAU", max_side: int = 640) -> np.ndarray:
    """Downscaled preview with a repeated diagonal watermark that cannot be cropped away."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    diag = int(np.hypot(sh, sw))
    layer = np.zeros((diag, diag), dtype=np.uint8)
    step = max(70, sw // 5)
    font_scale = max(0.5, sw / 900)
    for row, yy in enumerate(range(0, diag, step)):
        offset = (row % 2) * step
        for xx in range(-step + offset, diag, step * 3):
            cv2.putText(layer, label, (xx, yy), cv2.FONT_HERSHEY_DUPLEX, font_scale, 255, 2, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((diag / 2, diag / 2), 30, 1.0)
    layer = cv2.warpAffine(layer, rot, (diag, diag))
    oy, ox = (diag - sh) // 2, (diag - sw) // 2
    mark = layer[oy:oy + sh, ox:ox + sw].astype(np.float32)[..., None] / 255.0
    out = small.astype(np.float32) * (1 - mark * 0.45) + 255.0 * mark * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)

"""Image I/O utilities: EXIF orientation, base64 encoding/decoding, color conversions."""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR numpy array."""
    # Strip data URI prefix if present
    if "," in data:
        data = data.split(",", 1)[1]

    if len(data) > 20_000_000:
        raise ValueError("Photo exceeds the 15 MB input limit.")
    img_bytes = base64.b64decode(data, validate=True)
    pil_img = Image.open(io.BytesIO(img_bytes))
    if pil_img.format not in {"JPEG", "PNG", "WEBP"} or getattr(pil_img, "n_frames", 1) != 1:
        raise ValueError("Upload a single JPG, PNG or WebP photograph.")
    width, height = pil_img.size
    if max(width, height) > 8192 or width * height > 24_000_000:
        raise ValueError("Photo dimensions exceed 8192 pixels per side or 24 megapixels.")

    # Auto-correct EXIF orientation (phone cameras encode rotation in EXIF)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Colour-manage to sRGB. Phones often tag photos Display P3; reading those values as sRGB gives dull,
    # slightly wrong skin tones. Untagged and sRGB-tagged photos are left exactly as they are.
    pil_img = _to_srgb(pil_img)

    # Normalize color mode to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Convert PIL (RGB) -> OpenCV (BGR)
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


_SRGB_PROFILE = None


def _to_srgb(pil_img: Image.Image) -> Image.Image:
    """Convert an image with an embedded non-sRGB ICC profile to sRGB. Never fails: on any error the
    image is returned unchanged (the previous behaviour)."""
    icc = pil_img.info.get("icc_profile")
    if not icc:
        return pil_img
    try:
        from PIL import ImageCms

        global _SRGB_PROFILE
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        description = (ImageCms.getProfileDescription(source) or "").lower()
        if "srgb" in description:
            return pil_img
        if _SRGB_PROFILE is None:
            _SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        if pil_img.mode not in ("RGB", "CMYK"):
            return pil_img
        return ImageCms.profileToProfile(pil_img, source, _SRGB_PROFILE,
                                         renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
    except Exception:
        return pil_img


def resize_hq(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize for output. Shrinking uses Pillow's antialiased Lanczos: OpenCV's INTER_LANCZOS4 keeps a fixed
    8x8 kernel, so a 3-5x reduction aliases (jagged hair edges, moire on clothing; measured 2026-11-14).
    Enlarging keeps OpenCV Lanczos4 (no aliasing when upscaling)."""
    out_w, out_h = int(size[0]), int(size[1])
    h, w = image.shape[:2]
    if out_w >= w and out_h >= h:
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    # Channel order is irrelevant to resampling, so the BGR array goes through Pillow as is.
    return np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((out_w, out_h), Image.LANCZOS))


def encode_image_base64(
    image: np.ndarray,
    fmt: str = "JPEG",
    quality: int = 98,
    dpi: int = 600,
) -> str:
    """Encode a BGR numpy array to a base64 string with guaranteed metadata stripping and calibrated DPI."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0  # 4:4:4 chroma
        save_kwargs["dpi"] = (dpi, dpi)
        save_kwargs["exif"] = b""
        save_kwargs["comment"] = b""
    pil_img.save(buffer, **save_kwargs)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_bytes(
    image: np.ndarray,
    fmt: str = "JPEG",
    quality: int = 98,
    dpi: int = 600,
    max_size_kb: Optional[int] = None,
) -> bytes:
    """Encode BGR array to bytes with optional binary-search compression for size limit."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if max_size_kb is None:
        buffer = io.BytesIO()
        save_kwargs: dict = {"format": fmt, "dpi": (dpi, dpi)}
        if fmt.upper() == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0
            save_kwargs["exif"] = b""
            save_kwargs["comment"] = b""
        pil_img.save(buffer, **save_kwargs)
        return buffer.getvalue()

    # Binary-search for optimal quality within file size constraint
    max_bytes = max_size_kb * 1024
    lo, hi = 10, quality
    best_bytes = b""

    while lo <= hi:
        mid = (lo + hi) // 2
        buffer = io.BytesIO()
        save_kwargs = {"format": fmt, "quality": mid, "subsampling": 0, "dpi": (dpi, dpi), "exif": b"", "comment": b""}
        pil_img.save(buffer, **save_kwargs)
        result = buffer.getvalue()

        if len(result) <= max_bytes:
            best_bytes = result
            lo = mid + 1
        else:
            hi = mid - 1

    return best_bytes if best_bytes else buffer.getvalue()


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    """Convert RGB tuple to CIE Lab color space (D65 illuminant)."""
    pixel = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)
    return lab[0, 0].astype(np.float64)


def ciede2000_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute CIEDE2000 color difference between two Lab colors.

    Simplified implementation of Sharma et al. 2005.
    More perceptually accurate than CIE76 Delta-E.
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    C_avg_7 = C_avg**7
    G = 0.5 * (1.0 - np.sqrt(C_avg_7 / (C_avg_7 + 25.0**7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360) / 2.0
    else:
        hp_avg = (h1p + h2p - 360) / 2.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_avg - 30))
        + 0.24 * np.cos(np.radians(2 * hp_avg))
        + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
        - 0.20 * np.cos(np.radians(4 * hp_avg - 63))
    )

    SL = 1.0 + 0.015 * (Lp_avg - 50) ** 2 / np.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg_7 = Cp_avg**7
    RT = (
        -2.0
        * np.sqrt(Cp_avg_7 / (Cp_avg_7 + 25.0**7))
        * np.sin(np.radians(60.0 * np.exp(-((hp_avg - 275) / 25.0) ** 2)))
    )

    dE = np.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)
    )

    return float(dE)


def encode_document_image(image, doc_spec, dpi, max_size_kb=None):
    """Shared preview/download encoding contract; never release out-of-range bytes."""
    limits = [n for n in (doc_spec.max_file_size_bytes, max_size_kb * 1024 if max_size_kb else None) if n]
    maximum = min(limits) if limits else None
    encoded = encode_image_bytes(image, dpi=dpi, max_size_kb=maximum / 1024 if maximum else None)
    if maximum and len(encoded) > maximum:
        raise ValueError("Photo could not be encoded within the upload size limit. Retake with a simpler background.")
    if doc_spec.min_file_size_bytes and len(encoded) < doc_spec.min_file_size_bytes:
        raise ValueError("Photo is below the portal minimum file size. Upload a higher-detail original.")
    return encoded

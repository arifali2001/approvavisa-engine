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

    img_bytes = base64.b64decode(data)
    pil_img = Image.open(io.BytesIO(img_bytes))

    # Auto-correct EXIF orientation (phone cameras encode rotation in EXIF)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Normalize color mode to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Convert PIL (RGB) -> OpenCV (BGR)
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def encode_image_base64(image: np.ndarray, fmt: str = "JPEG", quality: int = 98) -> str:
    """Encode a BGR numpy array to a base64 string."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt.upper() == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0  # 4:4:4 chroma
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
        pil_img.save(buffer, **save_kwargs)
        return buffer.getvalue()

    # Binary-search for optimal quality within file size constraint
    max_bytes = max_size_kb * 1024
    lo, hi = 10, quality
    best_bytes = b""

    while lo <= hi:
        mid = (lo + hi) // 2
        buffer = io.BytesIO()
        save_kwargs = {"format": fmt, "quality": mid, "subsampling": 0, "dpi": (dpi, dpi)}
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

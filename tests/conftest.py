"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def spec_registry():
    from approvavisa_engine.core.spec_registry import JSONSpecRegistry
    return JSONSpecRegistry()


@pytest.fixture
def sample_image():
    """Generate a synthetic test image with a face-like pattern."""
    import numpy as np
    # 800x800 white image with a simple oval "face"
    img = np.full((800, 800, 3), 240, dtype=np.uint8)  # light gray background

    # Draw oval "face"
    import cv2
    cv2.ellipse(img, (400, 400), (120, 160), 0, 0, 360, (180, 150, 130), -1)

    # Draw "eyes"
    cv2.circle(img, (360, 360), 15, (40, 40, 40), -1)
    cv2.circle(img, (440, 360), 15, (40, 40, 40), -1)

    # Draw "mouth"
    cv2.ellipse(img, (400, 450), (30, 10), 0, 0, 360, (150, 100, 100), -1)

    return img

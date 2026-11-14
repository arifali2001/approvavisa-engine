"""Stage-level timing of the ApprovaVisa engine's editing flow (what the sites call), on this PC.
Usage: .venv python scripts/bench_engine.py <photo> [scale]   (scale 3.2 turns a 1 MP sample into ~11 MP)"""
import base64, collections, functools, io, sys, time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(ENGINE))

import cv2, numpy as np
from PIL import Image

photo, scale = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
img = Image.open(photo).convert("RGB")
if scale != 1.0:
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
buf = io.BytesIO(); img.save(buf, "JPEG", quality=92)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"input {img.size[0]}x{img.size[1]} = {img.size[0]*img.size[1]/1e6:.1f} MP, {len(buf.getvalue())/1e6:.1f} MB")

from approvavisa_engine.api import deps
from approvavisa_engine.core import background, crown_detector, face_analyzer, image_quality, photo_processor, validator, preview, image_utils

stats = collections.defaultdict(lambda: [0, 0.0])
def wrap(cls, name, label):
    orig = getattr(cls, name)
    @functools.wraps(orig)
    def inner(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            s = stats[label]; s[0] += 1; s[1] += time.perf_counter() - t
    setattr(cls, name, inner)

wrap(background.RembgBackgroundEngine, "remove_background", "bg.remove_background (rembg+refine)")
wrap(background.RembgBackgroundEngine, "_portrait_confidence", "  bg._portrait_confidence (selfie multiclass)")
wrap(background.RembgBackgroundEngine, "_refine_alpha", "  bg._refine_alpha")
wrap(background.RembgBackgroundEngine, "analyze_background", "bg.analyze_background")
wrap(crown_detector.SegmentationCrownDetector, "detect_crown", "crown.detect_crown")
wrap(face_analyzer.MediaPipeFaceAnalyzer, "analyze", "face.analyze")
wrap(image_quality.OpenCVQualityAnalyzer, "analyze", "quality.analyze")
wrap(validator.ICAOValidator, "validate", "validator.validate (total)")
wrap(photo_processor.StandardPhotoProcessor, "process", "processor.process (total, incl. recursion)")
wrap(preview.AnnotatedPreviewGenerator, "generate_preview_specimen", "preview.specimen")

t0 = time.perf_counter(); deps.get_validator(); deps.get_processor(); deps.get_preview_generator()
from approvavisa_engine.main import _warm_up; _warm_up()
print(f"warm-up {time.perf_counter()-t0:.1f}s"); stats.clear()

from fastapi.testclient import TestClient
from approvavisa_engine.main import app
from approvavisa_engine.config import settings
c = TestClient(app)
H = {"X-API-Key": settings.api_key}

for label, path, body in [
    ("PREVIEW  /v1/validate editing_mode", "/v1/validate", {"image": b64, "country_code": "ES", "document_type": "DNI", "editing_mode": True}),
    ("DOWNLOAD /v1/process editing_mode", "/v1/process", {"image": b64, "country_code": "ES", "document_type": "DNI", "editing_mode": True}),
]:
    stats.clear(); t = time.perf_counter()
    r = c.post(path, json=body, headers=H)
    total = time.perf_counter() - t
    j = r.json()
    print(f"\n== {label}: HTTP {r.status_code}, {total:.2f}s, success/compliant={j.get('success', j.get('compliant'))} {str(j.get('message',''))[:80]}")
    for k, (n, s) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        print(f"   {s:6.2f}s  x{n:<2} {k}")

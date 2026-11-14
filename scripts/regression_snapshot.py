"""Before/after safety net for engine changes.

Runs real photos through the same calls the sites make and stores every check result and measurement,
plus the delivered image, so two engine versions can be compared.

  python scripts/regression_snapshot.py snapshot OUT_DIR PHOTO [PHOTO ...]
  python scripts/regression_snapshot.py compare BEFORE_DIR AFTER_DIR

`compare` fails (exit 1) when a decision changes (success, compliant, preparable, the set of failed checks) or a
geometry metric moves more than its tolerance. Pixel changes are reported but expected (that's the point of a
quality fix); side-by-side JPEGs are written to AFTER_DIR/sbs/ for a visual check.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DOCS = [("ES", "DNI"), ("IT", "Passport"), ("US", "Passport"), ("GB", "Passport")]
MODES = [
    ("validate_edit", "/v1/validate", {"editing_mode": True}),
    ("process_edit", "/v1/process", {"editing_mode": True}),
    ("validate_plain", "/v1/validate", {"editing_mode": False}),
]
# Geometry/measurement tolerances (absolute) for metrics present in the response.
TOL = {"eyeLevelMm": 0.4, "headHeightPercent": 1.5, "opticalYawDegrees": 1.5, "opticalPitchDegrees": 2.0,
       "aspectRatio": 0.01, "backgroundDeltaE": 3.0, "exposureEv": 0.3}


def _load(path: str, scale: float) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def snapshot(out: Path, photos: list[str]) -> None:
    from fastapi.testclient import TestClient

    from approvavisa_engine.config import settings
    from approvavisa_engine.main import app

    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    results = {}
    with TestClient(app) as client:
        headers = {"X-API-Key": settings.api_key}
        for photo in photos:
            for scale in ([1.0, 3.0] if "upscale" in photo else [1.0]):
                path = photo.replace("::upscale", "")
                b64 = _load(path, scale)
                stem = Path(path).stem + ("@x3" if scale != 1.0 else "")
                for cc, doc in DOCS:
                    for mode, url, extra in MODES:
                        key = f"{stem}|{cc}-{doc}|{mode}"
                        t = time.perf_counter()
                        r = client.post(url, headers=headers,
                                        json={"image": b64, "country_code": cc, "document_type": doc, **extra})
                        dt = time.perf_counter() - t
                        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                        rec = {"status": r.status_code, "seconds": round(dt, 2),
                               "success": j.get("success"), "compliant": j.get("compliant"),
                               "preparable": j.get("preparable"),
                               "failed": sorted(c["id"] for c in j.get("checks", []) if not c.get("passed")),
                               "measured": {c["id"]: c.get("measured") for c in j.get("checks", [])},
                               "metrics": j.get("metrics") or {},
                               "width_px": j.get("width_px"), "height_px": j.get("height_px"),
                               "message": (j.get("message") or "")[:120]}
                        img = j.get("processed_image")
                        if img:
                            name = key.replace("|", "__") + ".jpg"
                            (out / "img" / name).write_bytes(base64.b64decode(img))
                            rec["image"] = name
                        results[key] = rec
                        print(f"{dt:5.1f}s {r.status_code} {key}: compliant={rec['compliant']} success={rec['success']} failed={rec['failed']}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def compare(before: Path, after: Path) -> int:
    import cv2
    import numpy as np

    a = json.loads((before / "results.json").read_text(encoding="utf-8"))
    b = json.loads((after / "results.json").read_text(encoding="utf-8"))
    (after / "sbs").mkdir(exist_ok=True)
    problems, t_a, t_b = [], 0.0, 0.0
    for key, ra in a.items():
        rb = b.get(key)
        if rb is None:
            problems.append(f"{key}: missing in after")
            continue
        t_a += ra["seconds"]; t_b += rb["seconds"]
        for f in ("status", "success", "compliant", "preparable", "failed"):
            if ra[f] != rb[f]:
                problems.append(f"{key}: {f} {ra[f]} -> {rb[f]}")
        for m, tol in TOL.items():
            va, vb = ra["metrics"].get(m), rb["metrics"].get(m)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and abs(va - vb) > tol:
                problems.append(f"{key}: {m} {va:.2f} -> {vb:.2f} (tol {tol})")
        if (ra.get("width_px"), ra.get("height_px")) != (rb.get("width_px"), rb.get("height_px")):
            problems.append(f"{key}: size {ra.get('width_px')}x{ra.get('height_px')} -> {rb.get('width_px')}x{rb.get('height_px')}")
        if ra.get("image") and rb.get("image"):
            ia = cv2.imread(str(before / "img" / ra["image"])); ib = cv2.imread(str(after / "img" / rb["image"]))
            if ia is not None and ib is not None:
                if ia.shape != ib.shape:
                    ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
                cv2.imwrite(str(after / "sbs" / rb["image"]), np.hstack([ia, ib]))
    print(f"\ncalls: {len(a)}  total time before {t_a:.0f}s, after {t_b:.0f}s")
    print("\n".join(problems) if problems else "No decision or geometry changes.")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(Path(sys.argv[2]), sys.argv[3:])
    else:
        sys.exit(compare(Path(sys.argv[2]), Path(sys.argv[3])))

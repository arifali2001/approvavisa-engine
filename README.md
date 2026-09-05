# ApprovaVisa Engine

[![Live Website](https://img.shields.io/badge/Live-approvavisa.com-00C853?style=flat&logo=google-chrome&logoColor=white)](https://www.approvavisa.com)
[![Created by Arif](https://img.shields.io/badge/Created%20by-Arif%20%7C%20Glyphash-6C5CE7)](https://www.glyphash.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade ICAO Doc 9303 compliant biometric passport photo validation and processing engine.**

> **Live in Production**: Powering **[www.approvavisa.com](https://www.approvavisa.com)**  
> **Author**: Built with love, sweat, and ungodly amounts of caffeine by **Arif**, Founder of **[Glyphash](https://www.glyphash.com)** ([www.glyphash.com](https://www.glyphash.com))

---

### What is this? (And why does it exist?)

Have you ever tried submitting a passport photo online only to have an automated consular portal reject it because your left eyeball was allegedly **0.4mm too high**, your white wall had a **microscopic shadow**, or you dared to look **3 degrees to the left**?

Welcome to **ICAO Doc 9303** — roughly 150 pages of bureaucratic geometric nightmares written by international committees who seem to believe human skulls are mathematically uniform spheroids with zero hair and permanent robot expressions.

Most online "passport photo validators" are lazy wrappers that generate a random score between 80 and 95 and hope for the best. 

**This is not that.**

ApprovaVisa Engine is a genuine computer vision microservice running on **FastAPI**:
- **MediaPipe Tasks 478-point 3D Face Mesh** + **OpenCV `solvePnP`**: Computes genuine 3D Euler angles (Yaw, Pitch, Roll) using iterative Levenberg-Marquardt optimization against a canonical 3D human skull model. If you're tilted 4 degrees, we know it down to the decimal point.
- **IS-Net Neural Background Removal + 3-Stage Guided Filtering**: Because plain background removal turns curly flyaway hair into a jagged PS2 video game sprite. We defringe and composite so cleanly you look like you posed in an expensive studio.
- **CIEDE2000 ($\Delta E_{00}$) Color Delta**: The human eye perceives color non-linearly. We calculate perceptual color difference in Lab color space so muddy off-white hallway walls don't slip through.
- **Neural Hair Crown Detection**: Standard face detectors stop dead at the forehead. We scan the segmented alpha silhouette row-by-row so people with afros, high curls, buns, and turbans don't get accidentally decapitated during cropping.
- **Seamless Torso Extrusion**: If someone uploads a photo cropped tightly at the collarbone, we gently extrude the bottom row of clothing downward so they don't look like a floating decapitated head.
- **Lanczos-4 600 DPI Calibration**: Resamples and unsharp-masks directly to the exact millimeter specs of 190+ countries.

---

## The 22-Point Biometric Gauntlet

Every photo must survive 4 rigorous pillars:

1. **Spatial Geometry (35% Weight)**: Eye baseline elevation ($56-58\%$), crown clearance, horizontal symmetry offset ($\le 3.0\text{ mm}$), and head-to-frame ratio.
2. **Photometric Balance (25% Weight)**: CIEDE2000 background uniformity ($\Delta E < 5.0$), bilateral shadow leveling, exposure dynamic range, and specular hotspot elimination.
3. **Facial Biometrics (25% Weight)**: Zero optical rotation ($\le 5^\circ$ yaw/pitch), unobstructed eye apertures (Eye Aspect Ratio $> 0.20$), closed mouth (Mouth Aspect Ratio $< 0.15$), and neutral smile detection.
4. **Digital Output (15% Weight)**: Resolution density ($\ge 8\text{ px/mm}$), 600 DPI target print calibration, JPEG compression artifacts, and adaptive megapixel blur detection.

---

## Quickstart

Spin it up locally before your next flight:

```bash
# 1. Grab the code
git clone https://github.com/approvavisa/approvavisa-engine.git
cd approvavisa-engine

# 2. Install dependencies (grab a snack while PyTorch/OpenCV downloads)
pip install -e ".[dev]"

# 3. Configure your secrets
cp .env.example .env
# Set your API_KEY in .env (or leave it as 'changeme' if you like living dangerously)

# 4. Fire up the engine
uvicorn approvavisa_engine.main:app --reload

# 5. Run the test suite
pytest tests/ -v
```

---

## API Endpoints

Interactive Swagger documentation is available out of the box at **`http://localhost:8000/docs`**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/health` | Health probe (returns healthy if the server hasn't melted) |
| `GET` | `/v1/specs` | Search or list biometric requirements for 190+ countries |
| `GET` | `/v1/specs/{code}` | Get exact dimensions, DPI, and background rules for a country (e.g. `US`, `GB`, `CA`, `IN`) |
| `POST` | `/v1/validate` | Run the 22-point ICAO check and get back actionable retake coaching + watermarked specimen |
| `POST` | `/v1/process` | Crop to spec, strip backgrounds, calibrate to 600 DPI, and generate a 4x6 tiled print sheet |

### Authentication

Pass your key via header, bearer token, or query parameter (because testing 5MB base64 JSON payloads in Swagger without query params is pure torture):

```http
X-API-Key: your-secret-api-key
# OR
Authorization: Bearer your-secret-api-key
# OR
GET /v1/specs?api_key=your-secret-api-key
```

### Validate Request Body

```json
{
  "image": "<base64-encoded-image>",
  "country_code": "US",
  "document_type": "Passport"
}
```

---

## Architecture

```
src/approvavisa_engine/
├── main.py                  # FastAPI app factory, lifespan lifecycle & CORS
├── config.py                # Pydantic Settings (loads .env without blowing up)
├── api/
│   ├── deps.py              # Lazy singletons (keeps heavy ML models warm in RAM)
│   ├── router.py            # Central v1 route aggregator
│   └── v1/                  # REST endpoints (health, specs, validate, process)
├── core/
│   ├── face_analyzer.py     # MediaPipe Tasks + OpenCV solvePnP 3D pose math
│   ├── crown_detector.py    # Neural segmentation (saving afros & turbans from bad crops)
│   ├── background.py        # Rembg IS-Net + 3-stage guided filter & CIEDE2000
│   ├── image_quality.py     # Adaptive Laplacian blur & histogram dynamic range
│   ├── photo_processor.py   # True visual centering, torso extrusion & Lanczos-4 600 DPI
│   ├── preview.py           # Watermarked specimen generation with baked measuring scales
│   ├── validator.py         # 22-point ICAO compliance engine with SHA-256 certificate hashing
│   ├── spec_registry.py     # 190+ country specifications loaded from vendored database
│   └── image_utils.py       # EXIF auto-rotation, color space conversions, base64 IO
├── models/                  # Pydantic v2 schemas matching frontend TypeScript contracts
└── data/                    # Vendored countries.json database & ML models
```

---

## Extensibility

Don't like MediaPipe? Want to bring your own custom YOLO or TensorRT model? 
Every core engine component inherits from an abstract base class. Override them with zero code surgery using FastAPI's dependency injection:

```python
from approvavisa_engine.api.deps import get_face_analyzer
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer

class MySupercomputerFaceAnalyzer(BaseFaceAnalyzer):
    def analyze(self, image):
        # Your custom high-octane model implementation here
        ...

# Override in your app:
app.dependency_overrides[get_face_analyzer] = lambda: MySupercomputerFaceAnalyzer()
```

---

## Docker Deployment

Deploy with one command:

```bash
docker compose up --build
```

---

## Author & Credits

- **Creator**: **Arif**, Founder of **[Glyphash](https://www.glyphash.com)** ([www.glyphash.com](https://www.glyphash.com))
- **Live Production App**: **[ApprovaVisa](https://www.approvavisa.com)** ([www.approvavisa.com](https://www.approvavisa.com))

## License

MIT License — see [LICENSE](LICENSE) for details. (Free to use, but please don't use it to submit fake passport photos to Interpol).

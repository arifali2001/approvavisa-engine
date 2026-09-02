# ApprovaVisa Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade ICAO Doc 9303 compliant biometric passport photo validation and processing engine.**

Real computer vision analysis using MediaPipe Face Mesh (468-point 3D landmarks), solvePnP head pose estimation, CIEDE2000 color science, and rembg background removal. Not simulated scores.

## Features

- **22-point validation pipeline** across 4 pillars (Spatial Geometry, Photometric Balance, Facial Biometrics, Digital Output)
- **190+ countries** supported via vendored spec database
- **solvePnP 3D head pose** for sub-degree yaw/pitch/roll accuracy
- **CIEDE2000 color difference** for perceptually accurate background validation
- **rembg background removal** with alpha matting edge refinement
- **Image quality analysis**: blur, noise, exposure, red-eye, sharpness
- **Hair crown detection** via MediaPipe segmentation
- **Print sheet generation** (4x6 tiling with cut guides)
- **Extensible architecture** with 5 abstract base classes and DI
- **API key authentication**
- **Docker ready**

## Quickstart

```bash
# Clone
git clone https://github.com/approvavisa/approvavisa-engine.git
cd approvavisa-engine

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API key

# Run
uvicorn approvavisa_engine.main:app --reload

# Test
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check |
| GET | `/v1/specs` | List all supported countries |
| GET | `/v1/specs/{code}` | Get spec for a country |
| POST | `/v1/validate` | Validate a photo |
| POST | `/v1/process` | Process (crop + reformat) a photo |

### Validate Request

```json
{
  "image": "<base64-encoded-image>",
  "country_code": "US",
  "document_type": "Passport"
}
```

### Headers

```
X-API-Key: your-secret-api-key
```

## Architecture

```
src/approvavisa_engine/
+-- main.py                  # FastAPI app factory
+-- config.py                # Pydantic Settings (12+ env vars)
+-- api/                     # REST endpoints
+-- core/                    # Engine modules
|   +-- face_analyzer.py     # MediaPipe + solvePnP
|   +-- crown_detector.py    # Hair crown detection
|   +-- background.py        # rembg + CIEDE2000
|   +-- image_quality.py     # Blur, noise, exposure, red-eye
|   +-- validator.py         # 22-point ICAO validator
|   +-- photo_processor.py   # Crop, resize, print sheet
|   +-- preview.py           # Annotated preview generation
|   +-- spec_registry.py     # Country spec database
|   +-- image_utils.py       # IO, EXIF, color science
+-- models/                  # Pydantic models
+-- data/                    # Vendored countries.json
```

## Extensibility

All core components have abstract base classes. Override via FastAPI dependency injection:

```python
from approvavisa_engine.core.face_analyzer import BaseFaceAnalyzer

class MyFaceAnalyzer(BaseFaceAnalyzer):
    def analyze(self, image):
        # Your custom implementation
        ...

# In your app setup:
app.dependency_overrides[get_face_analyzer] = lambda: MyFaceAnalyzer()
```

## Docker

```bash
docker compose up --build
```

## License

MIT License - see [LICENSE](LICENSE) for details.

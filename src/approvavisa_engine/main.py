"""FastAPI application factory with lifespan management.

Welcome to the engine room of ApprovaVisa.
This FastAPI service orchestrates MediaPipe Tasks, OpenCV solvePnP 3D pose estimation,
Rembg neural matting, and 22 ICAO biometric checks without melting your server or blowing up your RAM.
Built with clean dependency injection so you can swap out any CV component without
having an emotional breakdown in the REST layer.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from approvavisa_engine.api.router import api_router
from approvavisa_engine.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _warm_up() -> None:
    """Load every model and run one tiny inference before this worker takes traffic.

    The models load lazily, so without this the first photo after a (re)start took ~55 s, close
    to the sites' 60 s timeout. Failures are logged, never fatal: the lazy path still works.
    """
    import time

    import numpy as np
    from PIL import Image

    from approvavisa_engine.api import deps

    started = time.perf_counter()
    try:
        deps.get_spec_registry()
        deps.get_validator()
        deps.get_processor()
        deps.get_preview_generator()
        blank = np.full((256, 256, 3), 255, np.uint8)
        deps.get_face_analyzer().analyze(blank)
        deps.get_crown_detector()._get_segmenter()
        background = deps.get_background_engine()
        session = background._get_session()
        if session is not None:
            session.predict(Image.new("RGB", (64, 64), "white"))
        background._portrait_confidence(blank)
        logger.info(f"Warm-up done in {time.perf_counter() - started:.1f}s")
    except Exception as e:  # defensive: never block startup
        logger.warning(f"Warm-up incomplete after {time.perf_counter() - started:.1f}s: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("ApprovaVisa Engine starting up...")
    logger.info(f"  Workers: {settings.workers}")
    logger.info(f"  Max image size: {settings.max_image_size_mb}MB")
    logger.info(f"  rembg model: {settings.rembg_model}")
    logger.info(f"  Output DPI: {settings.output_dpi}")
    logger.info(f"  Background removal: {settings.enable_background_removal}")
    logger.info(f"  Alpha matting: {settings.enable_alpha_matting}")
    if settings.warm_up_on_start:
        import asyncio

        await asyncio.to_thread(_warm_up)
    yield
    logger.info("ApprovaVisa Engine shutting down.")


app = FastAPI(
    title="ApprovaVisa Engine",
    description=(
        "Production-grade ICAO Doc 9303 compliant biometric passport photo "
        "validation and processing engine. Supports 190+ countries with real "
        "computer vision analysis — not simulated scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "engine": "ApprovaVisa Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health",
    }

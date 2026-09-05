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

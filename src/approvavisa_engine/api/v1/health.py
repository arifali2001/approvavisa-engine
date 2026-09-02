"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check — returns engine status."""
    return {
        "status": "healthy",
        "engine": "approvavisa-engine",
        "version": "1.0.0",
    }

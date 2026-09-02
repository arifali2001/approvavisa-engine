"""Central API router — aggregates all v1 routes."""

from __future__ import annotations

from fastapi import APIRouter

from approvavisa_engine.api.v1 import health, process, specs, validate

api_router = APIRouter()

# Health (no auth required)
api_router.include_router(health.router, prefix="/v1", tags=["Health"])

# Authenticated endpoints
api_router.include_router(specs.router, prefix="/v1", tags=["Specifications"])
api_router.include_router(validate.router, prefix="/v1", tags=["Validation"])
api_router.include_router(process.router, prefix="/v1", tags=["Processing"])

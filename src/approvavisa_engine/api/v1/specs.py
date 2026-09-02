"""Spec listing and lookup endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from approvavisa_engine.api.deps import get_spec_registry, verify_api_key
from approvavisa_engine.core.spec_registry import BaseSpecRegistry
from approvavisa_engine.models.specs import CountrySpec, CountrySummary

router = APIRouter()


@router.get("/specs", response_model=List[CountrySummary])
async def list_specs(
    q: Optional[str] = Query(None, description="Search query (country name or code)"),
    registry: BaseSpecRegistry = Depends(get_spec_registry),
    _: str = Depends(verify_api_key),
):
    """List all supported countries or search by name/code."""
    if q:
        return registry.search(q)
    specs = registry.get_all()
    return [
        CountrySummary(
            code=s.code,
            name=s.name,
            flag=s.flag,
            document_types=[d.type for d in s.documents],
            popular=s.popular,
        )
        for s in specs
    ]


@router.get("/specs/{code}", response_model=CountrySpec)
async def get_spec(
    code: str,
    registry: BaseSpecRegistry = Depends(get_spec_registry),
    _: str = Depends(verify_api_key),
):
    """Get full specification for a country by code."""
    spec = registry.get_by_code(code)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Country '{code}' not found")
    return spec

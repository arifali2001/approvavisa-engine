"""Job-application portrait endpoint (Bewerbungsfoto). Independent of document specs and checks."""

from __future__ import annotations

import base64
import logging
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from approvavisa_engine.api.deps import get_background_engine, get_face_analyzer, verify_api_key
from approvavisa_engine.core.image_utils import decode_base64_image, encode_image_bytes
from approvavisa_engine.core.portrait import BACKDROPS, FRAMES, PortraitError, make_portrait, watermark_preview

logger = logging.getLogger(__name__)
router = APIRouter()

Background = Literal["white", "light-grey", "studio-grey", "blue-grey", "warm"]
Frame = Literal["cv", "square"]


class PortraitRequest(BaseModel):
    image: str  # base64 (data URI allowed)
    background: Background = "light-grey"
    frames: List[Frame] = Field(default_factory=lambda: ["cv", "square"], min_length=1, max_length=2)
    retouch: bool = True
    # True: watermarked, downscaled previews only. False: full-resolution files (paid delivery).
    preview: bool = True
    watermark_label: str = Field(default="VORSCHAU", max_length=32)


class PortraitImage(BaseModel):
    frame: Frame
    image: str  # base64 JPEG
    width_px: int
    height_px: int


class PortraitResult(BaseModel):
    success: bool
    preview: bool
    background: str
    images: List[PortraitImage] = []
    warnings: List[str] = []
    message: str = ""


@router.get("/portrait/options")
async def portrait_options(_: str = Depends(verify_api_key)) -> Dict[str, list]:
    return {"backgrounds": list(BACKDROPS), "frames": [{"id": k, "width": w, "height": h} for k, (w, h) in FRAMES.items()]}


@router.post("/portrait", response_model=PortraitResult)
async def portrait(request: PortraitRequest, _: str = Depends(verify_api_key)):
    try:
        image = decode_base64_image(request.image)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {error}")
    frames = list(dict.fromkeys(request.frames))
    try:
        result = make_portrait(image, request.background, get_face_analyzer(), get_background_engine(),
                               frames=frames, retouch=request.retouch)
    except PortraitError as error:
        return PortraitResult(success=False, preview=request.preview, background=request.background, message=str(error))

    images: List[PortraitImage] = []
    for frame in frames:
        out = result.images[frame]
        if request.preview:
            out = watermark_preview(out, request.watermark_label.upper())
        data = encode_image_bytes(out, quality=82 if request.preview else 95, dpi=300)
        images.append(PortraitImage(frame=frame, image=base64.b64encode(data).decode("ascii"),
                                    width_px=out.shape[1], height_px=out.shape[0]))
    return PortraitResult(success=True, preview=request.preview, background=result.background, images=images,
                          warnings=result.warnings)

"""Photo management endpoints — score, cull, suggest categories, group."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


# --- Photo Cull (Segment 21) ---


class ScorePhotoRequest(BaseModel):
    path: str


class CullPhotosRequest(BaseModel):
    folder: str
    keep_pct: int = 80
    rejects_folder: str | None = None


@router.post("/score-photo")
def api_score_photo(req: ScorePhotoRequest) -> dict:
    """Score a photo's technical + aesthetic quality (Gemini vision, /100)."""
    from photo_cull import score_photo

    return score_photo(req.path)


@router.post("/cull-photos")
def api_cull_photos(req: CullPhotosRequest) -> dict:
    """Auto-cull photos: score all, move bottom rejects to _rejects folder. Background job."""
    from photo_cull import cull_photos

    return cull_photos(req.folder, req.keep_pct, req.rejects_folder)


@router.get("/cull-photos/status")
def api_cull_status(
    folder: str = Query(..., description="Folder being culled"),
    cancel: bool = Query(False, description="Set true to cancel the job"),
) -> dict:
    """Poll cull job progress. Set cancel=true to stop."""
    from photo_cull import get_cull_status

    return get_cull_status(folder, cancel=cancel)


# --- Photo Group (Segment 21B) ---


class SuggestCategoriesRequest(BaseModel):
    folder: str


@router.post("/suggest-categories")
def api_suggest_categories(req: SuggestCategoriesRequest) -> dict:
    """Sample photos and suggest grouping categories via Gemini vision."""
    from photo_cull import suggest_categories

    return suggest_categories(req.folder)


class GroupPhotosRequest(BaseModel):
    folder: str
    categories: list
    uncategorized_folder: str | None = None


@router.post("/group-photos")
def api_group_photos(req: GroupPhotosRequest) -> dict:
    """Auto-group photos: classify ALL images via Gemini vision, move to category subfolders. Background job."""
    from photo_cull import group_photos

    return group_photos(req.folder, req.categories, req.uncategorized_folder)


@router.get("/group-photos/status")
def api_group_status(
    folder: str = Query(..., description="Folder being grouped"),
    cancel: bool = Query(False, description="Set true to cancel the job"),
) -> dict:
    """Poll group job progress. Set cancel=true to stop."""
    from photo_cull import get_group_status

    return get_group_status(folder, cancel=cancel)

"""Media search & processing router — visual search, video, audio, OCR."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.helpers import _check_safe

router = APIRouter()

# Background embedding/scoring jobs: folder -> {status, total, done, error?}
_embedding_jobs = {}


# --- Pydantic models ---


class VisualSearchRequest(BaseModel):
    folder: str
    query: str
    limit: int = 10
    recursive: bool = False


class VideoSearchRequest(BaseModel):
    video_path: str
    query: str
    fps: float = 1.0
    limit: int = 5


class ExtractFrameRequest(BaseModel):
    video_path: str
    seconds: float
    output_path: str = None


class TranscribeAudioRequest(BaseModel):
    path: str


class SearchAudioRequest(BaseModel):
    audio_path: str
    query: str
    limit: int = 5


class OcrRequest(BaseModel):
    path: str = None
    folder: str = None


# --- Visual image search (Segment 18C: SigLIP2 embeddings) ---


@router.post("/search-images-visual")
def search_images_visual_endpoint(req: VisualSearchRequest) -> dict:
    """Search images in a folder by text description using SigLIP2 embeddings."""
    folder = os.path.abspath(req.folder)
    _check_safe(folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    from image_search import _HAS_SIGLIP, _get_image_files, _load_cached_embeddings, _mem_cache, search_images

    # Check if embedding is needed (folder not in memory cache)
    # Auto-recurse if no images at top level but subfolders exist
    recursive = req.recursive
    files = _get_image_files(folder, recursive=recursive)
    if not files and not recursive:
        files = _get_image_files(folder, recursive=True)
        if files:
            recursive = True
    if not files:
        return {
            "error": f"No images found in {folder}",
            "results": [],
            "_hint": "No images found. Try search_documents(query=..., file_type='image', folder=...) to search indexed image captions instead.",
        }
    # If auto-recursed into many images, suggest search_documents first (free + instant)
    if recursive and len(files) > 200:
        return {
            "error": f"Folder has {len(files)} images across subfolders — too many for visual search. Use search_documents(query=..., file_type='image', folder='{folder}') instead — it searches indexed image captions instantly for free.",
            "results": [],
            "total_images": len(files),
            "_hint": f"Too many images ({len(files)}). Try search_documents with file_type='image' first — it's free and instant.",
        }

    # No SigLIP — dispatch to Gemini vision (no embedding needed)
    if not _HAS_SIGLIP:
        from image_search import _search_images_gemini

        if len(files) > 500:
            # Large folder — run in background
            if folder in _embedding_jobs:
                job = _embedding_jobs[folder]
                if job["status"] == "running":
                    return {
                        "status": "scoring",
                        "total_batches": job["total"],
                        "done_batches": job["done"],
                        "_hint": f"Still scoring images with Gemini ({job['done']}/{job['total']} batches). Tell the user to wait and try again.",
                    }
                if job["status"] == "done":
                    result = job.get("result", {})
                    del _embedding_jobs[folder]
                    result["_hint"] = (
                        "Visual search complete. Results are AI-analyzed — trust them to answer, categorize, or group."
                    )
                    return result
                if job["status"] == "error":
                    del _embedding_jobs[folder]

            import math
            import threading

            total_batches = math.ceil(len(files) / 200)
            _embedding_jobs[folder] = {"status": "running", "total": total_batches, "done": 0}

            def _bg_gemini_search() -> None:
                try:

                    def _progress(done: int, total: int) -> None:
                        _embedding_jobs[folder]["done"] = done

                    result = _search_images_gemini(
                        folder, req.query, limit=req.limit, progress_callback=_progress, recursive=recursive
                    )
                    _embedding_jobs[folder]["status"] = "done"
                    _embedding_jobs[folder]["result"] = result
                except Exception as e:
                    _embedding_jobs[folder]["status"] = "error"
                    _embedding_jobs[folder]["error"] = str(e)

            threading.Thread(target=_bg_gemini_search, daemon=True).start()
            return {
                "status": "scoring",
                "total_images": len(files),
                "total_batches": total_batches,
                "_hint": f"Scoring {len(files)} images with Gemini vision in {total_batches} batches. Tell the user it's processing and will be ready soon.",
            }

        # Small folder — do inline
        result = search_images(folder, req.query, limit=req.limit, recursive=recursive)
        if "error" in result and not result.get("results"):
            raise HTTPException(status_code=404, detail=result["error"])
        result["_hint"] = (
            "Visual search complete. Results are AI-analyzed — trust them to answer, categorize, or group."
        )
        return result

    mem = _mem_cache.get(folder)
    if mem and len(mem["paths"]) == len(files):
        # Already cached — search instantly
        result = search_images(folder, req.query, limit=req.limit, recursive=recursive)
        if "error" in result and not result.get("results"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    # Check how many need embedding
    cached = _load_cached_embeddings(files)
    to_embed = len(files) - len(cached)

    if to_embed > 50:
        # Large job — run in background
        if folder in _embedding_jobs:
            job = _embedding_jobs[folder]
            if job["status"] == "running":
                return {
                    "status": "embedding",
                    "total": job["total"],
                    "done": job["done"],
                    "_hint": f"Still embedding ({job['done']}/{job['total']}). Tell the user to wait and try again in a bit.",
                }
            if job["status"] == "done":
                # Background embedding finished — search now
                result = search_images(folder, req.query, limit=req.limit, recursive=recursive)
                if "error" in result and not result.get("results"):
                    raise HTTPException(status_code=404, detail=result["error"])
                result["_hint"] = (
                    "Visual search complete. Results are AI-analyzed — trust them to answer, categorize, or group."
                )
                return result
            if job["status"] == "error":
                del _embedding_jobs[folder]  # Clear failed job, will retry below

        import threading

        _embedding_jobs[folder] = {"status": "running", "total": len(files), "done": len(cached)}

        def _bg_embed() -> None:
            try:
                from image_search import embed_images

                def _progress(done: int, total: int) -> None:
                    _embedding_jobs[folder]["done"] = done + len(cached)

                embed_images(folder, progress_callback=_progress)
                _embedding_jobs[folder]["status"] = "done"
                _embedding_jobs[folder]["done"] = len(files)
            except Exception as e:
                _embedding_jobs[folder]["status"] = "error"
                _embedding_jobs[folder]["error"] = str(e)

        threading.Thread(target=_bg_embed, daemon=True).start()
        return {
            "status": "embedding",
            "total": len(files),
            "to_embed": to_embed,
            "cached": len(cached),
            "_hint": f"Embedding {to_embed} images in background. Tell the user it's processing and will be ready in a few minutes. They can search once done.",
        }

    # Small job — do inline
    result = search_images(folder, req.query, limit=req.limit, recursive=recursive)
    if "error" in result and not result.get("results"):
        raise HTTPException(status_code=404, detail=result["error"])
    result["_hint"] = "Visual search complete. Results are AI-analyzed — trust them to answer, categorize, or group."
    return result


@router.get("/embedding-status")
def embedding_status(folder: str = Query(...)) -> dict:
    """Check background embedding progress."""
    folder = os.path.abspath(folder)
    _check_safe(folder)
    job = _embedding_jobs.get(folder)
    if not job:
        return {"status": "none", "folder": folder}
    return {"folder": folder, **job}


# --- Video search (Segment 18H: SigLIP2 + FFmpeg) ---


@router.post("/search-video")
def search_video_endpoint(req: VideoSearchRequest) -> dict:
    """Search inside a video by text description using SigLIP2 frame embeddings."""
    video_path = os.path.abspath(req.video_path)
    _check_safe(video_path)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    from video_search import search_video

    result = search_video(video_path, req.query, fps=req.fps, limit=req.limit)
    if "error" in result and not result.get("results"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/extract-frame")
def extract_frame_endpoint(req: ExtractFrameRequest) -> dict:
    """Extract a single frame from a video at a given timestamp."""
    video_path = os.path.abspath(req.video_path)
    _check_safe(video_path)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    from video_search import extract_frame_image

    try:
        out = extract_frame_image(video_path, req.seconds, req.output_path)
        return {"path": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")


# --- Audio (Gemini native) ---


@router.post("/transcribe-audio")
def transcribe_audio_endpoint(req: TranscribeAudioRequest) -> dict:
    """Transcribe an audio file to text using Gemini."""
    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    from audio_search import transcribe_audio

    result = transcribe_audio(path)
    if "error" in result and "text" not in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/search-audio")
def search_audio_endpoint(req: SearchAudioRequest) -> dict:
    """Search within an audio file for specific content."""
    path = os.path.abspath(req.audio_path)
    _check_safe(path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")
    from audio_search import search_audio

    result = search_audio(path, req.query, limit=req.limit)
    if "error" in result and not result.get("results"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# --- OCR ---


@router.post("/ocr")
def ocr_endpoint(req: OcrRequest) -> dict:
    """Extract text from an image, PDF, or all files in a folder using OCR."""
    from api.files import _ocr_single

    if req.folder:
        folder = os.path.abspath(req.folder)
        _check_safe(folder)
        if not os.path.isdir(folder):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
        ocr_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".pdf"}
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in ocr_exts]
        )
        if not files:
            raise HTTPException(status_code=404, detail=f"No images or PDFs found in: {folder}")
        _BATCH_CAP = 50  # prevent multi-minute blocking on huge folders
        capped = len(files) > _BATCH_CAP
        files = files[:_BATCH_CAP]
        results = {}
        for fpath in files:
            result = _ocr_single(fpath)
            if "error" not in result:
                results[os.path.basename(fpath)] = result.get("text", "")
        resp = {"folder": folder, "files_processed": len(results), "results": results}
        if capped:
            resp["_hint"] = f"Capped at {_BATCH_CAP} files. Process remaining files in separate calls."
        return resp

    if not req.path:
        raise HTTPException(status_code=400, detail="Provide path or folder")
    result = _ocr_single(req.path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

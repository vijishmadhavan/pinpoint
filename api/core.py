"""Core endpoints: health, status, indexing, and single-file indexing helpers."""

from __future__ import annotations

import logging
import os
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.helpers import _check_safe, _get_conn
from database import DB_PATH, get_stats
from indexer import index_folder
from indexing_service import index_single_file
from job_service import (
    get_or_create_job,
    is_job_cancelling,
    mark_job_cancelled,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    update_job_progress,
)
from pinpoint import __version__

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Health check ---


@router.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


# --- Indexing with progress tracking (Segment 18L: Supermemory pattern) ---

_indexing_progress = {}  # folder -> { total, processed, current_file, status, percent }
_indexing_lock = threading.Lock()
_embedding_jobs = {}  # folder -> {status, total, done, error?}


class _JobCancelled(Exception):
    pass


class IndexRequest(BaseModel):
    folder: str


@router.post("/index")
def index_endpoint(req: IndexRequest) -> dict:
    """Index all supported files in a folder. Large folders run in background automatically."""
    folder = os.path.abspath(req.folder)
    _check_safe(folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    # Already running?
    with _indexing_lock:
        p = _indexing_progress.get(folder)
    if p and p.get("status") == "indexing":
        return {
            "status": "indexing",
            "folder": folder,
            "total": p["total"],
            "processed": p["processed"],
            "percent": p["percent"],
            "_hint": f"Already indexing ({p['processed']}/{p['total']}). Tell the user to wait.",
        }

    # Count files to decide sync vs async
    supported_exts = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".txt",
        ".csv",
        ".epub",
        ".md",
        ".json",
        ".log",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
        ".heif",
    }
    file_count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported_exts:
                file_count += 1
        if file_count > 50:
            break  # Enough to decide

    if file_count > 50:
        # Large folder — background
        conn = _get_conn()
        job_id, created = get_or_create_job(conn, "folder_index", target_path=folder, current_stage="queued")
        if not created:
            return {
                "status": "indexing",
                "folder": folder,
                "job_id": job_id,
                "total": _indexing_progress.get(folder, {}).get("total", file_count),
                "background": True,
                "_hint": "Folder indexing already running. Tell the user they can check progress.",
            }
        _indexing_progress[folder] = {
            "total": file_count,
            "processed": 0,
            "current_file": "",
            "status": "indexing",
            "percent": 0,
            "job_id": job_id,
        }
        update_job_progress(
            conn,
            job_id,
            total_items=file_count,
            completed_items=0,
            details={"folder": folder, "current_file": ""},
            current_stage="queued",
        )

        def _bg_index() -> None:
            job_conn = _get_conn()
            try:
                if is_job_cancelling(job_conn, job_id):
                    _indexing_progress[folder]["status"] = "cancelled"
                    mark_job_cancelled(job_conn, job_id)
                    return
                mark_job_running(job_conn, job_id, current_stage="indexing")
                index_folder(folder, DB_PATH, progress_callback=_update_progress)
                _indexing_progress[folder]["status"] = "done"
                _indexing_progress[folder]["percent"] = 100
                update_job_progress(
                    job_conn,
                    job_id,
                    total_items=file_count,
                    completed_items=file_count,
                    details={"folder": folder, "current_file": ""},
                    current_stage="completed",
                )
                mark_job_completed(job_conn, job_id, current_stage="completed")
            except _JobCancelled:
                _indexing_progress[folder]["status"] = "cancelled"
                mark_job_cancelled(job_conn, job_id)
            except Exception as exc:
                _indexing_progress[folder]["status"] = "error"
                _indexing_progress[folder]["error"] = "Indexing failed"
                mark_job_failed(job_conn, job_id, str(exc), current_stage="failed")
                logger.exception("folder_index_background_failed", extra={"folder": folder})

        threading.Thread(target=_bg_index, daemon=True).start()
        return {
            "status": "indexing",
            "folder": folder,
            "job_id": job_id,
            "total": file_count,
            "background": True,
            "_hint": f"Indexing {file_count}+ files in background. Tell the user it's started and they can check progress or search once done.",
        }

    # Small folder — sync
    try:
        result = index_folder(folder, DB_PATH, progress_callback=_update_progress)
    except Exception:
        logger.exception("folder_index_failed", extra={"folder": folder})
        raise
    _indexing_progress.pop(folder, None)
    return result


def _update_progress(folder: str, total: int, processed: int, current_file: str) -> None:
    """Callback for indexing progress."""
    state = _indexing_progress.get(folder, {})
    current_name = os.path.basename(current_file) if current_file else ""
    _indexing_progress[folder] = {
        "total": total,
        "processed": processed,
        "current_file": current_name,
        "status": "indexing" if processed < total else "done",
        "percent": round(processed / total * 100) if total > 0 else 0,
        "job_id": state.get("job_id"),
    }
    job_id = state.get("job_id")
    if not job_id:
        return
    conn = _get_conn()
    if is_job_cancelling(conn, job_id):
        raise _JobCancelled()
    update_job_progress(
        conn,
        job_id,
        total_items=total,
        completed_items=processed,
        details={"folder": folder, "current_file": current_name},
        current_stage="indexing" if processed < total else "completed",
    )


@router.get("/indexing/status")
def indexing_status() -> dict:
    """Get current indexing progress for all active jobs. Includes both indexing and embedding."""
    jobs = []
    for k, v in _indexing_progress.items():
        jobs.append({"folder": k, "type": "index", **v})
    for k, v in _embedding_jobs.items():
        jobs.append({"folder": k, "type": "embed", **v})
    if not jobs:
        return {"active": False, "jobs": []}
    return {
        "active": any(j.get("status") in ("indexing", "running") for j in jobs),
        "jobs": jobs,
    }


# --- Stats ---


@router.get("/status")
def status_endpoint() -> dict:
    """Return indexing statistics."""
    conn = _get_conn()
    stats = get_stats(conn)
    stats["api_version"] = __version__
    stats["db_path"] = DB_PATH
    stats["db_size_mb"] = round(os.path.getsize(DB_PATH) / 1024 / 1024, 2) if os.path.exists(DB_PATH) else 0
    return stats


# --- Index file on demand (Segment 15) ---


class IndexFileRequest(BaseModel):
    path: str


@router.post("/index-file")
def index_file_endpoint(req: IndexFileRequest) -> dict:
    """Index a single file into the search database on demand. Extracts facts if text is substantial."""
    path = os.path.abspath(req.path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Use POST /index for folders. This endpoint is for single files.")

    conn = _get_conn()
    try:
        outcome = index_single_file(conn, path, skip_unchanged=True, facts_enabled=True)
    except Exception:
        logger.exception("index_file_failed", extra={"path": path})
        raise
    if outcome["status"] == "skipped":
        if outcome["reason"] == "unchanged":
            return {
                "success": True,
                "path": path,
                "already_indexed": True,
                "hash": outcome["hash"],
                "_hint": "File already indexed (unchanged). Use search_documents to search it.",
            }
        if outcome["reason"] == "unextractable":
            raise HTTPException(status_code=400, detail=f"Cannot extract text from {os.path.basename(path)}")
        raise HTTPException(status_code=400, detail=f"File indexing skipped: {outcome['reason']}")

    return {
        "success": True,
        "path": path,
        "file_type": outcome["file_type"],
        "text_length": outcome["text_length"],
        "hash": outcome["hash"],
        "chunks": outcome["chunks"],
        "facts_extracted": outcome["facts_extracted"],
        "_hint": "File is now searchable. Use search_documents to find specific sections within it.",
    }

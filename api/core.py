"""Core endpoints: health, status, indexing, file watcher."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.helpers import _check_safe, _get_conn
from database import DB_PATH, get_stats
from indexer import index_folder

router = APIRouter()

# --- Health check ---


@router.get("/ping")
def ping() -> dict:
    return {"status": "ok"}


# --- Indexing with progress tracking (Segment 18L: Supermemory pattern) ---

_indexing_progress = {}  # folder -> { total, processed, current_file, status, percent }
_embedding_jobs = {}  # folder -> {status, total, done, error?}


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
    if folder in _indexing_progress and _indexing_progress[folder].get("status") == "indexing":
        p = _indexing_progress[folder]
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
        _indexing_progress[folder] = {
            "total": file_count,
            "processed": 0,
            "current_file": "",
            "status": "indexing",
            "percent": 0,
        }

        def _bg_index() -> None:
            try:
                index_folder(folder, DB_PATH, progress_callback=_update_progress)
                _indexing_progress[folder]["status"] = "done"
                _indexing_progress[folder]["percent"] = 100
            except Exception as e:
                _indexing_progress[folder]["status"] = "error"
                _indexing_progress[folder]["error"] = str(e)

        threading.Thread(target=_bg_index, daemon=True).start()
        return {
            "status": "indexing",
            "folder": folder,
            "total": file_count,
            "background": True,
            "_hint": f"Indexing {file_count}+ files in background. Tell the user it's started and they can check progress or search once done.",
        }

    # Small folder — sync
    result = index_folder(folder, DB_PATH, progress_callback=_update_progress)
    _indexing_progress.pop(folder, None)
    return result


def _update_progress(folder: str, total: int, processed: int, current_file: str) -> None:
    """Callback for indexing progress."""
    _indexing_progress[folder] = {
        "total": total,
        "processed": processed,
        "current_file": os.path.basename(current_file) if current_file else "",
        "status": "indexing" if processed < total else "done",
        "percent": round(processed / total * 100) if total > 0 else 0,
    }


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

    from database import chunk_document, upsert_document
    from extractors import extract

    # DB-first: skip extraction if file unchanged (same mtime = same content)
    conn = _get_conn()
    existing = conn.execute("SELECT hash, modified_at FROM documents WHERE path = ? AND active = 1", (path,)).fetchone()
    if existing:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        if existing["modified_at"] and existing["modified_at"] >= file_mtime:
            return {
                "success": True,
                "path": path,
                "already_indexed": True,
                "hash": existing["hash"][:16],
                "_hint": "File already indexed (unchanged). Use search_documents to search it.",
            }

    result = extract(path)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Cannot extract text from {os.path.basename(path)}")

    conn = _get_conn()
    h = upsert_document(conn, path, result["text"], result["file_type"], result.get("page_count", 0))

    # Chunk the document for section-level search (Segment 18P)
    chunks_count = 0
    doc_row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
    if doc_row:
        doc_id = doc_row["id"]
        try:
            chunks_count = chunk_document(conn, doc_id, result["text"])
        except Exception as e:
            print(f"[Chunk] Failed for {path}: {e}")

    # Extract facts from substantial documents (Supermemory pattern)
    facts_count = 0
    text = result["text"]
    if len(text) > 200:  # Only for non-trivial documents
        if doc_row:
            # Check if facts already extracted for this document
            existing = conn.execute("SELECT COUNT(*) as c FROM facts WHERE document_id = ?", (doc_id,)).fetchone()
            if existing["c"] == 0:
                try:
                    facts_count = _extract_facts(conn, doc_id, text[:4000], os.path.basename(path))
                except Exception as e:
                    print(f"[Facts] Extraction failed for {path}: {e}")

    return {
        "success": True,
        "path": path,
        "file_type": result["file_type"],
        "text_length": len(text),
        "hash": h[:16],
        "chunks": chunks_count,
        "facts_extracted": facts_count,
        "_hint": "File is now searchable. Use search_documents to find specific sections within it.",
    }


def _extract_facts(conn: sqlite3.Connection, doc_id: int, text: str, filename: str) -> int:
    """Extract key facts from document text using Gemini."""
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        return 0

    prompt = f"""Extract 3-10 key facts from this document. One fact per line, no numbering.
Include: names, dates, amounts, topics, key details.
Keep each fact short (1 sentence max).
File: {filename}

Text:
{text}

Facts:"""

    try:
        from google.genai import types as genai_types

        from extractors import _get_gemini, gemini_call_with_retry

        _facts_schema = {
            "type": "OBJECT",
            "properties": {"facts": {"type": "ARRAY", "items": {"type": "STRING"}}},
            "required": ["facts"],
        }
        client = _get_gemini()
        if not client:
            return 0
        response = gemini_call_with_retry(
            client,
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_facts_schema,
            ),
        )
        import json as _json_mod

        data = _json_mod.loads(response.text)
        facts_list = data.get("facts", [])
        if not facts_list:
            return 0

        now = datetime.utcnow().isoformat()
        count = 0
        for fact in facts_list:
            fact = str(fact).strip()
            if fact and len(fact) > 10:
                conn.execute(
                    "INSERT INTO facts(document_id, fact_text, created_at) VALUES (?, ?, ?)", (doc_id, fact, now)
                )
                count += 1
        conn.commit()
        return count
    except Exception as e:
        print(f"[Facts] Gemini error: {e}")
        return 0


# --- Folder Watcher (auto-index new/modified files) ---

_watchers = {}  # folder -> Observer
_watch_debounce = {}  # filepath -> Timer

# Supported file extensions for auto-indexing (same as extractors.py)
_WATCH_EXTS = {
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


def _auto_index_file(path: str) -> None:
    """Auto-index a single file (called by watcher after debounce)."""
    _watch_debounce.pop(path, None)
    if not os.path.exists(path):
        return
    ext = os.path.splitext(path)[1].lower()
    if ext not in _WATCH_EXTS:
        return
    try:
        from database import chunk_document, upsert_document
        from extractors import extract

        result = extract(path)
        if result is None:
            return
        conn = _get_conn()
        upsert_document(conn, path, result["text"], result["file_type"], result.get("page_count", 0))
        doc_row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if doc_row:
            try:
                chunk_document(conn, doc_row["id"], result["text"])
            except Exception:
                pass
        print(f"[Watcher] Auto-indexed: {os.path.basename(path)}")
    except Exception as e:
        print(f"[Watcher] Failed to index {path}: {e}")


def _start_watcher(folder: str) -> None:
    """Start watching a folder for file changes."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class IndexHandler(FileSystemEventHandler):
        def _handle(self, event: object) -> None:
            if event.is_directory:
                return
            path = event.src_path
            ext = os.path.splitext(path)[1].lower()
            if ext not in _WATCH_EXTS:
                return
            # Debounce: wait 5s after last change before indexing
            old = _watch_debounce.pop(path, None)
            if old:
                old.cancel()
            t = threading.Timer(5.0, _auto_index_file, args=[path])
            _watch_debounce[path] = t
            t.start()

        def on_created(self, event: object) -> None:
            self._handle(event)

        def on_modified(self, event: object) -> None:
            self._handle(event)

    observer = Observer()
    observer.schedule(IndexHandler(), folder, recursive=True)
    observer.daemon = True
    observer.start()
    _watchers[folder] = observer
    print(f"[Watcher] Watching: {folder}")


def _stop_watcher(folder: str) -> None:
    """Stop watching a folder."""
    obs = _watchers.pop(folder, None)
    if obs:
        obs.stop()
        print(f"[Watcher] Stopped: {folder}")


@router.post("/watch")
def watch_folder_endpoint(folder: str = Query(..., description="Folder to watch")) -> dict:
    """Start auto-indexing new/modified files in a folder."""
    folder = os.path.abspath(folder)
    _check_safe(folder)
    if not os.path.isdir(folder):
        raise HTTPException(400, f"Not a directory: {folder}")
    if folder in _watchers:
        return {"success": True, "folder": folder, "note": "Already watching"}
    _start_watcher(folder)
    # Persist to settings
    conn = _get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = 'watched_folders'").fetchone()
    watched = set(row["value"].split("|")) if row and row["value"] else set()
    watched.add(folder)
    conn.execute(
        "INSERT INTO settings(key, value) VALUES ('watched_folders', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("|".join(watched),),
    )
    conn.commit()
    return {
        "success": True,
        "folder": folder,
        "_hint": f"Now watching {folder}. New/modified files will be auto-indexed.",
    }


@router.post("/unwatch")
def unwatch_folder_endpoint(folder: str = Query(..., description="Folder to stop watching")) -> dict:
    """Stop auto-indexing a folder."""
    folder = os.path.abspath(folder)
    _check_safe(folder)
    if folder not in _watchers:
        return {"success": False, "error": f"Not watching: {folder}"}
    _stop_watcher(folder)
    # Remove from settings
    conn = _get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = 'watched_folders'").fetchone()
    if row and row["value"]:
        watched = set(row["value"].split("|"))
        watched.discard(folder)
        conn.execute(
            "INSERT INTO settings(key, value) VALUES ('watched_folders', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("|".join(watched) if watched else "",),
        )
        conn.commit()
    return {"success": True, "folder": folder, "_hint": f"Stopped watching {folder}."}


@router.get("/watched")
def list_watched() -> dict:
    """List all currently watched folders."""
    return {"folders": list(_watchers.keys()), "count": len(_watchers)}


def restore_watchers_on_startup() -> None:
    """Restore watchers from settings DB. Call from app startup."""
    try:
        conn = _get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = 'watched_folders'").fetchone()
        if row and row["value"]:
            for folder in row["value"].split("|"):
                if folder and os.path.isdir(folder):
                    _start_watcher(folder)
    except Exception as e:
        print(f"[Watcher] Failed to restore watchers: {e}")

"""Shared helpers, constants, and utilities for all API router modules."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import HTTPException

from database import DB_PATH, init_db
from indexing_service import index_single_file
from job_service import (
    get_or_create_job,
    is_job_cancelling,
    mark_job_cancelled,
    mark_job_completed,
    mark_job_failed,
    mark_job_running,
    update_job_stage,
)

logger = logging.getLogger(__name__)

# --- Thread-local DB connections ---
_local = threading.local()
_migrations_done = False
_migrations_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _migrations_done
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = init_db(DB_PATH)
        _local.conn = conn
    # Run migrations once across all threads
    if not _migrations_done:
        with _migrations_lock:
            if not _migrations_done:
                try:
                    try:
                        conn.execute("SELECT superseded_by FROM memories LIMIT 1")
                    except Exception:
                        try:
                            conn.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER DEFAULT NULL")
                        except Exception:
                            pass
                    try:
                        conn.execute("SELECT 1 FROM facts LIMIT 1")
                    except Exception:
                        conn.execute("""CREATE TABLE IF NOT EXISTS facts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            document_id INTEGER NOT NULL,
                            fact_text TEXT NOT NULL,
                            category TEXT DEFAULT 'general',
                            created_at TEXT NOT NULL
                        )""")
                        conn.commit()
                    _migrations_done = True
                except Exception:
                    logger.exception("db_helper_migration_failed")
    return conn


# --- Generated file tracking ---


def record_generated_file(path: str, tool_name: str, description: str = "") -> None:
    """Record a file created by a tool for future retrieval."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO generated_files (path, tool_name, description, created_at) VALUES (?, ?, ?, ?)",
            (path, tool_name, description, datetime.now().isoformat()),
        )
        conn.commit()
    except Exception:
        logger.exception("record_generated_file_failed", extra={"path": path, "tool_name": tool_name})


# --- Formatting helpers ---


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _human_date(timestamp: float) -> str:
    """Convert Unix timestamp to human-readable date."""
    return datetime.fromtimestamp(timestamp).strftime("%d %b %Y, %I:%M %p")


# --- Path safety ---

BLOCKED_PREFIXES = (
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/proc",
    "/sys",
    "/dev",
    "/var/run",
    "/var/lock",
    "/var/lib",
    "/root",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
)


_BLOCKED_BASENAMES = {".ssh", ".gnupg", ".env", ".git", "auth"}


def _is_safe_path(path: str) -> bool:
    """Check if a path is safe to operate on (not a system directory or sensitive dotfile)."""
    abs_path = os.path.realpath(os.path.abspath(path))
    for prefix in BLOCKED_PREFIXES:
        if abs_path.startswith(prefix):
            return False
    # Block sensitive dotfiles/directories anywhere in the path
    parts = abs_path.replace("\\", "/").split("/")
    for part in parts:
        if part in _BLOCKED_BASENAMES:
            return False
    return True


def _check_safe(path: str) -> None:
    """Raise 403 if path is in a blocked system directory."""
    if not _is_safe_path(path):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is in a protected system directory",
        )


# --- URL safety (SSRF prevention) ---

_PRIVATE_RANGES = None


def _check_url_safe(url: str) -> None:
    """Raise 403 if URL resolves to a private/loopback IP (SSRF prevention)."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    global _PRIVATE_RANGES
    if _PRIVATE_RANGES is None:
        _PRIVATE_RANGES = [
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("fc00::/7"),
            ipaddress.ip_network("fe80::/10"),
        ]

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname")
    try:
        addr = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)[0][4][0]
        ip = ipaddress.ip_address(addr)
        for net in _PRIVATE_RANGES:
            if ip in net:
                raise HTTPException(status_code=403, detail="Access denied: URL resolves to private/loopback address")
    except socket.gaierror:
        raise HTTPException(status_code=403, detail="Access denied: cannot resolve hostname")


# --- File type constants ---

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif", ".heic"}

TEXT_EXTS = {
    ".txt",
    ".csv",
    ".md",
    ".log",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".js",
    ".py",
    ".sh",
}

PDF_EXTS = {".pdf"}
OFFICE_EXTS = {".docx", ".pptx", ".epub"}
EXCEL_EXTS = {".xlsx", ".xlsm"}

MAX_READ_SIZE = 10 * 1024 * 1024  # 10 MB max
MAX_TEXT_CHARS = 8000  # truncate text for Gemini context
_AUTO_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="auto-index")
_AUTO_INDEX_IN_FLIGHT: set[str] = set()
_AUTO_INDEX_LOCK = threading.Lock()


def _background_index(path: str) -> None:
    """Fire-and-forget: index + chunk + embed a file in a background thread.
    Skips if already indexed and unchanged (mtime check). Best-effort."""
    abs_path = os.path.abspath(path)
    conn = init_db(DB_PATH)
    try:
        job_id, created = get_or_create_job(conn, "auto_index_file", target_path=abs_path, current_stage="queued")
    finally:
        conn.close()
    if not created:
        return

    with _AUTO_INDEX_LOCK:
        if abs_path in _AUTO_INDEX_IN_FLIGHT:
            return
        _AUTO_INDEX_IN_FLIGHT.add(abs_path)

    def _do_index():
        job_conn = init_db(DB_PATH)
        try:
            if is_job_cancelling(job_conn, job_id):
                mark_job_cancelled(job_conn, job_id)
                return
            mark_job_running(job_conn, job_id, current_stage="indexing")
            outcome = index_single_file(job_conn, abs_path, skip_unchanged=True, facts_enabled=False)
            if is_job_cancelling(job_conn, job_id):
                mark_job_cancelled(job_conn, job_id)
                logger.info("background_index_cancelled", extra={"path": abs_path})
                return
            if outcome["status"] == "skipped":
                mark_job_completed(job_conn, job_id, current_stage=f"skipped:{outcome.get('reason', 'unknown')}")
                logger.info(
                    "background_index_skipped",
                    extra={"path": abs_path, "reason": outcome.get("reason", "unknown")},
                )
                return
            if outcome["status"] != "indexed":
                mark_job_failed(job_conn, job_id, f"unexpected status: {outcome['status']}", current_stage="failed")
                logger.warning("background_index_unknown_status", extra={"path": abs_path, "status": outcome["status"]})
                return
            update_job_stage(job_conn, job_id, "embedding")
            mark_job_completed(job_conn, job_id, current_stage="completed")
            logger.info(
                "background_index_completed",
                extra={"path": abs_path, "chunks": outcome["chunks"], "embedded_chunks": outcome["embedded_chunks"]},
            )
        except Exception as exc:
            try:
                mark_job_failed(job_conn, job_id, str(exc), current_stage="failed")
            except Exception:
                logger.info(
                    "background_index_job_update_failed",
                    extra={"path": abs_path, "job_id": job_id},
                )
            logger.exception("background_index_failed", extra={"path": abs_path})
        finally:
            job_conn.close()
            with _AUTO_INDEX_LOCK:
                _AUTO_INDEX_IN_FLIGHT.discard(abs_path)

    _AUTO_INDEX_EXECUTOR.submit(_do_index)


def _get_images_in_folder(folder: str) -> list[str]:
    """List all image files in a folder."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

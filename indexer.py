"""
Pinpoint — File indexer

Scans folders → extracts text (via extractors.py) → stores in DB (via database.py).
Hash-based change detection: skip unchanged files (instant re-scan).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import UTC
from typing import Any

from database import chunk_document, get_stats, init_db, soft_delete_missing, upsert_document
from extractors import IMAGE_EXTENSIONS, OFFICE_EXTENSIONS, TEXT_EXTENSIONS, extract

SUPPORTED_EXTENSIONS = {".pdf"} | OFFICE_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def scan_folder(folder: str) -> list[str]:
    """Recursively find all supported files in a folder."""
    files = []
    for root, dirs, filenames in os.walk(folder):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


def index_folder(
    folder: str, db_path: str | None = None, progress_callback: Callable[[str, int, int, str], None] | None = None
) -> dict[str, Any]:
    """
    Index all supported files in a folder.

    Args:
        progress_callback: Optional function(folder, total, processed, current_file)
                          called after each file for progress tracking.

    Returns stats dict with counts and timing.
    """
    from database import DB_PATH

    db_path = db_path or DB_PATH

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"[ERROR] Not a directory: {folder}")
        return {}

    conn = init_db(db_path)
    files = scan_folder(folder)
    print(f"[Indexer] Found {len(files)} supported files in {folder}")

    if progress_callback:
        progress_callback(folder, len(files), 0, "")

    indexed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, path in enumerate(files, 1):
        abs_path = os.path.abspath(path)

        # Check if file is already indexed with same content
        # Quick check: read file, hash it, compare with stored hash
        # But we can't hash the *extracted* text without extracting first.
        # Instead, check if path exists in DB and file hasn't been modified.
        row = conn.execute(
            "SELECT hash, modified_at FROM documents WHERE path = ? AND active = 1", (abs_path,)
        ).fetchone()

        if row is not None:
            # File already indexed — check if file modification time is newer
            file_mtime = os.path.getmtime(abs_path)
            db_mtime = row["modified_at"]
            # If file hasn't been modified since indexing, skip
            try:
                from datetime import datetime

                db_dt = datetime.fromisoformat(db_mtime)
                file_dt = datetime.fromtimestamp(file_mtime, tz=UTC)
                if file_dt <= db_dt:
                    skipped += 1
                    if progress_callback:
                        progress_callback(folder, len(files), indexed + skipped + failed, abs_path)
                    continue
            except (ValueError, OSError):
                pass  # Can't compare → re-extract to be safe

        # Extract text
        ext = os.path.splitext(path)[1].lower()
        print(f"  [{i}/{len(files)}] Extracting: {os.path.basename(path)}", end=" … ")
        t0 = time.time()

        result = extract(path)
        if result is None:
            print("FAILED")
            failed += 1
            if progress_callback:
                progress_callback(folder, len(files), indexed + skipped + failed, abs_path)
            continue

        elapsed = time.time() - t0

        # Store in DB
        upsert_document(conn, path, result["text"], result["file_type"], result.get("page_count", 0))

        # Chunk for section-level search (Segment 18P)
        doc_row = conn.execute("SELECT id FROM documents WHERE path = ?", (os.path.abspath(path),)).fetchone()
        n_chunks = 0
        if doc_row:
            try:
                n_chunks = chunk_document(conn, doc_row["id"], result["text"])
            except Exception as e:
                print(f"[Chunk] {e}")

        indexed += 1
        print(f"OK ({elapsed:.2f}s, {len(result['text'])} chars, {n_chunks} chunks)")

        if progress_callback:
            progress_callback(folder, len(files), indexed + skipped + failed, abs_path)

    # Soft-delete files that no longer exist
    all_paths = {os.path.abspath(f) for f in files}
    deleted = soft_delete_missing(conn, all_paths)
    if deleted > 0:
        print(f"[Indexer] Soft-deleted {deleted} removed files")

    elapsed_total = time.time() - t_start
    stats = get_stats(conn)
    conn.close()

    print(f"\n[Indexer] Done in {elapsed_total:.1f}s")
    print(f"  Indexed: {indexed}, Skipped (unchanged): {skipped}, Failed: {failed}")
    print(f"  Total in DB: {stats['total_documents']} documents, {stats['unique_content']} unique content")
    print(f"  By type: {stats['by_type']}")

    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "deleted": deleted,
        "elapsed": elapsed_total,
        **stats,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        db_arg = sys.argv[2] if len(sys.argv) > 2 else None
        index_folder(folder, db_arg)
    else:
        print("Usage: python indexer.py <folder> [db_path]")
        print("Example: python indexer.py ~/Documents")

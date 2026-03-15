"""Shared document indexing helpers used by API endpoints and folder indexing."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import time
from datetime import UTC, datetime
from typing import Any

from database import chunk_document, upsert_document
from extractors import extract

logger = logging.getLogger(__name__)


def file_is_unchanged(conn: sqlite3.Connection, path: str) -> bool:
    """Return True when the indexed file is present and not newer on disk."""
    abs_path = os.path.abspath(path)
    row = conn.execute(
        "SELECT modified_at FROM documents WHERE path = ? AND active = 1",
        (abs_path,),
    ).fetchone()
    if not row:
        return False

    try:
        file_mtime = os.path.getmtime(abs_path)
        db_dt = datetime.fromisoformat(row["modified_at"])
        file_dt = datetime.fromtimestamp(file_mtime, tz=UTC)
        return file_dt <= db_dt
    except (ValueError, OSError, TypeError):
        return False


def embed_chunks(conn: sqlite3.Connection, doc_id: int) -> int:
    """Embed document chunks for semantic search. Returns count embedded."""
    from google.genai import types

    from image_search import EMBED_DIM, _get_client

    rows = conn.execute(
        """SELECT ch.id, ch.text FROM chunks ch
           LEFT JOIN chunk_embeddings ce ON ce.chunk_id = ch.id
           WHERE ch.document_id = ? AND ce.chunk_id IS NULL""",
        (doc_id,),
    ).fetchall()
    if not rows:
        return 0

    try:
        client = _get_client()
    except RuntimeError as exc:
        logger.warning("embed_client_unavailable", extra={"doc_id": doc_id, "reason": str(exc)})
        return 0
    except Exception:
        logger.exception("embed_client_init_failed", extra={"doc_id": doc_id})
        return 0
    now = datetime.now(UTC).isoformat()
    embedded = 0
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r["text"][:2000] for r in batch]
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIM,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            for j, emb_result in enumerate(result.embeddings):
                chunk_id = batch[j]["id"]
                emb_bytes = struct.pack(f"{EMBED_DIM}f", *emb_result.values)
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding, embedded_at) VALUES (?, ?, ?)",
                    (chunk_id, emb_bytes, now),
                )
                embedded += 1
        except Exception:
            logger.exception("embed_batch_failed", extra={"doc_id": doc_id})
            continue

    if embedded > 0:
        conn.commit()
        logger.info("embed_completed", extra={"doc_id": doc_id, "embedded": embedded})
    return embedded


def extract_facts(conn: sqlite3.Connection, doc_id: int, text: str, filename: str) -> int:
    """Extract key facts from document text using Gemini."""
    if not os.getenv("GEMINI_API_KEY"):
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

        schema = {
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
                response_json_schema=schema,
            ),
        )
        data = json.loads(response.text)
        facts_list = data.get("facts", [])
        if not facts_list:
            return 0

        now = datetime.now(UTC).isoformat()
        count = 0
        for fact in facts_list:
            fact = str(fact).strip()
            if fact and len(fact) > 10:
                conn.execute(
                    "INSERT INTO facts(document_id, fact_text, created_at) VALUES (?, ?, ?)",
                    (doc_id, fact, now),
                )
                count += 1
        conn.commit()
        return count
    except Exception:
        logger.exception("fact_extraction_failed", extra={"doc_id": doc_id, "filename": filename})
        return 0


def index_single_file(
    conn: sqlite3.Connection,
    path: str,
    *,
    skip_unchanged: bool = True,
    skip_scanned_pdf: bool = False,
    facts_enabled: bool = False,
    embeddings_enabled: bool = True,
) -> dict[str, Any]:
    """Index one file and return a structured status dict."""
    abs_path = os.path.abspath(path)
    t0 = time.time()

    if skip_unchanged and file_is_unchanged(conn, abs_path):
        row = conn.execute("SELECT hash FROM documents WHERE path = ? AND active = 1", (abs_path,)).fetchone()
        return {
            "status": "skipped",
            "reason": "unchanged",
            "path": abs_path,
            "hash": row["hash"][:16] if row and row["hash"] else "",
            "elapsed": time.time() - t0,
        }

    result = extract(abs_path)
    if result is None:
        return {"status": "skipped", "reason": "unextractable", "path": abs_path, "elapsed": time.time() - t0}

    text = result.get("text", "")
    if skip_scanned_pdf and abs_path.lower().endswith(".pdf") and len(text.strip()) < 50:
        return {"status": "skipped", "reason": "scanned_pdf", "path": abs_path, "elapsed": time.time() - t0}

    content_hash = upsert_document(conn, abs_path, text, result["file_type"], result.get("page_count", 0))
    doc_row = conn.execute("SELECT id FROM documents WHERE path = ?", (abs_path,)).fetchone()
    if not doc_row:
        raise RuntimeError(f"Document row missing after upsert for {abs_path}")

    doc_id = doc_row["id"]
    chunks_count = chunk_document(conn, doc_id, text)
    embedded_count = embed_chunks(conn, doc_id) if embeddings_enabled and chunks_count > 0 else 0

    facts_count = 0
    if facts_enabled and len(text) > 200:
        existing = conn.execute("SELECT COUNT(*) as c FROM facts WHERE document_id = ?", (doc_id,)).fetchone()
        if existing and existing["c"] == 0:
            facts_count = extract_facts(conn, doc_id, text[:4000], os.path.basename(abs_path))

    return {
        "status": "indexed",
        "path": abs_path,
        "doc_id": doc_id,
        "hash": content_hash[:16],
        "file_type": result["file_type"],
        "text_length": len(text),
        "page_count": result.get("page_count", 0),
        "chunks": chunks_count,
        "embedded_chunks": embedded_count,
        "facts_extracted": facts_count,
        "elapsed": time.time() - t0,
    }

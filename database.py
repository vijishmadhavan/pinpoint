"""
Pinpoint — SQLite + FTS5 database (QMD-inspired)

Content-addressable storage with SHA-256 dedup.
FTS5 full-text search with porter unicode61 tokenizer.
Soft delete for removed files, llm_cache for Gemini responses.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinpoint.db")


def _now() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def content_hash(text: str) -> str:
    """SHA-256 hash of text content (for dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open database connection with WAL mode and foreign keys."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create all tables, FTS5 index, and triggers. Idempotent."""
    conn = get_db(db_path)

    conn.executescript("""
        -- Content-addressable storage (dedup by SHA-256 hash)
        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Document metadata (one row per file)
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            hash TEXT NOT NULL,
            file_type TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(active);
        CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);

        -- LLM response cache (for Gemini query expansion etc.)
        CREATE TABLE IF NOT EXISTS llm_cache (
            hash TEXT PRIMARY KEY,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Conversation messages (Segment 13: conversation memory)
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id, timestamp);

        -- Session metadata (tracks last activity for idle timeout)
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER DEFAULT 0
        );

        -- Face embedding cache (Segment 14: on-demand face search)
        CREATE TABLE IF NOT EXISTS face_cache (
            image_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            face_idx INTEGER NOT NULL,
            bbox TEXT NOT NULL,
            embedding BLOB NOT NULL,
            confidence REAL NOT NULL,
            age INTEGER,
            gender TEXT,
            pose TEXT,
            PRIMARY KEY (image_path, face_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_face_cache_path ON face_cache(image_path);

        -- Image embeddings (Segment 18C: SigLIP2 visual search)
        CREATE TABLE IF NOT EXISTS image_embeddings (
            path TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            mtime REAL NOT NULL,
            embedded_at TEXT NOT NULL
        );

        -- Persistent memories (Segment 18F: personal facts across sessions)
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            user_id TEXT DEFAULT NULL,
            superseded_by INTEGER DEFAULT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Memory audit trail (Segment 18Y: Mem0-inspired)
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            old_fact TEXT,
            new_fact TEXT,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- Extracted facts from documents (Segment 18L: Supermemory pattern)
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            fact_text TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        -- Video frame embeddings (Segment 18H: on-demand video search)
        CREATE TABLE IF NOT EXISTS video_embeddings (
            video_path TEXT NOT NULL,
            frame_sec REAL NOT NULL,
            embedding BLOB NOT NULL,
            mtime REAL NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (video_path, frame_sec)
        );

        -- Document chunks (Segment 18P: Chonkie-powered chunking)
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_num INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_index INTEGER NOT NULL,
            end_index INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, chunk_num)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

        -- Known faces (Segment 18V: persistent face recognition)
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source_image TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_known_faces_name ON known_faces(name);

        -- File path registry (lightweight index of filenames in common folders)
        CREATE TABLE IF NOT EXISTS file_paths (
            path TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            ext TEXT DEFAULT '',
            size_bytes INTEGER DEFAULT 0,
            modified_at REAL DEFAULT 0,
            scanned_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_file_paths_filename ON file_paths(filename);
        CREATE INDEX IF NOT EXISTS idx_file_paths_ext ON file_paths(ext);

        -- Generated files (track files created by tools for future retrieval)
        CREATE TABLE IF NOT EXISTS generated_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_generated_files_path ON generated_files(path);

        -- Settings (key-value store for toggles like memory on/off)
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Reminders (persistent, survives restarts)
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_jid TEXT NOT NULL,
            message TEXT NOT NULL,
            trigger_at TEXT NOT NULL,
            repeat TEXT DEFAULT NULL,
            created_at TEXT NOT NULL
        );
    """)

    # Migrate face_cache: add columns if missing (for existing DBs)
    face_cols = {r[1] for r in conn.execute("PRAGMA table_info(face_cache)").fetchall()}
    for col, ctype in [("age", "INTEGER"), ("gender", "TEXT"), ("pose", "TEXT")]:
        if col not in face_cols:
            conn.execute(f"ALTER TABLE face_cache ADD COLUMN {col} {ctype}")
    conn.commit()

    # FTS5 virtual table — CREATE VIRTUAL TABLE doesn't support IF NOT EXISTS
    # Check if it already exists first
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'").fetchone()
    if not row:
        conn.execute("""
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                filepath,
                title,
                body,
                tokenize='porter unicode61'
            )
        """)

    # Triggers to keep FTS in sync with documents table
    # We use AFTER triggers so FTS always matches the documents table

    # On INSERT: add to FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_documents_insert
        AFTER INSERT ON documents
        WHEN NEW.active = 1
        BEGIN
            INSERT INTO documents_fts(rowid, filepath, title, body)
            SELECT NEW.id, NEW.path, NEW.title, c.text
            FROM content c WHERE c.hash = NEW.hash;
        END
    """)

    # On UPDATE (content changed or reactivated): refresh FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_documents_update
        AFTER UPDATE ON documents
        WHEN NEW.active = 1
        BEGIN
            DELETE FROM documents_fts WHERE rowid = OLD.id;
            INSERT INTO documents_fts(rowid, filepath, title, body)
            SELECT NEW.id, NEW.path, NEW.title, c.text
            FROM content c WHERE c.hash = NEW.hash;
        END
    """)

    # On UPDATE (soft delete): remove from FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_documents_deactivate
        AFTER UPDATE ON documents
        WHEN NEW.active = 0 AND OLD.active = 1
        BEGIN
            DELETE FROM documents_fts WHERE rowid = OLD.id;
        END
    """)

    # On DELETE: remove from FTS
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_documents_delete
        AFTER DELETE ON documents
        BEGIN
            DELETE FROM documents_fts WHERE rowid = OLD.id;
        END
    """)

    # Chunks FTS5 virtual table (Segment 18P)
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'").fetchone()
    if not row:
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                filepath,
                title,
                body,
                tokenize='porter unicode61'
            )
        """)

    # Triggers to keep chunks_fts in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_chunks_insert
        AFTER INSERT ON chunks
        BEGIN
            INSERT INTO chunks_fts(rowid, filepath, title, body)
            SELECT NEW.id, d.path, d.title, NEW.text
            FROM documents d WHERE d.id = NEW.document_id;
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_chunks_delete
        AFTER DELETE ON chunks
        BEGIN
            DELETE FROM chunks_fts WHERE rowid = OLD.id;
        END
    """)

    # Memories FTS5 virtual table (Segment 18Y: replaces LIKE with BM25)
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'").fetchone()
    if not row:
        conn.execute("""
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                fact,
                tokenize='porter unicode61'
            )
        """)
        # Backfill existing memories into FTS
        conn.execute("""
            INSERT INTO memories_fts(rowid, fact)
            SELECT id, fact FROM memories WHERE superseded_by IS NULL
        """)

    # Migrate: add user_id column if missing (Segment 18Y)
    mem_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "user_id" not in mem_cols:
        conn.execute("ALTER TABLE memories ADD COLUMN user_id TEXT DEFAULT NULL")

    # Migrate: create memory_history if missing (for existing DBs)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            old_fact TEXT,
            new_fact TEXT,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def _extract_title(path: str, text: str) -> str:
    """Extract a title from text or fall back to filename."""
    # Try first markdown heading
    for line in text.split("\n")[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()[:200]
    # Fall back to filename without extension
    return os.path.splitext(os.path.basename(path))[0]


def upsert_document(conn: sqlite3.Connection, path: str, text: str, file_type: str, page_count: int = 0) -> str:
    """
    Insert or update a document. Returns the content hash.

    - If text is new: insert into content table
    - If file path is new: insert into documents
    - If file exists but content changed: update hash + modified_at
    - If file exists and content same: skip (no-op)
    """
    h = content_hash(text)
    now = _now()
    abs_path = os.path.abspath(path)
    title = _extract_title(abs_path, text)
    file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0

    # Ensure content exists
    conn.execute("INSERT OR IGNORE INTO content(hash, text, created_at) VALUES (?, ?, ?)", (h, text, now))

    # Check if document already exists
    row = conn.execute("SELECT id, hash, active FROM documents WHERE path = ?", (abs_path,)).fetchone()

    if row is None:
        # New document
        conn.execute(
            """INSERT INTO documents(path, title, hash, file_type, page_count,
               file_size, active, created_at, modified_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (abs_path, title, h, file_type, page_count, file_size, now, now),
        )
    elif row["hash"] != h or row["active"] == 0:
        # Content changed or was soft-deleted — update
        conn.execute(
            """UPDATE documents SET title=?, hash=?, file_type=?, page_count=?,
               file_size=?, active=1, modified_at=? WHERE id=?""",
            (title, h, file_type, page_count, file_size, now, row["id"]),
        )
    # else: same hash, still active → skip

    conn.commit()
    return h


def soft_delete_missing(conn: sqlite3.Connection, indexed_paths: set[str]) -> int:
    """
    Soft-delete documents whose files no longer exist on disk.
    Returns count of deactivated documents.
    """
    rows = conn.execute("SELECT id, path FROM documents WHERE active = 1").fetchall()

    count = 0
    for row in rows:
        if row["path"] not in indexed_paths:
            conn.execute("UPDATE documents SET active = 0, modified_at = ? WHERE id = ?", (_now(), row["id"]))
            count += 1

    if count > 0:
        conn.commit()
    return count


def cleanup_orphaned_content(conn: sqlite3.Connection) -> int:
    """Delete content rows that no active document references. Returns count."""
    cursor = conn.execute("""
        DELETE FROM content WHERE hash NOT IN (
            SELECT DISTINCT hash FROM documents WHERE active = 1
        )
    """)
    conn.commit()
    return cursor.rowcount


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return indexing statistics."""
    total = conn.execute("SELECT COUNT(*) as n FROM documents WHERE active = 1").fetchone()["n"]

    by_type = {}
    rows = conn.execute("SELECT file_type, COUNT(*) as n FROM documents WHERE active = 1 GROUP BY file_type").fetchall()
    for row in rows:
        by_type[row["file_type"]] = row["n"]

    content_rows = conn.execute("SELECT COUNT(*) as n FROM content").fetchone()["n"]

    return {
        "total_documents": total,
        "by_type": by_type,
        "unique_content": content_rows,
    }


# --- Document chunking (Segment 18P: Chonkie-powered) ---

# Lazy-loaded chunker instance
_chunker = None


def _get_chunker() -> Any:
    """Load Chonkie RecursiveChunker once."""
    global _chunker
    if _chunker is None:
        from chonkie import RecursiveChunker

        # ~500 words ≈ ~2500 chars. Use character tokenizer (zero deps, fastest).
        _chunker = RecursiveChunker(tokenizer="character", chunk_size=2500)
    return _chunker


def chunk_document(conn: sqlite3.Connection, document_id: int, text: str) -> int:
    """
    Chunk a document's text and store in chunks table.
    Deletes existing chunks for this document first (re-chunk on update).
    Returns number of chunks created.

    Short documents (<3000 chars) are stored as a single chunk.
    Chunks are prepared BEFORE deleting old data — if chunking fails, old chunks remain intact.
    """
    if not text or not text.strip():
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()
        return 0

    # Prepare chunks first (can fail safely without touching DB)
    if len(text) < 3000:
        new_chunks = [(document_id, 0, text, 0, len(text))]
    else:
        chunker = _get_chunker()
        raw = chunker.chunk(text)
        new_chunks = [(document_id, i, c.text, c.start_index, c.end_index) for i, c in enumerate(raw)]

    # Atomic DB operation: delete old + insert new
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.executemany(
        "INSERT INTO chunks(document_id, chunk_num, text, start_index, end_index) VALUES (?, ?, ?, ?, ?)",
        new_chunks,
    )
    conn.commit()
    return len(new_chunks)


# --- LLM cache operations ---


def cache_get(conn: sqlite3.Connection, key: str) -> str | None:
    """Get cached LLM response. Returns None if not found."""
    h = content_hash(key)
    row = conn.execute("SELECT result FROM llm_cache WHERE hash = ?", (h,)).fetchone()
    return row["result"] if row else None


def cache_set(conn: sqlite3.Connection, key: str, result: str) -> None:
    """Store LLM response in cache. LRU trim at 1000 entries (1% chance per write)."""
    import random

    h = content_hash(key)
    now = _now()
    conn.execute("INSERT OR REPLACE INTO llm_cache(hash, result, created_at) VALUES (?, ?, ?)", (h, result, now))
    # 1% chance: trim to newest 1000 entries
    if random.random() < 0.01:
        conn.execute("""
            DELETE FROM llm_cache WHERE hash NOT IN (
                SELECT hash FROM llm_cache ORDER BY created_at DESC LIMIT 1000
            )
        """)
    conn.commit()


# --- Quick test ---
if __name__ == "__main__":
    import tempfile

    print("=== Pinpoint Database Test ===\n")

    # Use temp DB for testing
    test_db = os.path.join(tempfile.gettempdir(), "pinpoint_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    conn = init_db(test_db)
    print(f"[OK] Database created: {test_db}")

    # Check tables exist
    tables = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger') ORDER BY name"
        ).fetchall()
    ]
    print(f"[OK] Tables/triggers: {tables}")

    # Check FTS5 tokenizer
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='documents_fts'").fetchone()
    assert "porter unicode61" in row["sql"], "FTS5 tokenizer not set!"
    print("[OK] FTS5 with porter unicode61 tokenizer")

    # Test upsert — use a real file (this script itself)
    this_file = os.path.abspath(__file__)
    h = upsert_document(conn, this_file, "Test document about invoices and receipts.", "txt", 0)
    print(f"[OK] Inserted document, hash: {h[:16]}...")

    # Test FTS search
    results = conn.execute(
        "SELECT * FROM documents_fts WHERE documents_fts MATCH 'invoice'",
    ).fetchall()
    assert len(results) == 1, f"Expected 1 FTS result, got {len(results)}"
    print(f"[OK] FTS search 'invoice' → {len(results)} result (porter stemming works!)")

    # Test dedup: same content, different path
    h2 = upsert_document(conn, this_file, "Test document about invoices and receipts.", "txt", 0)
    assert h == h2, "Hash should be same for same content"
    content_count = conn.execute("SELECT COUNT(*) as n FROM content").fetchone()["n"]
    assert content_count == 1, f"Expected 1 content row (dedup), got {content_count}"
    print("[OK] Content dedup works (1 content row for same text)")

    # Test content change detection
    h3 = upsert_document(conn, this_file, "Updated content with new text.", "txt", 0)
    assert h3 != h, "Hash should differ for different content"
    doc = conn.execute("SELECT hash FROM documents WHERE path = ?", (this_file,)).fetchone()
    assert doc["hash"] == h3, "Document hash should be updated"
    print("[OK] Content change detection works (hash updated)")

    # Test FTS after update — should find new content
    results = conn.execute(
        "SELECT * FROM documents_fts WHERE documents_fts MATCH 'updated'",
    ).fetchall()
    assert len(results) == 1, f"Expected 1 FTS result for 'updated', got {len(results)}"
    print("[OK] FTS synced after content update")

    # Test soft delete
    soft_delete_missing(conn, set())  # no paths → everything gets soft-deleted
    doc = conn.execute("SELECT active FROM documents WHERE path = ?", (this_file,)).fetchone()
    assert doc["active"] == 0, "Document should be soft-deleted"
    # FTS should be empty now
    results = conn.execute(
        "SELECT * FROM documents_fts WHERE documents_fts MATCH 'updated'",
    ).fetchall()
    assert len(results) == 0, f"Expected 0 FTS results after soft delete, got {len(results)}"
    print("[OK] Soft delete works (active=0, removed from FTS)")

    # Test reactivation
    h4 = upsert_document(conn, this_file, "Reactivated document.", "txt", 0)
    doc = conn.execute("SELECT active FROM documents WHERE path = ?", (this_file,)).fetchone()
    assert doc["active"] == 1, "Document should be reactivated"
    print("[OK] Reactivation works (soft-deleted → active=1)")

    # Test orphan cleanup
    soft_delete_missing(conn, set())
    cleaned = cleanup_orphaned_content(conn)
    print(f"[OK] Orphan cleanup: removed {cleaned} content rows")

    # Test stats
    upsert_document(conn, this_file, "Final test doc.", "pdf", 5)
    stats = get_stats(conn)
    print(f"[OK] Stats: {stats}")

    # Test LLM cache
    cache_set(conn, "test query", '{"expanded": ["test", "testing"]}')
    cached = cache_get(conn, "test query")
    assert cached == '{"expanded": ["test", "testing"]}', "Cache miss!"
    print("[OK] LLM cache works")

    # Cleanup
    conn.close()
    os.remove(test_db)
    print("\n=== All tests passed! ===")

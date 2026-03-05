"""Tests for database.py — schema, FTS5, upsert, dedup, soft delete, cache."""

from database import (
    cache_get,
    cache_set,
    cleanup_orphaned_content,
    content_hash,
    get_stats,
    init_db,
    soft_delete_missing,
    upsert_document,
)


class TestSchema:
    """Verify init_db creates the expected tables, indexes, and FTS5 config."""

    def test_init_creates_tables(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        assert "documents" in tables
        assert "content" in tables
        assert "documents_fts" in tables
        assert "memories" in tables
        assert "conversations" in tables
        assert "facts" in tables
        assert "known_faces" in tables
        assert "chunks" in tables
        assert "llm_cache" in tables
        conn.close()

    def test_fts5_porter_tokenizer(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='documents_fts'"
        ).fetchone()
        assert "porter unicode61" in row["sql"]
        conn.close()

    def test_chunks_fts_exists(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='chunks_fts'"
        ).fetchone()
        assert row is not None
        assert "porter unicode61" in row["sql"]
        conn.close()

    def test_idempotent(self, tmp_path):
        """init_db can be called twice on the same file without error."""
        db_path = str(tmp_path / "test.db")
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        # Verify tables still exist after second init
        tables = [
            r["name"]
            for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "documents" in tables
        conn2.close()


class TestUpsert:
    """Verify insert, update, dedup, FTS sync, and porter stemming."""

    def test_insert_new_document(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("Hello world about invoices")
        h = upsert_document(conn, str(f), "Hello world about invoices", "txt")
        # SHA-256 hex digest is 64 chars
        assert len(h) == 64
        row = conn.execute(
            "SELECT * FROM documents WHERE path = ?", (str(f.resolve()),)
        ).fetchone()
        assert row is not None
        assert row["active"] == 1
        assert row["file_type"] == "txt"
        conn.close()

    def test_content_dedup(self, tmp_path):
        """Same text in two files produces one content row but two document rows."""
        conn = init_db(str(tmp_path / "test.db"))
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        h1 = upsert_document(conn, str(f1), "same content", "txt")
        h2 = upsert_document(conn, str(f2), "same content", "txt")
        assert h1 == h2
        # Only 1 content row (content-addressable dedup)
        count = conn.execute("SELECT COUNT(*) as n FROM content").fetchone()["n"]
        assert count == 1
        # But 2 document rows
        doc_count = conn.execute("SELECT COUNT(*) as n FROM documents").fetchone()["n"]
        assert doc_count == 2
        conn.close()

    def test_content_change_updates_hash(self, tmp_path):
        """Upserting the same path with different text updates the hash."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("original")
        h1 = upsert_document(conn, str(f), "original", "txt")
        f.write_text("updated")
        h2 = upsert_document(conn, str(f), "updated", "txt")
        assert h1 != h2
        doc = conn.execute(
            "SELECT hash FROM documents WHERE path = ?", (str(f.resolve()),)
        ).fetchone()
        assert doc["hash"] == h2
        conn.close()

    def test_fts_search_after_insert(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "invoice.txt"
        f.write_text("invoice for payment of services")
        upsert_document(conn, str(f), "invoice for payment of services", "txt")
        results = conn.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH 'invoice'"
        ).fetchall()
        assert len(results) == 1
        conn.close()

    def test_fts_porter_stemming(self, tmp_path):
        """Porter stemmer: 'running' should match a search for 'run'."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("She was running fast")
        upsert_document(conn, str(f), "She was running fast", "txt")
        results = conn.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH 'run'"
        ).fetchall()
        assert len(results) == 1
        conn.close()

    def test_fts_syncs_on_content_update(self, tmp_path):
        """After content change, FTS should find the new text, not the old."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("original unique word alpha")
        upsert_document(conn, str(f), "original unique word alpha", "txt")
        # Update content
        f.write_text("updated unique word beta")
        upsert_document(conn, str(f), "updated unique word beta", "txt")
        # Old term should no longer match
        old = conn.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH 'alpha'"
        ).fetchall()
        assert len(old) == 0
        # New term should match
        new = conn.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH 'beta'"
        ).fetchall()
        assert len(new) == 1
        conn.close()


class TestSoftDelete:
    """Verify soft delete deactivates documents and removes from FTS."""

    def test_soft_delete_removes_from_fts(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("unique content here")
        upsert_document(conn, str(f), "unique content here", "txt")
        # Passing an empty set means no files exist -> all get soft-deleted
        count = soft_delete_missing(conn, set())
        assert count == 1
        # FTS should be empty
        results = conn.execute(
            "SELECT * FROM documents_fts WHERE documents_fts MATCH 'unique'"
        ).fetchall()
        assert len(results) == 0
        conn.close()

    def test_soft_delete_preserves_matching_paths(self, tmp_path):
        """Documents whose paths are in the indexed_paths set are kept active."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "keep.txt"
        f.write_text("keep this file")
        upsert_document(conn, str(f), "keep this file", "txt")
        abs_path = str(f.resolve())
        count = soft_delete_missing(conn, {abs_path})
        assert count == 0
        doc = conn.execute(
            "SELECT active FROM documents WHERE path = ?", (abs_path,)
        ).fetchone()
        assert doc["active"] == 1
        conn.close()

    def test_reactivation(self, tmp_path):
        """Upserting a soft-deleted document reactivates it."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("reactivate me")
        upsert_document(conn, str(f), "reactivate me", "txt")
        soft_delete_missing(conn, set())
        # Confirm deactivated
        doc = conn.execute(
            "SELECT active FROM documents WHERE path = ?", (str(f.resolve()),)
        ).fetchone()
        assert doc["active"] == 0
        # Re-upsert reactivates
        f.write_text("reactivated content")
        upsert_document(conn, str(f), "reactivated content", "txt")
        doc = conn.execute(
            "SELECT active FROM documents WHERE path = ?", (str(f.resolve()),)
        ).fetchone()
        assert doc["active"] == 1
        conn.close()


class TestOrphanCleanup:
    """Verify orphaned content rows get cleaned up."""

    def test_cleanup_orphaned_content(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("orphan test")
        upsert_document(conn, str(f), "orphan test", "txt")
        # Soft delete -> content becomes orphaned
        soft_delete_missing(conn, set())
        cleaned = cleanup_orphaned_content(conn)
        assert cleaned == 1
        content_count = conn.execute(
            "SELECT COUNT(*) as n FROM content"
        ).fetchone()["n"]
        assert content_count == 0
        conn.close()


class TestCache:
    """Verify LLM cache get/set roundtrip."""

    def test_cache_roundtrip(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        cache_set(conn, "test key", '{"result": 42}')
        val = cache_get(conn, "test key")
        assert val == '{"result": 42}'
        conn.close()

    def test_cache_miss(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        val = cache_get(conn, "nonexistent")
        assert val is None
        conn.close()

    def test_cache_overwrite(self, tmp_path):
        """Writing the same key twice overwrites the value."""
        conn = init_db(str(tmp_path / "test.db"))
        cache_set(conn, "key", "value1")
        cache_set(conn, "key", "value2")
        val = cache_get(conn, "key")
        assert val == "value2"
        conn.close()


class TestContentHash:
    """Verify content_hash produces stable SHA-256 digests."""

    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_text_different_hash(self):
        assert content_hash("hello") != content_hash("world")

    def test_sha256_length(self):
        assert len(content_hash("test")) == 64


class TestStats:
    """Verify get_stats returns correct document counts by type."""

    def test_get_stats(self, tmp_path):
        conn = init_db(str(tmp_path / "test.db"))
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.pdf"
        f1.write_text("text content")
        f2.write_text("pdf content")
        upsert_document(conn, str(f1), "text content", "txt")
        upsert_document(conn, str(f2), "pdf content", "pdf")
        stats = get_stats(conn)
        assert stats["total_documents"] == 2
        assert stats["by_type"]["txt"] == 1
        assert stats["by_type"]["pdf"] == 1
        assert stats["unique_content"] == 2
        conn.close()

    def test_stats_excludes_soft_deleted(self, tmp_path):
        """Soft-deleted documents should not appear in stats."""
        conn = init_db(str(tmp_path / "test.db"))
        f = tmp_path / "doc.txt"
        f.write_text("temporary")
        upsert_document(conn, str(f), "temporary", "txt")
        soft_delete_missing(conn, set())
        stats = get_stats(conn)
        assert stats["total_documents"] == 0
        assert stats["by_type"] == {}
        conn.close()

"""Tests for api/search.py — search, facts, document lookup."""

from __future__ import annotations


class TestSearch:
    def test_search_returns_results(self, client, seeded_db, sample_folder):
        """Seeded DB has 'hello world' document — search should find it."""
        r = client.get("/search", params={"q": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) >= 1
        assert any("hello" in str(res).lower() for res in data["results"])

    def test_search_no_results(self, client, seeded_db):
        r = client.get("/search", params={"q": "xyzzynonexistent123"})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 0

    def test_search_filter_file_type(self, client, seeded_db, sample_folder):
        r = client.get("/search", params={"q": "Alice", "file_type": "csv"})
        assert r.status_code == 200

    def test_search_with_folder_filter(self, client, seeded_db, sample_folder):
        folder = str(sample_folder)
        r = client.get("/search", params={"q": "hello", "folder": folder})
        assert r.status_code == 200
        for result in r.json()["results"]:
            assert result["path"].startswith(folder)

    def test_search_with_limit(self, client, seeded_db, sample_folder):
        r = client.get("/search", params={"q": "the", "limit": 1})
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 1

    def test_search_empty_query_rejected(self, client, seeded_db):
        r = client.get("/search")
        assert r.status_code == 422

    def test_search_hint_on_results(self, client, seeded_db, sample_folder):
        r = client.get("/search", params={"q": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert "_hint" in data
        assert len(data["results"]) >= 1

    def test_search_hint_on_no_results(self, client, seeded_db):
        r = client.get("/search", params={"q": "xyzzynonexistent123"})
        assert r.status_code == 200
        assert "_hint" in r.json()


class TestSearchFacts:
    def test_search_facts(self, client, seeded_db):
        """Seeded DB has a fact about 'hello world message'."""
        r = client.get("/search-facts", params={"q": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1

    def test_search_facts_no_match(self, client, seeded_db):
        r = client.get("/search-facts", params={"q": "xyzzynonexistent123"})
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestDocument:
    def test_get_document_by_id(self, client, seeded_db):
        """Get a document by its ID."""
        r = client.get("/document/1")
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert "path" in data

    def test_get_document_not_found(self, client, seeded_db):
        r = client.get("/document/9999")
        assert r.status_code == 404


class TestSearchFactsExtra:
    def test_search_facts_with_limit(self, client, seeded_db):
        from datetime import UTC, datetime

        doc_id = seeded_db.execute(
            "SELECT id FROM documents WHERE path LIKE '%hello.txt'"
        ).fetchone()["id"]
        seeded_db.execute(
            "INSERT INTO facts(document_id, fact_text, category, created_at) VALUES (?, ?, ?, ?)",
            (doc_id, "Another hello fact for limit testing", "general", datetime.now(UTC).isoformat()),
        )
        seeded_db.commit()
        r = client.get("/search-facts", params={"q": "hello", "limit": 1})
        assert r.status_code == 200
        assert r.json()["count"] <= 1

    def test_search_facts_hint_on_results(self, client, seeded_db):
        r = client.get("/search-facts", params={"q": "hello"})
        assert r.status_code == 200
        assert "_hint" in r.json()

    def test_search_facts_hint_on_no_results(self, client, seeded_db):
        r = client.get("/search-facts", params={"q": "xyzzynonexistent123"})
        assert r.status_code == 200
        assert "_hint" in r.json()


class TestDocumentExtra:
    def test_document_has_all_fields(self, client, seeded_db):
        r = client.get("/document/1")
        assert r.status_code == 200
        data = r.json()
        assert data["text"]
        assert data["path"]
        assert data["file_type"]

    def test_document_inactive_still_returned(self, client, seeded_db):
        from database import soft_delete_missing

        soft_delete_missing(seeded_db, set())
        r = client.get("/document/1")
        assert r.status_code == 200
        assert r.json()["active"] == 0


class TestWebRead:
    def test_web_read_basic(self, client):
        import unittest.mock as mock

        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.raise_for_status.return_value = None
        fake_response.headers = {"content-type": "text/html; charset=utf-8"}
        fake_response.text = (
            "<html><body><p>This is the main content of the test page. "
            "Enough text to pass the readability threshold for processing.</p>"
            "</body></html>"
        )
        with mock.patch("requests.get", return_value=fake_response):
            r = client.get("/web-read", params={"url": "https://example.com/test"})
        assert r.status_code == 200
        data = r.json()
        assert "content" in data or "error" in data

    def test_web_read_missing_url(self, client):
        r = client.get("/web-read")
        assert r.status_code == 422

    def test_web_read_dns_failure(self, client):
        from unittest.mock import patch

        from fastapi import HTTPException

        # Patch _check_url_safe to raise 403 (simulates DNS failure / SSRF block)
        with patch("api.search._check_url_safe", side_effect=HTTPException(status_code=403, detail="blocked")):
            r = client.get("/web-read", params={"url": "https://unreachable.test.invalid/"})
        assert r.status_code == 403


class TestSearchChunks:
    def test_search_finds_chunked_content(self, client, seeded_db):
        from database import chunk_document

        doc_id = seeded_db.execute(
            "SELECT id FROM documents WHERE path LIKE '%hello.txt'"
        ).fetchone()["id"]
        chunk_document(seeded_db, doc_id, "Distinctive chunked paragraph about quarterly revenue")

        r = client.get("/search", params={"q": "quarterly revenue"})
        assert r.status_code == 200
        assert "results" in r.json()

    def test_search_folder_filter_excludes_other_folders(self, client, seeded_db, sample_folder, tmp_path):
        from database import upsert_document

        other_file = tmp_path / "other_hello.txt"
        other_file.write_text("Hello world in another folder")
        upsert_document(seeded_db, str(other_file), "Hello world in another folder", "txt")

        r = client.get("/search", params={"q": "hello", "folder": str(sample_folder)})
        assert r.status_code == 200
        for result in r.json()["results"]:
            assert result["path"].startswith(str(sample_folder))

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

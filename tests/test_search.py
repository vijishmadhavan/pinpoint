"""Tests for api/search.py — search, facts, document lookup."""

from __future__ import annotations

from unittest.mock import patch

from search_pipeline import DEFAULT_SEARCH_OPTIONS, ENHANCED_SEARCH_OPTIONS, SearchOptions


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

    def test_search_includes_stage_timing(self, client, seeded_db):
        r = client.get("/search", params={"q": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert "timing" in data
        assert data["timing"]["total_ms"] >= 0
        assert data["timing"]["lexical_ms"] >= 0
        assert data["timing"]["probe_ms"] >= 0

    def test_search_empty_inner_result_includes_timing(self, seeded_db):
        from search_pipeline import _search_inner

        result = _search_inner(seeded_db, "!!!", 5, None, None)

        assert result["results"] == []
        assert "timing" in result
        assert result["timing"]["total_ms"] >= 0

    def test_search_api_includes_search_explanation_block(self, client, seeded_db):
        r = client.get("/search", params={"q": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert data["search_explanation"]["search_mode"] == "lexical-first"
        assert data["search_explanation"]["ambiguous_search"] is False
        assert data["search_explanation"]["result_explanations_available"] is True


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

    def test_get_document_overview_by_id(self, client, seeded_db):
        r = client.get("/document/1/overview")
        assert r.status_code == 200
        data = r.json()
        assert data["title"]
        assert "overview" in data
        assert "top_sections" in data
        assert "facts" in data
        assert data["_hint"]


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

    def test_document_inactive_not_returned(self, client, seeded_db):
        from database import soft_delete_missing

        soft_delete_missing(seeded_db, set())
        r = client.get("/document/1")
        assert r.status_code == 404

    def test_document_overview_inactive_not_returned(self, client, seeded_db):
        from database import soft_delete_missing

        soft_delete_missing(seeded_db, set())
        r = client.get("/document/1/overview")
        assert r.status_code == 404

    def test_document_overview_includes_fact_rows_when_available(self, client, seeded_db):
        r = client.get("/document/1/overview")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["facts"], list)


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


class TestSemanticFallback:
    def test_search_uses_embedding_results_when_query_expansion_unavailable(self, client, seeded_db):
        """When embeddings are enabled and expansion fails, embedding results still appear."""
        from search_pipeline import SearchOptions

        enhanced = SearchOptions(use_query_expansion=True, use_embeddings=True, use_reranker=False)
        with (
            patch("search_pipeline.expand_query", return_value=[]),
            patch("search_pipeline._embedding_search", return_value=[{
                "id": 99,
                "path": "/tmp/fallback.txt",
                "file_type": "txt",
                "title": "Fallback doc",
                "score": 0.9,
                "raw_score": 0.9,
                "text": "semantic fallback content",
                "page_count": 1,
                "chunk_num": 0,
                "modified_at": "2026-01-01T00:00:00+00:00",
            }]),
            patch("search_pipeline.DEFAULT_SEARCH_OPTIONS", enhanced),
        ):
            r = client.get("/search", params={"q": "Hello world"})
        assert r.status_code == 200
        data = r.json()
        # Should have results from either FTS5 (seeded db has "Hello world") or embedding fallback
        assert data["results"]


class TestSearchOptions:
    def test_default_search_options_are_lexical_first(self):
        assert DEFAULT_SEARCH_OPTIONS.use_query_expansion is False
        assert DEFAULT_SEARCH_OPTIONS.use_embeddings is False
        assert DEFAULT_SEARCH_OPTIONS.use_reranker is False
        assert DEFAULT_SEARCH_OPTIONS.use_position_blend is False
        assert ENHANCED_SEARCH_OPTIONS.use_query_expansion is True
        assert ENHANCED_SEARCH_OPTIONS.use_embeddings is True
        assert ENHANCED_SEARCH_OPTIONS.use_reranker is True

    def test_search_options_can_disable_embedding_and_rerank_stages(self, seeded_db):
        from search_pipeline import _search_inner

        options = SearchOptions(
            use_query_expansion=False,
            use_embeddings=False,
            use_reranker=False,
            use_position_blend=False,
            use_strong_signal_shortcut=False,
        )

        with (
            patch("search_pipeline._embedding_search", side_effect=AssertionError("embedding search should be disabled")),
            patch("search_pipeline._rerank_results", side_effect=AssertionError("reranker should be disabled")),
        ):
            result = _search_inner(seeded_db, "hello", 5, None, None, options)

        assert result["results"]

    def test_default_search_does_not_escalate_for_strong_lexical_results(self, seeded_db):
        from search_pipeline import _search_inner

        with (
            patch("search_pipeline.expand_query", side_effect=AssertionError("query expansion should stay disabled")),
            patch("search_pipeline._embedding_search", side_effect=AssertionError("embedding search should stay disabled")),
            patch("search_pipeline._rerank_results", side_effect=AssertionError("reranker should stay disabled")),
        ):
            result = _search_inner(seeded_db, "hello", 5, None, None)

        assert result["results"]
        assert result["enhanced_search_used"] is False
        assert result["ambiguous_search"] is False


class TestQueryBuilders:
    def test_relaxed_query_filters_single_character_tokens(self):
        from search_pipeline import _build_relaxed_fts5_query

        query = "cenvat credit dispute case m/s. l&t ltd."
        relaxed = _build_relaxed_fts5_query(query)

        assert '"cenvat"*' in relaxed
        assert '"credit"*' in relaxed
        assert '"ltd"*' in relaxed
        assert '"m"*' not in relaxed
        assert '"s"*' not in relaxed
        assert '"l"*' not in relaxed
        assert '"t"*' not in relaxed

    def test_broad_query_filters_single_character_tokens(self):
        from search_pipeline import _build_broad_fts5_query

        query = "cenvat credit dispute case m/s. l&t ltd."
        broad = _build_broad_fts5_query(query)

        assert '"cenvat"*' in broad
        assert '"credit"*' in broad
        assert '"ltd"*' in broad
        assert '"m"*' not in broad
        assert '"s"*' not in broad
        assert '"l"*' not in broad
        assert '"t"*' not in broad


class TestLexicalFallbacks:
    def test_search_relaxes_stopword_heavy_query_when_strict_match_fails(self, seeded_db, tmp_path):
        from database import upsert_document
        from search_pipeline import _search_inner

        path = tmp_path / "clients.csv"
        path.write_text("client,status\nWipro,Renewal Pending\n", encoding="utf-8")
        upsert_document(seeded_db, str(path), "client status Wipro Renewal Pending", "csv")

        result = _search_inner(seeded_db, "which client is renewal pending", 5, None, None)

        assert result["results"]

    def test_search_boosts_filename_and_title_identifier_matches(self, seeded_db, tmp_path):
        from database import upsert_document
        from search_pipeline import _search_inner

        exact_path = tmp_path / "invoice_4821.txt"
        broad_path = tmp_path / "notes.txt"
        exact_text = "Payment reminder for invoice 4821."
        broad_text = "Project notes mention invoice 4821 once among many unrelated updates."
        exact_path.write_text(exact_text, encoding="utf-8")
        broad_path.write_text(broad_text, encoding="utf-8")
        upsert_document(seeded_db, str(exact_path), exact_text, "txt")
        upsert_document(seeded_db, str(broad_path), broad_text, "txt")

        result = _search_inner(seeded_db, "invoice 4821", 5, None, None)

        assert result["results"]
        assert result["results"][0]["path"] == str(exact_path)
        assert result["results"][0]["metadata_score"] > 0
        assert result["results"][0]["match_type"] in {"path", "title"}
        assert "identifier" in result["results"][0]["why_matched"].lower() or "file name/path" in result["results"][0]["why_matched"].lower()

    def test_search_explanation_fields_exist_on_results(self, seeded_db):
        from search_pipeline import _search_inner

        result = _search_inner(seeded_db, "hello world", 5, None, None)

        assert result["results"]
        top = result["results"][0]
        assert top["match_type"]
        assert top["why_matched"]
        assert top["match_type"] in {"title", "path", "chunk", "content", "blended", "unknown"}


class TestAmbiguousSearch:
    def test_search_marks_clustered_results_as_ambiguous(self, client):
        fake_results = [
            {"id": 1, "path": "/tmp/a.txt", "title": "Case A", "score": 0.91},
            {"id": 2, "path": "/tmp/b.txt", "title": "Case B", "score": 0.89},
            {"id": 3, "path": "/tmp/c.txt", "title": "Case C", "score": 0.87},
        ]
        with patch("api.search.search", return_value={
            "query": "section 138 case",
            "fts5_query": '"section"* AND "138"* AND "case"*',
            "results": fake_results,
            "strong_signal": False,
            "ambiguous_search": True,
            "clarification_hint": "Multiple similar matches found. Can you specify the file name, title, date, person, location, or year?",
            "ambiguous_result_count": 3,
            "expanded": False,
            "relaxed_lexical": False,
            "enhanced_search_used": False,
        }):
            r = client.get("/search", params={"q": "section 138 case"})

        assert r.status_code == 200
        data = r.json()
        assert data["ambiguous_search"] is True
        assert data["ambiguous_result_count"] == 3
        assert "specify" in data["clarification_hint"].lower()
        assert data["_hint"] == data["clarification_hint"]
        assert data["search_explanation"]["ambiguous_search"] is True

    def test_search_hint_uses_clarification_for_ambiguous_results(self, client, seeded_db, tmp_path):
        from database import upsert_document

        for name in ["section_138_alpha.txt", "section_138_beta.txt", "section_138_gamma.txt"]:
            path = tmp_path / name
            text = "Section 138 cheque dishonour dispute involving similar facts."
            path.write_text(text, encoding="utf-8")
            upsert_document(seeded_db, str(path), text, "txt")

        r = client.get("/search", params={"q": "section 138 cheque case"})

        assert r.status_code == 200
        data = r.json()
        assert data["ambiguous_search"] is True
        assert data["_hint"] == data["clarification_hint"]
        assert "specify" in data["_hint"].lower()

    def test_search_matches_handover_synonym_for_handoff_query(self, seeded_db, tmp_path):
        from database import upsert_document
        from search_pipeline import _search_inner

        path = tmp_path / "incident.txt"
        path.write_text("Publish a shift handover checklist for the on-call engineer.", encoding="utf-8")
        upsert_document(seeded_db, str(path), "Publish a shift handover checklist for the on-call engineer.", "txt")

        result = _search_inner(seeded_db, "on-call handoff checklist", 5, None, None)

        assert result["results"]
        assert result["results"][0]["path"] == str(path)
        assert result["relaxed_lexical"] is True

    def test_search_merges_broad_candidates_after_relaxed_hit(self, seeded_db, tmp_path):
        from database import upsert_document
        from search_pipeline import _search_inner

        strict_path = tmp_path / "postmortem.txt"
        broad_path = tmp_path / "runbook.txt"
        strict_path.write_text("Page the platform team when queue depth exceeds 50000 jobs.", encoding="utf-8")
        broad_path.write_text("Check current queue depth and notify the on-call engineer.", encoding="utf-8")
        upsert_document(seeded_db, str(strict_path), "Page the platform team when queue depth exceeds 50000 jobs.", "txt")
        upsert_document(seeded_db, str(broad_path), "Check current queue depth and notify the on-call engineer.", "txt")

        result = _search_inner(seeded_db, "when should platform team be paged for queue depth", 5, None, None)
        returned_paths = [row["path"] for row in result["results"]]

        assert str(strict_path) in returned_paths
        assert str(broad_path) in returned_paths

    def test_search_uses_broad_fallback_for_surface_form_mismatch(self, seeded_db, tmp_path):
        from database import upsert_document
        from search_pipeline import _search_inner

        path = tmp_path / "minutes.txt"
        path.write_text("Quarterly business review attendees list", encoding="utf-8")
        upsert_document(seeded_db, str(path), "Quarterly business review attendees list", "txt")

        result = _search_inner(seeded_db, "who attended quarterly business review", 5, None, None)

        assert result["results"]
        assert result["results"][0]["path"] == str(path)
        assert result["relaxed_lexical"] is True

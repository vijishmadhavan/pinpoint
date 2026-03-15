"""Focused unit tests for indexing_service failure handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_index_single_file_raises_when_document_row_missing(tmp_path):
    from indexing_service import index_single_file

    path = tmp_path / "missing-doc-row.txt"
    path.write_text("hello world", encoding="utf-8")

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    with (
        patch("indexing_service.extract", return_value={
            "text": "hello world",
            "file_type": "txt",
            "page_count": 0,
        }),
        patch("indexing_service.upsert_document", return_value="abc123"),
    ):
        with pytest.raises(RuntimeError, match="Document row missing after upsert"):
            index_single_file(conn, str(path), skip_unchanged=False, facts_enabled=False, embeddings_enabled=False)

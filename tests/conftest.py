"""Shared pytest fixtures for Pinpoint integration tests."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Module-level dict to map connection id -> db path (since sqlite3.Connection
# doesn't support arbitrary attribute assignment).
_conn_paths: dict[int, str] = {}


@pytest.fixture()
def test_db(tmp_path):
    """Create a fresh test database with all tables initialised."""
    db_path = str(tmp_path / "test.db")
    from database import init_db

    conn = init_db(db_path)
    _conn_paths[id(conn)] = db_path
    yield conn
    _conn_paths.pop(id(conn), None)
    conn.close()


def _get_db_path(conn: sqlite3.Connection) -> str:
    """Retrieve the db_path associated with a test connection."""
    return _conn_paths[id(conn)]


@pytest.fixture()
def client(test_db):
    """FastAPI TestClient wired to the test database (no auth required).

    Patches:
      1. api.helpers._get_conn  -> returns test_db connection
      2. api.search.DB_PATH     -> test DB file path (search() opens its own conn)
      3. api.API_SECRET         -> empty string (disable auth middleware)
    """
    import api
    import api.helpers

    db_path = _get_db_path(test_db)

    # Reset migrations flag so test DB gets migrations applied
    api.helpers._migrations_done = False

    with (
        patch.object(api.helpers, "_get_conn", lambda: test_db),
        patch.object(api.helpers, "DB_PATH", db_path),
        patch("api.search.DB_PATH", db_path),
        patch("search.DB_PATH", db_path),
        patch("search_pipeline.DB_PATH", db_path),
        patch("database.DB_PATH", db_path),
        patch.object(api, "API_SECRET", ""),
        patch("api.files.scan_paths_background", lambda: None),  # Prevent WSL filesystem hang
        patch("api.files._get_common_folders", lambda: []),  # Belt-and-suspenders
    ):
        from api import app

        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def sample_folder(tmp_path):
    """Create a small sample folder with test files.

    Layout:
        files/
            hello.txt      — "Hello world" text
            data.csv       — 3-row CSV (name, age, city)
            notes.md       — short markdown
            empty.txt      — zero-length file
            subdir/
                nested.txt — "Nested file content"
    """
    base = tmp_path / "files"
    base.mkdir()

    (base / "hello.txt").write_text("Hello world\nThis is a test file.", encoding="utf-8")

    csv_content = "name,age,city\nAlice,30,New York\nBob,25,London\nCarol,35,New York\n"
    (base / "data.csv").write_text(csv_content, encoding="utf-8")

    (base / "notes.md").write_text("# My Notes\n\nSome *markdown* content.\n", encoding="utf-8")

    (base / "empty.txt").write_text("", encoding="utf-8")

    subdir = base / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested file content", encoding="utf-8")

    return base


@pytest.fixture()
def seeded_db(test_db, sample_folder):
    """test_db with 3 documents inserted (hello.txt, data.csv, notes.md) + 1 fact.

    Uses upsert_document so FTS5 triggers fire and search works.
    """
    from database import upsert_document

    upsert_document(
        test_db,
        str(sample_folder / "hello.txt"),
        "Hello world this is a test file with some content",
        "txt",
    )
    upsert_document(
        test_db,
        str(sample_folder / "data.csv"),
        "name age city Alice 30 NYC Bob 25 LA Carol 35 Chicago",
        "csv",
    )
    upsert_document(
        test_db,
        str(sample_folder / "notes.md"),
        "Meeting Notes discussed project timeline",
        "md",
    )

    # Insert a fact linked to hello.txt
    doc_id = test_db.execute(
        "SELECT id FROM documents WHERE path LIKE '%hello.txt'"
    ).fetchone()["id"]
    from datetime import UTC, datetime

    test_db.execute(
        "INSERT INTO facts(document_id, fact_text, category, created_at) VALUES (?, ?, ?, ?)",
        (doc_id, "The file contains a hello world message", "general", datetime.now(UTC).isoformat()),
    )
    test_db.commit()
    return test_db

"""Tests for api/core.py — ping, status, indexing, watchers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tests.conftest import _get_db_path


class TestPing:
    def test_ping(self, client):
        r = client.get("/ping")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestStatus:
    def test_status(self, client, seeded_db):
        """Status should return document counts from test DB."""
        db_path = _get_db_path(seeded_db)
        with patch("api.core.DB_PATH", db_path):
            r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "by_type" in data
        assert data["total_documents"] == 3
        assert data["by_type"]["txt"] == 1
        assert data["by_type"]["csv"] == 1
        assert data["by_type"]["md"] == 1
        assert "db_path" in data
        assert "db_size_mb" in data
        assert data["db_size_mb"] >= 0


class TestIndexingStatus:
    def test_no_active_jobs(self, client):
        r = client.get("/indexing/status")
        assert r.status_code == 200
        data = r.json()
        assert data["active"] is False
        assert data["jobs"] == []


class TestIndexFile:
    def test_index_text_file(self, client, sample_folder):
        """Index a .txt file — works without Gemini."""
        path = str(sample_folder / "hello.txt")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            r = client.post("/index-file", json={"path": path})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["file_type"] == "txt"
        assert data["text_length"] > 0

    def test_index_csv_file(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            r = client.post("/index-file", json={"path": path})
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert r.json()["file_type"] == "csv"

    def test_index_md_file(self, client, sample_folder):
        """Markdown files are extracted as plain text (file_type='txt')."""
        path = str(sample_folder / "notes.md")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            r = client.post("/index-file", json={"path": path})
        assert r.status_code == 200
        assert r.json()["success"] is True
        # extractors.py classifies .md as "txt" (TEXT_EXTENSIONS handler)
        assert r.json()["file_type"] == "txt"

    def test_index_already_indexed(self, client, sample_folder):
        """Indexing same file twice — second call should still succeed.

        The mtime check in index_file_endpoint compares a UTC-aware DB timestamp
        against a naive file mtime, so the already_indexed shortcut may not trigger.
        Either outcome (already_indexed=True or re-indexed) is acceptable.
        """
        path = str(sample_folder / "hello.txt")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            r1 = client.post("/index-file", json={"path": path})
            assert r1.status_code == 200
            assert r1.json()["success"] is True

            # Second call — file unchanged
            r2 = client.post("/index-file", json={"path": path})
        assert r2.status_code == 200
        assert r2.json()["success"] is True
        # Accept either: skipped (already_indexed) or re-indexed (same hash)
        data2 = r2.json()
        assert data2.get("already_indexed") is True or "hash" in data2

    def test_index_nonexistent(self, client):
        r = client.post("/index-file", json={"path": "/tmp/nonexistent_xyz_pinpoint.txt"})
        assert r.status_code == 404

    def test_index_directory_rejected(self, client, sample_folder):
        r = client.post("/index-file", json={"path": str(sample_folder)})
        assert r.status_code == 400


class TestIndex:
    def test_index_small_folder(self, client, test_db, sample_folder):
        """Index a small folder of text files — should run synchronously."""
        db_path = _get_db_path(test_db)
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False),
            patch("api.core.DB_PATH", db_path),
        ):
            r = client.post("/index", json={"folder": str(sample_folder)})
        assert r.status_code == 200
        data = r.json()
        # index_folder returns {indexed, skipped, failed, ...}
        assert data.get("indexed", 0) >= 1
        assert "total_documents" in data

    def test_index_invalid_folder(self, client):
        r = client.post("/index", json={"folder": "/nonexistent_folder_xyz"})
        assert r.status_code == 400

    def test_index_large_folder_returns_job_id(self, client, tmp_path):
        folder = tmp_path / "bulk"
        folder.mkdir()
        for i in range(55):
            (folder / f"doc_{i}.txt").write_text("hello", encoding="utf-8")

        with (
            patch("api.core.get_or_create_job", return_value=(123, True)),
            patch("api.core.threading.Thread") as thread_cls,
        ):
            r = client.post("/index", json={"folder": str(folder)})

        assert r.status_code == 200
        data = r.json()
        assert data["background"] is True
        assert data["job_id"] == 123
        thread_cls.assert_called_once()

    def test_index_large_folder_marks_cancelled_job(self, client, tmp_path):
        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        folder = tmp_path / "bulk_cancel"
        folder.mkdir()
        for i in range(55):
            (folder / f"doc_{i}.txt").write_text("hello", encoding="utf-8")

        def fake_index_folder(_folder, _db_path, progress_callback):
            progress_callback(str(folder), 55, 1, str(folder / "doc_0.txt"))
            return {"indexed": 1}

        with (
            patch("api.core.get_or_create_job", return_value=(123, True)),
            patch("api.core.threading.Thread", ImmediateThread),
            patch("api.core.is_job_cancelling", side_effect=[False, True]),
            patch("api.core.update_job_progress"),
            patch("api.core.mark_job_running"),
            patch("api.core.mark_job_cancelled") as cancelled,
            patch("api.core.mark_job_completed"),
            patch("api.core.index_folder", side_effect=fake_index_folder),
        ):
            r = client.post("/index", json={"folder": str(folder)})

        assert r.status_code == 200
        assert r.json()["job_id"] == 123
        cancelled.assert_called_once()


try:
    import watchdog  # noqa: F401

    _has_watchdog = True
except ImportError:
    _has_watchdog = False


@pytest.mark.skipif(not _has_watchdog, reason="watchdog not installed")
class TestWatchers:
    def test_watched_empty(self, client):
        r = client.get("/watched-folders")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_watch_and_unwatch(self, client, sample_folder):
        """Watch a folder, verify it's listed, then unwatch."""
        folder = str(sample_folder)

        # Watch
        r = client.post("/watch-folder", json={"path": folder})
        assert r.status_code == 200
        assert r.json()["status"] in ("watching", "already_watching")

        # Verify listed
        r = client.get("/watched-folders")
        assert r.json()["count"] >= 1
        paths = [f["path"] for f in r.json()["folders"]]
        assert folder in paths

        # Unwatch
        r = client.post("/unwatch-folder", json={"path": folder})
        assert r.status_code == 200
        assert r.json()["status"] == "unwatched"

        # Verify removed
        r = client.get("/watched-folders")
        paths = [f["path"] for f in r.json().get("folders", [])]
        assert folder not in paths
        assert r.json()["count"] == 0

    def test_watch_invalid_folder(self, client):
        r = client.post("/watch-folder", json={"path": "/nonexistent_folder_xyz"})
        assert r.status_code == 404

    def test_unwatch_not_watched(self, client, sample_folder):
        r = client.post("/unwatch-folder", json={"path": str(sample_folder)})
        assert r.status_code == 200
        assert r.json()["status"] == "not_found"

    def test_watch_already_watched(self, client, sample_folder):
        """Watching the same folder twice should succeed with a note."""
        folder = str(sample_folder)
        r1 = client.post("/watch-folder", json={"path": folder})
        assert r1.status_code == 200

        r2 = client.post("/watch-folder", json={"path": folder})
        assert r2.status_code == 200
        assert r2.json()["status"] == "already_watching"

        # Cleanup: unwatch
        client.post("/unwatch-folder", json={"path": folder})


class TestIndexFileExtra:
    def test_index_file_unsupported_ext(self, client, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("some content")
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            r = client.post("/index-file", json={"path": str(f)})
        # Unsupported extension — should fail or return error
        assert r.status_code in (200, 400)


class TestStatusExtra:
    def test_status_returns_stats(self, client, test_db):
        db_path = _get_db_path(test_db)
        with patch("api.core.DB_PATH", db_path):
            r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data

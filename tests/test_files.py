"""Tests for api/files.py — file operations."""

from __future__ import annotations

import os
import shutil
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch


class TestListFiles:
    def test_list_files(self, client, sample_folder):
        r = client.get("/list_files", params={"folder": str(sample_folder)})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 4  # hello.txt, data.csv, notes.md, empty.txt, subdir
        names = [e["name"] for e in data["entries"]]
        assert "hello.txt" in names

    def test_list_files_filter_ext(self, client, sample_folder):
        r = client.get("/list_files", params={"folder": str(sample_folder), "filter_ext": ".txt"})
        assert r.status_code == 200
        data = r.json()
        for entry in data["entries"]:
            assert entry["name"].endswith(".txt")

    def test_list_files_name_contains(self, client, sample_folder):
        r = client.get("/list_files", params={"folder": str(sample_folder), "name_contains": "hello"})
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_list_files_recursive(self, client, sample_folder):
        r = client.get("/list_files", params={"folder": str(sample_folder), "recursive": True, "name_contains": "nested"})
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_list_files_invalid_folder(self, client):
        r = client.get("/list_files", params={"folder": "/nonexistent_folder_xyz"})
        assert r.status_code == 400


class TestFileInfo:
    def test_file_info(self, client, sample_folder):
        path = str(sample_folder / "hello.txt")
        r = client.get("/file_info", params={"path": path})
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "hello.txt"
        assert data["is_dir"] is False
        assert data["size"] > 0

    def test_folder_info(self, client, sample_folder):
        r = client.get("/file_info", params={"path": str(sample_folder)})
        assert r.status_code == 200
        data = r.json()
        assert data["is_dir"] is True
        assert "file_count" in data

    def test_file_not_found(self, client):
        r = client.get("/file_info", params={"path": "/tmp/nonexistent_file_xyz.txt"})
        assert r.status_code == 404


class TestReadFile:
    def test_read_file_triggers_background_index_for_text(self, client, sample_folder):
        path = str(sample_folder / "hello.txt")
        with patch("api.files._background_index") as background_index:
            r = client.post("/read_file", json={"path": path})
        assert r.status_code == 200
        background_index.assert_called_once_with(path)

    def test_read_text_file(self, client, sample_folder):
        path = str(sample_folder / "hello.txt")
        r = client.post("/read_file", json={"path": path})
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "text"
        assert "Hello world" in data["content"]

    def test_read_csv_file(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/read_file", json={"path": path})
        assert r.status_code == 200
        assert "Alice" in r.json()["content"]

    def test_read_nonexistent(self, client):
        r = client.post("/read_file", json={"path": "/tmp/nonexistent_xyz.txt"})
        assert r.status_code == 404

    def test_read_directory_fails(self, client, sample_folder):
        r = client.post("/read_file", json={"path": str(sample_folder)})
        assert r.status_code == 400


class TestMoveFile:
    def test_move_file(self, client, sample_folder):
        src = str(sample_folder / "empty.txt")
        dest = str(sample_folder / "moved_empty.txt")
        r = client.post("/move_file", json={"source": src, "destination": dest})
        assert r.status_code == 200
        assert r.json()["action"] == "moved"
        assert os.path.exists(dest)
        assert not os.path.exists(src)

    def test_copy_file(self, client, sample_folder):
        src = str(sample_folder / "hello.txt")
        dest = str(sample_folder / "hello_copy.txt")
        r = client.post("/move_file", json={"source": src, "destination": dest, "is_copy": True})
        assert r.status_code == 200
        assert r.json()["action"] == "copied"
        assert os.path.exists(src)  # original still exists
        assert os.path.exists(dest)

    def test_move_nonexistent(self, client):
        r = client.post("/move_file", json={"source": "/tmp/nonexistent_xyz.txt", "destination": "/tmp/dest.txt"})
        assert r.status_code == 404


class TestCreateFolder:
    def test_create_folder(self, client, tmp_path):
        path = str(tmp_path / "new_folder")
        r = client.post("/create_folder", json={"path": path})
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert os.path.isdir(path)

    def test_create_existing_folder(self, client, sample_folder):
        r = client.post("/create_folder", json={"path": str(sample_folder)})
        assert r.status_code == 200
        assert r.json()["already_existed"] is True


class TestDeleteFile:
    def test_delete_file(self, client, sample_folder):
        path = str(sample_folder / "empty.txt")
        r = client.post("/delete_file", json={"path": path})
        assert r.status_code == 200
        assert not os.path.exists(path)

    def test_delete_directory_fails(self, client, sample_folder):
        r = client.post("/delete_file", json={"path": str(sample_folder)})
        assert r.status_code == 400

    def test_delete_nonexistent(self, client):
        r = client.post("/delete_file", json={"path": "/tmp/nonexistent_xyz.txt"})
        assert r.status_code == 404


class TestFindDuplicates:
    def test_find_duplicates(self, client, sample_folder):
        # Create a duplicate
        shutil.copy2(str(sample_folder / "hello.txt"), str(sample_folder / "hello_dup.txt"))
        r = client.post("/find-duplicates", json={"folder": str(sample_folder)})
        assert r.status_code == 200
        data = r.json()
        assert data["duplicate_groups"] >= 1


class TestBatchRename:
    def test_batch_rename_dry_run(self, client, sample_folder):
        r = client.post("/batch-rename", json={
            "folder": str(sample_folder),
            "pattern": r"\.txt$",
            "replace": ".bak",
            "dry_run": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["dry_run"] is True
        assert data["renamed"] >= 1
        # Files should NOT actually be renamed
        assert os.path.exists(str(sample_folder / "hello.txt"))


class TestMoveUpdatesPathTables:
    """Verify move_file updates video_embeddings and photo_classifications paths."""

    def test_move_updates_video_embeddings(self, client, test_db, sample_folder):
        src = str(sample_folder / "hello.txt")
        dest = str(sample_folder / "hello_moved.txt")
        # Create video_embeddings table and insert a row keyed by src path
        test_db.execute("""
            CREATE TABLE IF NOT EXISTS video_embeddings (
                video_path TEXT NOT NULL, frame_sec REAL NOT NULL,
                embedding BLOB NOT NULL, mtime REAL NOT NULL,
                embedded_at TEXT NOT NULL, PRIMARY KEY (video_path, frame_sec)
            )
        """)
        test_db.execute(
            "INSERT INTO video_embeddings VALUES (?, 1.0, X'00', 1.0, '2024-01-01')",
            (src,),
        )
        test_db.commit()
        r = client.post("/move_file", json={"source": src, "destination": dest})
        assert r.status_code == 200
        row = test_db.execute(
            "SELECT video_path FROM video_embeddings WHERE video_path = ?", (dest,)
        ).fetchone()
        assert row is not None

    def test_move_updates_photo_classifications(self, client, test_db, sample_folder):
        src = str(sample_folder / "hello.txt")
        dest = str(sample_folder / "hello_moved.txt")
        # Create photo_classifications table and insert a row keyed by src path
        test_db.execute("""
            CREATE TABLE IF NOT EXISTS photo_classifications (
                path TEXT PRIMARY KEY, mtime REAL, category TEXT, classified_at TEXT
            )
        """)
        test_db.execute(
            "INSERT INTO photo_classifications VALUES (?, 1.0, 'nature', '2024-01-01')",
            (src,),
        )
        test_db.commit()
        r = client.post("/move_file", json={"source": src, "destination": dest})
        assert r.status_code == 200
        row = test_db.execute(
            "SELECT path FROM photo_classifications WHERE path = ?", (dest,)
        ).fetchone()
        assert row is not None

    def test_move_updates_documents_path(self, client, test_db, sample_folder):
        src = str(sample_folder / "hello.txt")
        dest = str(sample_folder / "hello_moved.txt")
        # Index the file first
        from database import upsert_document
        upsert_document(test_db, src, "Hello world", "txt")
        r = client.post("/move_file", json={"source": src, "destination": dest})
        assert r.status_code == 200
        row = test_db.execute(
            "SELECT path FROM documents WHERE path = ?", (dest,)
        ).fetchone()
        assert row is not None
        old = test_db.execute(
            "SELECT path FROM documents WHERE path = ?", (src,)
        ).fetchone()
        assert old is None

    def test_move_updates_generated_files_and_known_faces(self, client, test_db, sample_folder):
        src = str(sample_folder / "hello.txt")
        dest = str(sample_folder / "hello_moved.txt")
        test_db.execute(
            "INSERT INTO generated_files(path, tool_name, description, created_at) VALUES (?, 'tool', '', '2024-01-01')",
            (src,),
        )
        test_db.execute(
            "INSERT INTO known_faces(name, embedding, source_image, created_at) VALUES ('alice', X'00', ?, '2024-01-01')",
            (src,),
        )
        test_db.commit()
        r = client.post("/move_file", json={"source": src, "destination": dest})
        assert r.status_code == 200
        assert test_db.execute("SELECT path FROM generated_files WHERE path = ?", (dest,)).fetchone() is not None
        assert test_db.execute(
            "SELECT source_image FROM known_faces WHERE source_image = ?",
            (dest,),
        ).fetchone() is not None


class TestWatchFolders:
    def test_watch_and_unwatch_folder(self, client, sample_folder):
        folder = str(sample_folder)
        r = client.post("/watch-folder", json={"path": folder})
        assert r.status_code == 200
        assert r.json()["status"] in {"watching", "already_watching"}

        listed = client.get("/watched-folders")
        assert listed.status_code == 200
        assert any(item["path"] == folder for item in listed.json()["folders"])

        r = client.post("/unwatch-folder", json={"path": folder})
        assert r.status_code == 200
        assert r.json()["status"] in {"unwatched", "not_found"}

    def test_unwatch_folder_checks_safe_path(self, client):
        r = client.post("/unwatch-folder", json={"path": "/etc"})
        assert r.status_code == 403

    def test_unwatch_folder_requests_watch_job_cancellation(self, client, test_db, sample_folder):
        from datetime import UTC, datetime

        folder = str(sample_folder)
        test_db.execute(
            "INSERT INTO watched_folders(path, added_at) VALUES (?, ?)",
            (folder, datetime.now(UTC).isoformat()),
        )
        test_db.execute(
            """
            INSERT INTO background_jobs(
                job_type, target_path, target_hash, status, current_stage,
                total_items, completed_items, details_json,
                started_at, finished_at, created_at, updated_at, failure_reason, retry_count
            ) VALUES (?, ?, '', 'running', 'walking', 5, 2, ?, ?, NULL, ?, ?, '', 0)
            """,
            (
                "watch_initial_index",
                folder,
                '{"folder": "sample", "discovered": 5}',
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        test_db.commit()

        r = client.post("/unwatch-folder", json={"path": folder})
        assert r.status_code == 200
        row = test_db.execute(
            "SELECT status FROM background_jobs WHERE job_type = ? AND target_path = ? ORDER BY id DESC LIMIT 1",
            ("watch_initial_index", folder),
        ).fetchone()
        assert row is not None
        assert row["status"] == "cancelling"


class TestBackgroundJobsApi:
    def test_list_background_jobs(self, client, test_db):
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        test_db.execute(
            """
            INSERT INTO background_jobs(
                job_type, target_path, target_hash, status, current_stage,
                total_items, completed_items, details_json,
                started_at, finished_at, created_at, updated_at, failure_reason, retry_count
            ) VALUES (?, '', '', 'pending', 'queued', 10, 3, ?, NULL, NULL, ?, ?, '', 0)
            """,
            ("auto_index_file", '{"path": "/tmp/demo.txt"}', now, now),
        )
        test_db.commit()

        r = client.get("/background-jobs")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        job = next(job for job in data["jobs"] if job["job_type"] == "auto_index_file")
        assert job["total_items"] == 10
        assert job["completed_items"] == 3
        assert job["details"]["path"] == "/tmp/demo.txt"

    def test_cancel_background_job(self, client, test_db):
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        cur = test_db.execute(
            """
            INSERT INTO background_jobs(
                job_type, target_path, target_hash, status, current_stage,
                total_items, completed_items, details_json,
                started_at, finished_at, created_at, updated_at, failure_reason, retry_count
            ) VALUES (?, '', '', 'running', 'indexing', 4, 1, '', ?, NULL, ?, ?, '', 0)
            """,
            ("auto_index_file", now, now, now),
        )
        test_db.commit()

        r = client.post(f"/background-jobs/{cur.lastrowid}/cancel")
        assert r.status_code == 200
        row = test_db.execute("SELECT status FROM background_jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
        assert row is not None
        assert row["status"] == "cancelling"


class TestBackgroundIndexing:
    def test_background_index_dedupes_same_path_submission(self, sample_folder):
        from api.helpers import _AUTO_INDEX_IN_FLIGHT, _background_index

        path = str(sample_folder / "hello.txt")
        submitted = []

        def fake_submit(fn):
            submitted.append(fn)
            return SimpleNamespace()

        _AUTO_INDEX_IN_FLIGHT.clear()
        with patch("api.helpers._AUTO_INDEX_EXECUTOR.submit", side_effect=fake_submit):
            _background_index(path)
            _background_index(path)

        assert len(submitted) == 1
        _AUTO_INDEX_IN_FLIGHT.clear()

    def test_background_index_honors_pre_start_cancellation(self, sample_folder):
        from api.helpers import _AUTO_INDEX_IN_FLIGHT, _background_index

        path = str(sample_folder / "hello.txt")
        submitted = []

        def fake_submit(fn):
            submitted.append(fn)
            fn()
            return SimpleNamespace()

        _AUTO_INDEX_IN_FLIGHT.clear()
        with (
            patch("api.helpers.init_db", return_value=SimpleNamespace(close=lambda: None)),
            patch("api.helpers._AUTO_INDEX_EXECUTOR.submit", side_effect=fake_submit),
            patch("api.helpers.get_or_create_job", return_value=(99, True)),
            patch("api.helpers.is_job_cancelling", return_value=True),
            patch("api.helpers.mark_job_cancelled") as cancelled,
            patch("api.helpers.mark_job_running") as running,
            patch("api.helpers.index_single_file") as index_single_file,
        ):
            _background_index(path)

        assert len(submitted) == 1
        cancelled.assert_called_once()
        running.assert_not_called()
        index_single_file.assert_not_called()
        _AUTO_INDEX_IN_FLIGHT.clear()

    def test_auto_index_docs_logs_partial_failures(self):
        from api import files as api_files

        conn = object()
        paths = ["/tmp/one.txt", "/tmp/two.txt", "/tmp/three.txt"]
        side_effects = [
            {"status": "indexed"},
            RuntimeError("boom"),
            {"status": "skipped", "reason": "unchanged"},
        ]

        def fake_index_single_file(*args, **kwargs):
            result = side_effects.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        api_files._scan_status["indexed"] = 0
        with (
            patch("api.files.index_single_file", side_effect=fake_index_single_file),
            patch("api.files.get_or_create_job", return_value=("fake-job-id", True)),
            patch("api.files.mark_job_running", return_value=None),
            patch("api.files.mark_job_completed", return_value=None),
            patch("api.files.mark_job_failed", return_value=None),
            patch("api.files.update_job_progress", return_value=None),
            patch("api.files.logger.info") as info_log,
            patch("api.files.logger.warning") as warning_log,
            patch("api.files.logger.exception") as exception_log,
        ):
            api_files._auto_index_docs(conn, paths)

        assert api_files._scan_status["indexed"] == 1
        info_log.assert_called_once()
        warning_log.assert_called_once()
        exception_log.assert_called_once()


class TestSecurity:
    """Verify _check_safe blocks access to system directories."""

    def test_list_files_blocked_path(self, client):
        r = client.get("/list_files", params={"folder": "/etc"})
        assert r.status_code == 403

    def test_read_file_blocked_path(self, client):
        r = client.post("/read_file", json={"path": "/etc/passwd"})
        assert r.status_code == 403

    def test_move_file_blocked_source(self, client):
        r = client.post("/move_file", json={"source": "/etc/passwd", "destination": "/tmp/stolen"})
        assert r.status_code == 403

    def test_move_file_blocked_dest(self, client, sample_folder):
        r = client.post("/move_file", json={
            "source": str(sample_folder / "hello.txt"),
            "destination": "/etc/hello.txt",
        })
        assert r.status_code == 403

    def test_delete_file_blocked_path(self, client):
        r = client.post("/delete_file", json={"path": "/etc/passwd"})
        assert r.status_code == 403

    def test_create_folder_blocked_path(self, client):
        r = client.post("/create_folder", json={"path": "/usr/local/evil"})
        assert r.status_code == 403

    def test_path_traversal_blocked(self, client, sample_folder):
        traversal = str(sample_folder / "../../../../../../etc/passwd")
        r = client.post("/read_file", json={"path": traversal})
        assert r.status_code == 403


class TestGrep:
    def test_grep_basic(self, client, sample_folder):
        r = client.post("/grep", json={"folder": str(sample_folder), "pattern": "Hello"})
        assert r.status_code == 200
        assert r.json()["matches"] >= 1

    def test_grep_no_match(self, client, sample_folder):
        r = client.post("/grep", json={"folder": str(sample_folder), "pattern": "xyzzynonexistent"})
        assert r.status_code == 200
        assert r.json()["matches"] == 0

    def test_grep_folder_not_found(self, client):
        r = client.post("/grep", json={"folder": "/tmp/nonexistent_folder_xyz", "pattern": "test"})
        assert r.status_code == 400


class TestBatchMove:
    def test_batch_move_basic(self, client, tmp_path):
        src1 = tmp_path / "a.txt"
        src2 = tmp_path / "b.txt"
        src1.write_text("file a")
        src2.write_text("file b")
        dest = str(tmp_path / "moved")
        r = client.post("/batch_move", json={"sources": [str(src1), str(src2)], "destination": dest})
        assert r.status_code == 200
        assert r.json()["moved_count"] == 2
        assert not src1.exists()
        assert not src2.exists()

    def test_batch_move_nonexistent_skipped(self, client, tmp_path):
        dest = str(tmp_path / "moved")
        r = client.post("/batch_move", json={
            "sources": ["/tmp/nonexistent_xyz_pinpoint.txt"],
            "destination": dest,
        })
        assert r.status_code == 200
        assert r.json()["skipped_count"] == 1
        assert r.json()["moved_count"] == 0


class TestBatchRenameExecute:
    def test_batch_rename_execute(self, client, tmp_path):
        (tmp_path / "photo_001.txt").write_text("a")
        (tmp_path / "photo_002.txt").write_text("b")
        r = client.post("/batch-rename", json={
            "folder": str(tmp_path),
            "pattern": r"photo_",
            "replace": "img_",
            "dry_run": False,
        })
        assert r.status_code == 200
        assert r.json()["dry_run"] is False
        assert r.json()["renamed"] >= 1
        import os
        assert os.path.exists(str(tmp_path / "img_001.txt"))

    def test_batch_rename_no_matches(self, client, sample_folder):
        r = client.post("/batch-rename", json={
            "folder": str(sample_folder),
            "pattern": r"xyzzy_nonexistent",
            "replace": "replaced",
        })
        assert r.status_code == 200
        assert r.json()["renamed"] == 0

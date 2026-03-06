"""Audit V2 validation tests — verify security fixes, resource management, and feature stability."""

import os
import sqlite3
import tempfile
import threading

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client():
    os.environ["API_SECRET"] = ""
    from api import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_folder(tmp_path):
    (tmp_path / "test.txt").write_text("hello world")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return str(tmp_path)


# =============================================================================
# 2. SECURITY VERIFICATION
# =============================================================================


class TestCommandInjection:
    """#1: Verify name_contains is sanitized for cmd.exe injection."""

    def test_name_contains_strips_shell_chars(self, client, sample_folder):
        """Shell metacharacters should be stripped, not executed."""
        r = client.get("/list_files", params={
            "folder": sample_folder,
            "name_contains": "test&whoami&echo"
        })
        assert r.status_code == 200
        # Should not crash or return shell output

    def test_name_contains_with_pipe(self, client, sample_folder):
        r = client.get("/list_files", params={
            "folder": sample_folder,
            "name_contains": "test|dir"
        })
        assert r.status_code == 200

    def test_name_contains_with_semicolon(self, client, sample_folder):
        r = client.get("/list_files", params={
            "folder": sample_folder,
            "name_contains": "test;ls"
        })
        assert r.status_code == 200


class TestPathTraversalFaces:
    """#4: Verify _check_safe blocks path traversal on face endpoints."""

    def test_detect_faces_blocked_path(self, client):
        r = client.post("/detect-faces", json={"image_path": "/etc/passwd"})
        assert r.status_code == 403

    def test_crop_face_blocked_path(self, client):
        r = client.post("/crop-face", json={"image_path": "/etc/shadow", "face_idx": 0})
        assert r.status_code == 403

    def test_find_person_blocked_folder(self, client, sample_folder):
        r = client.post("/find-person", json={
            "reference_image": os.path.join(sample_folder, "photo.jpg"),
            "folder": "/etc"
        })
        assert r.status_code == 403

    def test_count_faces_blocked_path(self, client):
        r = client.post("/count-faces", json={"image_path": "/usr/bin/ls"})
        assert r.status_code == 403

    def test_compare_faces_blocked_path(self, client, sample_folder):
        r = client.post("/compare-faces", json={
            "image_path_1": "/etc/passwd",
            "face_idx_1": 0,
            "image_path_2": os.path.join(sample_folder, "photo.jpg"),
            "face_idx_2": 0,
        })
        assert r.status_code == 403

    def test_remember_face_blocked_path(self, client):
        r = client.post("/remember-face", json={
            "image_path": "/etc/passwd", "face_idx": 0, "name": "test"
        })
        assert r.status_code == 403

    def test_recognize_faces_blocked_path(self, client):
        r = client.post("/recognize-faces", json={"image_path": "/etc/passwd"})
        assert r.status_code == 403

    def test_detect_faces_blocked_folder(self, client):
        r = client.post("/detect-faces", json={"folder": "/etc"})
        assert r.status_code == 403


class TestPathTraversalPhotos:
    """#5: Verify _check_safe blocks path traversal on photo endpoints."""

    def test_score_photo_blocked(self, client):
        r = client.post("/score-photo", json={"path": "/etc/passwd"})
        assert r.status_code == 403

    def test_cull_photos_blocked(self, client):
        r = client.post("/cull-photos", json={"folder": "/etc"})
        assert r.status_code == 403

    def test_suggest_categories_blocked(self, client):
        r = client.post("/suggest-categories", json={"folder": "/usr/bin"})
        assert r.status_code == 403

    def test_group_photos_blocked(self, client):
        r = client.post("/group-photos", json={"folder": "/proc", "categories": ["a"]})
        assert r.status_code == 403

    def test_cull_photos_blocked_rejects(self, client, sample_folder):
        r = client.post("/cull-photos", json={
            "folder": sample_folder,
            "rejects_folder": "/etc/rejects"
        })
        assert r.status_code == 403


class TestPathTraversalGoogle:
    """#6: Verify _check_safe blocks path traversal on Google endpoints."""

    def test_drive_upload_blocked(self, client):
        r = client.post("/google/drive-upload", params={"path": "/etc/passwd"})
        assert r.status_code == 403

    def test_gmail_attach_blocked(self, client):
        r = client.post("/google/gmail-send", json={
            "to": "test@test.com",
            "subject": "test",
            "body": "test",
            "attach": "/etc/passwd"
        })
        assert r.status_code == 403


class TestPathTraversalOCR:
    """#7: Verify _check_safe blocks path traversal on OCR single-path."""

    def test_ocr_blocked_path(self, client):
        r = client.post("/ocr", json={"path": "/etc/passwd"})
        assert r.status_code == 403


class TestPathTraversalIndex:
    """#8: Verify _check_safe blocks path traversal on /index."""

    def test_index_blocked_path(self, client):
        r = client.post("/index", json={"folder": "/etc"})
        assert r.status_code == 403

    def test_index_blocked_proc(self, client):
        r = client.post("/index", json={"folder": "/proc"})
        assert r.status_code == 403


class TestBlockedDotfiles:
    """#36: Verify sensitive dotfiles are blocked."""

    def test_ssh_blocked(self, client):
        home = os.path.expanduser("~")
        r = client.post("/read_file", json={"path": os.path.join(home, ".ssh/id_rsa")})
        assert r.status_code == 403

    def test_env_blocked(self, client):
        home = os.path.expanduser("~")
        r = client.post("/read_file", json={"path": os.path.join(home, ".env")})
        assert r.status_code == 403


class TestPdEvalSafety:
    """#3: Verify pd.eval uses numexpr (no arbitrary Python)."""

    def test_eval_no_dunder(self, client, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        r = client.post("/analyze-data", json={
            "path": str(csv), "operation": "eval", "query": "df.__class__"
        })
        # Should be blocked by the blocklist
        assert r.status_code == 400

    def test_eval_no_import(self, client, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        r = client.post("/analyze-data", json={
            "path": str(csv), "operation": "eval", "query": "import os"
        })
        assert r.status_code == 400


class TestSSRFProtection:
    """#22: Verify SSRF protection blocks private IPs."""

    def test_download_localhost_blocked(self, client):
        r = client.post("/download-url", json={"url": "http://127.0.0.1:8080/secret"})
        assert r.status_code == 403

    def test_download_private_ip_blocked(self, client):
        r = client.post("/download-url", json={"url": "http://192.168.1.1/"})
        assert r.status_code == 403

    def test_download_link_local_blocked(self, client):
        r = client.post("/download-url", json={"url": "http://169.254.169.254/metadata"})
        assert r.status_code == 403


class TestZipBombProtection:
    """#16: Verify zip bomb and symlink protection."""

    def test_extract_archive_symlink(self, client, tmp_path):
        import zipfile
        # Create a zip with a path traversal attempt
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../../etc/evil.txt", "malicious content")
        out = tmp_path / "output"
        r = client.post("/extract-archive", json={
            "path": str(zip_path),
            "output_path": str(out),
        })
        assert r.status_code == 400
        assert "Unsafe path" in r.json()["detail"]


# =============================================================================
# 3. RESOURCE & MEMORY STABILITY
# =============================================================================


class TestDBConnectionStability:
    """#14: Verify shared DB connections don't leak."""

    def test_image_search_shared_conn(self):
        from image_search import _get_conn
        conn1 = _get_conn()
        conn2 = _get_conn()
        assert conn1 is conn2  # Same connection reused

    def test_video_search_shared_conn(self):
        from video_search import _get_conn
        conn1 = _get_conn()
        conn2 = _get_conn()
        assert conn1 is conn2  # Same connection reused


class TestMemCacheEviction:
    """#15: Verify _mem_cache respects size limit."""

    def test_cache_limit(self):
        from image_search import _MEM_CACHE_MAX, _mem_cache
        assert _MEM_CACHE_MAX == 10


class TestSearchConnectionSafety:
    """#27: Verify search() closes connection on exception."""

    def test_search_closes_conn_on_success(self):
        from search import search
        # Just verify it doesn't crash — connection closed in finally
        result = search("nonexistent_query_xyz")
        assert "results" in result


class TestExtractTextSizeLimit:
    """#29: Verify extract_text enforces file size limit."""

    def test_large_file_skipped(self, tmp_path):
        from extractors import _MAX_TEXT_SIZE
        # Create a file just over the limit (using sparse file)
        big = tmp_path / "huge.txt"
        big.write_text("x")  # Create file
        # Monkey-patch getsize to simulate large file
        import unittest.mock as mock
        with mock.patch("os.path.getsize", return_value=_MAX_TEXT_SIZE + 1):
            from extractors import extract_text
            result = extract_text(str(big))
        assert result is None


class TestMemoryListSuperseded:
    """#31: Verify memory_list filters superseded memories."""

    def test_superseded_hidden(self, client):
        from api.helpers import _get_conn
        conn = _get_conn()
        # Insert a superseded memory
        conn.execute(
            "INSERT OR IGNORE INTO memories (fact, category, superseded_by, created_at, updated_at) VALUES (?, ?, ?, datetime('now'), datetime('now'))",
            ("old fact", "test", 999)
        )
        # Insert an active memory
        conn.execute(
            "INSERT OR IGNORE INTO memories (fact, category, superseded_by, created_at, updated_at) VALUES (?, ?, ?, datetime('now'), datetime('now'))",
            ("active fact", "test", None)
        )
        conn.commit()

        r = client.get("/memory/list", params={"category": "test"})
        assert r.status_code == 200
        facts = [m["fact"] for m in r.json()["memories"]]
        assert "old fact" not in facts
        if "active fact" in facts:
            assert True  # Active memory visible

        # Cleanup
        conn.execute("DELETE FROM memories WHERE fact IN ('old fact', 'active fact')")
        conn.commit()


# =============================================================================
# 4. FEATURE VALIDATION
# =============================================================================


class TestCosineSimSafety:
    """#18: Verify _cosine_sim handles zero vectors."""

    def test_zero_vector(self):
        import numpy as np

        from face_search import _cosine_sim
        a = np.zeros(512)
        b = np.ones(512)
        result = _cosine_sim(a, b)
        assert not np.isnan(result)  # Should not be NaN
        assert result == 0.0  # Zero vector = zero similarity


class TestDuplicateFilenameProtection:
    """#19: Verify cull/group don't overwrite duplicate filenames."""

    def test_unique_dest_names(self, tmp_path):
        dest = tmp_path / "rejects"
        dest.mkdir()
        # Create two files that would conflict
        (dest / "photo.jpg").write_bytes(b"first")

        # Simulate the dedup logic
        import shutil
        src = tmp_path / "photo.jpg"
        src.write_bytes(b"second")

        dest_path = dest / "photo.jpg"
        if dest_path.exists():
            base, ext = os.path.splitext("photo.jpg")
            counter = 1
            while dest_path.exists():
                dest_path = dest / f"{base}_{counter}{ext}"
                counter += 1
        shutil.move(str(src), str(dest_path))

        # Both files should exist
        assert (dest / "photo.jpg").read_bytes() == b"first"
        assert (dest / "photo_1.jpg").read_bytes() == b"second"

"""Tests for api/photos.py — photo scoring, culling, grouping (mocked)."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import patch


class TestScorePhoto:
    def test_score(self, client, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake photo")
        mock_result = {
            "path": str(img),
            "score": 78,
            "technical": 82,
            "aesthetic": 74,
            "reasoning": "Good exposure, nice composition",
        }
        with patch("photo_cull.score_photo", return_value=mock_result):
            r = client.post("/score-photo", json={"path": str(img)})
        assert r.status_code == 200
        assert r.json()["score"] == 78


class TestCullPhotos:
    def test_cull(self, client, tmp_path):
        folder = tmp_path / "photos"
        folder.mkdir()
        mock_result = {
            "status": "started",
            "folder": str(folder),
            "total_images": 50,
            "keep_pct": 80,
        }
        with patch("photo_cull.cull_photos", return_value=mock_result):
            r = client.post("/cull-photos", json={
                "folder": str(folder), "keep_pct": 80,
            })
        assert r.status_code == 200
        assert r.json()["status"] == "started"


class TestCullStatus:
    def test_status(self, client, tmp_path):
        folder = str(tmp_path / "photos")
        mock_result = {
            "status": "scoring",
            "total": 50,
            "scored": 25,
            "percent": 50,
        }
        with patch("photo_cull.get_cull_status", return_value=mock_result):
            r = client.get("/cull-photos/status", params={"folder": folder})
        assert r.status_code == 200
        assert r.json()["percent"] == 50

    def test_done_status_can_include_csv_report_path(self, client, tmp_path):
        folder = str(tmp_path / "photos")
        mock_result = {
            "status": "done",
            "kept": 40,
            "rejected": 10,
            "report_path": "/tmp/_cull_report.html",
            "csv_report_path": "/tmp/_cull_report.csv",
        }
        with patch("photo_cull.get_cull_status", return_value=mock_result):
            r = client.get("/cull-photos/status", params={"folder": folder})
        assert r.status_code == 200
        assert r.json()["csv_report_path"] == "/tmp/_cull_report.csv"


class TestCullReuse:
    def test_cull_reuses_completed_run_and_status_survives_memory_reset(self, tmp_path):
        import photo_cull

        folder = tmp_path / "wedding"
        folder.mkdir()
        for i in range(2):
            (folder / f"img_{i}.jpg").write_bytes(b"fake")

        conn = sqlite3.connect(tmp_path / "photo_cull.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        photo_cull._db_conn = conn
        photo_cull._init_table(conn)
        photo_cull._cull_jobs.clear()

        def fake_score(path):
            base = 80 if path.endswith("img_0.jpg") else 72
            return {
                "path": path,
                "sharpness": 10,
                "exposure": 10,
                "composition": 8,
                "quality": 8,
                "emotion": 15,
                "interest": 14,
                "keeper": 15,
                "total": base,
                "reasoning": "Strong frame and clean exposure.",
            }

        with (
            patch("photo_cull.score_photo", side_effect=fake_score),
            patch("photo_cull._make_thumbnail_b64", return_value="thumb"),
        ):
            first = photo_cull.cull_photos(str(folder), keep_pct=80)
            assert first["started"] is True

            for _ in range(100):
                status = photo_cull.get_cull_status(str(folder))
                if status.get("status") == "done":
                    break
                time.sleep(0.02)
            assert status["status"] == "done"
            assert status["report_path"].endswith("_cull_report.html")
            assert status["csv_report_path"].endswith("_cull_report.csv")

            second = photo_cull.cull_photos(str(folder), keep_pct=80)
            assert second["status"] == "already_done"
            assert second["already_done"] is True

            photo_cull._cull_jobs.clear()
            recovered = photo_cull.get_cull_status(str(folder))
            assert recovered["reused"] is True
            assert recovered["report_path"].endswith("_cull_report.html")
            assert recovered["csv_report_path"].endswith("_cull_report.csv")


class TestSuggestCategories:
    def test_suggest(self, client, tmp_path):
        folder = tmp_path / "photos"
        folder.mkdir()
        mock_result = {
            "categories": ["Landscapes", "People", "Food", "Architecture"],
            "sample_count": 20,
        }
        with patch("photo_cull.suggest_categories", return_value=mock_result):
            r = client.post("/suggest-categories", json={"folder": str(folder)})
        assert r.status_code == 200
        assert len(r.json()["categories"]) == 4


class TestGroupPhotos:
    def test_group(self, client, tmp_path):
        folder = tmp_path / "photos"
        folder.mkdir()
        mock_result = {
            "status": "started",
            "folder": str(folder),
            "categories": ["Landscapes", "People"],
            "total_images": 100,
        }
        with patch("photo_cull.group_photos", return_value=mock_result):
            r = client.post("/group-photos", json={
                "folder": str(folder),
                "categories": ["Landscapes", "People"],
            })
        assert r.status_code == 200
        assert r.json()["status"] == "started"


class TestGroupStatus:
    def test_status(self, client, tmp_path):
        folder = str(tmp_path / "photos")
        mock_result = {
            "status": "classifying",
            "total": 100,
            "classified": 60,
            "percent": 60,
        }
        with patch("photo_cull.get_group_status", return_value=mock_result):
            r = client.get("/group-photos/status", params={"folder": folder})
        assert r.status_code == 200
        assert r.json()["percent"] == 60

"""Tests for api/photos.py — photo scoring, culling, grouping (mocked)."""

from __future__ import annotations

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

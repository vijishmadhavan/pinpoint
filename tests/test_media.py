"""Tests for api/media.py — visual search, video, audio, OCR (mocked)."""

from __future__ import annotations

import os
from unittest.mock import patch


class TestEmbeddingStatus:
    def test_no_job(self, client, tmp_path):
        folder = str(tmp_path / "photos")
        r = client.get("/embedding-status", params={"folder": folder})
        assert r.status_code == 200
        assert r.json()["status"] == "none"


class TestSearchVideo:
    def test_search_video(self, client, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        mock_result = {
            "results": [{"timestamp": 5.0, "score": 0.85, "description": "A sunset"}],
            "count": 1,
        }
        with patch("video_search.search_video", return_value=mock_result):
            r = client.post("/search-video", json={
                "video_path": str(video), "query": "sunset",
            })
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_search_video_not_found(self, client):
        r = client.post("/search-video", json={
            "video_path": "/tmp/nonexistent.mp4", "query": "test",
        })
        assert r.status_code == 404


class TestExtractFrame:
    def test_extract_frame(self, client, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        out = str(tmp_path / "frame.png")
        with patch("video_search.extract_frame_image", return_value=out):
            r = client.post("/extract-frame", json={
                "video_path": str(video), "seconds": 5.0,
            })
        assert r.status_code == 200
        assert r.json()["path"] == out

    def test_extract_frame_not_found(self, client):
        r = client.post("/extract-frame", json={
            "video_path": "/tmp/nonexistent.mp4", "seconds": 1.0,
        })
        assert r.status_code == 404


class TestTranscribeAudio:
    def test_transcribe(self, client, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake audio")
        mock_result = {"text": "Hello world", "duration": 5.0}
        with patch("audio_search.transcribe_audio", return_value=mock_result):
            r = client.post("/transcribe-audio", json={"path": str(audio)})
        assert r.status_code == 200
        assert "Hello world" in r.json()["text"]

    def test_transcribe_not_found(self, client):
        r = client.post("/transcribe-audio", json={"path": "/tmp/nonexistent.mp3"})
        assert r.status_code == 404


class TestSearchAudio:
    def test_search_audio(self, client, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake audio")
        mock_result = {
            "results": [{"timestamp": "0:30", "text": "mentioned pizza"}],
            "count": 1,
        }
        with patch("audio_search.search_audio", return_value=mock_result):
            r = client.post("/search-audio", json={
                "audio_path": str(audio), "query": "pizza",
            })
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_search_audio_not_found(self, client):
        r = client.post("/search-audio", json={
            "audio_path": "/tmp/nonexistent.mp3", "query": "test",
        })
        assert r.status_code == 404


class TestOcr:
    def test_ocr_single_file(self, client, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(b"fake image")
        mock_result = {"path": str(img), "text": "Scanned text here", "method": "tesseract_ocr"}
        with patch("api.files._ocr_single", return_value=mock_result):
            r = client.post("/ocr", json={"path": str(img)})
        assert r.status_code == 200
        assert "Scanned text" in r.json()["text"]

    def test_ocr_no_path_or_folder(self, client):
        r = client.post("/ocr", json={})
        assert r.status_code == 400


class TestSearchImagesVisual:
    def test_search_images_visual(self, client, tmp_path):
        folder = str(tmp_path)
        mock_result = {
            "results": [{"path": "/tmp/sunset.jpg", "score": 0.9}],
            "count": 1,
        }
        with (
            patch("image_search.search_images", return_value=mock_result),
            patch("image_search._get_image_files", return_value=["/tmp/sunset.jpg"]),
        ):
            r = client.post("/search-images-visual", json={"folder": folder, "query": "sunset"})
        assert r.status_code == 200

    def test_search_images_visual_large_embedding_job_returns_job_id(self, client, tmp_path):
        folder = str(tmp_path)
        files = [f"/tmp/img_{i}.jpg" for i in range(60)]
        with (
            patch("image_search._get_image_files", return_value=files),
            patch("image_search._load_cached_embeddings", return_value=[]),
            patch("api.media.get_or_create_job", return_value=(456, True)),
            patch("api.media.threading.Thread") as thread_cls,
        ):
            r = client.post("/search-images-visual", json={"folder": folder, "query": "sunset"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "embedding"
        assert data["job_id"] == 456
        thread_cls.assert_called_once()

    def test_search_images_visual_marks_cancelled_job(self, client, tmp_path):
        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        folder = str(tmp_path)
        files = [f"/tmp/img_{i}.jpg" for i in range(60)]

        def fake_embed_images(_folder, progress_callback=None):
            assert progress_callback is not None
            progress_callback(1, 60)
            return {}

        with (
            patch("image_search._get_image_files", return_value=files),
            patch("image_search._load_cached_embeddings", return_value=[]),
            patch("api.media.get_or_create_job", return_value=(456, True)),
            patch("api.media.threading.Thread", ImmediateThread),
            patch("api.media.is_job_cancelling", side_effect=[False, True]),
            patch("api.media.update_job_progress"),
            patch("api.media.mark_job_running"),
            patch("api.media.mark_job_cancelled") as cancelled,
            patch("api.media.mark_job_completed"),
            patch("image_search.embed_images", side_effect=fake_embed_images),
        ):
            r = client.post("/search-images-visual", json={"folder": folder, "query": "sunset"})

        assert r.status_code == 200
        assert r.json()["job_id"] == 456
        cancelled.assert_called_once()

    def test_search_images_visual_no_folder(self, client):
        r = client.post("/search-images-visual", json={"query": "sunset"})
        assert r.status_code == 422


class TestOCRExtra:
    def test_ocr_folder(self, client, tmp_path):
        img1 = tmp_path / "scan1.png"
        img2 = tmp_path / "scan2.png"
        img1.write_bytes(b"fake image 1")
        img2.write_bytes(b"fake image 2")
        mock_result = {"path": "mock", "text": "OCR text", "method": "tesseract_ocr"}
        with patch("api.files._ocr_single", return_value=mock_result):
            r = client.post("/ocr", json={"folder": str(tmp_path)})
        assert r.status_code == 200

    def test_ocr_not_found(self, client):
        mock_result = {"error": "File not found: /tmp/nonexistent_scan.png"}
        with patch("api.files._ocr_single", return_value=mock_result):
            r = client.post("/ocr", json={"path": "/tmp/nonexistent_scan.png"})
        assert r.status_code == 400

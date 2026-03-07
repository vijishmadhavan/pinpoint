"""Tests for ML pipeline edge cases — embedding caches, NaN handling, batch failures."""

import os

import numpy as np
import pytest
from PIL import Image


class TestImageSearchPipeline:
    def test_get_image_files_empty_folder(self, tmp_path):
        from image_search import _get_image_files

        result = _get_image_files(str(tmp_path))
        assert result == []

    def test_get_image_files_filters_extensions(self, tmp_path):
        from image_search import _get_image_files

        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "img.png").write_bytes(b"\x89PNG")

        result = _get_image_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "photo.jpg" in basenames
        assert "img.png" in basenames
        assert "doc.pdf" not in basenames

    def test_get_image_files_nonexistent_folder(self):
        from image_search import _get_image_files

        result = _get_image_files("/tmp/nonexistent_folder_12345")
        assert result == []

    def test_get_image_files_recursive(self, tmp_path):
        from image_search import _get_image_files

        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"\xff\xd8\xff")
        (sub / "nested.jpg").write_bytes(b"\xff\xd8\xff")

        result = _get_image_files(str(tmp_path), recursive=True)
        assert len(result) == 2

    def test_embedding_bytes_roundtrip(self):
        from image_search import EMBED_DIM, _bytes_to_embedding, _embedding_to_bytes

        emb = np.random.randn(EMBED_DIM).astype(np.float32)
        data = _embedding_to_bytes(emb)
        recovered = _bytes_to_embedding(data)
        np.testing.assert_allclose(emb, recovered, rtol=1e-6)

    def test_normalize_handles_zero_vector(self):
        """Zero vector normalization should not produce NaN."""
        from image_search import _normalize

        zero_emb = np.zeros(768, dtype=np.float32)
        result = _normalize(zero_emb)
        assert not np.any(np.isnan(result))


class TestVideoSearchPipeline:
    def test_normalize_handles_nan(self):
        """NaN embeddings should be handled gracefully."""
        from image_search import _normalize

        nan_emb = np.full(768, np.nan, dtype=np.float32)
        result = _normalize(nan_emb)
        # Should not crash — result may be NaN or zero
        assert isinstance(result, np.ndarray)


class TestFaceSearchPipeline:
    def test_preprocess_downsizes_large_images(self, tmp_path):
        from face_search import MAX_FACE_DIM, _preprocess_for_face

        img_path = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 3000), "red").save(str(img_path))

        arr = _preprocess_for_face(str(img_path))
        h, w = arr.shape[:2]
        assert max(h, w) <= MAX_FACE_DIM

    def test_preprocess_small_image_unchanged(self, tmp_path):
        from face_search import MAX_FACE_DIM, _preprocess_for_face

        img_path = tmp_path / "small.jpg"
        Image.new("RGB", (200, 150), "blue").save(str(img_path))

        arr = _preprocess_for_face(str(img_path))
        h, w = arr.shape[:2]
        assert w == 200
        assert h == 150

    def test_preprocess_returns_bgr(self, tmp_path):
        """InsightFace expects BGR, not RGB."""
        from face_search import _preprocess_for_face

        img_path = tmp_path / "rgb_test.jpg"
        # Pure red image (255,0,0) in RGB should be (0,0,255) in BGR
        Image.new("RGB", (100, 100), (255, 0, 0)).save(str(img_path))

        arr = _preprocess_for_face(str(img_path))
        # BGR: blue channel (index 0) should be 0, red channel (index 2) should be ~255
        assert arr[50, 50, 0] < 10  # Blue channel ~ 0
        assert arr[50, 50, 2] > 245  # Red channel ~ 255


class TestIndexerPipeline:
    def test_extract_text_imports(self):
        """Verify extractors module is importable and has extract_text."""
        from extractors import extract_text

        assert callable(extract_text)

    def test_image_exts_defined(self):
        """Image extensions constant should exist."""
        from image_search import IMAGE_EXTS

        assert ".jpg" in IMAGE_EXTS
        assert ".png" in IMAGE_EXTS

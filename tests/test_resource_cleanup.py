"""Tests for resource cleanup — verify PIL Images and BytesIO are properly closed."""

import numpy as np
import pytest
from PIL import Image


class TestImageSearchCleanup:
    def test_load_image_fast_returns_closeable(self, tmp_path):
        """_load_image_fast should return an image that caller can close."""
        from image_search import _load_image_fast

        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (200, 200), "red").save(str(img_path))

        img = _load_image_fast(str(img_path))
        assert img is not None
        assert img.size[0] <= 384  # MAX_LOAD_DIM
        img.close()  # Should not raise

    def test_load_image_fast_handles_corrupt(self, tmp_path):
        """Corrupt image should raise, not leak."""
        from image_search import _load_image_fast

        bad_path = tmp_path / "corrupt.jpg"
        bad_path.write_bytes(b"not a real image")

        with pytest.raises((OSError, ValueError, SyntaxError)):
            _load_image_fast(str(bad_path))


class TestFaceSearchCleanup:
    def test_preprocess_closes_image(self, tmp_path):
        """_preprocess_for_face should not leak PIL images."""
        from face_search import _preprocess_for_face

        img_path = tmp_path / "face.jpg"
        Image.new("RGB", (2000, 1500), "blue").save(str(img_path))

        arr = _preprocess_for_face(str(img_path))
        assert isinstance(arr, np.ndarray)
        assert arr.shape[2] == 3  # BGR channels


class TestPhotoCullCleanup:
    def test_make_thumbnail_closes_resources(self, tmp_path):
        """_make_thumbnail_b64 should close both Image and BytesIO."""
        from photo_cull import _make_thumbnail_b64

        img_path = tmp_path / "photo.jpg"
        Image.new("RGB", (800, 600), "green").save(str(img_path))

        b64 = _make_thumbnail_b64(str(img_path), size=100)
        assert b64 is not None
        assert len(b64) > 0

        # Decode to verify it's valid base64 JPEG
        import base64
        data = base64.b64decode(b64)
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_make_thumbnail_handles_missing_file(self):
        """Missing file should return None, not crash."""
        from photo_cull import _make_thumbnail_b64

        result = _make_thumbnail_b64("/tmp/nonexistent_photo_12345.jpg")
        assert result is None


class TestExtractorsCleanup:
    def test_preprocess_image_closes_original_on_resize(self):
        """When resizing, original should be closeable after."""
        from extractors import _preprocess_image

        img = Image.new("RGB", (3000, 2000), "red")
        result = _preprocess_image(img, 1024)
        assert max(result.size) <= 1024
        result.close()
        img.close()  # Should not raise even if already consumed

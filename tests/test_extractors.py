"""Tests for extractors.py — PDF/DOCX/image extraction, OCR fallback, edge cases."""

import pytest

from extractors import _preprocess_image, extract_text


class TestPreprocessImage:
    def test_small_image_unchanged(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), "red")
        result = _preprocess_image(img, 1024)
        assert result.size == (100, 100)
        result.close()
        img.close()

    def test_large_image_resized(self):
        from PIL import Image

        img = Image.new("RGB", (4000, 3000), "blue")
        result = _preprocess_image(img, 1024)
        w, h = result.size
        assert max(w, h) <= 1024
        assert min(w, h) > 0
        result.close()
        img.close()

    def test_preserves_aspect_ratio(self):
        from PIL import Image

        img = Image.new("RGB", (2000, 1000), "green")
        result = _preprocess_image(img, 500)
        w, h = result.size
        ratio = w / h
        assert abs(ratio - 2.0) < 0.1
        result.close()
        img.close()


class TestExtractText:
    def _get_text(self, result):
        """Extract text from result (dict with 'text' key or None)."""
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get("text")
        return result

    def test_plain_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world\nThis is a test file.")
        result = extract_text(str(f))
        text = self._get_text(result)
        assert text is not None
        assert "Hello world" in text

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        result = extract_text(str(f))
        text = self._get_text(result)
        assert text is not None
        assert "Alice" in text

    def test_nonexistent_file_returns_none(self):
        result = extract_text("/tmp/this_file_does_not_exist_12345.pdf")
        assert result is None

    def test_empty_file_returns_none_or_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = extract_text(str(f))
        text = self._get_text(result)
        assert text is None or text.strip() == ""

    def test_unknown_extension_handled(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        result = extract_text(str(f))
        # Unknown extensions: returns None or dict with empty/minimal text
        if result is not None:
            text = self._get_text(result)
            assert text is not None  # at least returns something

    def test_large_text_file_within_limit(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("x" * (1024 * 1024))
        result = extract_text(str(f))
        text = self._get_text(result)
        assert text is not None
        assert len(text) > 0

    def test_pdf_extraction_no_crash(self, tmp_path):
        """PDF extraction should not crash."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF content here")
        doc.save(str(pdf_path))
        doc.close()

        result = extract_text(str(pdf_path))
        # Should return something (dict or None), not crash
        assert result is None or isinstance(result, dict)

    def test_encrypted_pdf_no_crash(self, tmp_path):
        """Encrypted PDFs should be handled gracefully."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        pdf_path = tmp_path / "encrypted.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Secret content")
        perm = fitz.PDF_PERM_ACCESSIBILITY
        encrypt_meth = fitz.PDF_ENCRYPT_AES_256
        doc.save(str(pdf_path), encryption=encrypt_meth, owner_pw="owner", user_pw="user", permissions=perm)
        doc.close()

        result = extract_text(str(pdf_path))
        assert result is None or isinstance(result, dict)

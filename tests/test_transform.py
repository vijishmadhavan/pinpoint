"""Tests for api/transform.py — write, generate, compress, image ops, run python."""

import os
from unittest.mock import patch


class TestWriteFile:
    def test_write_new_file(self, client, tmp_path):
        path = str(tmp_path / "output.txt")
        r = client.post("/write-file", json={"path": path, "content": "Hello from test"})
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "Hello from test"

    def test_append_to_file(self, client, tmp_path):
        path = str(tmp_path / "append.txt")
        client.post("/write-file", json={"path": path, "content": "Line 1\n"})
        client.post("/write-file", json={"path": path, "content": "Line 2\n", "append": True})
        with open(path) as f:
            content = f.read()
        assert "Line 1" in content
        assert "Line 2" in content

    def test_creates_parent_dirs(self, client, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "file.txt")
        r = client.post("/write-file", json={"path": path, "content": "nested"})
        assert r.status_code == 200
        assert os.path.exists(path)


class TestGenerateExcel:
    def test_generate_from_dicts(self, client, tmp_path):
        path = str(tmp_path / "output.xlsx")
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        r = client.post("/generate-excel", json={"path": path, "data": data})
        assert r.status_code == 200
        assert r.json()["rows"] == 2
        assert os.path.exists(path)

    def test_generate_from_lists(self, client, tmp_path):
        path = str(tmp_path / "output2.xlsx")
        data = [["Alice", 30], ["Bob", 25]]
        r = client.post("/generate-excel", json={"path": path, "data": data, "columns": ["name", "age"]})
        assert r.status_code == 200
        assert r.json()["columns"] == ["name", "age"]


class TestCompressExtract:
    def test_compress_and_extract(self, client, sample_folder, tmp_path):
        # Compress
        zip_path = str(tmp_path / "archive.zip")
        files = [str(sample_folder / "hello.txt"), str(sample_folder / "data.csv")]
        r = client.post("/compress-files", json={"paths": files, "output_path": zip_path})
        assert r.status_code == 200
        assert r.json()["files_added"] == 2
        assert os.path.exists(zip_path)

        # Extract
        extract_path = str(tmp_path / "extracted")
        r = client.post("/extract-archive", json={"path": zip_path, "output_path": extract_path})
        assert r.status_code == 200
        assert r.json()["files_extracted"] == 2
        assert os.path.exists(os.path.join(extract_path, "hello.txt"))


class TestImageOps:
    def _create_test_image(self, path, width=100, height=80):
        """Create a simple test PNG image."""
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        img.save(path)
        img.close()

    def test_resize_image(self, client, tmp_path):
        img_path = str(tmp_path / "test.png")
        self._create_test_image(img_path, 200, 150)
        out_path = str(tmp_path / "resized.png")
        r = client.post("/resize-image", json={"path": img_path, "width": 100, "output_path": out_path})
        assert r.status_code == 200
        assert r.json()["new_size"][0] == 100

    def test_crop_image(self, client, tmp_path):
        img_path = str(tmp_path / "test.png")
        self._create_test_image(img_path, 200, 150)
        out_path = str(tmp_path / "cropped.png")
        r = client.post("/crop-image", json={"path": img_path, "x": 10, "y": 10, "width": 50, "height": 50, "output_path": out_path})
        assert r.status_code == 200
        assert r.json()["new_size"] == [50, 50]

    def test_convert_image(self, client, tmp_path):
        img_path = str(tmp_path / "test.png")
        self._create_test_image(img_path)
        out_path = str(tmp_path / "test.jpg")
        r = client.post("/convert-image", json={"path": img_path, "format": "jpg", "output_path": out_path})
        assert r.status_code == 200
        assert r.json()["format"] == "jpg"
        assert os.path.exists(out_path)


class TestRunPython:
    """Tests for /run-python endpoint.

    signal.SIGALRM only works in the main thread, but TestClient runs
    requests in a worker thread.  We mock signal.signal and signal.alarm
    to no-ops so the endpoint's timeout mechanism doesn't raise.
    """

    def _post_run_python(self, client, **json_body):
        """Helper that patches signal for the run-python call.

        signal.SIGALRM only works in the main thread but TestClient
        dispatches requests in a worker thread, so we stub signal.signal
        and signal.alarm to prevent ValueError.
        """
        import signal

        with (
            patch.object(signal, "signal", return_value=None),
            patch.object(signal, "alarm", return_value=None),
        ):
            return client.post("/run-python", json=json_body)

    def test_simple_code(self, client):
        r = self._post_run_python(client, code="print('hello')")
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert "hello" in r.json()["stdout"]

    def test_math_calculation(self, client):
        r = self._post_run_python(client, code="print(sum(range(10)))")
        assert r.status_code == 200
        assert "45" in r.json()["stdout"]

    def test_error_handling(self, client):
        r = self._post_run_python(client, code="raise ValueError('test error')")
        assert r.status_code == 200
        assert r.json()["success"] is False
        assert "ValueError" in r.json()["error"]

    def test_file_creation(self, client):
        r = self._post_run_python(client, code="with open('test_output.txt', 'w') as f: f.write('hello')")
        assert r.status_code == 200
        assert r.json()["success"] is True
        # Should detect created file
        if r.json().get("files_created"):
            assert any("test_output" in f for f in r.json()["files_created"])


class TestImageMetadata:
    def _create_test_png(self, path, width=10, height=10):
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(128, 64, 32))
        img.save(path)
        img.close()

    def test_image_metadata(self, client, tmp_path):
        img_path = str(tmp_path / "meta_test.png")
        self._create_test_png(img_path, 10, 10)
        r = client.post("/image-metadata", json={"path": img_path})
        assert r.status_code == 200
        data = r.json()
        assert data["dimensions"]["width"] == 10
        assert data["dimensions"]["height"] == 10

    def test_image_metadata_not_found(self, client, tmp_path):
        r = client.post("/image-metadata", json={"path": str(tmp_path / "nope.png")})
        assert r.status_code == 404


class TestDownloadUrl:
    def test_download_url_invalid(self, client, tmp_path):
        r = client.post("/download-url", json={"url": "not-a-url", "save_path": str(tmp_path / "out")})
        assert r.status_code == 400

    def test_download_url_no_scheme(self, client, tmp_path):
        r = client.post("/download-url", json={"url": "ftp://example.com/file.txt", "save_path": str(tmp_path / "out")})
        assert r.status_code == 400


class TestGenerateChart:
    def test_generate_chart_bar(self, client, tmp_path):
        out = str(tmp_path / "chart.png")
        r = client.post("/generate-chart", json={
            "data": {"x": ["A", "B", "C"], "y": [10, 20, 15]},
            "chart_type": "bar",
            "title": "Test Chart",
            "output_path": out,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert os.path.exists(out)

    def test_generate_chart_line(self, client, tmp_path):
        out = str(tmp_path / "line.png")
        r = client.post("/generate-chart", json={
            "data": {"labels": ["Jan", "Feb", "Mar"], "values": [5, 10, 7]},
            "chart_type": "line",
            "output_path": out,
        })
        assert r.status_code == 200
        assert os.path.exists(out)


class TestSplitPdf:
    def test_split_pdf_not_found(self, client, tmp_path):
        r = client.post("/split-pdf", json={
            "path": str(tmp_path / "ghost.pdf"),
            "pages": "1",
            "output_path": str(tmp_path / "out.pdf"),
        })
        assert r.status_code == 404


class TestPdfToImages:
    def test_pdf_to_images_not_found(self, client, tmp_path):
        r = client.post("/pdf-to-images", json={"path": str(tmp_path / "ghost.pdf")})
        assert r.status_code == 404


class TestImagesToPdf:
    def _create_png(self, path):
        from PIL import Image
        img = Image.new("RGB", (20, 20), color=(200, 100, 50))
        img.save(str(path))
        img.close()

    def test_images_to_pdf(self, client, tmp_path):
        img1 = tmp_path / "page1.png"
        img2 = tmp_path / "page2.png"
        self._create_png(img1)
        self._create_png(img2)
        out = str(tmp_path / "combined.pdf")
        r = client.post("/images-to-pdf", json={
            "paths": [str(img1), str(img2)],
            "output_path": out,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert r.json()["pages"] == 2
        assert os.path.exists(out)

    def test_images_to_pdf_missing_image(self, client, tmp_path):
        r = client.post("/images-to-pdf", json={
            "paths": [str(tmp_path / "nope.png")],
            "output_path": str(tmp_path / "out.pdf"),
        })
        assert r.status_code == 404


class TestCropImageExtra:
    def _create_test_image(self, path, width=100, height=80):
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(0, 128, 255))
        img.save(str(path))
        img.close()

    def test_crop_image_not_found(self, client, tmp_path):
        r = client.post("/crop-image", json={
            "path": str(tmp_path / "nope.png"),
            "x": 0, "y": 0, "width": 50, "height": 50,
        })
        assert r.status_code == 404

    def test_crop_out_of_bounds(self, client, tmp_path):
        img_path = tmp_path / "small.png"
        self._create_test_image(img_path, width=50, height=40)
        out = str(tmp_path / "cropped_oob.png")
        r = client.post("/crop-image", json={
            "path": str(img_path),
            "x": 0, "y": 0, "width": 200, "height": 200,
            "output_path": out,
        })
        assert r.status_code == 200
        assert os.path.exists(out)

"""Tests for api/faces.py — face detection, recognition, memory (mocked)."""

import os
from unittest.mock import patch


def _mock_detect_faces(image_path, conn):
    """Mock detect_faces: return 2 fake faces."""
    return [
        {"bbox": [10, 10, 50, 50], "confidence": 0.98, "age": 30, "gender": "M"},
        {"bbox": [100, 10, 150, 50], "confidence": 0.95, "age": 25, "gender": "F"},
    ]


def _mock_detect_faces_error(image_path, conn):
    return {"error": "No face model available"}


def _mock_crop_face(image_path, face_idx, conn):
    return {"path": "/tmp/cropped_face.jpg", "bbox": [10, 10, 50, 50]}


def _mock_crop_face_error(image_path, face_idx, conn):
    return {"error": "Face index out of range"}


def _mock_find_person(ref_image, folder, conn, threshold):
    return [
        {"path": "/tmp/match1.jpg", "similarity": 0.85},
        {"path": "/tmp/match2.jpg", "similarity": 0.72},
    ]


def _mock_find_person_error(ref_image, folder, conn, threshold):
    return {"error": "No faces detected in reference image"}


def _mock_find_person_by_face(ref_image, face_idx, folder, conn, threshold):
    return [
        {"path": "/tmp/match1.jpg", "similarity": 0.90},
    ]


def _mock_find_person_by_face_error(ref_image, face_idx, folder, conn, threshold):
    return {"error": "Face index out of range"}


def _mock_count_faces(image_path, conn):
    return {"image_path": image_path, "count": 3}


def _mock_count_faces_error(image_path, conn):
    return {"error": "Cannot open image"}


def _mock_compare_faces(img1, idx1, img2, idx2, conn):
    return {"similarity": 0.78, "same_person": True}


def _mock_compare_faces_error(img1, idx1, img2, idx2, conn):
    return {"error": "No face found at index 0 in image"}


def _mock_remember_face(image_path, face_idx, name, conn):
    return {"success": True, "name": name, "id": 1}


def _mock_remember_face_error(image_path, face_idx, name, conn):
    return {"error": "No face detected in image"}


def _mock_forget_face(name, conn):
    return {"success": True, "name": name, "deleted": 1}


def _mock_forget_face_error(name, conn):
    return {"error": "No saved face found with name 'Unknown'"}


def _mock_recognize_faces(image_path, conn):
    return {"faces": [{"name": "John", "confidence": 0.92, "bbox": [10, 10, 50, 50]}]}


def _mock_recognize_faces_error(image_path, conn):
    return {"error": "No face model available"}


# ---------------------------------------------------------------------------
# /detect-faces
# ---------------------------------------------------------------------------


class TestDetectFaces:
    def test_detect_single_image(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake image data")
        with patch("face_search.detect_faces", _mock_detect_faces):
            r = client.post("/detect-faces", json={"image_path": str(img)})
        assert r.status_code == 200
        data = r.json()
        assert data["face_count"] == 2
        assert len(data["faces"]) == 2
        assert data["faces"][0]["confidence"] == 0.98

    def test_detect_error(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.detect_faces", _mock_detect_faces_error):
            r = client.post("/detect-faces", json={"image_path": str(img)})
        assert r.status_code == 400
        assert "No face model" in r.json()["detail"]

    def test_detect_folder(self, client, tmp_path):
        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"fake")
        (folder / "b.png").write_bytes(b"fake")
        with patch("face_search.detect_faces", _mock_detect_faces):
            r = client.post("/detect-faces", json={"folder": str(folder)})
        assert r.status_code == 200
        data = r.json()
        assert data["images_processed"] == 2
        assert "results" in data
        # Each image should have its own faces + face_count
        for img_name, img_data in data["results"].items():
            assert img_data["face_count"] == 2

    def test_detect_folder_empty(self, client, tmp_path):
        folder = tmp_path / "empty_folder"
        folder.mkdir()
        r = client.post("/detect-faces", json={"folder": str(folder)})
        assert r.status_code == 404
        assert "No images found" in r.json()["detail"]

    def test_detect_folder_skips_errors(self, client, tmp_path):
        """When detect_faces returns error for some images, they are skipped."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "good.jpg").write_bytes(b"fake")
        (folder / "bad.jpg").write_bytes(b"fake")
        call_count = [0]

        def _alternating(image_path, conn):
            call_count[0] += 1
            if "bad" in os.path.basename(image_path):
                return {"error": "corrupt image"}
            return _mock_detect_faces(image_path, conn)

        with patch("face_search.detect_faces", _alternating):
            r = client.post("/detect-faces", json={"folder": str(folder)})
        assert r.status_code == 200
        # Only the good image should be in results
        assert r.json()["images_processed"] == 1


# ---------------------------------------------------------------------------
# /crop-face
# ---------------------------------------------------------------------------


class TestCropFace:
    def test_crop(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.crop_face", _mock_crop_face):
            r = client.post("/crop-face", json={"image_path": str(img), "face_idx": 0})
        assert r.status_code == 200
        assert "path" in r.json()
        assert "bbox" in r.json()

    def test_crop_error(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.crop_face", _mock_crop_face_error):
            r = client.post("/crop-face", json={"image_path": str(img), "face_idx": 99})
        assert r.status_code == 400
        assert "Face index out of range" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /find-person
# ---------------------------------------------------------------------------


class TestFindPerson:
    def test_find(self, client, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")
        folder = tmp_path / "search"
        folder.mkdir()
        with patch("face_search.find_person", _mock_find_person):
            r = client.post("/find-person", json={
                "reference_image": str(img), "folder": str(folder)
            })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["matches"]) == 2
        assert data["matches"][0]["similarity"] == 0.85

    def test_find_error(self, client, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")
        folder = tmp_path / "search"
        folder.mkdir()
        with patch("face_search.find_person", _mock_find_person_error):
            r = client.post("/find-person", json={
                "reference_image": str(img), "folder": str(folder)
            })
        assert r.status_code == 400
        assert "No faces detected" in r.json()["detail"]

    def test_find_with_threshold(self, client, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")
        folder = tmp_path / "search"
        folder.mkdir()
        captured = {}

        def _capture_threshold(ref_image, fld, conn, threshold):
            captured["threshold"] = threshold
            return _mock_find_person(ref_image, fld, conn, threshold)

        with patch("face_search.find_person", _capture_threshold):
            r = client.post("/find-person", json={
                "reference_image": str(img), "folder": str(folder), "threshold": 0.7
            })
        assert r.status_code == 200
        assert captured["threshold"] == 0.7


# ---------------------------------------------------------------------------
# /find-person-by-face
# ---------------------------------------------------------------------------


class TestFindPersonByFace:
    def test_find_by_face(self, client, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")
        folder = tmp_path / "search"
        folder.mkdir()
        with patch("face_search.find_person_by_face", _mock_find_person_by_face):
            r = client.post("/find-person-by-face", json={
                "reference_image": str(img), "face_idx": 0, "folder": str(folder)
            })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["matches"][0]["similarity"] == 0.90

    def test_find_by_face_error(self, client, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"fake")
        folder = tmp_path / "search"
        folder.mkdir()
        with patch("face_search.find_person_by_face", _mock_find_person_by_face_error):
            r = client.post("/find-person-by-face", json={
                "reference_image": str(img), "face_idx": 99, "folder": str(folder)
            })
        assert r.status_code == 400
        assert "Face index out of range" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /count-faces
# ---------------------------------------------------------------------------


class TestCountFaces:
    def test_count_single(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.count_faces", _mock_count_faces):
            r = client.post("/count-faces", json={"image_path": str(img)})
        assert r.status_code == 200
        assert r.json()["count"] == 3

    def test_count_error(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.count_faces", _mock_count_faces_error):
            r = client.post("/count-faces", json={"image_path": str(img)})
        assert r.status_code == 400
        assert "Cannot open image" in r.json()["detail"]

    def test_count_folder(self, client, tmp_path):
        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"fake")
        (folder / "b.png").write_bytes(b"fake")
        with patch("face_search.count_faces", _mock_count_faces):
            r = client.post("/count-faces", json={"folder": str(folder)})
        assert r.status_code == 200
        data = r.json()
        assert data["images_processed"] == 2
        assert "results" in data

    def test_count_folder_empty(self, client, tmp_path):
        folder = tmp_path / "empty_photos"
        folder.mkdir()
        r = client.post("/count-faces", json={"folder": str(folder)})
        assert r.status_code == 404
        assert "No images found" in r.json()["detail"]

    def test_count_paths(self, client, tmp_path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")
        with patch("face_search.count_faces", _mock_count_faces):
            r = client.post("/count-faces", json={"paths": [str(img1), str(img2)]})
        assert r.status_code == 200
        data = r.json()
        assert data["images_processed"] == 2


# ---------------------------------------------------------------------------
# /compare-faces
# ---------------------------------------------------------------------------


class TestCompareFaces:
    def test_compare(self, client, tmp_path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")
        with patch("face_search.compare_faces", _mock_compare_faces):
            r = client.post("/compare-faces", json={
                "image_path_1": str(img1), "image_path_2": str(img2)
            })
        assert r.status_code == 200
        data = r.json()
        assert data["similarity"] == 0.78
        assert data["same_person"] is True

    def test_compare_error(self, client, tmp_path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")
        with patch("face_search.compare_faces", _mock_compare_faces_error):
            r = client.post("/compare-faces", json={
                "image_path_1": str(img1), "image_path_2": str(img2)
            })
        assert r.status_code == 400
        assert "No face found" in r.json()["detail"]

    def test_compare_with_face_indices(self, client, tmp_path):
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")
        captured = {}

        def _capture_indices(i1, idx1, i2, idx2, conn):
            captured["idx1"] = idx1
            captured["idx2"] = idx2
            return _mock_compare_faces(i1, idx1, i2, idx2, conn)

        with patch("face_search.compare_faces", _capture_indices):
            r = client.post("/compare-faces", json={
                "image_path_1": str(img1), "face_idx_1": 2,
                "image_path_2": str(img2), "face_idx_2": 3,
            })
        assert r.status_code == 200
        assert captured["idx1"] == 2
        assert captured["idx2"] == 3


# ---------------------------------------------------------------------------
# /remember-face
# ---------------------------------------------------------------------------


class TestRememberFace:
    def test_remember(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.remember_face", _mock_remember_face):
            r = client.post("/remember-face", json={
                "image_path": str(img), "face_idx": 0, "name": "Alice"
            })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Alice"
        assert data["success"] is True
        assert "_hint" in data

    def test_remember_error(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.remember_face", _mock_remember_face_error):
            r = client.post("/remember-face", json={
                "image_path": str(img), "face_idx": 0, "name": "Alice"
            })
        assert r.status_code == 400
        assert "No face detected" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /forget-face
# ---------------------------------------------------------------------------


class TestForgetFace:
    def test_forget(self, client):
        with patch("face_search.forget_face", _mock_forget_face):
            r = client.post("/forget-face", json={"name": "Alice"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["name"] == "Alice"
        assert data["deleted"] == 1

    def test_forget_error(self, client):
        with patch("face_search.forget_face", _mock_forget_face_error):
            r = client.post("/forget-face", json={"name": "Unknown"})
        assert r.status_code == 400
        assert "No saved face found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /recognize-faces
# ---------------------------------------------------------------------------


class TestRecognizeFaces:
    def test_recognize(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.recognize_faces", _mock_recognize_faces):
            r = client.post("/recognize-faces", json={"image_path": str(img)})
        assert r.status_code == 200
        data = r.json()
        assert len(data["faces"]) == 1
        assert data["faces"][0]["name"] == "John"
        assert data["faces"][0]["confidence"] == 0.92

    def test_recognize_error(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")
        with patch("face_search.recognize_faces", _mock_recognize_faces_error):
            r = client.post("/recognize-faces", json={"image_path": str(img)})
        assert r.status_code == 400
        assert "No face model" in r.json()["detail"]

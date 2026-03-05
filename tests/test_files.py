"""Tests for api/files.py — file operations."""

from __future__ import annotations

import os
import shutil


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

"""Tests for api/data.py — calculate, analyze-data."""


class TestCalculate:
    def test_basic_math(self, client):
        r = client.post("/calculate", json={"expression": "2 + 3"})
        assert r.status_code == 200
        assert r.json()["result"] == 5

    def test_complex_expression(self, client):
        r = client.post("/calculate", json={"expression": "sqrt(144) + pi"})
        assert r.status_code == 200
        result = r.json()["result"]
        assert abs(result - 15.14159265) < 0.01

    def test_division(self, client):
        r = client.post("/calculate", json={"expression": "100 / 3"})
        assert r.status_code == 200
        assert abs(r.json()["result"] - 33.333) < 0.01

    def test_empty_expression(self, client):
        r = client.post("/calculate", json={"expression": ""})
        assert r.status_code == 400

    def test_invalid_expression(self, client):
        r = client.post("/calculate", json={"expression": "import os"})
        assert r.status_code == 400

    def test_formatted_output(self, client):
        r = client.post("/calculate", json={"expression": "1000000 + 500000"})
        assert r.status_code == 200
        assert r.json()["formatted"] == "1,500,000"


class TestAnalyzeData:
    def test_describe_csv(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "describe"})
        assert r.status_code == 200
        data = r.json()
        assert data["shape"] == [3, 3]  # 3 rows, 3 columns
        assert "name" in data["columns"]

    def test_head(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "head", "head": 2})
        assert r.status_code == 200
        assert "Alice" in r.json()["data"]

    def test_columns(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "columns"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "name" in data
        assert "age" in data

    def test_search(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "search", "query": "Alice"})
        assert r.status_code == 200
        assert r.json()["matched"] >= 1

    def test_filter(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "filter", "query": "age > 28"})
        assert r.status_code == 200
        assert r.json()["matched_rows"] == 2  # Alice (30) and Carol (35)

    def test_sort(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "sort", "columns": "-age"})
        assert r.status_code == 200
        # Carol (35) should be first
        assert "Carol" in r.json()["data"].split("\n")[1]

    def test_value_counts(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "value_counts", "columns": "city"})
        assert r.status_code == 200

    def test_shape(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "shape"})
        assert r.status_code == 200
        assert "3 rows" in r.json()["data"]

    def test_file_not_found(self, client):
        r = client.post("/analyze-data", json={"path": "/tmp/nonexistent.csv", "operation": "describe"})
        assert r.status_code == 404

    def test_unknown_operation(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "foobar"})
        assert r.status_code == 400

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

    def test_groupby(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "groupby", "columns": "city"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "New York" in data
        assert "London" in data

    def test_describe_with_stats(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post("/analyze-data", json={"path": path, "operation": "describe"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "count" in data
        assert "mean" in data

    def test_analyze_blocked_path(self, client):
        r = client.post("/analyze-data", json={"path": "/etc/passwd", "operation": "describe"})
        assert r.status_code == 403


class TestCalculateExtra:
    def test_power(self, client):
        r = client.post("/calculate", json={"expression": "2**10"})
        assert r.status_code == 200
        assert r.json()["result"] == 1024

    def test_modulo(self, client):
        r = client.post("/calculate", json={"expression": "10 % 3"})
        assert r.status_code == 200
        assert r.json()["result"] == 1


class TestReadExcel:
    def test_read_excel_basic(self, client, tmp_path):
        import openpyxl

        path = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Name", "Age"])
        ws.append(["Alice", 30])
        wb.save(str(path))

        r = client.post("/read_excel", json={"path": str(path)})
        assert r.status_code == 200
        data = r.json()
        assert "sheet_names" in data
        assert data["rows"] >= 1

    def test_read_excel_not_found(self, client, tmp_path):
        r = client.post("/read_excel", json={"path": str(tmp_path / "nope.xlsx")})
        assert r.status_code == 404

    def test_read_excel_wrong_format(self, client, sample_folder):
        r = client.post("/read_excel", json={"path": str(sample_folder / "data.csv")})
        assert r.status_code == 400


class TestExtractTables:
    def test_extract_tables_not_found(self, client, tmp_path):
        missing = str(tmp_path / "nope.pdf")
        r = client.post(f"/extract-tables?path={missing}")
        assert r.status_code == 404

    def test_extract_tables_wrong_format(self, client, sample_folder):
        path = str(sample_folder / "data.csv")
        r = client.post(f"/extract-tables?path={path}")
        assert r.status_code == 400

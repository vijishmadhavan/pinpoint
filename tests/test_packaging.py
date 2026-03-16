from __future__ import annotations

import importlib
import runpy
from unittest.mock import patch


class TestImportSmoke:
    def test_core_modules_import(self):
        modules = [
            "database",
            "search",
            "search_pipeline",
            "indexing_service",
            "job_service",
            "api",
        ]

        for name in modules:
            mod = importlib.import_module(name)
            assert mod is not None, f"failed to import {name}"

    def test_packaged_namespace_imports(self):
        modules = [
            "pinpoint",
            "pinpoint.database",
            "pinpoint.search",
            "pinpoint.search_pipeline",
            "pinpoint.indexing_service",
            "pinpoint.job_service",
            "pinpoint.api",
        ]

        for name in modules:
            mod = importlib.import_module(name)
            assert mod is not None, f"failed to import {name}"


class TestSearchShim:
    def test_search_module_reexports_expected_symbols(self):
        import search
        import search_pipeline

        assert callable(search.search)
        assert callable(search.search_simple)
        assert callable(search.build_fts5_query)
        assert search.search is search_pipeline.search
        assert search.search_simple is search_pipeline.search_simple
        assert search.build_fts5_query is search_pipeline.build_fts5_query

    def test_packaged_search_shim_reexports_expected_symbols(self):
        import pinpoint.search as packaged_search
        import search_pipeline

        assert callable(packaged_search.search)
        assert callable(packaged_search.search_simple)
        assert callable(packaged_search.build_fts5_query)
        assert packaged_search.search is search_pipeline.search
        assert packaged_search.search_simple is search_pipeline.search_simple
        assert packaged_search.build_fts5_query is search_pipeline.build_fts5_query


class TestCliSmoke:
    def test_cli_parser_supports_help(self, capsys):
        from pinpoint.cli import main

        try:
            main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        captured = capsys.readouterr()
        assert "Pinpoint CLI" in captured.out
        assert "search" in captured.out


class TestRunApiEntrypoint:
    def test_run_api_main_imports_app_and_calls_uvicorn(self):
        with (
            patch("api.files.scan_paths_background", lambda: None),
            patch("api.files._get_common_folders", lambda: []),
            patch("uvicorn.run") as uvicorn_run,
        ):
            runpy.run_module("run_api", run_name="__main__")

        assert uvicorn_run.call_count == 1
        args, kwargs = uvicorn_run.call_args
        assert args[0] is importlib.import_module("api").app
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 5123
        assert kwargs["log_level"] == "info"

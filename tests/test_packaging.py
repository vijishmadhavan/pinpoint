from __future__ import annotations

import importlib
import runpy
from unittest.mock import MagicMock, patch


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

    def test_cli_help_lists_start(self, capsys):
        from pinpoint.cli import main

        try:
            main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        captured = capsys.readouterr()
        assert "start" in captured.out

    def test_start_api_only_when_bot_missing(self, capsys, tmp_path):
        from pinpoint.cli import main

        api_proc = MagicMock()
        api_proc.poll.return_value = None

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._wait_for_api", return_value=True),
            patch("pinpoint.cli.shutil.which", return_value=None),
            patch("pinpoint.cli.subprocess.Popen", return_value=api_proc) as popen,
            patch("pinpoint.cli.time.sleep", side_effect=KeyboardInterrupt),
        ):
            assert main(["start"]) == 0

        captured = capsys.readouterr()
        assert "pinpoint-bot not found" in captured.out
        assert popen.call_count == 1
        api_proc.terminate.assert_called_once()

    def test_start_bot_only_requires_running_api(self, capsys, tmp_path):
        from pinpoint.cli import main

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._api_ping", return_value=False),
        ):
            assert main(["start", "--bot"]) == 1

        captured = capsys.readouterr()
        assert "API is not reachable" in captured.out

    def test_start_launches_api_and_bot_when_available(self, capsys, tmp_path):
        from pinpoint.cli import main

        api_proc = MagicMock()
        bot_proc = MagicMock()
        api_proc.poll.return_value = None
        bot_proc.poll.return_value = None

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._wait_for_api", return_value=True),
            patch("pinpoint.cli.shutil.which", return_value="/usr/bin/pinpoint-bot"),
            patch("pinpoint.cli.subprocess.Popen", side_effect=[api_proc, bot_proc]) as popen,
            patch("pinpoint.cli.time.sleep", side_effect=KeyboardInterrupt),
        ):
            assert main(["start"]) == 0

        captured = capsys.readouterr()
        assert "Starting bot via /usr/bin/pinpoint-bot" in captured.out
        assert popen.call_count == 2
        bot_proc.terminate.assert_called_once()
        api_proc.terminate.assert_called_once()


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

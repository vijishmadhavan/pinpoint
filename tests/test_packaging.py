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
        assert "doctor" in captured.out

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

    def test_child_env_includes_skills_dir(self, tmp_path):
        from pinpoint.cli import _child_env

        with patch("pinpoint.cli.user_data_dir", return_value=tmp_path):
            env = _child_env("0.0.0.0", 5123)

        assert env["PINPOINT_SKILLS_DIR"]
        assert env["PINPOINT_AUTH_DIR"].endswith("bot-auth")
        assert env["PINPOINT_QR_DIR"].endswith("qr")

    def test_status_reports_cli_paths(self, capsys, tmp_path):
        from pinpoint.cli import main

        db_path = tmp_path / "pinpoint.sqlite3"
        auth_dir = tmp_path / "bot-auth"
        qr_dir = tmp_path / "qr"
        logs_dir = tmp_path / "logs"
        env_path = tmp_path / ".env"
        auth_dir.mkdir(parents=True)
        qr_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        env_path.write_text("TZ=UTC\n", encoding="utf-8")

        class FakeConn:
            def execute(self, sql):
                class _Cursor:
                    def __init__(self, value):
                        self._value = value

                    def fetchone(self):
                        return [self._value]

                if "FROM documents" in sql:
                    return _Cursor(12)
                if "FROM background_jobs" in sql:
                    return _Cursor(2)
                if "FROM watched_folders" in sql:
                    return _Cursor(1)
                raise AssertionError(sql)

            def close(self):
                return None

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._api_ping", return_value=False),
            patch("pinpoint.cli._db_path", return_value=str(db_path)),
            patch("pinpoint.cli._bot_installed", return_value=(True, "/usr/bin/pinpoint-bot")),
            patch("database.init_db", return_value=FakeConn()),
        ):
            assert main(["status"]) == 0

        captured = capsys.readouterr()
        assert "Config:" in captured.out
        assert "Bot: installed (/usr/bin/pinpoint-bot)" in captured.out
        assert "Bot auth dir:" in captured.out
        assert "QR dir:" in captured.out
        assert "Logs dir:" in captured.out

    def test_doctor_reports_missing_setup(self, capsys, tmp_path):
        from pinpoint.cli import main

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._db_path", return_value=str(tmp_path / "pinpoint.sqlite3")),
            patch("pinpoint.cli._api_ping", return_value=False),
            patch("pinpoint.cli._bot_installed", return_value=(False, "")),
        ):
            assert main(["doctor"]) == 1

        captured = capsys.readouterr()
        assert "Pinpoint doctor" in captured.out
        assert "[NO ] config file:" in captured.out
        assert "run `pinpoint setup`" in captured.out

    def test_doctor_reports_usable_setup(self, capsys, tmp_path):
        from pinpoint.cli import DEFAULT_ENV, main

        (tmp_path / "logs").mkdir(parents=True)
        (tmp_path / "bot-auth").mkdir(parents=True)
        (tmp_path / "qr").mkdir(parents=True)
        (tmp_path / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in {**DEFAULT_ENV, "GEMINI_API_KEY": "test-key"}.items()) + "\n",
            encoding="utf-8",
        )

        with (
            patch("pinpoint.cli.user_data_dir", return_value=tmp_path),
            patch("pinpoint.cli._db_path", return_value=str(tmp_path / "pinpoint.sqlite3")),
            patch("pinpoint.cli._api_ping", return_value=True),
            patch("pinpoint.cli._bot_installed", return_value=(True, "/usr/bin/pinpoint-bot")),
        ):
            assert main(["doctor"]) == 0

        captured = capsys.readouterr()
        assert "[OK ] config file:" in captured.out
        assert "[OK ] API ping:" in captured.out
        assert "[OK ] bot command:" in captured.out
        assert "Core setup looks usable." in captured.out


class TestSkillsPackaging:
    def test_skills_package_exists(self):
        import skills

        assert skills is not None

    def test_packaged_skills_dir_contains_core_rules(self):
        from pinpoint.cli import _skills_dir

        skills_dir = _skills_dir()
        assert (skills_dir / "core-rules.md").exists()


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

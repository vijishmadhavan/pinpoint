from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_cli_session_meta_round_trip(tmp_path):
    from pinpoint import cli_chat

    meta_path = tmp_path / "cli_sessions.json"
    session_id = cli_chat.create_cli_session("Invoice lookup", path=meta_path)
    cli_chat.touch_cli_session(session_id, title="Invoice lookup", path=meta_path)
    assert cli_chat.rename_cli_session(session_id, "Invoices", path=meta_path) is True

    sessions = cli_chat.get_recent_cli_sessions(path=meta_path)
    assert sessions
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["title"] == "Invoices"


def test_resolve_cli_session_can_resume_specific_session(tmp_path):
    from pinpoint import cli_chat

    meta_path = tmp_path / "cli_sessions.json"
    first = cli_chat.create_cli_session("First", path=meta_path)
    second = cli_chat.create_cli_session("Second", path=meta_path)

    resolved = cli_chat.resolve_cli_session(resume_id=first, path=meta_path)
    assert resolved == first

    recent = cli_chat.get_recent_cli_sessions(path=meta_path)
    assert recent[0]["session_id"] == first
    assert second in {item["session_id"] for item in recent}


def test_cli_history_and_reset_use_conversation_tables(tmp_path):
    from database import init_db
    from pinpoint import cli_chat

    db_path = str(tmp_path / "test.db")
    init_db(db_path).close()

    with patch.object(cli_chat, "DB_PATH", db_path):
        cli_chat._save_message("cli:test", "user", "find invoice 4821")
        cli_chat._save_message("cli:test", "assistant", "Top match: invoice_4821.pdf")

        history = cli_chat.format_history("cli:test")
        assert "find invoice 4821" in history
        assert "invoice_4821.pdf" in history

        deleted = cli_chat.reset_cli_session("cli:test")
        assert deleted == 2
        assert cli_chat.format_history("cli:test") == "No history."


def test_render_results_and_open_out_of_range():
    from pinpoint import cli_chat

    results = [
        {
            "source": "documents",
            "title": "Invoice March",
            "path": "/tmp/invoice-march.pdf",
            "snippet": "March invoice for customer 4821",
        }
    ]
    rendered = cli_chat.render_results(results)
    assert "Invoice March" in rendered
    assert "March invoice" in rendered
    assert "/tmp/invoice-march.pdf" in rendered

    ok, message = cli_chat.open_result(results, 2)
    assert ok is False
    assert "out of range" in message

    ok, message = cli_chat.reveal_result(results, 2)
    assert ok is False
    assert "out of range" in message


def test_format_status_reads_local_db(tmp_path):
    from database import init_db
    from pinpoint import cli_chat

    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO background_jobs(job_type, target_path, target_hash, status, current_stage, total_items, completed_items, details_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        ("folder_index", "/tmp/docs", "", "running", "indexing", 10, 3, "{}"),
    )
    conn.execute(
        "INSERT INTO watched_folders(path, added_at) VALUES (?, datetime('now'))",
        ("/tmp/docs",),
    )
    conn.commit()
    conn.close()

    with (
        patch.object(cli_chat, "DB_PATH", db_path),
        patch("pinpoint.cli._api_ping", return_value=False),
    ):
        text = cli_chat.format_status()

    assert "API: stopped" in text
    assert "Watched folders: 1" in text
    assert "Active jobs: 1" in text


def test_index_path_rejects_missing_folder():
    from pinpoint import cli_chat

    text = cli_chat.index_path("/tmp/definitely-not-a-real-pinpoint-folder")
    assert "Not a directory" in text

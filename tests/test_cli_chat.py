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

    ok, message = cli_chat.open_result(results, 2)
    assert ok is False
    assert "out of range" in message

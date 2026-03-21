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

    cli_chat.set_send_target("12345@s.whatsapp.net", path=meta_path)
    assert cli_chat.get_send_target(path=meta_path) == "12345@s.whatsapp.net"


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


def test_choose_resume_session_defaults_to_latest(tmp_path):
    from pinpoint import cli_chat

    meta_path = tmp_path / "cli_sessions.json"
    latest = cli_chat.create_cli_session("Latest", path=meta_path)
    older = cli_chat.create_cli_session("Older", path=meta_path)
    cli_chat.resolve_cli_session(resume_id=latest, path=meta_path)

    outputs = []
    picked = cli_chat.choose_resume_session(
        path=meta_path,
        input_fn=lambda _prompt="": "",
        output_fn=outputs.append,
    )
    assert picked == latest
    assert any("Recent CLI sessions:" in line for line in outputs)


def test_startup_banner_mentions_title_and_session():
    from pinpoint import cli_chat

    banner = cli_chat.startup_banner(cli_chat.ChatState(session_id="cli:abc", title="Invoices"), resumed=True)
    assert "Pinpoint chat — Invoices" in banner
    assert "Resumed session: cli:abc" in banner
    assert "Type /help for commands." in banner


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


def test_send_result_queues_file(tmp_path):
    from database import init_db
    from pinpoint import cli_chat

    db_path = str(tmp_path / "test.db")
    init_db(db_path).close()
    meta_path = tmp_path / "cli_sessions.json"
    file_path = tmp_path / "invoice.pdf"
    file_path.write_text("dummy", encoding="utf-8")
    results = [{"source": "documents", "path": str(file_path), "title": "Invoice"}]

    with (
        patch.object(cli_chat, "DB_PATH", db_path),
        patch.object(cli_chat, "SESSION_META_PATH", meta_path),
    ):
        cli_chat.set_send_target("12345@s.whatsapp.net", path=meta_path)
        ok, message = cli_chat.send_result(results, 1, path=meta_path)
        assert ok is True
        assert "Queued" in message

        conn = init_db(db_path)
        row = conn.execute("SELECT chat_jid, file_path, status FROM outgoing_file_queue").fetchone()
        conn.close()

    assert row["chat_jid"] == "12345@s.whatsapp.net"
    assert row["file_path"] == str(file_path)
    assert row["status"] == "pending"


def test_quick_chat_response_handles_greeting():
    from pinpoint import cli_chat

    text = cli_chat._quick_chat_response("hi")
    assert text is not None
    assert "search files" in text.lower()


def test_answer_query_uses_quick_chat_when_agent_unavailable(tmp_path):
    from database import init_db
    from pinpoint import cli_chat

    db_path = str(tmp_path / "test.db")
    init_db(db_path).close()
    state = cli_chat.ChatState(session_id="cli:test-hello", title="New CLI chat")

    with (
        patch.object(cli_chat, "DB_PATH", db_path),
        patch.object(cli_chat, "_answer_via_node_agent", side_effect=RuntimeError("unavailable")),
    ):
        text, results = cli_chat.answer_query("hi", {}, state, save=False)

    assert "help you search files" in text.lower()
    assert results == []

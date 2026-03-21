from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from database import DB_PATH, init_db
from pinpoint import user_data_dir
from search import search

CLI_SESSION_PREFIX = "cli:"
SESSION_META_PATH = user_data_dir() / "cli_sessions.json"
MAX_HISTORY_MESSAGES = 20


@dataclass
class ChatState:
    session_id: str
    title: str
    last_results: list[dict] = field(default_factory=list)


def _load_session_meta(path: Path = SESSION_META_PATH) -> dict:
    if not path.exists():
        return {"last_session_id": "", "sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_session_id": "", "sessions": {}}
    if not isinstance(data, dict):
        return {"last_session_id": "", "sessions": {}}
    data.setdefault("last_session_id", "")
    data.setdefault("sessions", {})
    return data


def _save_session_meta(data: dict, path: Path = SESSION_META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_send_target(path: Path = SESSION_META_PATH) -> str:
    meta = _load_session_meta(path)
    return str(meta.get("send_target_chat_jid") or "").strip()


def set_send_target(chat_jid: str, path: Path = SESSION_META_PATH) -> str:
    meta = _load_session_meta(path)
    meta["send_target_chat_jid"] = chat_jid.strip()
    _save_session_meta(meta, path)
    return meta["send_target_chat_jid"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_cli_session(title: str | None = None, path: Path = SESSION_META_PATH) -> str:
    session_id = f"{CLI_SESSION_PREFIX}{uuid.uuid4().hex[:12]}"
    meta = _load_session_meta(path)
    display = (title or "New CLI chat").strip() or "New CLI chat"
    meta["sessions"][session_id] = {"title": display, "created_at": _now_iso(), "updated_at": _now_iso()}
    meta["last_session_id"] = session_id
    _save_session_meta(meta, path)
    return session_id


def touch_cli_session(session_id: str, title: str | None = None, path: Path = SESSION_META_PATH) -> None:
    meta = _load_session_meta(path)
    session = meta["sessions"].setdefault(session_id, {"title": title or "CLI chat", "created_at": _now_iso()})
    if title:
        session["title"] = title
    session["updated_at"] = _now_iso()
    meta["last_session_id"] = session_id
    _save_session_meta(meta, path)


def get_recent_cli_sessions(limit: int = 10, path: Path = SESSION_META_PATH) -> list[dict]:
    meta = _load_session_meta(path)
    sessions = []
    for session_id, info in meta.get("sessions", {}).items():
        sessions.append(
            {
                "session_id": session_id,
                "title": info.get("title") or "CLI chat",
                "created_at": info.get("created_at") or "",
                "updated_at": info.get("updated_at") or "",
            }
        )
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions[:limit]


def rename_cli_session(session_id: str, title: str, path: Path = SESSION_META_PATH) -> bool:
    meta = _load_session_meta(path)
    session = meta.get("sessions", {}).get(session_id)
    if not session:
        return False
    session["title"] = (title or "").strip() or session.get("title") or "CLI chat"
    session["updated_at"] = _now_iso()
    meta["last_session_id"] = session_id
    _save_session_meta(meta, path)
    return True


def choose_resume_session(
    path: Path = SESSION_META_PATH,
    *,
    input_fn=input,
    output_fn=print,
    limit: int = 5,
) -> str | None:
    sessions = get_recent_cli_sessions(limit=limit, path=path)
    if not sessions:
        return None
    if len(sessions) == 1:
        return sessions[0]["session_id"]

    output_fn("Recent CLI sessions:")
    for i, item in enumerate(sessions, 1):
        output_fn(f"{i}. {item['title']} [{item['session_id']}] ({item['updated_at']})")
    output_fn("Press Enter to resume the latest session, or type a number.")
    choice = (input_fn("Resume> ") or "").strip()
    if not choice:
        return sessions[0]["session_id"]
    try:
        idx = int(choice)
    except ValueError:
        return sessions[0]["session_id"]
    if 1 <= idx <= len(sessions):
        return sessions[idx - 1]["session_id"]
    return sessions[0]["session_id"]


def resolve_cli_session(
    new: bool = False,
    resume: bool = False,
    resume_id: str | None = None,
    path: Path = SESSION_META_PATH,
) -> str:
    meta = _load_session_meta(path)
    if resume_id:
        if resume_id in meta.get("sessions", {}):
            meta["last_session_id"] = resume_id
            _save_session_meta(meta, path)
            touch_cli_session(resume_id, path=path)
            return resume_id
        return create_cli_session(path=path)
    if new or not meta.get("last_session_id") or meta["last_session_id"] not in meta.get("sessions", {}):
        return create_cli_session(path=path)
    if resume:
        return meta["last_session_id"]
    return meta["last_session_id"]


def startup_banner(state: ChatState, *, resumed: bool = False) -> str:
    action = "Resumed" if resumed else "Started"
    return "\n".join(
        [
            f"Pinpoint chat — {state.title}",
            f"{action} session: {state.session_id}",
            "Try: find invoice 4821",
            "Type /help for commands.",
        ]
    )


def _save_message(session_id: str, role: str, content: str) -> None:
    conn = init_db(DB_PATH)
    now = _now_iso()
    conn.execute(
        "INSERT INTO conversations(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content.strip(), now),
    )
    conn.execute(
        """
        INSERT INTO conversation_sessions(session_id, created_at, updated_at, message_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at, message_count = message_count + 1
        """,
        (session_id, now, now),
    )
    conn.commit()
    conn.close()


def guess_recent_whatsapp_chat() -> str:
    conn = init_db(DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT session_id
            FROM conversation_sessions
            WHERE session_id NOT LIKE 'cli:%' AND instr(session_id, '@') > 0
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        return str(row["session_id"]) if row else ""
    finally:
        conn.close()


def queue_outgoing_file(chat_jid: str, file_path: str, caption: str = "") -> int:
    conn = init_db(DB_PATH)
    try:
        now = _now_iso()
        cursor = conn.execute(
            """
            INSERT INTO outgoing_file_queue(chat_jid, file_path, caption, status, error, created_at, claimed_at, completed_at)
            VALUES (?, ?, ?, 'pending', '', ?, '', '')
            """,
            (chat_jid, os.path.abspath(file_path), caption.strip(), now),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _load_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    conn = init_db(DB_PATH)
    rows = conn.execute(
        """
        SELECT role, content, timestamp FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def reset_cli_session(session_id: str) -> int:
    conn = init_db(DB_PATH)
    cursor = conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    deleted = cursor.rowcount
    conn.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return deleted


def _truncate(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def format_status() -> str:
    from pinpoint.cli import _api_ping

    conn = init_db(DB_PATH)
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        job_count = conn.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE status IN ('pending', 'running', 'cancelling')"
        ).fetchone()[0]
        watch_count = conn.execute("SELECT COUNT(*) FROM watched_folders").fetchone()[0]
    finally:
        conn.close()
    api_running = _api_ping()
    return (
        f"API: {'running' if api_running else 'stopped'}\n"
        f"Documents: {doc_count}\n"
        f"Watched folders: {watch_count}\n"
        f"Active jobs: {job_count}"
    )


def index_path(path: str) -> str:
    from indexer import index_folder

    target = os.path.abspath(path)
    if not os.path.isdir(target):
        return f"Not a directory: {target}"
    result = index_folder(target, DB_PATH)
    return (
        f"Indexed {target}\n"
        f"Indexed: {result.get('indexed', 0)}\n"
        f"Skipped: {result.get('skipped', 0)}\n"
        f"Failed: {result.get('failed', 0)}"
    )


def watch_path(path: str) -> str:
    try:
        from api.files import WatchFolderRequest, watch_folder_endpoint
    except Exception as exc:
        return f"Watch folders unavailable: {exc}"

    target = os.path.abspath(path)
    try:
        result = watch_folder_endpoint(WatchFolderRequest(path=target))
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        return f"Could not watch {target}: {detail}"
    return result.get("_hint") or f"Watching {target}"


def _terminal_link(path: str) -> str:
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        file_url = "file:///" + abs_path.replace("\\", "/")
    else:
        file_url = "file://" + abs_path
    return f"\033]8;;{file_url}\033\\{abs_path}\033]8;;\033\\"


def _cli_agent_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "bot" / "bin" / "pinpoint-cli-agent.js"


def _child_env(env: dict[str, str]) -> dict[str, str]:
    child = os.environ.copy()
    child.update(env)
    user_dir = str(user_data_dir())
    child.setdefault("PINPOINT_USER_DIR", user_dir)
    child.setdefault("PINPOINT_ENV_PATH", str(user_data_dir() / ".env"))
    child.setdefault("PINPOINT_LOG_DIR", str(user_data_dir() / "logs"))
    child.setdefault("PINPOINT_AUTH_DIR", str(user_data_dir() / "bot-auth"))
    child.setdefault("PINPOINT_QR_DIR", str(user_data_dir() / "qr"))
    child.setdefault("PINPOINT_API_URL", child.get("PINPOINT_API_URL", "http://127.0.0.1:5123"))
    child["PINPOINT_SILENT_STDOUT"] = "1"
    return child


def _answer_via_node_agent(
    query: str,
    env: dict[str, str],
    session_id: str,
    *,
    event_callback=None,
) -> tuple[str, list[dict]]:
    script = _cli_agent_script_path()
    if not script.exists():
        raise FileNotFoundError(f"CLI agent runner not found: {script}")
    proc = subprocess.Popen(
        ["node", str(script), "--session", session_id, "--message", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(env),
    )
    stderr_lines: list[str] = []

    def _pump_stderr() -> None:
        if not proc.stderr:
            return
        for raw_line in proc.stderr:
            line = raw_line.rstrip("\n")
            if line.startswith("EVENT "):
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event_callback:
                    try:
                        event_callback(event)
                    except Exception:
                        pass
            else:
                stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_pump_stderr, daemon=True)
    stderr_thread.start()
    stdout_text, _ = proc.communicate()
    stderr_thread.join(timeout=1.0)
    if proc.returncode != 0:
        err_text = "\n".join(s for s in stderr_lines if s).strip() or f"CLI agent exited with {proc.returncode}"
        raise RuntimeError(err_text)

    data = json.loads((stdout_text or "").strip() or "{}")
    events = data.get("events") or []
    parts = []
    for item in events:
        if item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]).strip())
    final_text = str(data.get("text") or "").strip()
    if final_text:
        parts.append(final_text)
    return ("\n".join(p for p in parts if p).strip() or "No response.", data.get("results") or [])


def _quick_chat_response(query: str) -> str | None:
    q = (query or "").strip().lower()
    if not q:
        return None
    if q in {"hi", "hello", "hey", "yo", "hiya"}:
        return "Hi. I can help you search files, inspect results, and run Pinpoint workflows from the terminal."
    if q in {"thanks", "thank you", "thx"}:
        return "You're welcome."
    if q in {"what can you do?", "what can you do", "help me", "help"}:
        return (
            "I can search your indexed files, open or reveal results, queue files to WhatsApp, and run Pinpoint file workflows. "
            "Try: find invoice 4821, cull the wedding folder, or search this Excel for a phone number."
        )
    return None


def _retrieve_context(query: str, limit: int = 5) -> tuple[dict, list[dict]]:
    from api.memory import _memory_fts_search
    from api.search import _detect_retrieval_intent, _document_overview, _search_facts

    conn = init_db(DB_PATH)
    intent = _detect_retrieval_intent(query)
    payload: dict = {"query": query, "intent": intent, "results": [], "overview": None}
    results: list[dict] = []
    if intent == "memory":
        results = [{"source": "memory", **r} for r in _memory_fts_search(conn, query, limit=min(limit, 5))]
    elif intent == "facts":
        results = [{"source": "facts", **r} for r in _search_facts(conn, query, limit=min(limit, 5))]
    if not results:
        doc_results = search(query, DB_PATH, limit=limit).get("results", [])
        results = [{"source": "documents", **r} for r in doc_results]
        if doc_results:
            try:
                payload["overview"] = _document_overview(conn, int(doc_results[0]["id"]), query=query)
            except Exception:
                payload["overview"] = None
    payload["results"] = results[:limit]
    conn.close()
    return payload, results[:limit]


def _llm_answer(query: str, context: dict, history: list[dict], gemini_key: str, model: str) -> str:
    from google import genai

    history_text = "\n".join(f"{m['role']}: {_truncate(m['content'], 180)}" for m in history[-6:])
    result_lines = []
    for i, item in enumerate(context.get("results", [])[:5], 1):
        source = item.get("source", "documents")
        if source == "documents":
            result_lines.append(
                f"{i}. [doc] {item.get('title') or item.get('filename') or item.get('path')} :: {_truncate(item.get('snippet') or '')}"
            )
        elif source == "facts":
            result_lines.append(f"{i}. [fact] {item.get('fact_text')} ({item.get('path') or ''})")
        else:
            result_lines.append(f"{i}. [memory] {item.get('fact')}")
    overview = context.get("overview") or {}
    overview_text = ""
    if overview:
        sections = overview.get("top_sections") or []
        facts = overview.get("facts") or []
        overview_text = (
            f"Top document: {overview.get('title') or overview.get('path')}\n"
            f"Overview: {_truncate(overview.get('overview') or '', 500)}\n"
            f"Sections: " + " | ".join(_truncate(s.get("preview") or "", 160) for s in sections[:3]) + "\n"
            "Facts: " + " | ".join(_truncate(f.get('fact_text') or '', 120) for f in facts[:4])
        )

    prompt = f"""Answer the user's request using the retrieved local Pinpoint context.
Be concise and practical. If the answer is uncertain, say that and point to the most relevant file.
Do not mention internal pipelines or implementation details.

Recent session context:
{history_text or "(none)"}

User request:
{query}

Retrieved results:
{chr(10).join(result_lines) or "(none)"}

{overview_text}
"""
    client = genai.Client(api_key=gemini_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def answer_query(
    query: str,
    env: dict[str, str],
    state: ChatState,
    save: bool = True,
    *,
    event_callback=None,
) -> tuple[str, list[dict]]:
    try:
        text, results = _answer_via_node_agent(query, env, state.session_id, event_callback=event_callback)
        if save:
            _save_message(state.session_id, "user", query)
            _save_message(state.session_id, "assistant", text)
            if state.title == "New CLI chat":
                state.title = _truncate(query, 60)
            touch_cli_session(state.session_id, title=state.title)
        state.last_results = results
        return text, results
    except Exception:
        pass

    quick = _quick_chat_response(query)
    if quick:
        if save:
            _save_message(state.session_id, "user", query)
            _save_message(state.session_id, "assistant", quick)
            if state.title == "New CLI chat":
                state.title = _truncate(query, 60)
            touch_cli_session(state.session_id, title=state.title)
        state.last_results = []
        return quick, []

    history = _load_history(state.session_id, limit=MAX_HISTORY_MESSAGES)
    context, results = _retrieve_context(query)
    gemini_key = env.get("GEMINI_API_KEY", "").strip()
    model = env.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    if gemini_key:
        try:
            text = _llm_answer(query, context, history, gemini_key, model)
        except Exception:
            text = ""
    else:
        text = ""

    if not text:
        if not results:
            text = "I couldn't find anything relevant in your indexed files."
        else:
            top = results[0]
            if top.get("source") == "memory":
                text = top.get("fact") or "I found a matching memory."
            elif top.get("source") == "facts":
                text = top.get("fact_text") or "I found a matching fact."
            else:
                name = top.get("title") or top.get("filename") or top.get("path")
                snippet = _truncate(top.get("snippet") or "", 180)
                text = f"Top match: {name}"
                if snippet:
                    text += f"\n{snippet}"

    if save:
        _save_message(state.session_id, "user", query)
        _save_message(state.session_id, "assistant", text)
        if state.title == "New CLI chat":
            state.title = _truncate(query, 60)
        touch_cli_session(state.session_id, title=state.title)
    state.last_results = results
    return text, results


def render_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["", "Results:"]
    for i, item in enumerate(results[:5], 1):
        source = item.get("source", "documents")
        if source == "documents":
            name = item.get("title") or item.get("filename") or item.get("path")
            lines.append(f"{i}. {name}")
            if item.get("path"):
                lines.append(f"   {_terminal_link(item['path'])}")
            snippet = _truncate(item.get("snippet") or "", 140)
            if snippet:
                lines.append(f"   {snippet}")
        elif source == "facts":
            lines.append(f"{i}. {item.get('fact_text')}")
        else:
            lines.append(f"{i}. {item.get('fact')}")
    return "\n".join(lines)


def format_history(session_id: str, limit: int = 10) -> str:
    history = _load_history(session_id, limit=limit)
    if not history:
        return "No history."
    lines = []
    for item in history[-limit:]:
        role = "You" if item["role"] == "user" else "Pinpoint"
        lines.append(f"{role}: {_truncate(item['content'], 120)}")
    return "\n".join(lines)


def format_sessions(limit: int = 10, path: Path = SESSION_META_PATH) -> str:
    sessions = get_recent_cli_sessions(limit=limit, path=path)
    if not sessions:
        return "No saved CLI sessions."
    lines = []
    for item in sessions:
        lines.append(f"{item['session_id']}  {item['title']}  ({item['updated_at']})")
    return "\n".join(lines)


def open_result(results: list[dict], index: int) -> tuple[bool, str]:
    if index < 1 or index > len(results):
        return False, f"Result {index} is out of range."
    item = results[index - 1]
    path = item.get("path")
    if not path:
        return False, "That result has no openable path."
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True, f"Opened {path}"
    except Exception as exc:
        return False, f"Could not open {path}: {exc}"


def reveal_result(results: list[dict], index: int) -> tuple[bool, str]:
    if index < 1 or index > len(results):
        return False, f"Result {index} is out of range."
    item = results[index - 1]
    path = item.get("path")
    if not path:
        return False, "That result has no revealable path."
    abs_path = os.path.abspath(path)
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", abs_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", abs_path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(abs_path)])
        return True, f"Opened containing folder for {abs_path}"
    except Exception as exc:
        return False, f"Could not reveal {abs_path}: {exc}"


def send_result(
    results: list[dict],
    index: int,
    *,
    chat_jid: str = "",
    path: Path = SESSION_META_PATH,
) -> tuple[bool, str]:
    if index < 1 or index > len(results):
        return False, f"Result {index} is out of range."
    item = results[index - 1]
    file_path = item.get("path")
    if not file_path:
        return False, "That result has no sendable path."
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return False, f"File not found: {abs_path}"

    target = (chat_jid or get_send_target(path) or guess_recent_whatsapp_chat()).strip()
    if not target:
        return False, "No WhatsApp send target is set. Use /send-target CHAT_JID first."

    queue_id = queue_outgoing_file(target, abs_path, caption=os.path.basename(abs_path))
    return True, f"Queued {abs_path} to {target} (queue #{queue_id})."


def cli_help() -> str:
    return "\n".join(
        [
            "Commands:",
            "/help           Show this help",
            "/reset          Clear the current session",
            "/history        Show recent turns",
            "/sessions       List recent CLI sessions",
            "/resume ID      Switch to a saved session",
            "/rename NAME    Rename the current session",
            "/status         Show local Pinpoint status",
            "/index PATH     Index a folder into the local database",
            "/watch PATH     Start watching a folder for background indexing",
            "/open N         Open result N from the latest search",
            "/reveal N       Open the containing folder for result N",
            "/send-target    Show current WhatsApp send target",
            "/send-target JID Set the WhatsApp JID used by /send",
            "/send N         Queue result N to be sent to WhatsApp",
            "/quit           Exit chat",
            "",
            'Try: find invoice 4821',
        ]
    )


def run_chat_loop(
    env: dict[str, str],
    *,
    new: bool = False,
    resume: bool = False,
    resume_id: str | None = None,
    initial_message: str | None = None,
) -> int:
    if resume and not resume_id and sys.stdin.isatty():
        selected = choose_resume_session()
        if selected:
            resume_id = selected
    session_id = resolve_cli_session(new=new, resume=resume, resume_id=resume_id)
    meta = _load_session_meta()
    title = meta.get("sessions", {}).get(session_id, {}).get("title") or "New CLI chat"
    state = ChatState(session_id=session_id, title=title)

    print(startup_banner(state, resumed=bool(resume or resume_id)))
    print("Ctrl+C or /quit to exit.")

    last_progress = {"text": ""}

    def _handle_event(event: dict) -> None:
        kind = event.get("type")
        text = str(event.get("text") or "").strip()
        if kind == "progress" and text and text != last_progress["text"]:
            last_progress["text"] = text
            print(f"[working] {text}")

    if initial_message:
        text, results = answer_query(initial_message, env, state, event_callback=_handle_event)
        print(text)
        extra = render_results(results)
        if extra:
            print(extra)
        return 0

    while True:
        try:
            user_msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if not user_msg:
            continue
        if user_msg.startswith("/find "):
            user_msg = user_msg[len("/find ") :].strip()
            if not user_msg:
                print("Usage: /find QUERY")
                continue
        if user_msg == "/quit":
            return 0
        if user_msg == "/help":
            print(cli_help())
            continue
        if user_msg == "/reset":
            deleted = reset_cli_session(state.session_id)
            state.last_results = []
            print(f"Session reset: {deleted} message(s) cleared.")
            continue
        if user_msg == "/history":
            print(format_history(state.session_id))
            continue
        if user_msg == "/sessions":
            print(format_sessions())
            continue
        if user_msg == "/status":
            print(format_status())
            continue
        if user_msg.startswith("/resume "):
            target = user_msg.split(maxsplit=1)[1].strip()
            meta = _load_session_meta()
            if target not in meta.get("sessions", {}):
                print(f"Unknown session: {target}")
                continue
            state.session_id = target
            state.title = meta["sessions"][target].get("title") or "CLI chat"
            state.last_results = []
            touch_cli_session(state.session_id, title=state.title)
            print(f"Resumed {state.session_id} — {state.title}")
            continue
        if user_msg.startswith("/rename "):
            new_title = user_msg.split(maxsplit=1)[1].strip()
            if not new_title:
                print("Usage: /rename NAME")
                continue
            rename_cli_session(state.session_id, new_title)
            state.title = new_title
            print(f"Renamed session to: {new_title}")
            continue
        if user_msg.startswith("/index "):
            target = user_msg.split(maxsplit=1)[1].strip()
            if not target:
                print("Usage: /index PATH")
                continue
            print(index_path(target))
            continue
        if user_msg.startswith("/watch "):
            target = user_msg.split(maxsplit=1)[1].strip()
            if not target:
                print("Usage: /watch PATH")
                continue
            print(watch_path(target))
            continue
        if user_msg == "/send-target":
            target = get_send_target()
            guessed = guess_recent_whatsapp_chat()
            if target:
                print(f"Current send target: {target}")
            elif guessed:
                print(f"No explicit send target set. Latest WhatsApp chat: {guessed}")
            else:
                print("No send target set. Use /send-target CHAT_JID")
            continue
        if user_msg.startswith("/send-target "):
            target = user_msg.split(maxsplit=1)[1].strip()
            if not target or "@" not in target:
                print("Usage: /send-target CHAT_JID")
                continue
            set_send_target(target)
            print(f"Send target set to: {target}")
            continue
        if user_msg.startswith("/open "):
            try:
                index = int(user_msg.split(maxsplit=1)[1])
            except Exception:
                print("Usage: /open N")
                continue
            ok, message = open_result(state.last_results, index)
            print(message)
            continue
        if user_msg.startswith("/reveal "):
            try:
                index = int(user_msg.split(maxsplit=1)[1])
            except Exception:
                print("Usage: /reveal N")
                continue
            ok, message = reveal_result(state.last_results, index)
            print(message)
            continue
        if user_msg.startswith("/send "):
            try:
                index = int(user_msg.split(maxsplit=1)[1])
            except Exception:
                print("Usage: /send N")
                continue
            ok, message = send_result(state.last_results, index)
            print(message)
            continue

        last_progress["text"] = ""
        text, results = answer_query(user_msg, env, state, event_callback=_handle_event)
        print(text)
        extra = render_results(results)
        if extra:
            print(extra)

"""Conversation, memory, settings, and reminders endpoints."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.helpers import _get_conn

router = APIRouter()


# --- Conversation memory (Segment 13) ---


class ConversationMessage(BaseModel):
    session_id: str
    role: str
    content: str


class ConversationResetRequest(BaseModel):
    session_id: str


@router.post("/conversation")
def conversation_save(msg: ConversationMessage) -> dict:
    """Save a conversation message (user or assistant)."""
    if msg.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'assistant'")
    if not msg.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    conn = _get_conn()
    now = datetime.utcnow().isoformat()

    cursor = conn.execute(
        "INSERT INTO conversations(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (msg.session_id, msg.role, msg.content.strip(), now),
    )

    # Upsert session metadata
    conn.execute(
        """
        INSERT INTO conversation_sessions(session_id, created_at, updated_at, message_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at = excluded.updated_at,
            message_count = message_count + 1
    """,
        (msg.session_id, now, now),
    )

    conn.commit()
    return {"success": True, "id": cursor.lastrowid}


@router.get("/conversation/history")
def conversation_history(
    session_id: str = Query(..., description="Session ID (chat JID)"),
    limit: int = Query(20, ge=1, le=100, description="Max messages to return"),
) -> dict:
    """Load recent conversation history for a session."""
    conn = _get_conn()

    # Get session metadata (for idle timeout check by caller)
    session = conn.execute(
        "SELECT updated_at, message_count FROM conversation_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    if not session:
        return {"session_id": session_id, "messages": [], "updated_at": None, "message_count": 0}

    # Get last N messages ordered by timestamp
    rows = conn.execute(
        """
        SELECT role, content, timestamp FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (session_id, limit),
    ).fetchall()

    # Reverse so oldest first (chronological order)
    messages = [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    return {
        "session_id": session_id,
        "messages": messages,
        "updated_at": session["updated_at"],
        "message_count": session["message_count"],
    }


@router.post("/conversation/reset")
def conversation_reset(req: ConversationResetRequest) -> dict:
    """Delete all messages for a session (reset conversation)."""
    conn = _get_conn()

    cursor = conn.execute("DELETE FROM conversations WHERE session_id = ?", (req.session_id,))
    deleted = cursor.rowcount

    conn.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (req.session_id,))

    conn.commit()
    return {"success": True, "deleted_count": deleted}


@router.get("/conversation/search")
def conversation_search(
    q: str = Query(..., description="Search keywords"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> dict:
    """Search past conversation messages by keyword."""
    conn = _get_conn()

    if session_id:
        rows = conn.execute(
            """
            SELECT session_id, role, content, timestamp FROM conversations
            WHERE session_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (session_id, f"%{q}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT session_id, role, content, timestamp FROM conversations
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (f"%{q}%", limit),
        ).fetchall()

    results = [dict(r) for r in rows]
    return {"query": q, "count": len(results), "results": results}


# --- Persistent Memory (Segment 18F) ---


class MemorySaveRequest(BaseModel):
    fact: str
    category: str = "general"


def _memory_fts_search(conn: sqlite3.Connection, query: str, limit: int = 10, user_id: str | None = None) -> list:
    """Search memories using FTS5 BM25 (porter stemming). Falls back to LIKE if FTS fails."""
    try:
        # Build FTS query: extract words, quote each, join with OR for recall
        words = query.strip().split()
        stop = {
            "i",
            "my",
            "is",
            "am",
            "the",
            "a",
            "an",
            "to",
            "in",
            "on",
            "at",
            "for",
            "of",
            "and",
            "or",
            "that",
            "this",
            "it",
        }
        keywords = [w for w in words if w.lower() not in stop and len(w) > 1]
        if not keywords:
            keywords = [w for w in words if len(w) > 1][:3]
        if not keywords:
            return []
        fts_query = " OR ".join(f'"{k}"' for k in keywords)
        rows = conn.execute(
            """SELECT m.id, m.fact, m.category, bm25(memories_fts) AS rank
               FROM memories_fts f
               JOIN memories m ON m.id = f.rowid
               WHERE memories_fts MATCH ? AND m.superseded_by IS NULL
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        results = [{"id": r["id"], "fact": r["fact"], "category": r["category"]} for r in rows]
        if results:
            return results
    except Exception as e:
        print(f"[Memory] FTS search failed ({e}), falling back to LIKE")
    # Fallback: LIKE search
    candidates = {}
    for kw in keywords[:4]:
        rows = conn.execute(
            "SELECT id, fact, category FROM memories WHERE superseded_by IS NULL AND fact LIKE ? LIMIT 10", (f"%{kw}%",)
        ).fetchall()
        for row in rows:
            if row["id"] not in candidates:
                candidates[row["id"]] = {"id": row["id"], "fact": row["fact"], "category": row["category"]}
    return list(candidates.values())[:limit]


def _memory_find_similar(conn: sqlite3.Connection, fact: str, limit: int = 5) -> list:
    """Find existing memories similar to a new fact using FTS5 BM25."""
    return _memory_fts_search(conn, fact, limit)


def _memory_log_history(
    conn: sqlite3.Connection, memory_id: int, old_fact: str | None, new_fact: str | None, action: str
) -> None:
    """Log a memory change to the audit trail."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO memory_history(memory_id, old_fact, new_fact, action, created_at) VALUES (?, ?, ?, ?, ?)",
        (memory_id, old_fact, new_fact, action, now),
    )


def _memory_fts_sync(conn: sqlite3.Connection, memory_id: int, fact: str, delete_only: bool = False) -> None:
    """Keep memories_fts in sync with memories table."""
    try:
        conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
        if not delete_only:
            conn.execute("INSERT INTO memories_fts(rowid, fact) VALUES (?, ?)", (memory_id, fact))
    except Exception:
        pass  # FTS sync is best-effort


def _memory_decide_with_llm(new_fact: str, new_category: str, existing: list[dict]) -> dict:
    """Use Gemini flash-lite to decide how new fact relates to existing memories.
    Returns: {"action": "ADD|UPDATE|DELETE|NONE", "target_id": int|null, "merged_text": str|null}
    """
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY or not existing:
        return {"action": "ADD", "target_id": None, "merged_text": None}

    # Map real IDs to integers (anti-hallucination)
    id_map = {}
    display = []
    for i, m in enumerate(existing):
        id_map[str(i)] = m["id"]
        display.append(f'{{"id": {i}, "text": "{m["fact"]}", "category": "{m["category"]}"}}')

    prompt = f"""You manage a personal memory store. A new fact is being saved. Compare it to existing memories and decide what to do.

Existing memories:
[{", ".join(display)}]

New fact: "{new_fact}" (category: {new_category})

Decide ONE action:
- NONE: New fact is semantically the same as an existing memory. Don't save. Example: "Likes pizza" and "Loves pizza" = same meaning.
- ADD: New fact is genuinely new, not related to any existing memory.
- UPDATE: New fact is about the same subject but more specific/detailed. Replace the old memory. Example: "Likes hiking" → "Likes hiking in the Western Ghats on weekends".
- MERGE: New fact adds complementary info to an existing memory. Combine them. Example: "Likes cheese pizza" + "Likes chicken pizza" → "Likes cheese and chicken pizza".
- DELETE: New fact directly contradicts an existing memory. Remove old, save new. Example: "Loves pizza" → "Dislikes pizza".

Decide the best action."""

    _memory_decision_schema = {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": ["ADD", "UPDATE", "MERGE", "DELETE", "NONE"]},
            "target_id": {"type": "INTEGER", "nullable": True},
            "merged_text": {"type": "STRING", "nullable": True},
        },
        "required": ["action"],
    }

    try:
        from google.genai import types as genai_types

        from extractors import _get_gemini, gemini_call_with_retry

        client = _get_gemini()
        if not client:
            return {"action": "ADD", "target_id": None, "merged_text": None}
        response = gemini_call_with_retry(
            client,
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_memory_decision_schema,
            ),
        )

        import json as _json

        decision = _json.loads(response.text)

        # Map integer ID back to real DB ID
        if decision.get("target_id") is not None:
            real_id = id_map.get(str(decision["target_id"]))
            if real_id is None:
                return {"action": "ADD", "target_id": None, "merged_text": None}
            decision["target_id"] = real_id

        if decision.get("action") not in ("ADD", "UPDATE", "MERGE", "DELETE", "NONE"):
            return {"action": "ADD", "target_id": None, "merged_text": None}

        return decision
    except Exception as e:
        print(f"[Memory] LLM decision failed ({e}), falling back to ADD")
        return {"action": "ADD", "target_id": None, "merged_text": None}


@router.post("/memory")
def memory_save(req: MemorySaveRequest) -> dict:
    """Save a personal fact to persistent memory. Uses LLM to detect duplicates, merge related facts, and handle contradictions."""
    if not req.fact.strip():
        raise HTTPException(status_code=400, detail="fact cannot be empty")
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    category = req.category.strip().lower()

    # Step 1: Find similar existing memories
    similar = _memory_find_similar(conn, req.fact)

    # Step 2: LLM decides what to do (or fallback to ADD if no similar)
    decision = _memory_decide_with_llm(req.fact.strip(), category, similar)
    action = decision.get("action", "ADD")
    target_id = decision.get("target_id")
    merged_text = decision.get("merged_text")

    result = {"success": True, "action": action}

    if action == "NONE":
        result["message"] = "Already remembered (semantically equivalent)"
        return result

    elif action == "UPDATE" and target_id:
        old_row = conn.execute("SELECT fact FROM memories WHERE id = ?", (target_id,)).fetchone()
        new_text = merged_text or req.fact.strip()
        conn.execute("UPDATE memories SET fact = ?, updated_at = ? WHERE id = ?", (new_text, now, target_id))
        _memory_log_history(conn, target_id, old_row["fact"] if old_row else None, new_text, "UPDATE")
        _memory_fts_sync(conn, target_id, new_text)
        conn.commit()
        result["id"] = target_id
        result["updated_text"] = new_text
        return result

    elif action == "MERGE" and target_id:
        old_row = conn.execute("SELECT fact FROM memories WHERE id = ?", (target_id,)).fetchone()
        new_text = merged_text or req.fact.strip()
        conn.execute("UPDATE memories SET fact = ?, updated_at = ? WHERE id = ?", (new_text, now, target_id))
        _memory_log_history(conn, target_id, old_row["fact"] if old_row else None, new_text, "MERGE")
        _memory_fts_sync(conn, target_id, new_text)
        conn.commit()
        result["id"] = target_id
        result["merged_text"] = new_text
        return result

    elif action == "DELETE" and target_id:
        old_row = conn.execute("SELECT fact FROM memories WHERE id = ?", (target_id,)).fetchone()
        conn.execute("UPDATE memories SET superseded_by = -1 WHERE id = ?", (target_id,))
        _memory_fts_sync(conn, target_id, "", delete_only=True)
        _memory_log_history(conn, target_id, old_row["fact"] if old_row else None, req.fact.strip(), "DELETE")
        cursor = conn.execute(
            "INSERT INTO memories(fact, category, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (req.fact.strip(), category, now, now),
        )
        new_id = cursor.lastrowid
        _memory_fts_sync(conn, new_id, req.fact.strip())
        _memory_log_history(conn, new_id, None, req.fact.strip(), "ADD")
        conn.commit()
        result["id"] = new_id
        result["superseded_id"] = target_id
        return result

    else:
        # ADD
        cursor = conn.execute(
            "INSERT INTO memories(fact, category, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (req.fact.strip(), category, now, now),
        )
        new_id = cursor.lastrowid
        _memory_fts_sync(conn, new_id, req.fact.strip())
        _memory_log_history(conn, new_id, None, req.fact.strip(), "ADD")
        conn.commit()
        result["id"] = new_id
        return result


@router.get("/memory/search")
def memory_search(
    q: str = Query(..., description="Search keywords"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Search persistent memories using FTS5 BM25."""
    conn = _get_conn()
    results = _memory_fts_search(conn, q, limit)
    return {"query": q, "count": len(results), "results": results}


@router.get("/memory/list")
def memory_list(
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """List all persistent memories (optionally filter by category)."""
    conn = _get_conn()
    if category:
        rows = conn.execute(
            "SELECT id, fact, category, created_at FROM memories WHERE category = ? AND superseded_by IS NULL ORDER BY updated_at DESC LIMIT ?",
            (category.lower(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, fact, category, created_at FROM memories WHERE superseded_by IS NULL ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"count": len(rows), "memories": [dict(r) for r in rows]}


@router.delete("/memory/{memory_id}")
def memory_delete(memory_id: int) -> dict:
    """Delete a memory by ID."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"success": True, "deleted_id": memory_id}


class MemoryForgetRequest(BaseModel):
    description: str


@router.post("/memory/forget")
def memory_forget(req: MemoryForgetRequest) -> dict:
    """Forget a memory by description — uses FTS5 to find best match and deletes it."""
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description cannot be empty")
    conn = _get_conn()
    # Use FTS5 to find matching memories
    candidates = _memory_fts_search(conn, req.description, limit=5)

    if not candidates:
        return {"success": False, "error": "No matching memory found", "searched_for": req.description}

    # Pick best match (FTS5 returns in BM25 rank order, first is best)
    best = candidates[0]
    _memory_log_history(conn, best["id"], best["fact"], None, "FORGET")
    _memory_fts_sync(conn, best["id"], "", delete_only=True)
    conn.execute("DELETE FROM memories WHERE id = ?", (best["id"],))
    conn.commit()
    return {"success": True, "deleted_id": best["id"], "deleted_fact": best["fact"]}


@router.get("/memory/context")
def memory_context(
    q: str | None = Query(None, description="Current user message for query-relevant retrieval"),
) -> dict:
    """Get active memories for system prompt. Static always included, dynamic filtered by relevance if query given."""
    conn = _get_conn()
    STATIC = ["preferences", "people", "places"]

    # Always include static memories (preferences/people/places)
    static_rows = conn.execute(
        "SELECT fact, category FROM memories WHERE superseded_by IS NULL AND category IN ('preferences', 'people', 'places') ORDER BY category, updated_at DESC"
    ).fetchall()

    lines = []
    groups = {}
    for r in static_rows:
        cat = r["category"] or "general"
        groups.setdefault(cat, []).append(r["fact"])
    for cat in STATIC:
        if cat in groups:
            lines.append(f"[{cat}]")
            for f in groups[cat]:
                lines.append(f"- {f}")
    static_count = len(static_rows)

    # Dynamic memories: query-relevant (FTS5) if query given, else all
    if q and q.strip():
        dynamic = _memory_fts_search(conn, q, limit=10)
        # Filter out static categories (already included above)
        dynamic = [d for d in dynamic if d.get("category", "general") not in STATIC]
        if dynamic:
            lines.append("[relevant]")
            for d in dynamic:
                lines.append(f"- {d['fact']}")
        total_count = static_count + len(dynamic)
    else:
        # No query — return all (backward compatible)
        dynamic_rows = conn.execute(
            "SELECT fact, category FROM memories WHERE superseded_by IS NULL AND category NOT IN ('preferences', 'people', 'places') ORDER BY category, updated_at DESC"
        ).fetchall()
        dyn_groups = {}
        for r in dynamic_rows:
            cat = r["category"] or "general"
            dyn_groups.setdefault(cat, []).append(r["fact"])
        for cat, facts in sorted(dyn_groups.items()):
            lines.append(f"[{cat}]")
            for f in facts:
                lines.append(f"- {f}")
        total_count = static_count + len(dynamic_rows)

    if not lines:
        return {"text": "", "count": 0}
    return {"text": "\n".join(lines), "count": total_count}


# --- Settings ---


@router.get("/setting")
def setting_get(key: str = Query(...)) -> dict:
    """Get a setting value."""
    conn = _get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return {"key": key, "value": row["value"] if row else None}


@router.post("/setting")
def setting_set(key: str = Query(...), value: str = Query(...)) -> dict:
    """Set a setting value."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    return {"success": True, "key": key, "value": value}


# --- Reminders (persistent) ---


class ReminderRequest(BaseModel):
    chat_jid: str
    message: str
    trigger_at: str  # ISO format
    repeat: str | None = None  # daily, weekly, monthly, weekdays


@router.post("/reminders")
def save_reminder(req: ReminderRequest) -> dict:
    """Save a reminder to the database."""
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO reminders(chat_jid, message, trigger_at, repeat, created_at) VALUES (?, ?, ?, ?, ?)",
        (req.chat_jid, req.message, req.trigger_at, req.repeat, now),
    )
    conn.commit()
    return {"id": cur.lastrowid, "message": req.message, "trigger_at": req.trigger_at, "repeat": req.repeat}


@router.get("/reminders")
def list_reminders_endpoint(chat_jid: str | None = Query(None)) -> dict:
    """Load all reminders (optionally filtered by chat_jid)."""
    conn = _get_conn()
    if chat_jid:
        rows = conn.execute("SELECT * FROM reminders WHERE chat_jid = ? ORDER BY trigger_at", (chat_jid,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM reminders ORDER BY trigger_at").fetchall()
    return {"reminders": [dict(r) for r in rows]}


@router.delete("/reminders/{reminder_id}")
def delete_reminder(reminder_id: int) -> dict:
    """Delete a reminder."""
    conn = _get_conn()
    conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    return {"success": True, "id": reminder_id}


@router.put("/reminders/{reminder_id}")
def update_reminder(reminder_id: int, trigger_at: str = Query(...)) -> dict:
    """Update a reminder's trigger time (used for rescheduling recurring reminders)."""
    conn = _get_conn()
    conn.execute("UPDATE reminders SET trigger_at = ? WHERE id = ?", (trigger_at, reminder_id))
    conn.commit()
    return {"success": True, "id": reminder_id, "trigger_at": trigger_at}

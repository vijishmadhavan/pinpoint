"""Persistent background job helpers for indexing and scan work."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any

from database import DB_PATH, init_db

RETRYABLE_SQLITE_ERRORS = ("database is locked", "database schema is locked", "unable to open database file")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_target(path: str | None) -> str:
    if not path:
        return ""
    return os.path.abspath(path)


def _decode_job_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if not row:
        return None
    job = dict(row)
    raw_details = job.get("details_json") or ""
    try:
        job["details"] = json.loads(raw_details) if raw_details else {}
    except json.JSONDecodeError:
        job["details"] = {}
    return job


def _execute_retry(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = (), *, commit: bool = False):
    delay = 0.05
    for attempt in range(4):
        try:
            cur = conn.execute(sql, params)
            if commit:
                conn.commit()
            return cur
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if attempt == 3 or not any(token in message for token in RETRYABLE_SQLITE_ERRORS):
                raise
            time.sleep(delay)
            delay *= 2


def create_job(
    conn: sqlite3.Connection,
    job_type: str,
    *,
    target_path: str = "",
    target_hash: str = "",
    current_stage: str = "queued",
) -> int:
    now = _now()
    cur = _execute_retry(
        conn,
        """
        INSERT INTO background_jobs(
            job_type, target_path, target_hash, status, current_stage,
            total_items, completed_items, details_json,
            started_at, finished_at, created_at, updated_at, failure_reason, retry_count
        ) VALUES (?, ?, ?, 'pending', ?, 0, 0, '', NULL, NULL, ?, ?, '', 0)
        """,
        (job_type, _normalize_target(target_path), target_hash, current_stage, now, now),
        commit=True,
    )
    return int(cur.lastrowid)


def get_active_job_for_target(conn: sqlite3.Connection, job_type: str, target_path: str = "") -> dict[str, Any] | None:
    row = _execute_retry(
        conn,
        """
        SELECT * FROM background_jobs
        WHERE job_type = ? AND target_path = ? AND status IN ('pending', 'running', 'cancelling')
        ORDER BY id DESC
        LIMIT 1
        """,
        (job_type, _normalize_target(target_path)),
    ).fetchone()
    return _decode_job_row(row)


def get_or_create_job(
    conn: sqlite3.Connection,
    job_type: str,
    *,
    target_path: str = "",
    current_stage: str = "queued",
) -> tuple[int, bool]:
    existing = get_active_job_for_target(conn, job_type, target_path)
    if existing:
        return int(existing["id"]), False
    return create_job(conn, job_type, target_path=target_path, current_stage=current_stage), True


def mark_job_running(conn: sqlite3.Connection, job_id: int, *, current_stage: str) -> None:
    now = _now()
    _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'running',
            current_stage = ?,
            started_at = COALESCE(started_at, ?),
            updated_at = ?,
            failure_reason = ''
        WHERE id = ?
        """,
        (current_stage, now, now, job_id),
        commit=True,
    )


def update_job_stage(conn: sqlite3.Connection, job_id: int, current_stage: str) -> None:
    _execute_retry(
        conn,
        "UPDATE background_jobs SET current_stage = ?, updated_at = ? WHERE id = ?",
        (current_stage, _now(), job_id),
        commit=True,
    )


def update_job_progress(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    total_items: int | None = None,
    completed_items: int | None = None,
    details: dict[str, Any] | None = None,
    current_stage: str | None = None,
) -> None:
    assignments = ["updated_at = ?"]
    params: list[Any] = [_now()]
    if total_items is not None:
        assignments.append("total_items = ?")
        params.append(total_items)
    if completed_items is not None:
        assignments.append("completed_items = ?")
        params.append(completed_items)
    if details is not None:
        assignments.append("details_json = ?")
        params.append(json.dumps(details))
    if current_stage is not None:
        assignments.append("current_stage = ?")
        params.append(current_stage)
    params.append(job_id)
    _execute_retry(
        conn,
        f"UPDATE background_jobs SET {', '.join(assignments)} WHERE id = ?",
        tuple(params),
        commit=True,
    )


def mark_job_completed(conn: sqlite3.Connection, job_id: int, *, current_stage: str = "done") -> None:
    now = _now()
    _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'completed',
            current_stage = ?,
            finished_at = ?,
            updated_at = ?,
            failure_reason = ''
        WHERE id = ?
        """,
        (current_stage, now, now, job_id),
        commit=True,
    )


def mark_job_failed(conn: sqlite3.Connection, job_id: int, reason: str, *, current_stage: str = "failed") -> None:
    now = _now()
    _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'failed',
            current_stage = ?,
            finished_at = ?,
            updated_at = ?,
            failure_reason = ?,
            retry_count = retry_count + 1
        WHERE id = ?
        """,
        (current_stage, now, now, reason[:500], job_id),
        commit=True,
    )


def request_job_cancel(conn: sqlite3.Connection, job_id: int) -> bool:
    cur = _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'cancelling', updated_at = ?
        WHERE id = ? AND status IN ('pending', 'running')
        """,
        (_now(), job_id),
        commit=True,
    )
    return cur.rowcount > 0


def cancel_jobs_for_target(conn: sqlite3.Connection, job_type: str, target_path: str) -> int:
    cur = _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'cancelling', updated_at = ?
        WHERE job_type = ? AND target_path = ? AND status IN ('pending', 'running')
        """,
        (_now(), job_type, _normalize_target(target_path)),
        commit=True,
    )
    return int(cur.rowcount)


def is_job_cancelling(conn: sqlite3.Connection, job_id: int) -> bool:
    row = _execute_retry(conn, "SELECT status FROM background_jobs WHERE id = ?", (job_id,)).fetchone()
    return bool(row and row["status"] == "cancelling")


def mark_job_cancelled(conn: sqlite3.Connection, job_id: int, *, current_stage: str = "cancelled") -> None:
    now = _now()
    _execute_retry(
        conn,
        """
        UPDATE background_jobs
        SET status = 'cancelled',
            current_stage = ?,
            finished_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (current_stage, now, now, job_id),
        commit=True,
    )


def list_jobs(conn: sqlite3.Connection, *, limit: int = 50, status: str = "") -> list[dict[str, Any]]:
    sql = "SELECT * FROM background_jobs"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = _execute_retry(conn, sql, tuple(params)).fetchall()
    return [_decode_job_row(row) for row in rows]


def get_job(job_id: int) -> dict[str, Any] | None:
    conn = init_db(DB_PATH)
    try:
        row = _execute_retry(conn, "SELECT * FROM background_jobs WHERE id = ?", (job_id,)).fetchone()
        return _decode_job_row(row)
    finally:
        conn.close()

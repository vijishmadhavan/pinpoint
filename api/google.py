"""Google Workspace endpoints: Gmail, Calendar, Drive — via gws CLI."""

from __future__ import annotations

import json
import shutil
import subprocess

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.helpers import _check_safe

router = APIRouter(prefix="/google")

GWS_BIN = shutil.which("gws")


def _run_gws(args: list[str], timeout: int = 30) -> dict:
    """Run a gws CLI command and return parsed JSON output."""
    if not GWS_BIN:
        raise HTTPException(status_code=503, detail="gws CLI not installed. Run: npm install -g @googleworkspace/cli")
    try:
        result = subprocess.run(
            [GWS_BIN, *args, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "not authenticated" in err.lower() or "no credentials" in err.lower() or "token" in err.lower():
                return {"error": "Not authenticated. Run: gws auth login", "details": err[:500]}
            return {"error": f"gws command failed (exit {result.returncode})", "details": err[:500]}
        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            return {"success": True, "message": "Command completed (no output)"}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"success": True, "raw_output": output[:2000]}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out"}
    except Exception as e:
        return {"error": str(e)}


# --- Gmail ---


class GmailSendRequest(BaseModel):
    to: str
    subject: str
    body: str
    attach: str | None = None


@router.post("/gmail-send")
def gmail_send(req: GmailSendRequest) -> dict:
    """Send an email via Gmail."""
    args = ["gmail", "+send", "--to", req.to, "--subject", req.subject, "--body", req.body]
    if req.attach:
        import os

        path = os.path.abspath(req.attach)
        _check_safe(path)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Attachment not found: {path}")
        args.extend(["--attach", path])
    result = _run_gws(args)
    if not result.get("error"):
        result["_hint"] = f"Email sent to {req.to}. Tell the user it's done."
    return result


@router.get("/gmail-search")
def gmail_search(
    q: str = Query(..., description="Gmail search query (same syntax as Gmail search bar)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Search Gmail messages."""
    result = _run_gws([
        "gmail", "users", "messages", "list",
        "--params", json.dumps({"userId": "me", "q": q, "maxResults": limit}),
    ])
    if result.get("error"):
        return result
    # Messages list only returns IDs — fetch snippets for top results
    messages = result.get("messages", [])
    if not messages:
        return {"query": q, "count": 0, "messages": [], "_hint": "No emails match. Try broader search terms."}
    # Get details for first few messages
    details = []
    for msg in messages[:limit]:
        msg_id = msg.get("id")
        if not msg_id:
            continue
        detail = _run_gws([
            "gmail", "users", "messages", "get",
            "--params", json.dumps({"userId": "me", "id": msg_id, "format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date"]}),
        ])
        if detail.get("error"):
            continue
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        details.append({
            "id": msg_id,
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "snippet": detail.get("snippet", ""),
        })
    return {
        "query": q,
        "count": len(details),
        "messages": details,
        "_hint": f"{len(details)} email(s) found. Summarize for the user.",
    }


@router.get("/gmail-triage")
def gmail_triage() -> dict:
    """Show unread inbox summary."""
    result = _run_gws(["gmail", "+triage"])
    if not result.get("error"):
        result["_hint"] = "Inbox summary loaded. Present it clearly to the user."
    return result


# --- Calendar ---


class CalendarCreateRequest(BaseModel):
    summary: str
    start: str  # ISO 8601
    end: str  # ISO 8601
    location: str | None = None
    description: str | None = None
    attendees: list[str] | None = None


@router.post("/calendar-create")
def calendar_create(req: CalendarCreateRequest) -> dict:
    """Create a calendar event."""
    args = ["calendar", "+insert", "--summary", req.summary, "--start", req.start, "--end", req.end]
    if req.location:
        args.extend(["--location", req.location])
    if req.description:
        args.extend(["--description", req.description])
    for email in req.attendees or []:
        args.extend(["--attendee", email])
    result = _run_gws(args)
    if not result.get("error"):
        result["_hint"] = f"Event '{req.summary}' created. Tell the user it's done."
    return result


@router.get("/calendar-events")
def calendar_events(
    days: int = Query(7, ge=1, le=30, description="Number of days ahead to show"),
    today: bool = Query(False, description="Show only today's events"),
) -> dict:
    """List upcoming calendar events."""
    args = ["calendar", "+agenda"]
    if today:
        args.append("--today")
    else:
        args.extend(["--days", str(days)])
    result = _run_gws(args)
    if not result.get("error"):
        result["_hint"] = "Calendar events loaded. Present them clearly to the user."
    return result


# --- Drive ---


@router.get("/drive-list")
def drive_list(
    q: str = Query("", description="Drive search query (empty = recent files)"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """List or search Google Drive files."""
    params = {"pageSize": limit}
    if q:
        params["q"] = f"name contains '{q}' or fullText contains '{q}'"
    else:
        params["orderBy"] = "modifiedTime desc"
    result = _run_gws([
        "drive", "files", "list",
        "--params", json.dumps(params),
    ])
    if result.get("error"):
        return result
    files = result.get("files", [])
    items = []
    for f in files:
        items.append({
            "id": f.get("id"),
            "name": f.get("name"),
            "mimeType": f.get("mimeType"),
            "modifiedTime": f.get("modifiedTime"),
        })
    return {
        "query": q or "(recent files)",
        "count": len(items),
        "files": items,
        "_hint": f"{len(items)} Drive file(s) found. Summarize for the user." if items else "No files found.",
    }


@router.post("/drive-upload")
def drive_upload(
    path: str = Query(..., description="Local file path to upload"),
    name: str | None = Query(None, description="Target filename on Drive"),
) -> dict:
    """Upload a local file to Google Drive."""
    import os

    path = os.path.abspath(path)
    _check_safe(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    args = ["drive", "+upload", path]
    if name:
        args.extend(["--name", name])
    result = _run_gws(args, timeout=120)
    if not result.get("error"):
        result["_hint"] = "File uploaded to Google Drive. Tell the user it's done."
    return result

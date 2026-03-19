"""Search endpoints: full-text search, facts search, document lookup, web search, web read."""

from __future__ import annotations

import os
import re
import threading
from urllib.parse import parse_qs, unquote, urlparse

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.helpers import _check_url_safe, _get_conn
from database import DB_PATH
from search import search
from search_pipeline import _coverage_score, _snippet

router = APIRouter()


def _search_explanation(result: dict) -> dict:
    """Compact top-level explanation of how the search behaved."""
    return {
        "search_mode": "lexical-first",
        "relaxed_lexical": bool(result.get("relaxed_lexical")),
        "enhanced_search_used": bool(result.get("enhanced_search_used")),
        "ambiguous_search": bool(result.get("ambiguous_search")),
        "result_explanations_available": bool(result.get("results")),
    }


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _document_overview(conn, doc_id: int, query: str | None = None) -> dict:
    row = conn.execute(
        """
        SELECT d.id, d.path, d.title, d.file_type, d.page_count, d.active, c.text
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.id = ? AND d.active = 1
    """,
        (doc_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    text = row["text"] or ""
    chunk_rows = conn.execute(
        """
        SELECT chunk_num, text, start_index, end_index
        FROM chunks
        WHERE document_id = ?
        ORDER BY chunk_num
    """,
        (doc_id,),
    ).fetchall()
    facts = conn.execute(
        """
        SELECT fact_text, category
        FROM facts
        WHERE document_id = ?
        ORDER BY id
        LIMIT 5
    """,
        (doc_id,),
    ).fetchall()

    preview = _truncate_text(text, 700)
    if query and query.strip() and chunk_rows:
        query_terms = [t for t in re.sub(r"[^\w\s]", "", query).split() if len(t) > 1]
        ranked_chunks = []
        for r in chunk_rows:
            coverage = _coverage_score(r["text"], row["title"], query)
            ranked_chunks.append({
                "chunk_num": r["chunk_num"],
                "start_index": r["start_index"],
                "end_index": r["end_index"],
                "preview": _snippet(r["text"], query_terms, max_len=240),
                "coverage_score": round(coverage, 4),
            })
        ranked_chunks.sort(key=lambda item: (item["coverage_score"], -item["chunk_num"]), reverse=True)
        section_previews = ranked_chunks[:3]
    else:
        section_previews = [
            {
                "chunk_num": r["chunk_num"],
                "start_index": r["start_index"],
                "end_index": r["end_index"],
                "preview": _truncate_text(r["text"], 240),
            }
            for r in chunk_rows[:3]
        ]
    fact_list = [dict(r) for r in facts]
    total_chunks = len(chunk_rows)

    return {
        "id": row["id"],
        "path": row["path"],
        "title": row["title"],
        "file_type": row["file_type"],
        "page_count": row["page_count"],
        "overview": preview,
        "query": query or "",
        "full_text_chars": len(text),
        "chunk_count": total_chunks,
        "top_sections": section_previews,
        "facts": fact_list,
        "_hint": "Document overview loaded. Use this first; read_document only if you need the full text.",
    }


# --- Search (enhanced with filters) ---


@router.get("/search")
def search_endpoint(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    file_type: str | None = Query(None, description="Filter by type: pdf, docx, xlsx, image, etc."),
    folder: str | None = Query(None, description="Filter by folder path prefix"),
) -> dict:
    """Search across all indexed documents. Supports file_type and folder filters."""
    result = search(q, DB_PATH, limit, file_type=file_type, folder=folder)
    result["search_explanation"] = _search_explanation(result)
    if not result.get("results"):
        result["_hint"] = (
            "No results. File may not be indexed — use index_file first, then retry. Or try search_facts for quick factual lookups."
        )
    elif result.get("ambiguous_search"):
        result["_hint"] = result.get("clarification_hint") or (
            "Multiple similar matches found. Ask the user to narrow it down with a file name, title, date, person, location, or year."
        )
    else:
        result["_hint"] = f"{len(result['results'])} result(s) found. Answer the user's question using these."
    return result


# --- Search facts ---


@router.get("/search-facts")
def search_facts_endpoint(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Search extracted facts from documents using FTS5."""
    conn = _get_conn()
    # Use FTS5 for fast full-text search, fall back to LIKE if FTS table missing
    try:
        rows = conn.execute(
            """SELECT f.id, f.fact_text, f.category, d.path, d.file_type
               FROM facts_fts fts
               JOIN facts f ON f.id = fts.rowid
               JOIN documents d ON f.document_id = d.id
               WHERE facts_fts MATCH ? AND d.active = 1 LIMIT ?""",
            (q, limit),
        ).fetchall()
    except Exception:
        rows = conn.execute(
            """SELECT f.id, f.fact_text, f.category, d.path, d.file_type
               FROM facts f JOIN documents d ON f.document_id = d.id
               WHERE f.fact_text LIKE ? AND d.active = 1 LIMIT ?""",
            (f"%{q}%", limit),
        ).fetchall()
    resp = {"query": q, "count": len(rows), "results": [dict(r) for r in rows]}
    if not rows:
        resp["_hint"] = "No facts match. Try search_documents for full-text search across document content."
    else:
        resp["_hint"] = f"{len(rows)} fact(s) found. Answer the user's question using these."
    return resp


# --- Get document by ID ---


@router.get("/document/{doc_id}")
def document_endpoint(doc_id: int) -> dict:
    """Get full document text by ID. Used by Gemini read_document tool."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT d.id, d.path, d.title, d.file_type, d.page_count, d.active, c.text
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.id = ? AND d.active = 1
    """,
        (doc_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return dict(row)


@router.get("/document/{doc_id}/overview")
def document_overview_endpoint(
    doc_id: int,
    q: str | None = Query(None, description="Optional user query to rank the most relevant sections"),
) -> dict:
    """Get a compact document overview before reading full text."""
    conn = _get_conn()
    return _document_overview(conn, doc_id, q)


# --- Web Search (LangSearch API, Jina fallback) ---


class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    count: int = Field(10, ge=1, le=20, description="Number of results")
    freshness: str = Field("noLimit", description="Time filter: noLimit, day, week, month")


class SearchFeedbackRequest(BaseModel):
    query: str
    signal: str = Field(..., description="helpful, not_helpful, wrong_result, opened_overview, opened_full_document")
    document_id: int | None = None
    document_path: str = ""
    session_id: str = ""
    notes: str = ""


@router.post("/search-feedback")
def search_feedback_endpoint(req: SearchFeedbackRequest) -> dict:
    """Record lightweight search feedback. Logging only — does not alter ranking."""
    signal = req.signal.strip().lower()
    allowed = {"helpful", "not_helpful", "wrong_result", "opened_overview", "opened_full_document"}
    if signal not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid signal: {req.signal}")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO search_feedback(query, document_id, document_path, signal, session_id, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """,
        (
            req.query.strip(),
            req.document_id,
            req.document_path.strip(),
            signal,
            req.session_id.strip(),
            req.notes.strip(),
        ),
    )
    conn.commit()
    return {"success": True, "id": cursor.lastrowid, "signal": signal}


@router.get("/search-feedback")
def search_feedback_list(
    q: str | None = Query(None, description="Optional query filter"),
    signal: str | None = Query(None, description="Optional signal filter"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """List recent search feedback for review."""
    conn = _get_conn()
    params: list[object] = []
    where: list[str] = []
    if q:
        where.append("query LIKE ?")
        params.append(f"%{q}%")
    if signal:
        where.append("signal = ?")
        params.append(signal.strip().lower())

    sql = """
        SELECT id, query, document_id, document_path, signal, session_id, notes, created_at
        FROM search_feedback
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return {"count": len(rows), "results": [dict(r) for r in rows]}


@router.post("/web-search")
def web_search(req: WebSearchRequest) -> dict:
    """Search the web using LangSearch API. Falls back to Jina Reader + Brave scraping."""
    import requests as http

    api_key = os.getenv("LANGSEARCH_API_KEY")
    if api_key:
        try:
            resp = http.post(
                "https://api.langsearch.com/v1/web-search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": req.query, "count": req.count, "freshness": req.freshness, "summary": True},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("data", {}).get("webPages", {}).get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("summary", item.get("snippet", ""))[:500],
                })
            return {
                "query": req.query,
                "count": len(results),
                "results": results,
                "source": "langsearch",
                "_hint": f"{len(results)} web result(s). Answer from these. Use web_read on a URL for full content.",
            }
        except Exception as e:
            print(f"[WebSearch] LangSearch failed: {e}, falling back to Jina/Brave")

    # Fallback: Jina Reader search
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        try:
            resp = http.get(
                f"https://s.jina.ai/{req.query}",
                headers={"Authorization": f"Bearer {jina_key}", "Accept": "application/json", "X-Retain-Images": "none"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("data", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", item.get("content", ""))[:500],
                })
            return {
                "query": req.query,
                "count": len(results),
                "results": results,
                "source": "jina",
                "_hint": f"{len(results)} web result(s). Answer from these. Use web_read on a URL for full content.",
            }
        except Exception as e:
            print(f"[WebSearch] Jina failed: {e}, falling back to Brave scrape")

    # Last resort: Brave HTML scraping (no API key needed)
    return {"error": "No web search API key configured. Set LANGSEARCH_API_KEY or JINA_API_KEY in .env."}


# --- Web Read (Segment 18U: browser tool, readability + html2text) ---

_WEB_READ_MAX = 8000  # max chars per chunk

# Thread-local html2text converter (each thread gets its own instance)
_h2t_local = threading.local()


def _get_html2text() -> object:
    h = getattr(_h2t_local, "h2t", None)
    if h is None:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_images = True
        h.body_width = 0  # no line wrapping
        h.ignore_emphasis = False
        h.protect_links = True
        _h2t_local.h2t = h
    return h


def _paginate(text: str, start: int) -> dict:
    """Paginate text: cut at nearest newline, return chunk + metadata."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    total = len(text)
    end = start + _WEB_READ_MAX
    if end < total:
        nl = text.rfind("\n", start, end)
        if nl > start:
            end = nl + 1
    chunk = text[start:end]
    has_more = end < total
    if has_more:
        chunk += f"\n...(truncated — {total} total chars, call with start={end} for more)"
    return {"content": chunk, "total": total, "start": start, "end": end, "has_more": has_more}


@router.get("/web-read")
def web_read(
    url: str = Query(..., description="URL to fetch and read"),
    start: int = Query(0, ge=0, description="Character offset for pagination"),
) -> dict:
    """Fetch any URL and return clean markdown text. Works like a browser."""
    import requests as req

    try:
        from readability import Document
    except ImportError:
        return {"url": url, "title": "", "content": "", "error": "Missing deps: pip install readability-lxml html2text"}
    _check_url_safe(url)
    try:
        resp = req.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=15,
            allow_redirects=True,
        )
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        # Non-HTML: return raw text (JSON, plain text, etc.)
        if "html" not in ct:
            result = _paginate(resp.text, start)
            result.update({"url": url, "title": ""})
            return result
        # Clean up search engine HTML before conversion
        html = resp.text

        # Decode DDG redirect URLs to actual URLs
        def _decode_ddg(m: re.Match) -> str:
            try:
                params = parse_qs(urlparse(m.group(0)).query)
                return unquote(params.get("uddg", [m.group(0)])[0])
            except Exception:
                return m.group(0)

        html = re.sub(r'//duckduckgo\.com/l/\?[^"\'>\s]+', _decode_ddg, html)
        # Convert relative URLs to absolute (for Brave, Google, etc.)
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        html = re.sub(r'href="/', f'href="{base}/', html)
        # Strip search engine nav/UI elements
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL)
        html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL)
        # HTML -> readability (extract main content) -> html2text (convert to markdown)
        h = _get_html2text()
        doc = Document(html)
        title = doc.title() or ""
        md = h.handle(doc.summary())
        if len(md.strip()) < 200:
            # Readability failed (search results, etc.) — convert full page
            md = h.handle(html)
        # Remove lines that are just relative links or empty brackets
        md = re.sub(r"^\s*\[?\s*\]\(</[^)]*>\)\s*$", "", md, flags=re.MULTILINE)
        md = re.sub(r"\n{3,}", "\n\n", md)
        result = _paginate(md, start)
        result.update({"url": url, "title": title})
        return result
    except Exception as e:
        return {"url": url, "title": "", "content": "", "error": str(e)}

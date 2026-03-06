"""Search endpoints: full-text search, facts search, document lookup, web read."""

from __future__ import annotations

import re
import threading
from urllib.parse import parse_qs, unquote, urlparse

from fastapi import APIRouter, HTTPException, Query

from api.helpers import _check_url_safe, _get_conn
from database import DB_PATH
from search import search

router = APIRouter()


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
    if not result.get("results"):
        result["_hint"] = (
            "No results. File may not be indexed — use index_file first, then retry. Or try search_facts for quick factual lookups."
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
    """Search extracted facts from documents."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT f.id, f.fact_text, f.category, d.path, d.file_type
           FROM facts f JOIN documents d ON f.document_id = d.id
           WHERE f.fact_text LIKE ? LIMIT ?""",
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
        WHERE d.id = ?
    """,
        (doc_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return dict(row)


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

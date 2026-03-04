"""
Pinpoint — Search function (QMD-inspired pipeline)

FTS5 query builder + BM25 scoring + normalization + strong signal detection.
Gemini query expansion + RRF fusion (Segment 16).
"""

import json
import os
import re
import sqlite3

from database import init_db, cache_get, cache_set, DB_PATH


def build_fts5_query(user_query: str) -> str:
    """
    Convert a user query into FTS5 MATCH syntax.

    Rules (from QMD buildFTS5Query):
      - Quoted phrases → keep as "exact phrase" (no prefix *)
      - Plain terms → "term"* (prefix match: "inv" matches "invoice")
      - Terms starting with - → NOT "term"*

    Examples:
      "sharma invoice"           → "sharma"* "invoice"*
      "reliance -cancelled"      → "reliance"* NOT "cancelled"*
      '"exact phrase" other'     → "exact phrase" "other"*
      "inv"                      → "inv"*
    """
    if not user_query or not user_query.strip():
        return ""

    parts = []
    query = user_query.strip()

    # Extract quoted phrases first
    quoted = re.findall(r'"([^"]+)"', query)
    for phrase in quoted:
        parts.append(f'"{phrase}"')
    # Remove quoted phrases from query
    query = re.sub(r'"[^"]*"', "", query).strip()

    # Process remaining tokens
    for token in query.split():
        token = token.strip()
        if not token:
            continue
        # Remove non-alphanumeric chars (except leading -)
        if token.startswith("-") and len(token) > 1:
            # Negation
            clean = re.sub(r"[^\w]", "", token[1:])
            if clean:
                parts.append(f'NOT "{clean}"*')
        else:
            clean = re.sub(r"[^\w]", "", token)
            if clean:
                parts.append(f'"{clean}"*')

    return " ".join(parts)


def _normalize_bm25(raw_score: float) -> float:
    """BM25 normalization: abs(score) / (1 + abs(score)) → always [0, 1)."""
    return abs(raw_score) / (1 + abs(raw_score))


def _snippet(text: str, query_terms: list[str], max_len: int = 200) -> str:
    """Extract a text snippet around the first query term match."""
    text_lower = text.lower()
    best_pos = -1

    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        # No match found — return start of text
        return text[:max_len].strip() + ("…" if len(text) > max_len else "")

    # Center snippet around the match
    start = max(0, best_pos - max_len // 2)
    end = start + max_len
    snippet = text[start:end].strip()

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + snippet + suffix


# --- Gemini Query Expansion (Segment 16) ---

_gemini_client = None


def _get_gemini():
    """Lazy-load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def expand_query(query: str, conn: sqlite3.Connection) -> list[str]:
    """
    Use Gemini to expand a search query into keyword variants.
    Returns list of variant queries. Cached in llm_cache.
    """
    cache_key = f"expand:{query}"
    cached = cache_get(conn, cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass

    client = _get_gemini()
    if not client:
        return []

    from google.genai import types
    _expand_schema = {
        "type": "OBJECT",
        "properties": {"queries": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["queries"],
    }
    try:
        from extractors import gemini_call_with_retry
        response = gemini_call_with_retry(
            client,
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            contents=f'Expand this search query into 3-5 keyword variants for full-text search. Query: "{query}"',
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_expand_schema,
            ),
        )
        data = json.loads(response.text)
        variants = data.get("queries", [])
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            cache_set(conn, cache_key, json.dumps(variants))
            return variants
    except Exception as e:
        print(f"[Search] Query expansion failed: {e}")

    return []


def _has_chunks(conn: sqlite3.Connection) -> bool:
    """Check if any chunks exist in the database."""
    row = conn.execute("SELECT COUNT(*) as n FROM chunks LIMIT 1").fetchone()
    return row["n"] > 0 if row else False


def _fts5_search(conn: sqlite3.Connection, fts5_query: str, limit: int,
                 file_type: str = None, folder: str = None) -> list[dict]:
    """Run FTS5 search. Searches chunks_fts if chunks exist, else documents_fts."""
    if not fts5_query:
        return []

    where_extra = ""
    params = [fts5_query]
    if file_type:
        where_extra += " AND d.file_type = ?"
        params.append(file_type.lower())
    if folder:
        where_extra += " AND d.path LIKE ?"
        params.append(folder.rstrip("/") + "/%")
    params.append(limit)

    # Try chunk-level search first (more precise)
    try:
        if _has_chunks(conn):
            rows = conn.execute(f"""
                SELECT d.id, d.path, d.file_type, d.title, d.page_count,
                       bm25(chunks_fts, 10.0, 10.0, 1.0) as raw_score,
                       ch.text, ch.chunk_num
                FROM chunks_fts cfts
                JOIN chunks ch ON ch.id = cfts.rowid
                JOIN documents d ON d.id = ch.document_id
                WHERE chunks_fts MATCH ?
                AND d.active = 1
                {where_extra}
                ORDER BY raw_score
                LIMIT ?
            """, params).fetchall()
            if rows:
                return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        pass

    # Fallback: document-level search
    try:
        rows = conn.execute(f"""
            SELECT d.id, d.path, d.file_type, d.title, d.page_count,
                   bm25(documents_fts, 10.0, 10.0, 1.0) as raw_score,
                   c.text, -1 as chunk_num
            FROM documents_fts fts
            JOIN documents d ON d.id = fts.rowid
            JOIN content c ON c.hash = d.hash
            WHERE documents_fts MATCH ?
            AND d.active = 1
            {where_extra}
            ORDER BY raw_score
            LIMIT ?
        """, params).fetchall()
    except sqlite3.OperationalError:
        return []

    return [dict(row) for row in rows]


def _rrf_fusion(result_lists: list[tuple[float, list[dict]]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — merge multiple ranked result lists.
    Each entry: (weight, results_list).
    Original query gets 2x weight, expanded variants get 1x.
    """
    scores = {}  # doc_id → { rrf_score, doc_data }

    for weight, results in result_lists:
        for rank, row in enumerate(results):
            doc_id = row["id"]
            rrf_score = weight / (k + rank + 1)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += rrf_score
            else:
                scores[doc_id] = {"rrf_score": rrf_score, "row": row}

    # Sort by RRF score descending
    sorted_docs = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_docs


def search(query: str, db_path: str = DB_PATH, limit: int = 20,
           file_type: str = None, folder: str = None) -> dict:
    """
    Search indexed documents using FTS5 + BM25.

    Args:
      query: search keywords
      db_path: path to SQLite database
      limit: max results
      file_type: filter by type (pdf, docx, xlsx, image, etc.)
      folder: filter by folder prefix (only files within this folder)

    Returns:
      {
        "query": original query,
        "fts5_query": the FTS5 MATCH string,
        "results": [ { id, path, file_type, title, score, snippet, page_count } ],
        "strong_signal": True if top result is a clear winner,
      }
    """
    conn = init_db(db_path)

    fts5_query = build_fts5_query(query)
    if not fts5_query:
        conn.close()
        return {"query": query, "fts5_query": "", "results": [], "strong_signal": False, "expanded": False}

    # Step 1: BM25 probe with original query
    original_rows = _fts5_search(conn, fts5_query, limit, file_type, folder)
    query_terms = [t for t in re.sub(r"[^\w\s]", "", query).split() if len(t) > 1]

    # Build initial results
    probe_results = []
    for row in original_rows:
        score = _normalize_bm25(row["raw_score"])
        chunk_num = row.get("chunk_num", -1)
        # For chunks, use more text as snippet (chunks are already small ~500 words)
        snippet_len = 500 if chunk_num >= 0 else 200
        probe_results.append({
            "id": row["id"],
            "path": row["path"],
            "file_type": row["file_type"],
            "title": row["title"],
            "score": round(score, 4),
            "raw_score": round(row["raw_score"], 4),
            "snippet": _snippet(row["text"], query_terms, max_len=snippet_len),
            "page_count": row["page_count"],
            "chunk_num": chunk_num,
        })

    # Step 2: Strong signal detection — skip expansion if clear winner
    strong_signal = False
    if len(probe_results) >= 1:
        top = probe_results[0]["score"]
        second = probe_results[1]["score"] if len(probe_results) >= 2 else 0
        strong_signal = top >= 0.85 and (top - second) >= 0.15

    if strong_signal:
        conn.close()
        return {
            "query": query,
            "fts5_query": fts5_query,
            "results": probe_results[:limit],
            "strong_signal": True,
            "expanded": False,
        }

    # Step 3: Query expansion via Gemini (cached)
    variants = expand_query(query, conn)

    if not variants:
        # Expansion failed or not available — return probe results
        conn.close()
        return {
            "query": query,
            "fts5_query": fts5_query,
            "results": probe_results[:limit],
            "strong_signal": False,
            "expanded": False,
        }

    # Step 4: Run FTS5 for each variant
    result_lists = [(2.0, original_rows)]  # Original query gets 2x weight
    for variant in variants:
        variant_fts = build_fts5_query(variant)
        if variant_fts and variant_fts != fts5_query:
            variant_rows = _fts5_search(conn, variant_fts, limit, file_type, folder)
            if variant_rows:
                result_lists.append((1.0, variant_rows))

    # Step 5: RRF fusion
    if len(result_lists) > 1:
        fused = _rrf_fusion(result_lists)
        results = []
        for item in fused[:limit]:
            row = item["row"]
            chunk_num = row.get("chunk_num", -1)
            snippet_len = 500 if chunk_num >= 0 else 200
            results.append({
                "id": row["id"],
                "path": row["path"],
                "file_type": row["file_type"],
                "title": row["title"],
                "score": round(item["rrf_score"], 4),
                "snippet": _snippet(row["text"], query_terms, max_len=snippet_len),
                "page_count": row["page_count"],
                "chunk_num": chunk_num,
            })
    else:
        results = probe_results[:limit]

    conn.close()

    return {
        "query": query,
        "fts5_query": fts5_query,
        "results": results,
        "strong_signal": False,
        "expanded": True,
        "variants": variants,
    }


def search_simple(query: str, db_path: str = DB_PATH, limit: int = 10) -> list[dict]:
    """Convenience: just return the results list."""
    return search(query, db_path, limit)["results"]


# --- Quick test ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = search(q)
        print(f"Query: {result['query']}")
        print(f"FTS5:  {result['fts5_query']}")
        print(f"Strong signal: {result['strong_signal']}")
        print(f"Results: {len(result['results'])}")
        for i, r in enumerate(result["results"], 1):
            print(f"\n  [{i}] {r['title']} ({r['file_type']})")
            print(f"      Score: {r['score']} (raw: {r['raw_score']})")
            print(f"      Path: {r['path']}")
            print(f"      Snippet: {r['snippet']}")
    else:
        print("Usage: python search.py <query>")
        print("Example: python search.py reliance invoice")

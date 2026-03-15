"""Search pipeline implementation for lexical, semantic, fusion, and rerank stages."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from database import DB_PATH, cache_get, cache_set, init_db

logger = logging.getLogger(__name__)

_gemini_client = None
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "get",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_TERM_SYNONYMS = {
    "attend": {"attendee", "attendees", "attended"},
    "attended": {"attend", "attendee", "attendees"},
    "attendees": {"attend", "attendee", "attended"},
    "handoff": {"handover"},
    "handover": {"handoff"},
}


@dataclass(frozen=True)
class SearchOptions:
    use_query_expansion: bool = False
    use_embeddings: bool = False
    use_reranker: bool = False
    use_position_blend: bool = False
    use_strong_signal_shortcut: bool = True
    lexical_candidate_limit: int = 20
    weak_result_score_threshold: float = 0.55
    weak_result_gap_threshold: float = 0.12
    rerank_min_results: int = 4


DEFAULT_SEARCH_OPTIONS = SearchOptions()
ENHANCED_SEARCH_OPTIONS = SearchOptions(use_query_expansion=True, use_embeddings=True, use_reranker=True)


def build_fts5_query(user_query: str) -> str:
    """Convert a user query into FTS5 MATCH syntax."""
    if not user_query or not user_query.strip():
        return ""

    parts = []
    query = user_query.strip()
    quoted = re.findall(r'"([^"]+)"', query)
    for phrase in quoted:
        parts.append(f'"{phrase}"')
    query = re.sub(r'"[^"]*"', "", query).strip()

    for token in query.split():
        token = token.strip()
        if not token:
            continue
        if token.startswith("-") and len(token) > 1:
            clean = re.sub(r"[^\w]", "", token[1:])
            if clean:
                parts.append(f'NOT "{clean}"*')
        else:
            clean = re.sub(r"[^\w]", "", token)
            if clean:
                parts.append(f'"{clean}"*')

    return " ".join(parts)


def _normalize_query_token(token: str) -> str:
    clean = re.sub(r"[^\w]", "", token.lower())
    if len(clean) <= 3:
        return clean
    if clean.endswith("ies") and len(clean) > 4:
        return clean[:-3] + "y"
    if clean.endswith("ing") and len(clean) > 5:
        return clean[:-3]
    if clean.endswith("ed") and len(clean) > 4:
        return clean[:-2]
    if clean.endswith("es") and len(clean) > 4:
        return clean[:-2]
    if clean.endswith("s") and len(clean) > 4:
        return clean[:-1]
    return clean


def _extract_query_terms(user_query: str) -> list[str]:
    terms = []
    for raw in re.findall(r"[A-Za-z0-9]+", user_query.lower()):
        normalized = _normalize_query_token(raw)
        if not normalized or normalized in _STOPWORDS:
            continue
        terms.append(normalized)
    return terms


def _query_concepts(user_query: str) -> list[set[str]]:
    concepts = []
    seen = set()
    for term in _extract_query_terms(user_query):
        if term in seen:
            continue
        seen.add(term)
        concept = {term}
        for synonym in _TERM_SYNONYMS.get(term, set()):
            normalized = _normalize_query_token(synonym)
            if normalized and normalized not in _STOPWORDS:
                concept.add(normalized)
        concepts.append(concept)
    return concepts


def _build_relaxed_fts5_query(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return ""

    parts = []
    for concept in _query_concepts(user_query):
        variants = sorted(v for v in concept if len(v) >= 2)
        if not variants:
            continue
        if len(variants) == 1:
            parts.append(f'"{variants[0]}"*')
        else:
            parts.append("(" + " OR ".join(f'"{variant}"*' for variant in variants) + ")")
    return " ".join(parts)


def _build_broad_fts5_query(user_query: str) -> str:
    parts = []
    seen = set()
    for concept in _query_concepts(user_query):
        for variant in sorted(concept):
            if variant in seen or len(variant) < 2:
                continue
            seen.add(variant)
            parts.append(f'"{variant}"*')
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def _normalize_text_terms(text: str) -> set[str]:
    terms = set()
    for raw in re.findall(r"[A-Za-z0-9]+", text.lower()):
        normalized = _normalize_query_token(raw)
        if normalized:
            terms.add(normalized)
    return terms


def _coverage_score(text: str, title: str, query: str) -> float:
    concepts = _query_concepts(query)
    if not concepts:
        return 0.0

    terms = _normalize_text_terms(f"{title} {text}")
    matched = 0
    for concept in concepts:
        if any(any(term.startswith(variant) or variant.startswith(term) for term in terms) for variant in concept):
            matched += 1
    return matched / len(concepts)


def _metadata_score(path: str, title: str, query: str) -> float:
    concepts = _query_concepts(query)
    if not concepts:
        return 0.0

    filename = os.path.splitext(os.path.basename(path))[0]
    metadata_terms = _normalize_text_terms(f"{filename} {title} {path.replace(os.sep, ' ')}")
    matched = 0
    for concept in concepts:
        if any(any(term.startswith(variant) or variant.startswith(term) for term in metadata_terms) for variant in concept):
            matched += 1

    base = matched / len(concepts)
    lower_filename = filename.lower()
    lower_title = title.lower()
    query_terms = _extract_query_terms(query)
    identifier_hits = sum(1 for term in query_terms if any(ch.isdigit() for ch in term) and (term in lower_filename or term in lower_title))
    exact_phrase = query.lower() in lower_filename or query.lower() in lower_title
    bonus = 0.0
    if exact_phrase:
        bonus += 0.2
    if identifier_hits:
        bonus += min(0.2, 0.1 * identifier_hits)
    return min(1.0, base + bonus)


def _normalize_bm25(raw_score: float) -> float:
    return abs(raw_score) / (1 + abs(raw_score))


def _snippet(text: str, query_terms: list[str], max_len: int = 200) -> str:
    text_lower = text.lower()
    best_pos = -1
    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        return text[:max_len].strip() + ("…" if len(text) > max_len else "")

    start = max(0, best_pos - max_len // 2)
    end = start + max_len
    snippet = text[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + snippet + suffix


def _get_gemini() -> Any | None:
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
        except ImportError:
            return None

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def expand_query(query: str, conn: sqlite3.Connection) -> list[str]:
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

    schema = {
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
                response_json_schema=schema,
            ),
        )
        data = json.loads(response.text)
        variants = data.get("queries", [])
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            cache_set(conn, cache_key, json.dumps(variants))
            return variants
    except Exception:
        logger.exception("query_expansion_failed")

    return []


def _has_chunks(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) as n FROM chunks LIMIT 1").fetchone()
    return row["n"] > 0 if row else False


def _fts5_search(
    conn: sqlite3.Connection, fts5_query: str, limit: int, file_type: str | None = None, folder: str | None = None
) -> list[dict[str, Any]]:
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

    try:
        if _has_chunks(conn):
            rows = conn.execute(
                f"""
                SELECT d.id, d.path, d.file_type, d.title, d.page_count,
                       d.modified_at,
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
            """,
                params,
            ).fetchall()
            if rows:
                return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        pass

    try:
        rows = conn.execute(
            f"""
            SELECT d.id, d.path, d.file_type, d.title, d.page_count,
                   d.modified_at,
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
        """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [dict(row) for row in rows]


def _lexical_search(
    conn: sqlite3.Connection, query: str, limit: int, file_type: str | None = None, folder: str | None = None
) -> tuple[str, list[dict[str, Any]], bool]:
    strict_query = build_fts5_query(query)
    rows = _fts5_search(conn, strict_query, limit, file_type, folder)

    def _merge_rows(base_rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = list(base_rows)
        seen = {row["id"] for row in base_rows}
        for row in extra_rows:
            if row["id"] in seen:
                continue
            merged.append(row)
            seen.add(row["id"])
            if len(merged) >= limit:
                break
        return merged

    final_query = strict_query
    used_fallback = False

    relaxed_query = _build_relaxed_fts5_query(query)
    if relaxed_query and relaxed_query != strict_query:
        relaxed_rows = _fts5_search(conn, relaxed_query, limit, file_type, folder)
        if relaxed_rows:
            if rows:
                rows = _merge_rows(rows, relaxed_rows)
                final_query = strict_query
            else:
                rows = relaxed_rows
                final_query = relaxed_query
            used_fallback = True

    broad_query = _build_broad_fts5_query(query)
    if broad_query and broad_query not in {strict_query, relaxed_query}:
        broad_rows = _fts5_search(conn, broad_query, limit, file_type, folder)
        if broad_rows:
            if rows:
                rows = _merge_rows(rows, broad_rows)
                final_query = strict_query if strict_query else final_query
            else:
                rows = broad_rows
                final_query = broad_query
            used_fallback = True

    return final_query, rows, used_fallback


def _embedding_search(
    conn: sqlite3.Connection, query: str, limit: int, file_type: str | None = None, folder: str | None = None
) -> list[dict[str, Any]]:
    import struct

    import numpy as np

    row = conn.execute("SELECT COUNT(*) as n FROM chunk_embeddings LIMIT 1").fetchone()
    if not row or row["n"] == 0:
        return []

    from image_search import EMBED_DIM, _normalize

    try:
        from image_search import _embed_text_query

        query_emb = _embed_text_query(query)
    except Exception:
        logger.exception("embedding_query_failed")
        return []

    where_extra = ""
    params: list[Any] = []
    if file_type:
        where_extra += " AND d.file_type = ?"
        params.append(file_type.lower())
    if folder:
        where_extra += " AND d.path LIKE ?"
        params.append(folder.rstrip("/") + "/%")

    cursor = conn.execute(
        f"""
        SELECT ce.chunk_id, ce.embedding, ch.text, ch.chunk_num, ch.document_id,
               d.id, d.path, d.file_type, d.title, d.page_count, d.modified_at
        FROM chunk_embeddings ce
        JOIN chunks ch ON ch.id = ce.chunk_id
        JOIN documents d ON d.id = ch.document_id
        WHERE d.active = 1
        {where_extra}
        """,
        params,
    )

    best_by_doc: dict[int, dict[str, Any]] = {}
    batch_size = 512
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        valid_rows = []
        embeddings = []
        for row in rows:
            try:
                emb = np.array(struct.unpack(f"{EMBED_DIM}f", row["embedding"]), dtype=np.float32)
            except struct.error:
                logger.warning("malformed_chunk_embedding", extra={"chunk_id": row["chunk_id"]})
                continue
            valid_rows.append(row)
            embeddings.append(emb)

        if not embeddings:
            continue

        emb_matrix = _normalize(np.stack(embeddings), axis=-1)
        similarities = emb_matrix @ query_emb
        for row, sim in zip(valid_rows, similarities, strict=False):
            doc_id = row["id"]
            sim_f = float(sim)
            existing = best_by_doc.get(doc_id)
            if existing is not None and sim_f <= existing["raw_score"]:
                continue
            best_by_doc[doc_id] = {
                "id": doc_id,
                "path": row["path"],
                "file_type": row["file_type"],
                "title": row["title"],
                "score": round(max(0.0, sim_f), 4),
                "raw_score": round(sim_f, 4),
                "text": row["text"],
                "page_count": row["page_count"],
                "chunk_num": row["chunk_num"],
                "modified_at": row["modified_at"],
            }

    return sorted(best_by_doc.values(), key=lambda row: row["raw_score"], reverse=True)[:limit]


def _is_weak_result_set(results: list[dict[str, Any]], options: SearchOptions) -> bool:
    if not results:
        return True
    top = results[0]["score"]
    second = results[1]["score"] if len(results) > 1 else 0.0
    if top < options.weak_result_score_threshold:
        return True
    return (top - second) < options.weak_result_gap_threshold


def _detect_ambiguity(results: list[dict[str, Any]], options: SearchOptions) -> tuple[bool, str, int]:
    if len(results) < 3:
        return False, "", 0

    top = float(results[0].get("score", 0.0))
    window = min(5, len(results))
    clustered = 0
    for row in results[:window]:
        score = float(row.get("score", 0.0))
        if (top - score) <= max(0.06, options.weak_result_gap_threshold):
            clustered += 1

    if clustered < 3:
        return False, "", clustered

    hint = "Multiple similar matches found. Can you specify the file name, title, date, person, location, or year?"
    return True, hint, clustered


def _rerank_results(
    query: str, results: list[dict[str, Any]], conn: sqlite3.Connection, limit: int = 10
) -> list[dict[str, Any]]:
    if len(results) <= 3:
        return results

    candidate_limit = min(12, len(results))
    cache_parts = [f"{r['id']}:{r.get('modified_at', '')}:{r.get('chunk_num', -1)}" for r in results[:candidate_limit]]
    cache_key = f"rerank:{query}:{'|'.join(cache_parts)}"
    cached = cache_get(conn, cache_key)
    if cached:
        try:
            cached_indices = json.loads(cached)
            reranked = []
            seen = set()
            for idx in cached_indices[:limit]:
                if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                    reranked.append(results[idx])
                    seen.add(idx)
            for i in range(min(limit, len(results))):
                if i not in seen:
                    reranked.append(results[i])
            return reranked[:limit]
        except (json.JSONDecodeError, IndexError):
            pass

    client = _get_gemini()
    if not client:
        return results[:limit]

    from google.genai import types

    candidates = []
    for i, row in enumerate(results[:candidate_limit]):
        text = (row.get("snippet") or row.get("text", ""))[:600]
        candidates.append(f"[{i}] {row['title']}: {text}")

    schema = {
        "type": "OBJECT",
        "properties": {"ranked_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
        "required": ["ranked_indices"],
    }

    try:
        from extractors import gemini_call_with_retry

        response = gemini_call_with_retry(
            client,
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            contents=f'Rerank these search results by relevance to the query: "{query}"\n\n' + "\n".join(candidates)
            + "\n\nReturn indices in order of relevance (most relevant first).",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        data = json.loads(response.text)
        indices = data.get("ranked_indices", [])
        cache_set(conn, cache_key, json.dumps(indices))

        reranked = []
        seen = set()
        for idx in indices[:limit]:
            if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                reranked.append(results[idx])
                seen.add(idx)
        for i in range(min(limit, len(results))):
            if i not in seen:
                reranked.append(results[i])
        return reranked[:limit]
    except Exception:
        logger.exception("reranking_failed")
        return results[:limit]


def _rrf_fusion(result_lists: list[tuple[float, list[dict[str, Any]]]], k: int = 60) -> list[dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}
    for weight, results in result_lists:
        for rank, row in enumerate(results):
            doc_id = row["id"]
            rrf_score = weight / (k + rank + 1)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += rrf_score
            else:
                scores[doc_id] = {"rrf_score": rrf_score, "row": row}
    return sorted(scores.values(), key=lambda row: row["rrf_score"], reverse=True)


def _build_probe_results(rows: list[dict[str, Any]], query_terms: list[str], query: str) -> list[dict[str, Any]]:
    probe_results = []
    for row in rows:
        bm25_score = _normalize_bm25(row["raw_score"])
        coverage = _coverage_score(row.get("text", ""), row.get("title", ""), query)
        metadata = _metadata_score(row.get("path", ""), row.get("title", ""), query)
        score = 0.65 * bm25_score + 0.2 * coverage + 0.15 * metadata
        chunk_num = row.get("chunk_num", -1)
        snippet_len = 500 if chunk_num >= 0 else 200
        probe_results.append(
            {
                "id": row["id"],
                "path": row["path"],
                "file_type": row["file_type"],
                "title": row["title"],
                "score": round(score, 4),
                "raw_score": round(row["raw_score"], 4),
                "coverage_score": round(coverage, 4),
                "metadata_score": round(metadata, 4),
                "snippet": _snippet(row["text"], query_terms, max_len=snippet_len),
                "page_count": row["page_count"],
                "chunk_num": chunk_num,
                "modified_at": row.get("modified_at", ""),
            }
        )
    return probe_results


def _has_strong_signal(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    top = results[0]["score"]
    second = results[1]["score"] if len(results) >= 2 else 0
    return top >= 0.85 and (top - second) >= 0.15


def _collect_retrieval_lists(
    conn: sqlite3.Connection,
    query: str,
    fts5_query: str,
    original_rows: list[dict[str, Any]],
    limit: int,
    file_type: str | None,
    folder: str | None,
    options: SearchOptions,
) -> tuple[list[str], list[tuple[float, list[dict[str, Any]]]]]:
    variants = expand_query(query, conn) if options.use_query_expansion else []
    result_lists = [(2.0, original_rows)]
    if variants:
        for variant in variants:
            variant_fts = build_fts5_query(variant)
            if variant_fts and variant_fts != fts5_query:
                variant_rows = _fts5_search(conn, variant_fts, limit, file_type, folder)
                if variant_rows:
                    result_lists.append((1.0, variant_rows))

    embedding_rows = _embedding_search(conn, query, limit, file_type, folder) if options.use_embeddings else []
    if embedding_rows:
        result_lists.append((1.5, embedding_rows))
    return variants, result_lists


def _fuse_result_lists(
    result_lists: list[tuple[float, list[dict[str, Any]]]],
    probe_results: list[dict[str, Any]],
    query_terms: list[str],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    if len(result_lists) <= 1:
        return probe_results[:limit]

    fused = _rrf_fusion(result_lists)
    results = []
    for item in fused[:limit]:
        row = item["row"]
        chunk_num = row.get("chunk_num", -1)
        snippet_len = 500 if chunk_num >= 0 else 200
        full_text = row.get("text", "")
        coverage = _coverage_score(full_text, row.get("title", ""), query)
        metadata = _metadata_score(row.get("path", ""), row.get("title", ""), query)
        blended_score = 0.72 * item["rrf_score"] + 0.16 * coverage + 0.12 * metadata
        results.append(
            {
                "id": row["id"],
                "path": row["path"],
                "file_type": row["file_type"],
                "title": row["title"],
                "score": round(blended_score, 4),
                "rrf_score": round(item["rrf_score"], 4),
                "coverage_score": round(coverage, 4),
                "metadata_score": round(metadata, 4),
                "snippet": _snippet(full_text, query_terms, max_len=snippet_len),
                "text": full_text,
                "page_count": row["page_count"],
                "chunk_num": chunk_num,
                "modified_at": row.get("modified_at", ""),
            }
        )
    return results


def _apply_rerank_stage(
    query: str,
    results: list[dict[str, Any]],
    conn: sqlite3.Connection,
    limit: int,
    options: SearchOptions,
) -> list[dict[str, Any]]:
    if len(results) < options.rerank_min_results or not options.use_reranker:
        return results

    reranked = _rerank_results(query, results, conn, limit)
    if not options.use_position_blend:
        return reranked
    pre_rerank_scores = {r["id"]: r["score"] for r in results}
    pre_rerank_rank = {r["id"]: i for i, r in enumerate(results)}
    for rank, row in enumerate(reranked):
        rrf_score = pre_rerank_scores.get(row["id"], 0)
        rerank_score = 1.0 / (1 + rank)
        original_rank = pre_rerank_rank.get(row["id"], rank)
        if original_rank < 3:
            rrf_weight = 0.75
        elif original_rank < 10:
            rrf_weight = 0.60
        else:
            rrf_weight = 0.40
        row["score"] = round(rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score, 4)
    return sorted(reranked, key=lambda row: row["score"], reverse=True)


def _finalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized = []
    for row in results:
        clean = dict(row)
        clean.pop("text", None)
        finalized.append(clean)
    return finalized


def search(
    query: str, db_path: str = DB_PATH, limit: int = 20, file_type: str | None = None, folder: str | None = None
) -> dict[str, Any]:
    return search_with_options(query, db_path, limit, file_type=file_type, folder=folder, options=DEFAULT_SEARCH_OPTIONS)


def search_with_options(
    query: str,
    db_path: str = DB_PATH,
    limit: int = 20,
    file_type: str | None = None,
    folder: str | None = None,
    options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
) -> dict[str, Any]:
    conn = init_db(db_path)
    try:
        return _search_inner(conn, query, limit, file_type, folder, options)
    finally:
        conn.close()


def _search_inner(
    conn: Any,
    query: str,
    limit: int,
    file_type: str | None,
    folder: str | None,
    options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    strict_fts5_query = build_fts5_query(query)
    fts5_query = strict_fts5_query
    if not fts5_query:
        timings["total_ms"] = _elapsed_ms(total_start)
        return {
            "query": query,
            "fts5_query": "",
            "results": [],
            "strong_signal": False,
            "ambiguous_search": False,
            "clarification_hint": "",
            "ambiguous_result_count": 0,
            "expanded": False,
            "relaxed_lexical": False,
            "timing": timings,
        }

    lexical_limit = max(limit, options.lexical_candidate_limit)
    stage_start = time.perf_counter()
    fts5_query, original_rows, relaxed = _lexical_search(conn, query, lexical_limit, file_type, folder)
    timings["lexical_ms"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    query_terms = [t for t in re.sub(r"[^\w\s]", "", query).split() if len(t) > 1]
    probe_results = _build_probe_results(original_rows, query_terms, query)
    timings["probe_ms"] = _elapsed_ms(stage_start)

    if options.use_strong_signal_shortcut and _has_strong_signal(probe_results):
        stage_start = time.perf_counter()
        ambiguous_search, clarification_hint, ambiguous_result_count = _detect_ambiguity(probe_results[:limit], options)
        timings["ambiguity_ms"] = _elapsed_ms(stage_start)
        timings["total_ms"] = _elapsed_ms(total_start)
        return {
            "query": query,
            "fts5_query": fts5_query,
            "results": probe_results[:limit],
            "strong_signal": True,
            "ambiguous_search": ambiguous_search,
            "clarification_hint": clarification_hint,
            "ambiguous_result_count": ambiguous_result_count,
            "expanded": False,
            "relaxed_lexical": relaxed,
            "timing": timings,
        }

    effective_options = options
    if not _is_weak_result_set(probe_results, options):
        effective_options = SearchOptions(
            use_query_expansion=False,
            use_embeddings=False,
            use_reranker=False,
            use_position_blend=False,
            use_strong_signal_shortcut=options.use_strong_signal_shortcut,
            lexical_candidate_limit=options.lexical_candidate_limit,
            weak_result_score_threshold=options.weak_result_score_threshold,
            weak_result_gap_threshold=options.weak_result_gap_threshold,
            rerank_min_results=options.rerank_min_results,
        )

    stage_start = time.perf_counter()
    variants, result_lists = _collect_retrieval_lists(
        conn, query, fts5_query, original_rows, lexical_limit, file_type, folder, effective_options
    )
    timings["retrieval_ms"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    results = _fuse_result_lists(result_lists, probe_results, query_terms, query, lexical_limit)
    timings["fusion_ms"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    results = _apply_rerank_stage(query, results, conn, lexical_limit, effective_options)
    timings["rerank_ms"] = _elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    results = _finalize_results(results)
    timings["finalize_ms"] = _elapsed_ms(stage_start)
    final_results = results[:limit]
    stage_start = time.perf_counter()
    ambiguous_search, clarification_hint, ambiguous_result_count = _detect_ambiguity(final_results, options)
    timings["ambiguity_ms"] = _elapsed_ms(stage_start)
    timings["total_ms"] = _elapsed_ms(total_start)

    return {
        "query": query,
        "fts5_query": fts5_query,
        "results": final_results,
        "strong_signal": False,
        "ambiguous_search": ambiguous_search,
        "clarification_hint": clarification_hint,
        "ambiguous_result_count": ambiguous_result_count,
        "expanded": bool(variants),
        "variants": variants,
        "relaxed_lexical": relaxed,
        "enhanced_search_used": effective_options.use_query_expansion or effective_options.use_embeddings or effective_options.use_reranker,
        "timing": timings,
    }


def search_simple(query: str, db_path: str = DB_PATH, limit: int = 10) -> list[dict[str, Any]]:
    return search(query, db_path, limit)["results"]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)

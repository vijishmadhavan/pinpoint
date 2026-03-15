"""Compatibility wrapper for the search pipeline."""

from __future__ import annotations

from search_pipeline import (  # noqa: F401 — re-exports for backward compat
    DB_PATH,
    DEFAULT_SEARCH_OPTIONS,
    SearchOptions,
    build_fts5_query,
    search,
    search_simple,
    search_with_options,
)

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
            print(f"      Score: {r['score']} (raw: {r.get('raw_score', 'n/a')})")
            print(f"      Path: {r['path']}")
            print(f"      Snippet: {r['snippet']}")
    else:
        print("Usage: python search.py <query>")
        print("Example: python search.py reliance invoice")

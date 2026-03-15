"""Offline search relevance benchmark for Pinpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database import init_db
from indexing_service import index_single_file
from search_pipeline import ENHANCED_SEARCH_OPTIONS, SearchOptions, search_with_options

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent / "bm25s"))
try:
    import bm25s  # noqa: E402, I001
    from bm25s.tokenization import Tokenizer  # noqa: E402, I001

    _HAS_BM25S = True
except ImportError:
    bm25s = None  # type: ignore[assignment]
    Tokenizer = None  # type: ignore[assignment, misc]
    _HAS_BM25S = False


@dataclass(frozen=True)
class VariantSpec:
    kind: str
    options: SearchOptions | None = None


FTS_ONLY_OPTIONS = SearchOptions(
        use_query_expansion=False,
        use_embeddings=False,
        use_reranker=False,
        use_position_blend=False,
        use_strong_signal_shortcut=False,
    )


VARIANTS: dict[str, VariantSpec] = {
    "fts_only": VariantSpec("search", FTS_ONLY_OPTIONS),
    "fts_embeddings": VariantSpec(
        "search",
        SearchOptions(
        use_query_expansion=False,
        use_embeddings=True,
        use_reranker=False,
        use_position_blend=False,
        use_strong_signal_shortcut=False,
    )),
    "full_pipeline": VariantSpec("search", ENHANCED_SEARCH_OPTIONS),
    "full_no_position_blend": VariantSpec(
        "search",
        SearchOptions(
        use_query_expansion=True,
        use_embeddings=True,
        use_reranker=True,
        use_position_blend=False,
        use_strong_signal_shortcut=True,
    )),
    "bm25l_only": VariantSpec("bm25l"),
    "fts_bm25l_rrf": VariantSpec("fts_bm25l_rrf"),
}


def _load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data.get("queries"), list):
        raise ValueError("Dataset must contain a 'queries' list")
    return data


def _open_eval_db(db_path: str):
    """Benchmark runs are single-process; DELETE mode avoids WAL issues on mounted filesystems."""
    conn = init_db(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _index_corpus(conn: Any, corpus_dir: Path, *, embeddings_enabled: bool) -> dict[str, Any]:
    indexed_files: list[str] = []
    skipped_files: list[dict[str, str]] = []
    embedded_chunks = 0

    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file():
            continue
        outcome = index_single_file(
            conn,
            str(path),
            skip_unchanged=False,
            facts_enabled=False,
            embeddings_enabled=embeddings_enabled,
        )
        if outcome["status"] == "indexed":
            indexed_files.append(path.name)
            embedded_chunks += int(outcome.get("embedded_chunks", 0) or 0)
            continue
        skipped_files.append({"file": path.name, "reason": outcome.get("reason", "unknown")})

    return {
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "embedded_chunks": embedded_chunks,
    }


def _build_bm25_index(db_path: str) -> dict[str, Any]:
    if not _HAS_BM25S:
        raise RuntimeError("bm25s not installed — clone https://github.com/xhluca/bm25s into project root")
    conn = _open_eval_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.path, d.title, c.text
            FROM documents d
            JOIN content c ON c.hash = d.hash
            WHERE d.active = 1
            ORDER BY d.id
            """
        ).fetchall()
    finally:
        conn.close()

    corpus = [f"{row['title']}\n{row['text']}" for row in rows]
    tokenizer = Tokenizer(stopwords="en")
    corpus_tokens = tokenizer.tokenize(corpus)
    retriever = bm25s.BM25(method="bm25l")
    retriever.index(corpus_tokens)
    return {
        "tokenizer": tokenizer,
        "retriever": retriever,
        "paths": [row["path"] for row in rows],
    }


def _run_bm25_query(bm25_index: dict[str, Any], query: str, limit: int) -> list[str]:
    query_tokens = bm25_index["tokenizer"].tokenize([query])
    k = min(limit, len(bm25_index["paths"]))
    doc_ids, _scores = bm25_index["retriever"].retrieve(query_tokens, k=k)
    ranked_paths: list[str] = []
    for doc_id in doc_ids[0]:
        if doc_id < 0:
            continue
        ranked_paths.append(bm25_index["paths"][int(doc_id)])
    return ranked_paths


def _rrf_fuse_rankings(rankings: list[list[str]], limit: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, path in enumerate(ranking):
            if path in seen:
                continue
            seen.add(path)
            scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank + 1)
    return [path for path, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _reciprocal_rank(ranked_docs: list[str], relevant: set[str]) -> float:
    for idx, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1.0 / idx
    return 0.0


def _recall_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc in ranked_docs[:k] if doc in relevant)
    return hits / len(relevant)


def _ndcg_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc in enumerate(ranked_docs[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _evaluate_variant(
    db_path: str,
    dataset: dict[str, Any],
    variant_name: str,
    variant: VariantSpec,
    limit: int,
    indexed_files: set[str],
    bm25_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = []
    success_at_1_total = 0.0
    mrr_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0
    latency_total_ms = 0.0
    timing_totals: dict[str, float] = {}
    evaluated = 0
    skipped = 0

    for query_case in dataset["queries"]:
        relevant = set(query_case["relevant"])
        preferred_top = query_case.get("preferred_top")
        missing = sorted(relevant - indexed_files)
        if missing:
            skipped += 1
            cases.append(
                {
                    "id": query_case["id"],
                    "query": query_case["query"],
                    "skipped": True,
                    "reason": f"relevant_docs_not_indexed: {', '.join(missing)}",
                }
            )
            continue

        t0 = time.perf_counter()
        response: dict[str, Any] | None = None
        if variant.kind == "search":
            response = search_with_options(
                query_case["query"],
                db_path=db_path,
                limit=limit,
                options=variant.options,
            )
            ranked_docs = [os.path.basename(row["path"]) for row in response.get("results", [])]
        elif variant.kind == "bm25l":
            if bm25_index is None:
                raise RuntimeError("bm25 index missing for bm25 variant")
            ranked_paths = _run_bm25_query(bm25_index, query_case["query"], limit)
            ranked_docs = [os.path.basename(path) for path in ranked_paths]
        elif variant.kind == "fts_bm25l_rrf":
            if bm25_index is None:
                raise RuntimeError("bm25 index missing for fusion variant")
            response = search_with_options(
                query_case["query"],
                db_path=db_path,
                limit=max(limit, 20),
                options=FTS_ONLY_OPTIONS,
            )
            fts_paths = [row["path"] for row in response.get("results", [])]
            bm25_paths = _run_bm25_query(bm25_index, query_case["query"], max(limit, 20))
            ranked_paths = _rrf_fuse_rankings([fts_paths, bm25_paths], limit)
            ranked_docs = [os.path.basename(path) for path in ranked_paths]
        else:
            raise RuntimeError(f"Unknown variant kind: {variant.kind}")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        stage_timing = response.get("timing", {}) if response else {}
        for key, value in stage_timing.items():
            timing_totals[key] = timing_totals.get(key, 0.0) + float(value)

        if preferred_top:
            success_at_1 = 1.0 if ranked_docs[:1] and ranked_docs[0] == preferred_top else 0.0
        else:
            success_at_1 = 1.0 if ranked_docs[:1] and ranked_docs[0] in relevant else 0.0
        rr = _reciprocal_rank(ranked_docs, relevant)
        recall = _recall_at_k(ranked_docs, relevant, limit)
        ndcg = _ndcg_at_k(ranked_docs, relevant, limit)

        success_at_1_total += success_at_1
        mrr_total += rr
        recall_total += recall
        ndcg_total += ndcg
        latency_total_ms += latency_ms
        evaluated += 1

        cases.append(
            {
                "id": query_case["id"],
                "query": query_case["query"],
                "relevant": sorted(relevant),
                "preferred_top": preferred_top or "",
                "returned": ranked_docs,
                "success_at_1": success_at_1,
                "reciprocal_rank": round(rr, 4),
                "recall_at_k": round(recall, 4),
                "ndcg_at_k": round(ndcg, 4),
                "latency_ms": latency_ms,
                "timing": stage_timing,
            }
        )

    denom = evaluated or 1
    avg_stage_timing = {key: round(total / denom, 3) for key, total in sorted(timing_totals.items())}
    return {
        "variant": variant_name,
        "kind": variant.kind,
        "evaluated_queries": evaluated,
        "skipped_queries": skipped,
        "success_at_1": round(success_at_1_total / denom, 4),
        "mrr": round(mrr_total / denom, 4),
        "recall_at_k": round(recall_total / denom, 4),
        "ndcg_at_k": round(ndcg_total / denom, 4),
        "avg_latency_ms": round(latency_total_ms / denom, 2),
        "avg_stage_timing_ms": avg_stage_timing,
        "cases": cases,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"Dataset: {report['dataset']['name']} (limit={report['limit']})")
    print(f"Indexed files: {len(report['indexing']['indexed_files'])}")
    if report["indexing"]["skipped_files"]:
        skipped = ", ".join(f"{item['file']}[{item['reason']}]" for item in report["indexing"]["skipped_files"])
        print(f"Skipped during indexing: {skipped}")
    print("")
    print("variant                  s@1    mrr    recall  ndcg   avg_ms  eval/skipped")
    print("--------------------------------------------------------------------------")
    for result in report["results"]:
        print(
            f"{result['variant']:<24} "
            f"{result['success_at_1']:<6.4f} "
            f"{result['mrr']:<6.4f} "
            f"{result['recall_at_k']:<7.4f} "
            f"{result['ndcg_at_k']:<6.4f} "
            f"{result['avg_latency_ms']:<7.2f} "
            f"{result['evaluated_queries']}/{result['skipped_queries']}"
        )
        if result.get("avg_stage_timing_ms"):
            stage_summary = ", ".join(f"{k}={v:.2f}" for k, v in result["avg_stage_timing_ms"].items())
            print(f"  timings: {stage_summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Pinpoint search variants on an offline benchmark.")
    parser.add_argument("--dataset", default="benchmarks/search_relevance.json", help="Path to relevance dataset JSON")
    parser.add_argument("--corpus", default="test_data", help="Folder to index for evaluation")
    parser.add_argument("--limit", type=int, default=5, help="Top-k cutoff for metrics")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), help="Variant names to evaluate")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    parser.add_argument("--db-path", default="", help="Optional SQLite DB path to use instead of a temp DB")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    corpus_dir = Path(args.corpus)
    dataset = _load_dataset(dataset_path)

    unknown_variants = [name for name in args.variants if name not in VARIANTS]
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {', '.join(unknown_variants)}")
    if not corpus_dir.is_dir():
        raise SystemExit(f"Corpus folder not found: {corpus_dir}")

    if args.db_path:
        db_path = args.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = _open_eval_db(db_path)
        try:
            indexing_info = _index_corpus(
                conn,
                corpus_dir,
                embeddings_enabled=any((VARIANTS[name].options and VARIANTS[name].options.use_embeddings) for name in args.variants),
            )
        finally:
            conn.close()
        indexed_files = set(indexing_info["indexed_files"])
        bm25_index = _build_bm25_index(db_path) if any(VARIANTS[name].kind in {"bm25l", "fts_bm25l_rrf"} for name in args.variants) else None
        results = [
            _evaluate_variant(db_path, dataset, variant_name, VARIANTS[variant_name], args.limit, indexed_files, bm25_index)
            for variant_name in args.variants
        ]
    else:
        with tempfile.TemporaryDirectory(prefix="pinpoint-search-eval-") as tmpdir:
            db_path = str(Path(tmpdir) / "search_eval.db")
            conn = _open_eval_db(db_path)
            try:
                indexing_info = _index_corpus(
                    conn,
                    corpus_dir,
                    embeddings_enabled=any((VARIANTS[name].options and VARIANTS[name].options.use_embeddings) for name in args.variants),
                )
            finally:
                conn.close()

            indexed_files = set(indexing_info["indexed_files"])
            bm25_index = _build_bm25_index(db_path) if any(VARIANTS[name].kind in {"bm25l", "fts_bm25l_rrf"} for name in args.variants) else None
            results = [
                _evaluate_variant(db_path, dataset, variant_name, VARIANTS[variant_name], args.limit, indexed_files, bm25_index)
                for variant_name in args.variants
            ]

    report = {
        "dataset": {"name": dataset.get("name", dataset_path.name), "path": str(dataset_path), "version": dataset.get("version", 1)},
        "corpus": str(corpus_dir),
        "limit": args.limit,
        "indexing": indexing_info,
        "environment": {
            "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            "semantic_search_effective": indexing_info["embedded_chunks"] > 0,
        },
        "results": results,
    }

    _print_summary(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Concurrent search load-test harness for Pinpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database import init_db
from evaluate_search import FTS_ONLY_OPTIONS, _index_corpus, _load_dataset
from search_pipeline import SearchOptions, search_with_options


@dataclass(frozen=True)
class LoadTestOptions:
    concurrency: int
    rounds: int
    limit: int
    options: SearchOptions


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
        "max_ms": round(max(values), 3),
    }


def _summarize_stage_timings(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for timing in timings for key in timing})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(timing[key]) for timing in timings if key in timing]
        summary[key] = _summarize_latency(values)
    return summary


def _run_one_query(query: str, db_path: str, limit: int, options: SearchOptions) -> dict[str, Any]:
    started = time.perf_counter()
    response = search_with_options(query, db_path=db_path, limit=limit, options=options)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "query": query,
        "result_count": len(response.get("results", [])),
        "wall_ms": round(wall_ms, 3),
        "timing": response.get("timing", {}),
        "ambiguous_search": bool(response.get("ambiguous_search")),
        "enhanced_search_used": bool(response.get("enhanced_search_used")),
    }


def _load_queries(dataset: dict[str, Any]) -> list[str]:
    return [str(item["query"]) for item in dataset.get("queries", []) if item.get("query")]


def run_load_test(
    *,
    db_path: str,
    dataset: dict[str, Any],
    options: LoadTestOptions,
) -> dict[str, Any]:
    queries = _load_queries(dataset)
    if not queries:
        raise ValueError("Dataset contains no queries")

    tasks = []
    for _round in range(options.rounds):
        tasks.extend(queries)

    total_requests = len(tasks)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=options.concurrency) as executor:
        futures = [
            executor.submit(_run_one_query, query, db_path, options.limit, options.options)
            for query in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration_s = max(time.perf_counter() - started, 1e-9)
    wall_values = [float(item["wall_ms"]) for item in results]
    stage_timings = [item["timing"] for item in results]
    ambiguous_count = sum(1 for item in results if item["ambiguous_search"])
    enhanced_count = sum(1 for item in results if item["enhanced_search_used"])

    return {
        "total_requests": total_requests,
        "concurrency": options.concurrency,
        "rounds": options.rounds,
        "throughput_qps": round(total_requests / duration_s, 3),
        "wall_latency_ms": _summarize_latency(wall_values),
        "stage_latency_ms": _summarize_stage_timings(stage_timings),
        "ambiguous_rate": round(ambiguous_count / total_requests, 4),
        "enhanced_search_rate": round(enhanced_count / total_requests, 4),
        "sample_results": results[: min(10, len(results))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concurrent search load testing against a benchmark corpus.")
    parser.add_argument("--dataset", default="benchmarks/search_relevance_v4_mixed.json", help="Path to relevance dataset JSON")
    parser.add_argument("--corpus", default="benchmarks/corpus_v4_mixed", help="Path to corpus directory")
    parser.add_argument("--db", default="", help="SQLite database path to use; defaults to a temp DB")
    parser.add_argument("--output", default="", help="Optional path to write JSON report")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent search workers")
    parser.add_argument("--rounds", type=int, default=10, help="How many times to replay the query set")
    parser.add_argument("--limit", type=int, default=5, help="Search result limit")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    corpus_path = Path(args.corpus)
    dataset = _load_dataset(dataset_path)

    if args.db:
        db_path = args.db
        tempdir = None
    else:
        tempdir = tempfile.TemporaryDirectory(prefix="pinpoint-load-")
        db_path = str(Path(tempdir.name) / "loadtest.sqlite3")

    conn = init_db(db_path)
    try:
        index_summary = _index_corpus(
            conn,
            corpus_path,
            embeddings_enabled=False,
        )
    finally:
        conn.close()

    report = {
        "dataset": str(dataset_path),
        "corpus": str(corpus_path),
        "db_path": db_path,
        "indexed_files": len(index_summary["indexed_files"]),
        "skipped_files": index_summary["skipped_files"],
        "options": {
            "concurrency": args.concurrency,
            "rounds": args.rounds,
            "limit": args.limit,
            "variant": "fts_only",
        },
        "summary": run_load_test(
            db_path=db_path,
            dataset=dataset,
            options=LoadTestOptions(
                concurrency=args.concurrency,
                rounds=args.rounds,
                limit=args.limit,
                options=FTS_ONLY_OPTIONS,
            ),
        ),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {output_path}")
    else:
        print(json.dumps(report, indent=2))

    if tempdir is not None:
        tempdir.cleanup()


if __name__ == "__main__":
    main()

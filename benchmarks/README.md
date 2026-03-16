# Search Benchmark

This folder holds the offline relevance benchmark for Pinpoint search.

## Run

```bash
python evaluate_search.py
python evaluate_search.py --limit 5 --output benchmarks/latest_report.json
python evaluate_search.py --dataset benchmarks/search_relevance_expanded.json --corpus benchmarks/corpus --output benchmarks/expanded_report.json
python evaluate_search.py --dataset benchmarks/search_relevance_semantic.json --corpus benchmarks/corpus_semantic --output benchmarks/semantic_report.json
python evaluate_search.py --dataset benchmarks/search_relevance_v2.json --corpus benchmarks/corpus_v2 --output benchmarks/v2_report.json
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --output benchmarks/v4_mixed_report.json
```

Benchmark preparation helpers live under `benchmarks/scripts/`.

Example:

```bash
python benchmarks/scripts/build_legal_benchmark_v3.py --help
```

## What it measures

- `fts_only`
- `fts_embeddings`
- `full_pipeline`
- `full_no_position_blend`

Metrics:

- `success_at_1`
- `mrr`
- `recall_at_k`
- `ndcg_at_k`
- `avg_latency_ms`

## Current scope

`search_relevance.json` is intentionally grounded in the local sample corpus that is extractable in a minimal environment:

- `client_database.csv`
- `expenses_2024.xlsx`
- `meeting_minutes_q1_review.txt`

This is the first Phase 3 baseline, not the final dataset. Expand it once richer extractors are available for PDF, DOCX, PPTX, and OCR-heavy images.

## Expanded corpus

`search_relevance_expanded.json` and `benchmarks/corpus/` provide a larger runnable benchmark that is independent of OCR and office-document extractors.

Use it to:

- validate lexical fallback behavior
- compare retrieval variants on more varied queries
- expose paraphrase gaps when embeddings are unavailable or unhelpful

## Semantic challenge

`search_relevance_semantic.json` and `benchmarks/corpus_semantic/` are intended to answer one question:

Does semantic retrieval help on paraphrase-style queries enough to justify its cost?

Check these report fields before drawing conclusions:

- `environment.gemini_api_key_configured`
- `environment.semantic_search_effective`
- `indexing.embedded_chunks`

If `semantic_search_effective` is `false`, the semantic variants are not actually exercising embeddings and should not be used to justify keeping them.

## Benchmark V2

`search_relevance_v2.json` is the harder benchmark:

- more overlapping documents
- paraphrase-style queries with weaker lexical overlap
- explicit `preferred_top` expectations for ambiguous ranking cases

This is the benchmark that should decide whether semantic retrieval is worth keeping for Pinpoint.

## Benchmark V4 Mixed Domain

`search_relevance_v4_mixed.json` and `benchmarks/corpus_v4_mixed/` are the closest benchmark to Pinpoint's intended day-to-day usage.

It mixes:

- notes and meeting minutes
- support/operations docs
- invoice and reimbursement text
- employee/policy documents
- CSV metadata lookups

Use it to judge:

- lexical-first quality on realistic product queries
- metadata-heavy retrieval behavior
- stage timing on a mixed-domain corpus before making more Phase 5 search changes

For concurrent lexical-first load profiling on the same mixed-domain benchmark, use:

```bash
python load_test_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8 --output benchmarks/v4_mixed_load_report.json
```

This records:
- wall-clock latency percentiles (`avg`, `p50`, `p95`, `max`)
- per-stage timing percentiles from the search pipeline
- throughput in queries/second
- ambiguity and enhanced-search rates under concurrent load

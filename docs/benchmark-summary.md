# Benchmark Summary

This page summarizes the benchmark results that currently matter for Pinpoint.

It is intentionally short and product-facing. For raw datasets, scripts, and reports, see [benchmarks/README.md](../benchmarks/README.md).

## What Pinpoint Measures

Pinpoint currently uses two benchmark styles:

- **Offline relevance benchmarks**: judged queries against a fixed local corpus
- **Concurrent load benchmarks**: repeated search under small concurrent load

Main metrics:

- `success@1`: how often the preferred result is ranked first
- `MRR`: whether relevant results appear near the top, even when not first
- `recall@5`: whether relevant results are found in the top 5
- `NDCG`: ranking quality across the result set
- latency / throughput: how fast the current search path responds

## Current Product-Shaped Benchmark

The most representative benchmark for day-to-day Pinpoint usage is:

- dataset: `benchmarks/search_relevance_v4_mixed.json`
- corpus: `benchmarks/corpus_v4_mixed/`
- report: `benchmarks/v4_mixed_lexical_compare_fixed.json`

This benchmark mixes:

- notes and meeting minutes
- support and runbook docs
- invoices and reimbursement text
- employee/policy content
- CSV lookups

Current `fts_only` result on that mixed-domain benchmark:

- `success@1`: `0.9444`
- `MRR`: `1.0000`
- `recall@5`: `1.0000`
- `NDCG`: `1.0000`
- average query latency: about `4.31 ms`

Practical takeaway:

- the default lexical-first search path is currently the best fit for Pinpoint's mixed-domain corpus
- embeddings are not the default retrieval path
- ambiguity handling matters more than adding more ranking layers

## Current Load Result

Concurrent load on the same mixed-domain benchmark:

- report: `benchmarks/v4_mixed_load_report.json`
- variant: `fts_only`
- concurrency: `2`
- rounds: `1`

Current result:

- throughput: about `216 QPS`
- average wall latency: about `9.16 ms`
- `p95` wall latency: about `11.51 ms`

Observed hotspot:

- the dominant search cost is still the lexical retrieval stage
- fusion and reranking are effectively negligible on the default path

## Stress Benchmark

Pinpoint also keeps a denser legal-text benchmark as a stress test:

- report: `benchmarks/v3_legal/bm25l_report.json`

Use that benchmark to:

- expose lexical fallback mistakes
- expose ambiguity-heavy ranking failures
- stress retrieval on highly overlapping text

Do **not** treat that legal benchmark as the product identity. It is a stress workload, not the target corpus.

## What These Benchmarks Do Not Prove

These results do **not** mean:

- Pinpoint has been validated on every corpus shape
- semantic retrieval is useless in all domains
- current latency numbers will hold at arbitrarily larger corpus sizes
- the WhatsApp bot quality is fully measured by search benchmarks alone

They do support these decisions:

- keep `FTS5` as the production baseline
- keep semantic retrieval optional/experimental
- measure search changes before making them default
- prefer product-facing fixes like clarification over speculative ranking complexity

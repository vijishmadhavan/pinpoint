from __future__ import annotations

from load_test_search import _percentile, _summarize_latency, _summarize_stage_timings


def test_percentile_interpolates_sorted_values():
    values = [1.0, 2.0, 3.0, 4.0]

    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 50) == 2.5
    assert round(_percentile(values, 95), 3) == 3.85
    assert _percentile(values, 100) == 4.0


def test_summarize_stage_timings_builds_per_stage_percentiles():
    summary = _summarize_stage_timings(
        [
            {"lexical_ms": 2.0, "total_ms": 3.0},
            {"lexical_ms": 4.0, "total_ms": 5.0},
            {"lexical_ms": 6.0, "total_ms": 7.0},
        ]
    )

    assert summary["lexical_ms"] == {
        "avg_ms": 4.0,
        "p50_ms": 4.0,
        "p95_ms": 5.8,
        "max_ms": 6.0,
    }
    assert summary["total_ms"]["avg_ms"] == 5.0


def test_summarize_latency_handles_empty_values():
    assert _summarize_latency([]) == {
        "avg_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }

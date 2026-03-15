from __future__ import annotations

from unittest.mock import patch

from evaluate_search import FTS_ONLY_OPTIONS, VariantSpec, _evaluate_variant


class TestEvaluateSearch:
    def test_run_bm25_query_caps_k_to_corpus_size(self):
        from evaluate_search import _run_bm25_query

        class FakeTokenizer:
            def tokenize(self, queries):
                return queries

        class FakeRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, query_tokens, k):
                self.calls.append(k)
                return [[0, 1]], [[1.0, 0.5]]

        retriever = FakeRetriever()
        bm25_index = {
            "tokenizer": FakeTokenizer(),
            "retriever": retriever,
            "paths": ["/tmp/a.txt", "/tmp/b.txt"],
        }

        ranked = _run_bm25_query(bm25_index, "alpha", 20)

        assert retriever.calls == [2]
        assert ranked == ["/tmp/a.txt", "/tmp/b.txt"]

    def test_search_variant_aggregates_stage_timing(self, tmp_path):
        db_path = str(tmp_path / "eval.db")
        dataset = {
            "queries": [
                {"id": "q1", "query": "alpha", "relevant": ["a.txt"]},
                {"id": "q2", "query": "beta", "relevant": ["b.txt"]},
            ]
        }
        indexed_files = {"a.txt", "b.txt"}
        variant = VariantSpec("search", FTS_ONLY_OPTIONS)
        responses = [
            {
                "results": [{"path": "/tmp/a.txt"}],
                "timing": {"lexical_ms": 2.0, "probe_ms": 1.0, "total_ms": 4.0},
            },
            {
                "results": [{"path": "/tmp/b.txt"}],
                "timing": {"lexical_ms": 4.0, "probe_ms": 3.0, "total_ms": 8.0},
            },
        ]

        with patch("evaluate_search.search_with_options", side_effect=responses):
            result = _evaluate_variant(db_path, dataset, "fts_only", variant, 5, indexed_files)

        assert result["avg_stage_timing_ms"]["lexical_ms"] == 3.0
        assert result["avg_stage_timing_ms"]["probe_ms"] == 2.0
        assert result["avg_stage_timing_ms"]["total_ms"] == 6.0
        assert result["cases"][0]["timing"]["lexical_ms"] == 2.0
        assert result["cases"][1]["timing"]["probe_ms"] == 3.0

    def test_non_search_variant_has_empty_stage_timing(self, tmp_path):
        db_path = str(tmp_path / "eval.db")
        dataset = {"queries": [{"id": "q1", "query": "alpha", "relevant": ["a.txt"]}]}
        indexed_files = {"a.txt"}
        variant = VariantSpec("bm25l")
        bm25_index = {"tokenizer": object(), "retriever": object(), "paths": ["/tmp/a.txt"]}

        with patch("evaluate_search._run_bm25_query", return_value=["/tmp/a.txt"]):
            result = _evaluate_variant(db_path, dataset, "bm25l_only", variant, 5, indexed_files, bm25_index)

        assert result["avg_stage_timing_ms"] == {}
        assert result["cases"][0]["timing"] == {}

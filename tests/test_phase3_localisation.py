"""
tests/test_phase3_localisation.py
──────────────────────────────────
Unit tests for Phase 3: BM25, RRF fusion, DeBERTa ranker, and pipeline.

All tests work without OpenAI API key or GPU — components degrade gracefully.

Run with: pytest tests/test_phase3_localisation.py -v
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_file_symbols(file_path: str, summary: str = ""):
    """Create a minimal FileSymbols for testing."""
    from ast_parser.python_parser import FileSymbols, FunctionInfo
    fs = FileSymbols(file_path=file_path, file_hash="aaa111")
    fs.module_docstring = summary
    # Also add a fake function whose name contains the summary words
    # so summary_text is fully populated
    if summary:
        fs.functions = [
            FunctionInfo(
                name=summary.split()[0] if summary.split() else "placeholder",
                qualified_name=summary.split()[0] if summary.split() else "placeholder",
                args=[], decorators=[], docstring=summary,
                start_line=1, end_line=5,
            )
        ]
    return fs


# ── BM25 tokeniser ────────────────────────────────────────────────────────────

class TestTokeniser:
    def test_lowercase(self):
        from localisation.bm25_retriever import _tokenise
        result = _tokenise("Hello World")
        assert all(t == t.lower() for t in result)

    def test_camel_case_split(self):
        from localisation.bm25_retriever import _tokenise
        result = _tokenise("QuerySet")
        assert "query" in result
        assert "set" in result

    def test_snake_case_split(self):
        from localisation.bm25_retriever import _tokenise
        result = _tokenise("get_queryset")
        assert "get" in result
        assert "queryset" in result

    def test_short_tokens_filtered(self):
        from localisation.bm25_retriever import _tokenise
        result = _tokenise("a b c def")
        assert "a" not in result
        assert "b" not in result
        assert "def" in result

    def test_path_tokenisation(self):
        from localisation.bm25_retriever import _tokenise
        result = _tokenise("django/db/models/query.py")
        assert "django" in result
        assert "models" in result
        assert "query" in result


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_index_and_query_basic(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        symbols = [
            make_file_symbols("models/user.py", "User authentication login password"),
            make_file_symbols("views/dashboard.py", "Dashboard render template"),
            make_file_symbols("utils/email.py", "Email sending SMTP"),
        ]
        retriever.index(symbols)
        hits = retriever.query("user authentication login", top_k=3)
        assert len(hits) >= 1
        assert hits[0].file_path == "models/user.py"

    def test_query_returns_positive_scores_only(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        symbols = [make_file_symbols(f"file{i}.py", f"content {i}") for i in range(5)]
        retriever.index(symbols)
        hits = retriever.query("content 3", top_k=5)
        assert all(h.score > 0 for h in hits)

    def test_ranks_are_sequential(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        symbols = [make_file_symbols(f"f{i}.py", f"word{i} text") for i in range(3)]
        retriever.index(symbols)
        hits = retriever.query("word0 text", top_k=3)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_empty_query_returns_empty(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        symbols = [make_file_symbols("a.py", "content")]
        retriever.index(symbols)
        hits = retriever.query("", top_k=5)
        assert hits == []

    def test_corpus_size(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        symbols = [make_file_symbols(f"f{i}.py", "text") for i in range(7)]
        retriever.index(symbols)
        assert retriever.corpus_size == 7

    def test_file_path_tokens_boost(self):
        from localisation.bm25_retriever import BM25Retriever
        # Both files have 'models' in content. But models.py ALSO has it in
        # path (doubled) — with a larger corpus that gives positive BM25 scores.
        retriever = BM25Retriever()
        symbols = [
            make_file_symbols("django/db/models.py", "handles database records"),
            make_file_symbols("utils/helper.py", "general utilities helper"),
            make_file_symbols("views/base.py", "base view rendering"),
            make_file_symbols("core/app.py", "application entry point"),
            make_file_symbols("api/serializers.py", "rest framework serializers"),
        ]
        retriever.index(symbols)
        hits = retriever.query("models", top_k=5)
        # models.py has 'models' in path (2x weight) — must appear in results
        paths = [h.file_path for h in hits]
        assert "django/db/models.py" in paths

    def test_not_indexed_raises(self):
        from localisation.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="not indexed"):
            retriever.query("test", top_k=5)

    def test_skips_parse_error_files(self):
        from localisation.bm25_retriever import BM25Retriever
        from ast_parser.python_parser import FileSymbols
        retriever = BM25Retriever()
        good = make_file_symbols("good.py", "good content")
        bad = FileSymbols(file_path="bad.py", file_hash="bbb", parse_error="SyntaxError")
        retriever.index([good, bad])
        assert retriever.corpus_size == 1


# ── RRF Fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:
    def test_basic_fusion(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [("a.py", 1.0, 1), ("b.py", 0.8, 2), ("c.py", 0.5, 3)]
        embed = [("b.py", 0.9, 1), ("a.py", 0.7, 2), ("d.py", 0.6, 3)]
        ppr = {"a.py": 0.5, "b.py": 0.3}

        result = reciprocal_rank_fusion(bm25, embed, ppr, top_k=4)
        assert len(result) <= 4
        # a.py appears in all three → should rank high
        top_paths = [h.file_path for h in result]
        assert "a.py" in top_paths[:2]

    def test_top_k_respected(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [(f"f{i}.py", 1.0, i + 1) for i in range(10)]
        result = reciprocal_rank_fusion(bm25, [], {}, top_k=3)
        assert len(result) == 3

    def test_empty_inputs(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [], {}, top_k=5)
        assert result == []

    def test_ranks_sequential(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [("a.py", 1.0, 1), ("b.py", 0.5, 2)]
        result = reciprocal_rank_fusion(bm25, [], {}, top_k=5)
        assert [h.rank for h in result] == list(range(1, len(result) + 1))

    def test_all_sources_tracked(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [("a.py", 1.0, 1)]
        embed = [("a.py", 0.9, 1)]
        ppr = {"a.py": 0.5}
        result = reciprocal_rank_fusion(bm25, embed, ppr, top_k=1)
        hit = result[0]
        assert hit.bm25_rank == 1
        assert hit.embed_rank == 1
        assert hit.ppr_rank == 1

    def test_ablation_no_ppr(self):
        from localisation.rrf_fusion import ablate
        bm25 = [("a.py", 1.0, 1)]
        ppr = {"b.py": 99.0}  # b.py has huge PPR score
        # With PPR zeroed out, b.py should NOT appear
        result = ablate(bm25, [], ppr, use_ppr=False, top_k=5)
        paths = [h.file_path for h in result]
        assert "b.py" not in paths

    def test_scores_descending(self):
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [("a.py", 1.0, 1), ("b.py", 0.5, 2), ("c.py", 0.1, 3)]
        result = reciprocal_rank_fusion(bm25, [], {}, top_k=3)
        scores = [h.fused_score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_all_lists(self):
        """File appearing only in PPR should still be in results."""
        from localisation.rrf_fusion import reciprocal_rank_fusion
        bm25 = [("a.py", 1.0, 1)]
        ppr = {"z.py": 1.0}  # only in PPR
        result = reciprocal_rank_fusion(bm25, [], ppr, top_k=10)
        paths = [h.file_path for h in result]
        assert "z.py" in paths


# ── DeBERTa Ranker — without GPU ──────────────────────────────────────────────

class TestDeBERTaRankerFallback:
    """Tests for graceful fallback when model is not loaded."""

    def test_rerank_fallback_returns_stage1_order(self):
        from localisation.deberta_ranker import DeBERTaRanker
        # Don't actually load the model
        ranker = DeBERTaRanker.__new__(DeBERTaRanker)
        ranker._available = False
        ranker._model = None
        ranker._tokenizer = None

        candidates = [("a.py", "summary a"), ("b.py", "summary b"), ("c.py", "summary c")]
        result = ranker.rerank("fix the bug", candidates, top_k=3)
        assert len(result) == 3
        assert result[0].file_path == "a.py"
        assert result[0].rank == 1

    def test_rerank_empty_candidates(self):
        from localisation.deberta_ranker import DeBERTaRanker
        ranker = DeBERTaRanker.__new__(DeBERTaRanker)
        ranker._available = False
        result = ranker.rerank("query", [], top_k=5)
        assert result == []

    def test_ranked_file_scores_are_positive(self):
        from localisation.deberta_ranker import DeBERTaRanker
        ranker = DeBERTaRanker.__new__(DeBERTaRanker)
        ranker._available = False
        candidates = [(f"f{i}.py", f"text {i}") for i in range(5)]
        result = ranker.rerank("test query", candidates, top_k=5)
        assert all(r.relevance_score > 0 for r in result)


# ── Recall metric ─────────────────────────────────────────────────────────────

class TestRecallMetric:
    def test_perfect_recall(self):
        from localisation.deberta_ranker import recall_at_k
        preds = ["a.py", "b.py", "c.py"]
        gold = ["a.py", "b.py"]
        assert recall_at_k(preds, gold, k=5) == 1.0

    def test_zero_recall(self):
        from localisation.deberta_ranker import recall_at_k
        preds = ["x.py", "y.py"]
        gold = ["a.py"]
        assert recall_at_k(preds, gold, k=5) == 0.0

    def test_partial_recall(self):
        from localisation.deberta_ranker import recall_at_k
        preds = ["a.py", "b.py", "c.py"]
        gold = ["a.py", "z.py"]
        assert recall_at_k(preds, gold, k=5) == 0.5

    def test_recall_at_k_respects_k(self):
        from localisation.deberta_ranker import recall_at_k
        preds = ["x.py", "a.py"]  # a.py is at position 2
        gold = ["a.py"]
        assert recall_at_k(preds, gold, k=1) == 0.0   # only looking at top-1
        assert recall_at_k(preds, gold, k=2) == 1.0

    def test_empty_gold(self):
        from localisation.deberta_ranker import recall_at_k
        assert recall_at_k(["a.py"], [], k=5) == 0.0


# ── Patch file extraction ─────────────────────────────────────────────────────

class TestExtractFilesFromPatch:
    def test_basic_unified_diff(self):
        from localisation.deberta_ranker import _extract_files_from_patch
        patch = textwrap.dedent("""
            diff --git a/django/db/models/query.py b/django/db/models/query.py
            --- a/django/db/models/query.py
            +++ b/django/db/models/query.py
            @@ -1 +1 @@
            -old
            +new
        """)
        files = _extract_files_from_patch(patch)
        assert "django/db/models/query.py" in files

    def test_multiple_files(self):
        from localisation.deberta_ranker import _extract_files_from_patch
        patch = textwrap.dedent("""
            --- a/foo.py
            +++ b/foo.py
            @@ -1 +1 @@ fix
            --- a/bar.py
            +++ b/bar.py
            @@ -1 +1 @@ fix
        """)
        files = _extract_files_from_patch(patch)
        assert "foo.py" in files
        assert "bar.py" in files

    def test_dev_null_excluded(self):
        from localisation.deberta_ranker import _extract_files_from_patch
        patch = "--- /dev/null\n+++ b/new_file.py\n"
        files = _extract_files_from_patch(patch)
        assert "/dev/null" not in files

    def test_empty_patch(self):
        from localisation.deberta_ranker import _extract_files_from_patch
        assert _extract_files_from_patch("") == []


# ── Failure categorisation ────────────────────────────────────────────────────

class TestFailureCategorisation:
    def test_success(self):
        from localisation.pipeline import categorise_localisation_failure
        result = categorise_localisation_failure(["a.py", "b.py"], ["a.py"], "good long detailed issue text here")
        assert result == "success"

    def test_wrong_file(self):
        from localisation.pipeline import categorise_localisation_failure
        # Long issue text (>10 words) + no gold file found → wrong_file
        long_issue = "there is a null pointer exception in the query filter method"
        result = categorise_localisation_failure(["x.py", "y.py"], ["z.py"], long_issue)
        assert result == "wrong_file"

    def test_partial_file(self):
        from localisation.pipeline import categorise_localisation_failure
        result = categorise_localisation_failure(["a.py"], ["a.py", "b.py"], "long enough issue text to be valid")
        assert result == "partial_file"

    def test_ambiguous_issue(self):
        from localisation.pipeline import categorise_localisation_failure
        result = categorise_localisation_failure(["x.py"], ["z.py"], "fix bug")  # very short
        assert result == "ambiguous_issue"


# ── Pipeline integration (no API required) ────────────────────────────────────

class TestLocalisationPipeline:
    def test_pipeline_bm25_only(self):
        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(
            use_embeddings=False,
            use_deberta=False,
            use_ppr=False,
        )
        symbols = [
            make_file_symbols("auth/models.py", "User model authentication password hash"),
            make_file_symbols("views/login.py", "Login view render form"),
            make_file_symbols("utils/email.py", "Email SMTP send message"),
        ]
        pipeline.index_repo(symbols)
        result = pipeline.localise("user authentication login password", top_k=3)
        assert len(result.hits) >= 1
        assert result.hits[0].file_path == "auth/models.py"

    def test_pipeline_empty_query(self):
        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(use_embeddings=False, use_deberta=False)
        symbols = [make_file_symbols("a.py", "content")]
        pipeline.index_repo(symbols)
        result = pipeline.localise("")
        assert result.failure_category == "empty_query"
        assert result.hits == []

    def test_pipeline_with_gold_files_computes_recall(self):
        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(use_embeddings=False, use_deberta=False, use_ppr=False)
        # Use a larger corpus so BM25 gives positive scores
        # 'queryset' appears in path AND content of target.py → guaranteed top-1
        symbols = [
            make_file_symbols("db/queryset.py", "queryset filter method database orm"),
            make_file_symbols("views/generic.py", "generic view rendering template"),
            make_file_symbols("utils/helper.py", "utility functions general purpose"),
            make_file_symbols("api/serializer.py", "rest framework serializer fields"),
            make_file_symbols("forms/widget.py", "html form widget rendering input"),
        ]
        pipeline.index_repo(symbols)
        result = pipeline.localise(
            "fix null pointer exception in queryset filter", top_k=5,
            gold_files=["db/queryset.py"]
        )
        assert result.recall_at_5 is not None
        assert result.recall_at_10 is not None
        assert result.recall_at_5 == 1.0  # queryset in path+content guarantees top rank

    def test_top_k_paths_property(self):
        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(use_embeddings=False, use_deberta=False, use_ppr=False)
        symbols = [make_file_symbols(f"f{i}.py", f"content {i}") for i in range(5)]
        pipeline.index_repo(symbols)
        result = pipeline.localise("content 1", top_k=3)
        assert len(result.top_k_paths) == len(result.hits)

    def test_hit_diagnostic_flags(self):
        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(use_embeddings=False, use_deberta=False, use_ppr=False)
        symbols = [make_file_symbols("a.py", "special word")]
        pipeline.index_repo(symbols)
        result = pipeline.localise("special word", top_k=1)
        if result.hits:
            hit = result.hits[0]
            assert hit.in_bm25 is True

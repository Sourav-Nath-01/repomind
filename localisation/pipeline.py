"""
localisation/pipeline.py
─────────────────────────
Full two-stage localisation pipeline.

Stage 1: BM25 + Embeddings (coarse ranking) → RRF fusion
Stage 2: DeBERTa cross-encoder (precision re-ranking)

Also handles:
  - Failure categorisation (wrong-file, partial-file, missing-dependency, ambiguous-issue)
  - MLflow cost tracking per retrieval call
  - Context budget enforcement (top-K files only)

Usage:
    pipeline = LocalisationPipeline(cache_dir=Path(".cache"))
    pipeline.index_repo(file_symbols, dependency_graph)
    
    result = pipeline.localise(
        issue_text="Fix null pointer in QuerySet.filter()",
        top_k=5,
    )
    for hit in result.hits:
        print(hit.file_path, hit.relevance_score)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class LocalisationHit:
    file_path: str
    relevance_score: float
    rank: int
    # Diagnostic: which stages contributed
    in_bm25: bool = False
    in_embed: bool = False
    in_ppr: bool = False
    bm25_rank: Optional[int] = None
    embed_rank: Optional[int] = None
    ppr_rank: Optional[int] = None


@dataclass
class LocalisationResult:
    hits: list[LocalisationHit]
    elapsed_seconds: float
    failure_category: Literal[
        "success",
        "wrong_file",
        "partial_file",
        "missing_dependency",
        "ambiguous_issue",
        "empty_query",
        "index_error",
    ] = "success"
    # For evaluation
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None

    @property
    def top_k_paths(self) -> list[str]:
        return [h.file_path for h in self.hits]


# ── Failure categorisation ────────────────────────────────────────────────────

def categorise_localisation_failure(
    predicted_files: list[str],
    gold_files: list[str],
    issue_text: str,
) -> Literal["wrong_file", "partial_file", "missing_dependency", "ambiguous_issue", "success"]:
    """
    Classify WHY localisation failed — generates signal for fine-tuning.

    Categories (from the roadmap):
      wrong_file:         Gold file not in predicted top-K at all
      partial_file:       Some gold files found but not all
      missing_dependency: Gold file has no BM25/embed match (needs graph)
      ambiguous_issue:    Issue text is very short / vague
      success:            All gold files found in predictions
    """
    gold_set = set(gold_files)
    pred_set = set(predicted_files)
    hits = gold_set & pred_set

    if len(hits) == len(gold_set):
        return "success"
    if not hits:
        # No gold files found at all
        if len(issue_text.strip().split()) < 10:
            return "ambiguous_issue"
        return "wrong_file"
    if len(hits) < len(gold_set):
        return "partial_file"
    return "missing_dependency"


# ── Main pipeline ─────────────────────────────────────────────────────────────

class LocalisationPipeline:
    """
    End-to-end file localisation pipeline:
        BM25 + Embeddings → RRF fusion → PPR graph propagation → DeBERTa re-rank

    The pipeline is stateful: index_repo() must be called before localise().
    """

    def __init__(
        self,
        cache_dir: Path = Path(".cache"),
        embedding_model: str = "text-embedding-3-small",
        deberta_model: str = "microsoft/deberta-v3-small",
        alpha_bm25: float = 0.4,
        alpha_embed: float = 0.4,
        alpha_ppr: float = 0.2,
        bm25_top_k: int = 20,
        embed_top_k: int = 20,
        ppr_top_k: int = 20,
        final_top_k: int = 10,
        use_deberta: bool = True,
        use_ppr: bool = True,
        use_embeddings: bool = True,
        track_mlflow: bool = False,
    ):
        self.alpha_bm25 = alpha_bm25
        self.alpha_embed = alpha_embed
        self.alpha_ppr = alpha_ppr
        self.bm25_top_k = bm25_top_k
        self.embed_top_k = embed_top_k
        self.ppr_top_k = ppr_top_k
        self.final_top_k = final_top_k
        self.use_ppr = use_ppr
        self.use_embeddings = use_embeddings
        self.track_mlflow = track_mlflow

        # Lazy-init components
        self._bm25: Optional[object] = None
        self._embed: Optional[object] = None
        self._graph: Optional[object] = None
        self._ranker: Optional[object] = None
        self._file_symbols: list = []

        # Build components
        from localisation.bm25_retriever import BM25Retriever
        self._bm25 = BM25Retriever()

        if use_embeddings:
            from localisation.embedding_retriever import EmbeddingRetriever
            self._embed = EmbeddingRetriever(
                model=embedding_model,
                cache_dir=cache_dir / "embeddings",
            )

        if use_deberta:
            from localisation.deberta_ranker import DeBERTaRanker
            self._ranker = DeBERTaRanker(model_name_or_path=deberta_model)

    def index_repo(
        self,
        file_symbols: list,
        dependency_graph=None,
        show_progress: bool = False,
    ) -> dict:
        """
        Index a repository for retrieval.

        Args:
            file_symbols:      list of FileSymbols from ast_parser
            dependency_graph:  RepoDependencyGraph (optional, enables PPR)
            show_progress:     log embedding progress

        Returns:
            stats dict with timing and cache info
        """
        self._file_symbols = file_symbols
        self._graph = dependency_graph

        start = time.monotonic()

        # BM25 index (fast — always runs)
        self._bm25.index(file_symbols)

        # Embedding index (slower, but cached)
        embed_stats = {}
        if self._embed:
            embed_stats = self._embed.index(file_symbols, show_progress=show_progress)

        elapsed = time.monotonic() - start
        logger.info(
            "Repo indexed in %.1fs — BM25: %d docs | Embed: %s",
            elapsed, self._bm25.corpus_size, embed_stats
        )
        return {"elapsed": elapsed, "bm25_docs": self._bm25.corpus_size, **embed_stats}

    def localise(
        self,
        issue_text: str,
        top_k: Optional[int] = None,
        gold_files: Optional[list[str]] = None,  # for evaluation only
    ) -> LocalisationResult:
        """
        Localise relevant files for a given issue.

        Args:
            issue_text:  the GitHub issue description
            top_k:       override final top-k (default: self.final_top_k)
            gold_files:  if provided, compute recall metrics

        Returns:
            LocalisationResult with ranked hits
        """
        if not issue_text.strip():
            return LocalisationResult(hits=[], elapsed_seconds=0.0, failure_category="empty_query")

        top_k = top_k or self.final_top_k
        start = time.monotonic()

        # ── Stage 1a: BM25 ────────────────────────────────────────────────
        bm25_results = self._bm25.query(issue_text, top_k=self.bm25_top_k)
        bm25_hits_for_rrf = [(h.file_path, h.score, h.rank) for h in bm25_results]

        # ── Stage 1b: Embeddings ──────────────────────────────────────────
        embed_hits_for_rrf = []
        if self._embed:
            embed_hits_for_rrf = self._embed.query(issue_text, top_k=self.embed_top_k)

        # ── Stage 1c: PPR graph propagation ──────────────────────────────
        ppr_scores = {}
        if self.use_ppr and self._graph:
            seed_scores = {h.file_path: 1.0 / h.rank for h in bm25_results[:10]}
            ppr_scores = self._graph.personalized_pagerank(
                seed_scores, top_k=self.ppr_top_k
            )

        # ── RRF fusion ────────────────────────────────────────────────────
        from localisation.rrf_fusion import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(
            bm25_hits=bm25_hits_for_rrf,
            embed_hits=embed_hits_for_rrf,
            ppr_scores=ppr_scores,
            alpha_bm25=self.alpha_bm25,
            alpha_embed=self.alpha_embed,
            alpha_ppr=self.alpha_ppr,
            top_k=top_k * 2,  # overshoot for Stage 2 input
        )

        # ── Stage 2: DeBERTa re-ranking ───────────────────────────────────
        fs_summary_map = {fs.file_path: fs.summary_text for fs in self._file_symbols}
        stage2_candidates = [
            (hit.file_path, fs_summary_map.get(hit.file_path, ""))
            for hit in fused
        ]

        if self._ranker and stage2_candidates:
            ranked_files = self._ranker.rerank(
                issue_text, stage2_candidates, top_k=top_k
            )
            hits = [
                LocalisationHit(
                    file_path=r.file_path,
                    relevance_score=r.relevance_score,
                    rank=r.rank,
                    in_bm25=any(h.file_path == r.file_path for h in bm25_results),
                    in_embed=any(h[0] == r.file_path for h in embed_hits_for_rrf),
                    in_ppr=r.file_path in ppr_scores,
                    bm25_rank=next(
                        (h.rank for h in bm25_results if h.file_path == r.file_path), None
                    ),
                    ppr_rank=next(
                        (i + 1 for i, (fp, _) in enumerate(
                            sorted(ppr_scores.items(), key=lambda x: -x[1])
                        ) if fp == r.file_path), None
                    ),
                )
                for r in ranked_files
            ]
        else:
            # Stage 1 output (no DeBERTa re-ranking)
            hits = [
                LocalisationHit(
                    file_path=h.file_path,
                    relevance_score=h.fused_score,
                    rank=h.rank,
                    in_bm25=h.bm25_rank is not None,
                    in_embed=h.embed_rank is not None,
                    in_ppr=h.ppr_rank is not None,
                    bm25_rank=h.bm25_rank,
                    embed_rank=h.embed_rank,
                    ppr_rank=h.ppr_rank,
                )
                for h in fused[:top_k]
            ]

        elapsed = time.monotonic() - start

        # ── Evaluation metrics ────────────────────────────────────────────
        result = LocalisationResult(hits=hits, elapsed_seconds=elapsed)
        if gold_files:
            from localisation.deberta_ranker import recall_at_k
            result.recall_at_5 = recall_at_k(result.top_k_paths, gold_files, k=5)
            result.recall_at_10 = recall_at_k(result.top_k_paths, gold_files, k=10)
            result.failure_category = categorise_localisation_failure(
                result.top_k_paths[:5], gold_files, issue_text
            )

        # ── MLflow tracking ────────────────────────────────────────────────
        if self.track_mlflow:
            self._log_to_mlflow(result)

        logger.debug(
            "Localised in %.2fs | top-%d files | recall@5=%.2f",
            elapsed, len(hits), result.recall_at_5 or 0.0
        )
        return result

    def _log_to_mlflow(self, result: LocalisationResult) -> None:
        try:
            import mlflow
            mlflow.log_metrics({
                "localisation_elapsed": result.elapsed_seconds,
                "recall_at_5": result.recall_at_5 or 0.0,
                "recall_at_10": result.recall_at_10 or 0.0,
            })
        except Exception:
            pass

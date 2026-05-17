"""
localisation/rrf_fusion.py
───────────────────────────
Reciprocal Rank Fusion (RRF) — merges three ranked lists into one.

RRF formula for document d:
    score(d) = Σ_i  α_i / (k + rank_i(d))

Where:
    rank_i(d) = rank of d in list i (1-indexed; ∞ if not in list)
    k         = 60  (standard smoothing constant)
    α_i       = weight for list i

Three input lists (configurable weights, defaults from settings):
    1. BM25 ranking           α = 0.4
    2. Embedding ranking      α = 0.4
    3. PPR graph propagation  α = 0.2

Default weights are tunable — α_bm25 + α_embed + α_ppr should sum to 1.0.
Weights can be ablated: set ppr α=0 to measure graph contribution.

Reference: Cormack et al. (2009) "Reciprocal rank fusion outperforms
condorcet and individual rank learning methods."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant (Cormack et al.)
RRF_K = 60


@dataclass
class FusedHit:
    file_path: str
    fused_score: float
    rank: int                     # final rank (1-indexed)
    bm25_rank: int | None         # rank in BM25 list (None if absent)
    embed_rank: int | None        # rank in embedding list
    ppr_rank: int | None          # rank in PPR list
    bm25_score: float = 0.0
    embed_score: float = 0.0
    ppr_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "fused_score": round(self.fused_score, 6),
            "rank": self.rank,
            "bm25_rank": self.bm25_rank,
            "embed_rank": self.embed_rank,
            "ppr_rank": self.ppr_rank,
        }


def reciprocal_rank_fusion(
    bm25_hits: list[tuple[str, float, int]],      # (file_path, score, rank)
    embed_hits: list[tuple[str, float, int]],
    ppr_scores: dict[str, float],                  # {file_path: ppr_score}
    alpha_bm25: float = 0.4,
    alpha_embed: float = 0.4,
    alpha_ppr: float = 0.2,
    k: int = RRF_K,
    top_k: int = 10,
) -> list[FusedHit]:
    """
    Fuse three ranked signals using Reciprocal Rank Fusion.

    Args:
        bm25_hits:   list of (file_path, score, rank) from BM25Retriever
        embed_hits:  list of (file_path, score, rank) from EmbeddingRetriever
        ppr_scores:  {file_path: ppr_score} from RepoDependencyGraph.personalized_pagerank()
        alpha_bm25:  weight for BM25 list
        alpha_embed: weight for embedding list
        alpha_ppr:   weight for PPR list (set to 0 to ablate graph component)
        k:           RRF smoothing constant (default 60)
        top_k:       number of results to return

    Returns:
        List of FusedHit sorted by fused_score descending
    """
    # Index each list by file_path → rank (1-indexed)
    bm25_rank_map: dict[str, int] = {fp: r for fp, _, r in bm25_hits}
    embed_rank_map: dict[str, int] = {fp: r for fp, _, r in embed_hits}

    # Convert PPR scores to ranks
    if ppr_scores:
        ppr_sorted = sorted(ppr_scores.items(), key=lambda x: -x[1])
        ppr_rank_map: dict[str, int] = {fp: i + 1 for i, (fp, _) in enumerate(ppr_sorted)}
    else:
        ppr_rank_map = {}

    # Keep raw scores for diagnostics
    bm25_score_map: dict[str, float] = {fp: s for fp, s, _ in bm25_hits}
    embed_score_map: dict[str, float] = {fp: s for fp, s, _ in embed_hits}

    # Union of all candidate files
    all_files = (
        set(bm25_rank_map.keys())
        | set(embed_rank_map.keys())
        | set(ppr_rank_map.keys())
    )

    fused: dict[str, float] = {}
    for fp in all_files:
        score = 0.0
        if fp in bm25_rank_map:
            score += alpha_bm25 / (k + bm25_rank_map[fp])
        if fp in embed_rank_map:
            score += alpha_embed / (k + embed_rank_map[fp])
        if fp in ppr_rank_map:
            score += alpha_ppr / (k + ppr_rank_map[fp])
        fused[fp] = score

    # Sort and build FusedHit list
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

    return [
        FusedHit(
            file_path=fp,
            fused_score=score,
            rank=i + 1,
            bm25_rank=bm25_rank_map.get(fp),
            embed_rank=embed_rank_map.get(fp),
            ppr_rank=ppr_rank_map.get(fp),
            bm25_score=bm25_score_map.get(fp, 0.0),
            embed_score=embed_score_map.get(fp, 0.0),
            ppr_score=ppr_scores.get(fp, 0.0),
        )
        for i, (fp, score) in enumerate(ranked)
    ]


def ablate(
    bm25_hits,
    embed_hits,
    ppr_scores,
    *,
    use_bm25: bool = True,
    use_embed: bool = True,
    use_ppr: bool = True,
    **kwargs,
) -> list[FusedHit]:
    """
    Convenience wrapper for ablation experiments.
    Set use_bm25/embed/ppr=False to zero out that component.
    """
    return reciprocal_rank_fusion(
        bm25_hits=bm25_hits if use_bm25 else [],
        embed_hits=embed_hits if use_embed else [],
        ppr_scores=ppr_scores if use_ppr else {},
        **kwargs,
    )

"""
localisation/colbert_ranker.py
───────────────────────────────
Stage 2 — ColBERT-v2 late-interaction ranker.

Replaces the DeBERTa cross-encoder with ColBERTv2, which is more
robust for code retrieval and often handles long contexts better.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "colbert-ir/colbertv2.0"


@dataclass
class RankedFile:
    file_path: str
    relevance_score: float
    rank: int
    stage1_rank: int


class ColBERTRanker:
    """
    Late-interaction re-ranker using ColBERT-v2 via RAGatouille.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL):
        self.model_name_or_path = model_name_or_path
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ragatouille import RAGPretrainedModel
            logger.info("Loading ColBERT ranker: %s", self.model_name_or_path)
            self._model = RAGPretrainedModel.from_pretrained(self.model_name_or_path)
            self._available = True
            logger.info("ColBERT ranker loaded successfully")
        except Exception as e:
            logger.warning(
                "ColBERT ranker not available (%s) — will use Stage 1 ordering as-is", e
            )

    def rerank(
        self,
        issue_text: str,
        candidates: list[tuple[str, str]],  # (file_path, file_summary)
        top_k: int = 10,
    ) -> list[RankedFile]:
        if not candidates:
            return []

        if not self._available:
            logger.debug("ColBERT unavailable — returning Stage 1 ordering")
            return [
                RankedFile(
                    file_path=fp,
                    relevance_score=1.0 / (i + 1),
                    rank=i + 1,
                    stage1_rank=i + 1,
                )
                for i, (fp, _) in enumerate(candidates[:top_k])
            ]

        # Extract documents for reranking (limit summary length to prevent OOM)
        docs = [summary[:4000] for _, summary in candidates]
        
        try:
            results = self._model.rerank(query=issue_text[:1000], documents=docs, k=top_k)
            # RAGatouille rerank returns a list of dicts: {'content': ..., 'score': ..., 'result_index': ...}
            
            ranked_files = []
            for rank, res in enumerate(results):
                idx = res["result_index"]
                fp = candidates[idx][0]
                ranked_files.append(
                    RankedFile(
                        file_path=fp,
                        relevance_score=float(res["score"]),
                        rank=rank + 1,
                        stage1_rank=idx + 1,
                    )
                )
            return ranked_files
        except Exception as e:
            logger.error("ColBERT reranking failed: %s", e)
            return [
                RankedFile(
                    file_path=fp,
                    relevance_score=1.0 / (i + 1),
                    rank=i + 1,
                    stage1_rank=i + 1,
                )
                for i, (fp, _) in enumerate(candidates[:top_k])
            ]

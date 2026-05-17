"""
localisation/bm25_retriever.py
───────────────────────────────
Stage 1a — BM25 retrieval over repo file corpus.

Indexes per file:
  - File path tokens  (e.g. 'django/db/models/query.py' → ['django','db','models','query'])
  - Docstrings        (module + function + class docstrings)
  - Function names    (tokenised by snake_case and CamelCase splitting)
  - Class names
  - Import targets

All text is lowercased and tokenised. BM25 (Okapi BM25 via rank-bm25)
scores each file given the issue query text.

Outputs: list of (file_path, bm25_score) sorted descending.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class BM25Hit:
    file_path: str
    score: float
    rank: int   # 1-indexed rank in BM25 ordering


def _tokenise(text: str) -> list[str]:
    """
    Tokenise text for BM25 indexing.
    - Lowercases
    - Splits on non-alphanumeric chars
    - Splits CamelCase: 'QuerySet' → ['query', 'set']
    - Splits snake_case: 'get_queryset' → ['get', 'queryset']
    - Removes tokens shorter than 2 chars
    """
    # Insert space before capital letters in CamelCase
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    # Split on non-alphanumeric
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def _build_document(file_path: str, summary_text: str) -> list[str]:
    """
    Build the BM25 document token list for one file.
    File path tokens are added with 2x weight (repeated).
    """
    path_tokens = _tokenise(file_path.replace("/", " ").replace("_", " ").replace(".", " "))
    content_tokens = _tokenise(summary_text)
    # Double-weight file path tokens — path relevance is strong signal
    return path_tokens + path_tokens + content_tokens


class BM25Retriever:
    """
    BM25 retriever over a corpus of Python files.

    Usage:
        retriever = BM25Retriever()
        retriever.index(file_symbols_list)
        hits = retriever.query("fix null pointer in QuerySet filter", top_k=20)
    """

    def __init__(self):
        self._bm25 = None
        self._file_paths: list[str] = []
        self._corpus: list[list[str]] = []

    def index(self, file_symbols_list) -> None:
        """
        Build BM25 index from a list of FileSymbols.

        Args:
            file_symbols_list: list of FileSymbols from ast_parser
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("Install rank-bm25: pip install rank-bm25") from e

        self._file_paths = []
        self._corpus = []

        for fs in file_symbols_list:
            if fs.parse_error:
                continue
            doc_tokens = _build_document(fs.file_path, fs.summary_text)
            if doc_tokens:
                self._file_paths.append(fs.file_path)
                self._corpus.append(doc_tokens)

        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index built: %d documents", len(self._file_paths))

    def query(self, query_text: str, top_k: int = 20) -> list[BM25Hit]:
        """
        Retrieve top-k files most relevant to query_text.

        Args:
            query_text: raw issue text or preprocessed query
            top_k: number of results to return

        Returns:
            List of BM25Hit sorted by score descending
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever is not indexed. Call .index() first.")

        query_tokens = _tokenise(query_text)
        if not query_tokens:
            logger.warning("Empty query tokens after tokenisation")
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair with file paths and sort
        ranked = sorted(
            zip(self._file_paths, scores),
            key=lambda x: -x[1],
        )

        return [
            BM25Hit(file_path=fp, score=float(score), rank=i + 1)
            for i, (fp, score) in enumerate(ranked[:top_k])
            if score > 0
        ]

    def query_batch(self, queries: list[str], top_k: int = 20) -> list[list[BM25Hit]]:
        """Query multiple issues at once."""
        return [self.query(q, top_k) for q in queries]

    @property
    def corpus_size(self) -> int:
        return len(self._file_paths)

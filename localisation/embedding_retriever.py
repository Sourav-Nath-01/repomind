"""
localisation/embedding_retriever.py
─────────────────────────────────────
Stage 1b — Dense embedding retrieval over repo file corpus.

Uses OpenAI text-embedding-3-small (1536-dim) to encode:
  - Each file's summary_text (docstrings + function/class names + imports)
  - The issue query text

Similarity is computed via cosine distance using FAISS IndexFlatIP
(Inner Product on L2-normalised vectors == cosine similarity).

Embedding cache:
  - Key: SHA-256 of the text being embedded
  - Backend: diskcache (local) or JSON fallback
  - A file whose content hasn't changed reuses its cached embedding
  - This is critical for latency: ~500 files × 0ms (cached) vs ~5s (fresh)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small dimension


# ── Embedding cache ───────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    SHA-256-keyed cache for embedding vectors.
    Avoids re-embedding files whose content hasn't changed.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dc = None
        self._try_init_diskcache()

    def _try_init_diskcache(self) -> None:
        try:
            import diskcache
            self._dc = diskcache.Cache(str(self.cache_dir / "embeddings"))
            logger.debug("EmbeddingCache: using diskcache backend")
        except ImportError:
            logger.debug("EmbeddingCache: using JSON fallback")

    def get(self, text_hash: str) -> Optional[np.ndarray]:
        key = f"emb:{text_hash}"
        if self._dc is not None:
            raw = self._dc.get(key)
        else:
            p = self.cache_dir / f"{text_hash}.json"
            raw = p.read_text() if p.exists() else None

        if raw is None:
            return None
        return np.array(json.loads(raw), dtype=np.float32)

    def set(self, text_hash: str, vector: np.ndarray) -> None:
        key = f"emb:{text_hash}"
        serialised = json.dumps(vector.tolist())
        if self._dc is not None:
            self._dc.set(key, serialised)
        else:
            p = self.cache_dir / f"{text_hash}.json"
            p.write_text(serialised)

    def stats(self) -> dict:
        if self._dc is not None:
            return {"backend": "diskcache", "size": len(self._dc)}
        return {"backend": "json_files"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Embedding retriever ───────────────────────────────────────────────────────

class EmbeddingRetriever:
    """
    Dense retrieval using OpenAI embeddings + FAISS index.

    Usage:
        retriever = EmbeddingRetriever(cache_dir=Path(".cache/embeddings"))
        retriever.index(file_symbols_list)
        hits = retriever.query("Fix null pointer in filter()", top_k=20)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: Path = Path(".cache/embeddings"),
        batch_size: int = 100,
    ):
        self.model = model
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_dir)

        self._index = None               # FAISS index
        self._file_paths: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def index(self, file_symbols_list, show_progress: bool = False) -> dict:
        """
        Build FAISS index from FileSymbols.

        Returns:
            stats dict: {total, cached, fresh, elapsed}
        """
        texts = []
        paths = []
        hashes = []

        for fs in file_symbols_list:
            if fs.parse_error or not fs.summary_text.strip():
                continue
            paths.append(fs.file_path)
            texts.append(fs.summary_text[:2000])  # token budget
            hashes.append(_sha256(fs.summary_text))

        # Check cache for each file
        cached_vecs: dict[int, np.ndarray] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, (text_hash, text) in enumerate(zip(hashes, texts)):
            vec = self.cache.get(text_hash)
            if vec is not None:
                cached_vecs[i] = vec
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        logger.info(
            "Embedding index: %d total, %d cached, %d to embed",
            len(texts), len(cached_vecs), len(uncached_texts)
        )

        # Embed uncached texts in batches
        start = time.monotonic()
        fresh_vecs: dict[int, np.ndarray] = {}
        if uncached_texts:
            all_fresh = self._embed_texts(uncached_texts, show_progress)
            for list_idx, (original_idx, text_hash) in enumerate(
                zip(uncached_indices, [hashes[i] for i in uncached_indices])
            ):
                vec = all_fresh[list_idx]
                fresh_vecs[original_idx] = vec
                self.cache.set(text_hash, vec)

        elapsed = time.monotonic() - start

        # Assemble all embeddings in order
        all_vecs = []
        self._file_paths = []
        for i, fp in enumerate(paths):
            vec = cached_vecs.get(i) or fresh_vecs.get(i)
            if vec is not None:
                all_vecs.append(vec)
                self._file_paths.append(fp)

        if not all_vecs:
            logger.warning("No embeddings produced — index is empty")
            return {"total": 0, "cached": 0, "fresh": 0, "elapsed": elapsed}

        self._embeddings = np.vstack(all_vecs).astype(np.float32)
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = self._embeddings / norms

        self._build_faiss_index()

        return {
            "total": len(texts),
            "cached": len(cached_vecs),
            "fresh": len(uncached_texts),
            "elapsed": round(elapsed, 2),
        }

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[str, float, int]]:
        """
        Retrieve top-k files by cosine similarity to query.

        Returns:
            List of (file_path, cosine_score, rank)
        """
        if self._index is None or not self._file_paths:
            raise RuntimeError("EmbeddingRetriever not indexed. Call .index() first.")

        query_vec = self._embed_texts([query_text[:2000]])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        k = min(top_k, len(self._file_paths))
        scores, indices = self._index.search(query_vec, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx >= 0:
                results.append((self._file_paths[idx], float(score), rank))

        return results

    def _embed_texts(self, texts: list[str], show_progress: bool = False) -> list[np.ndarray]:
        """Call OpenAI embeddings API in batches."""
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError as e:
            raise ImportError("Install openai: pip install openai") from e

        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            if show_progress:
                logger.info("Embedding batch %d/%d", i // self.batch_size + 1,
                            (len(texts) + self.batch_size - 1) // self.batch_size)
            response = client.embeddings.create(model=self.model, input=batch)
            for item in response.data:
                all_vecs.append(np.array(item.embedding, dtype=np.float32))
        return all_vecs

    def _build_faiss_index(self) -> None:
        """Build FAISS IndexFlatIP (inner product = cosine after normalisation)."""
        try:
            import faiss
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings)
            logger.info("FAISS index built: %d vectors, dim=%d", len(self._file_paths), dim)
        except ImportError:
            logger.warning("FAISS not available — falling back to numpy dot product search")
            self._index = _NumpyFallbackIndex(self._embeddings)


class _NumpyFallbackIndex:
    """Pure numpy inner-product search — no FAISS dependency needed."""

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def search(self, query: np.ndarray, k: int):
        scores = (self._matrix @ query.T).flatten()
        top_k = min(k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        return scores[indices].reshape(1, -1), indices.reshape(1, -1)

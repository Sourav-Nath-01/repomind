"""
localisation/deberta_ranker.py
───────────────────────────────
Stage 2 — DeBERTa-v3-small cross-encoder ranker.

Given a set of candidate files from Stage 1 (RRF fusion), this
re-ranks them using a fine-tuned DeBERTa-v3-small cross-encoder that
classifies (issue_text, file_summary) → relevant/not_relevant.

Cross-encoders are much more precise than bi-encoders because they see
both the query AND the document together — allowing full attention
across both. The trade-off is they can't be pre-indexed (must run at
query time), so we only apply them to the top-20 candidates from Stage 1.

Training data (for fine-tuning):
  - Positive: (issue_text, gold_file_summary) → label=1
  - Negative: (issue_text, random_file_summary) → label=0
  - Hard negatives: BM25 top-20 files that are NOT the gold file → label=0
  - Dataset built from SWE-bench Lite instances

This module has two modes:
  1. inference_only: loads a pre-trained checkpoint and scores candidates
  2. training: fine-tunes DeBERTa-v3-small on the SWE-bench training set

For Phase 3 we implement the inference path + training scaffold.
Fine-tuning happens in Phase 7 (after trajectory data is collected).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default model — can be swapped for a fine-tuned checkpoint
DEFAULT_MODEL = "microsoft/deberta-v3-small"

# Max token lengths for cross-encoder input
MAX_QUERY_LEN = 256   # issue text tokens
MAX_DOC_LEN = 256     # file summary tokens
MAX_TOTAL_LEN = 512   # total cross-encoder input length


@dataclass
class RankedFile:
    file_path: str
    relevance_score: float   # 0–1 probability of relevance
    rank: int                # final rank (1-indexed)
    stage1_rank: int         # rank before re-ranking


class DeBERTaRanker:
    """
    Cross-encoder re-ranker using DeBERTa-v3-small.

    Scores each (issue, file_summary) pair and re-orders Stage 1 candidates.
    Falls back gracefully to Stage 1 ordering if model unavailable.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = MAX_TOTAL_LEN,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = self._resolve_device(device)
        self._available = False
        self._try_load()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _try_load(self) -> None:
        """Attempt to load the model — log a warning if unavailable."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(
                "Loading DeBERTa ranker: %s on %s", self.model_name_or_path, self._device
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, num_labels=2
            )
            self._model.to(self._device)
            self._model.eval()
            self._available = True
            logger.info("DeBERTa ranker loaded successfully")
        except Exception as e:
            logger.warning(
                "DeBERTa ranker not available (%s) — will use Stage 1 ordering as-is", e
            )

    def rerank(
        self,
        issue_text: str,
        candidates: list[tuple[str, str]],  # list of (file_path, file_summary)
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[RankedFile]:
        """
        Re-rank candidates by relevance to issue_text.

        Args:
            issue_text: the GitHub issue description
            candidates: list of (file_path, file_summary) from Stage 1
            top_k: number of results to return
            batch_size: inference batch size

        Returns:
            List of RankedFile sorted by relevance_score descending
        """
        if not candidates:
            return []

        if not self._available:
            logger.debug("DeBERTa unavailable — returning Stage 1 ordering")
            return [
                RankedFile(
                    file_path=fp,
                    relevance_score=1.0 / (i + 1),  # inverse rank as score
                    rank=i + 1,
                    stage1_rank=i + 1,
                )
                for i, (fp, _) in enumerate(candidates[:top_k])
            ]

        # Score all candidates
        scores = self._score_batch(issue_text, candidates, batch_size)

        # Sort by score descending
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: -x[1],
        )

        return [
            RankedFile(
                file_path=fp,
                relevance_score=float(score),
                rank=i + 1,
                stage1_rank=next(
                    (j + 1 for j, (p, _) in enumerate(candidates) if p == fp), -1
                ),
            )
            for i, ((fp, _), score) in enumerate(ranked[:top_k])
        ]

    def _score_batch(
        self,
        issue_text: str,
        candidates: list[tuple[str, str]],
        batch_size: int,
    ) -> list[float]:
        """Run cross-encoder inference on all candidates in batches."""
        import torch
        import torch.nn.functional as F

        truncated_query = issue_text[:500]  # characters (tokenizer handles tokens)
        scores = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i: i + batch_size]
            texts_a = [truncated_query] * len(batch)
            texts_b = [summary[:500] for _, summary in batch]

            encoded = self._tokenizer(
                texts_a,
                texts_b,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = F.softmax(logits, dim=-1)
                # Class 1 = relevant
                batch_scores = probs[:, 1].cpu().tolist()
            scores.extend(batch_scores)

        return scores


# ── Training scaffold ─────────────────────────────────────────────────────────

class DeBERTaTrainer:
    """
    Fine-tunes DeBERTa-v3-small on (issue, file_summary) pairs.

    Training data format (JSONL):
        {"query": "<issue text>", "document": "<file summary>", "label": 0|1}

    Called in Phase 7 after collecting trajectory data from SWE-bench runs.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_MODEL,
        output_dir: Path = Path("models/deberta_ranker"),
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def prepare_training_data(
        self,
        swe_instances,       # list of SWEInstance
        file_symbols_map,    # {instance_id: list[FileSymbols]}
        hard_negatives_k: int = 5,  # BM25 top-k non-gold as hard negatives
    ) -> list[dict]:
        """
        Build training pairs from SWE-bench instances.

        Strategy:
          Positive: (issue, gold_file_summary) → label=1
          Hard-neg: BM25 top-5 files that are NOT in the gold patch → label=0
          Random-neg: random repo file → label=0 (1:2 pos:neg ratio)
        """
        from localisation.bm25_retriever import BM25Retriever
        import random

        training_pairs = []

        for inst in swe_instances:
            file_symbols = file_symbols_map.get(inst.instance_id, [])
            if not file_symbols:
                continue

            # Extract gold file paths from the patch
            gold_files = _extract_files_from_patch(inst.patch)

            # Build BM25 index for this repo
            retriever = BM25Retriever()
            retriever.index(file_symbols)
            bm25_hits = retriever.query(inst.problem_statement, top_k=hard_negatives_k + 5)

            fs_map = {fs.file_path: fs for fs in file_symbols}

            for gold_fp in gold_files:
                if gold_fp not in fs_map:
                    continue
                # Positive pair
                training_pairs.append({
                    "query": inst.problem_statement[:500],
                    "document": fs_map[gold_fp].summary_text[:500],
                    "label": 1,
                    "instance_id": inst.instance_id,
                })
                # Hard negatives
                for hit in bm25_hits[:hard_negatives_k]:
                    if hit.file_path not in gold_files and hit.file_path in fs_map:
                        training_pairs.append({
                            "query": inst.problem_statement[:500],
                            "document": fs_map[hit.file_path].summary_text[:500],
                            "label": 0,
                            "instance_id": inst.instance_id,
                        })

        logger.info(
            "Training data: %d pairs (%d positive, %d negative)",
            len(training_pairs),
            sum(1 for p in training_pairs if p["label"] == 1),
            sum(1 for p in training_pairs if p["label"] == 0),
        )
        return training_pairs

    def train(self, training_data: list[dict]) -> None:
        """Fine-tune DeBERTa on the prepared training data."""
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer
            )
            import torch
            from torch.utils.data import Dataset
        except ImportError as e:
            raise ImportError("Install transformers + torch for fine-tuning") from e

        class PairDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self): return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                enc = self.tokenizer(
                    item["query"], item["document"],
                    max_length=self.max_length,
                    padding="max_length", truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": torch.tensor(item["label"], dtype=torch.long),
                }

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=2
        )

        dataset = PairDataset(training_data, tokenizer, MAX_TOTAL_LEN)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=10,
            warmup_ratio=0.1,
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        logger.info("DeBERTa ranker saved to %s", self.output_dir)


# ── Metric helpers ────────────────────────────────────────────────────────────

def recall_at_k(
    predictions: list[str],
    gold_files: list[str],
    k: int,
) -> float:
    """Compute recall@k: fraction of gold files in top-k predictions."""
    if not gold_files:
        return 0.0
    top_k_set = set(predictions[:k])
    hits = sum(1 for gf in gold_files if gf in top_k_set)
    return hits / len(gold_files)


def _extract_files_from_patch(patch: str) -> list[str]:
    """Extract list of files modified in a unified diff."""
    import re
    # Match '--- a/path/to/file.py' or '+++ b/path/to/file.py'
    pattern = re.compile(r"^(?:\+\+\+|---)\s+(?:a/|b/)(.+?)(?:\s|$)", re.MULTILINE)
    files = list(set(pattern.findall(patch)))
    return [f for f in files if f and f != "/dev/null"]

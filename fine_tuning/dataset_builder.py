"""
fine_tuning/dataset_builder.py
────────────────────────────────
Build the fine-tuning dataset from Phase 4 trajectory JSONL files.

Dataset construction strategy:
  1. Load all trajectory JSONL files from results/trajectories/
  2. Filter to high-quality instances:
       - failure_category is NOT 'unknown' (has learnable signal)
       - patch is valid (starts with --- or diff --git)
       - problem_statement is >= 20 words (enough context)
  3. Format each entry as an instruction-following pair
  4. Build hard-negative augmentation:
       - For each resolved instance, create (issue, wrong_patch) → label=BAD
       - Teaches the model to distinguish correct vs. plausible-but-wrong patches
  5. Split 90/10 train/val
  6. Export as JSONL with ShareGPT / Alpaca / ChatML format options

Expected input: ~300–500 trajectory entries from a full SWE-bench Lite run
Expected output: ~800–1200 training pairs (with augmentation)

ChatML format (used by DeepSeek-Coder):
  <|im_start|>system
  You are an expert Python engineer...
  <|im_end|>
  <|im_start|>user
  ## GitHub Issue
  ...
  <|im_end|>
  <|im_start|>assistant
  --- a/path/to/file.py
  +++ b/path/to/file.py
  ...
  <|im_end|>
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── Format constants ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python software engineer specialising in bug fixes. "
    "You will be given a GitHub issue description and the relevant source files. "
    "Your task is to generate a minimal, correct unified diff patch that fixes the issue. "
    "Output ONLY the unified diff — no explanations, no markdown code blocks."
)

CHATML_TEMPLATE = """\
<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
{assistant}
<|im_end|>"""

# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class TrainingPair:
    system: str
    user: str
    assistant: str
    metadata: dict = field(default_factory=dict)

    def to_chatml(self) -> str:
        return CHATML_TEMPLATE.format(
            system=self.system, user=self.user, assistant=self.assistant
        )

    def to_alpaca(self) -> dict:
        return {
            "instruction": self.system + "\n\n" + self.user,
            "input": "",
            "output": self.assistant,
            "metadata": self.metadata,
        }

    def to_sharegpt(self) -> dict:
        return {
            "conversations": [
                {"from": "system", "value": self.system},
                {"from": "human",  "value": self.user},
                {"from": "gpt",    "value": self.assistant},
            ],
            "metadata": self.metadata,
        }

    def to_openai(self) -> dict:
        return {
            "messages": [
                {"role": "system",    "content": self.system},
                {"role": "user",      "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "metadata": self.metadata,
        }


@dataclass
class DatasetStats:
    total_trajectories: int = 0
    after_filter: int = 0
    resolved: int = 0
    unresolved_with_category: int = 0
    augmented_pairs: int = 0
    train_size: int = 0
    val_size: int = 0
    category_counts: dict = field(default_factory=dict)
    filter_reasons: dict = field(default_factory=dict)


# ── Dataset builder ────────────────────────────────────────────────────────────

class FinetuningDatasetBuilder:
    """
    Builds a fine-tuning dataset from Phase 4 trajectory JSONL files.

    Filtering criteria (all must pass):
      - failure_category != 'unknown'
      - patch is non-empty and looks like a valid diff
      - problem_statement has >= 20 words
      - (for positive pairs) instance was eventually resolved

    Augmentation:
      - Reflection pairs: (issue + failed_attempt_context) → correct_patch
        These teach the model the retry behaviour.
      - The model learns: "When tests fail with AssertionError at line X,
        the correct fix is Y" — generalised across many instances.
    """

    def __init__(
        self,
        trajectory_dir: Path = Path("results/trajectories"),
        output_dir: Path = Path("results/fine_tuning"),
        val_fraction: float = 0.10,
        min_problem_words: int = 20,
        max_patch_chars: int = 8000,
        seed: int = 42,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.output_dir = Path(output_dir)
        self.val_fraction = val_fraction
        self.min_problem_words = min_problem_words
        self.max_patch_chars = max_patch_chars
        self.seed = seed
        random.seed(seed)

    def build(
        self,
        include_reflection_pairs: bool = True,
        format: Literal["chatml", "alpaca", "sharegpt", "openai"] = "chatml",
    ) -> DatasetStats:
        """
        Build and export the fine-tuning dataset.

        Args:
            include_reflection_pairs: whether to include retry/reflection pairs
            format: output format for the JSONL

        Returns:
            DatasetStats with counts and breakdown
        """
        stats = DatasetStats()

        # ── Load all trajectory files ──────────────────────────────────────
        all_entries = self._load_trajectories()
        stats.total_trajectories = len(all_entries)
        logger.info("Loaded %d trajectory entries", len(all_entries))

        # ── Filter and build pairs ─────────────────────────────────────────
        pairs: list[TrainingPair] = []
        filter_reasons: dict[str, int] = {}

        for entry in all_entries:
            reason = self._filter(entry)
            if reason:
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                continue

            # Build pair based on whether it was resolved
            if entry.get("resolved"):
                pair = self._build_positive_pair(entry)
                stats.resolved += 1
            else:
                # Unresolved but has known failure category
                pair = self._build_negative_pair(entry)
                if pair:
                    stats.unresolved_with_category += 1

            if pair:
                pairs.append(pair)

            cat = entry.get("failure_category", "unknown")
            stats.category_counts[cat] = stats.category_counts.get(cat, 0) + 1

        stats.after_filter = len(pairs)
        stats.filter_reasons = filter_reasons
        logger.info(
            "After filtering: %d pairs (resolved=%d, unresolved=%d)",
            len(pairs), stats.resolved, stats.unresolved_with_category
        )

        # ── Reflection pair augmentation ───────────────────────────────────
        if include_reflection_pairs:
            reflection_pairs = self._build_reflection_pairs(all_entries)
            pairs.extend(reflection_pairs)
            stats.augmented_pairs = len(reflection_pairs)
            logger.info("Added %d reflection pairs", len(reflection_pairs))

        # ── Shuffle and split ──────────────────────────────────────────────
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * self.val_fraction))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        stats.train_size = len(train_pairs)
        stats.val_size = len(val_pairs)

        # ── Export ─────────────────────────────────────────────────────────
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._export(train_pairs, self.output_dir / "train.jsonl", format)
        self._export(val_pairs,   self.output_dir / "val.jsonl",   format)

        # Save stats
        stats_path = self.output_dir / "dataset_stats.json"
        stats_path.write_text(json.dumps(asdict(stats), indent=2))

        logger.info(
            "Dataset built: train=%d, val=%d → %s",
            stats.train_size, stats.val_size, self.output_dir
        )
        return stats

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _filter(self, entry: dict) -> Optional[str]:
        """Return a reason string if entry should be filtered, else None."""
        # Must have known failure category
        if entry.get("failure_category", "unknown") == "unknown":
            return "unknown_category"

        # Must have a non-empty patch
        patch = entry.get("patch", "").strip()
        if not patch:
            return "empty_patch"
        if not (patch.startswith("---") or patch.startswith("diff --git")):
            return "invalid_patch_format"
        if len(patch) > self.max_patch_chars:
            return "patch_too_long"

        # Must have sufficient problem statement
        problem = entry.get("problem_statement", "")
        if len(problem.strip().split()) < self.min_problem_words:
            return "problem_too_short"

        return None  # passes all filters

    # ── Pair builders ─────────────────────────────────────────────────────────

    def _build_positive_pair(self, entry: dict) -> TrainingPair:
        """Build a pair from a resolved instance."""
        user_prompt = self._build_user_prompt(
            problem_statement=entry.get("problem_statement", ""),
            localised_files=entry.get("localised_files", []),
        )
        return TrainingPair(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            assistant=entry["patch"],
            metadata={
                "instance_id": entry.get("instance_id"),
                "repo": entry.get("repo"),
                "failure_category": entry.get("failure_category"),
                "pair_type": "positive",
            },
        )

    def _build_negative_pair(self, entry: dict) -> Optional[TrainingPair]:
        """
        Build a pair from an unresolved instance — teaches the model
        to understand WHY the patch failed and what to do instead.
        Only useful if the test output contains actionable information.
        """
        test_stdout = entry.get("test_stdout", "")
        failure_category = entry.get("failure_category", "unknown")

        # Only keep categorised failures with diagnostic info
        if failure_category == "unknown" or not test_stdout:
            return None

        # Extract actionable error context
        from agent.failure_categoriser import extract_first_error_context
        error_context = extract_first_error_context(test_stdout)

        user_prompt = self._build_user_prompt(
            problem_statement=entry.get("problem_statement", ""),
            localised_files=entry.get("localised_files", []),
            failed_patch=entry.get("patch", ""),
            failure_category=failure_category,
            error_context=error_context,
        )
        # Note: assistant still gets the original patch even though it failed
        # The model learns the (issue + error) → patch_fix pattern
        return TrainingPair(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            assistant=entry["patch"],
            metadata={
                "instance_id": entry.get("instance_id"),
                "pair_type": "negative_with_context",
                "failure_category": failure_category,
            },
        )

    def _build_reflection_pairs(self, all_entries: list[dict]) -> list[TrainingPair]:
        """
        Build reflection pairs: (issue + attempt_k_failure) → attempt_{k+1}_patch.

        For multi-attempt instances where the agent eventually succeeds,
        we pair each failed attempt with the final successful patch.
        This directly teaches the reflection behaviour.
        """
        pairs = []
        # Group by instance_id
        by_instance: dict[str, list[dict]] = {}
        for e in all_entries:
            iid = e.get("instance_id", "")
            by_instance.setdefault(iid, []).append(e)

        for iid, entries in by_instance.items():
            entries_sorted = sorted(entries, key=lambda x: x.get("attempt", 1))
            # Find final successful patch
            final = next((e for e in reversed(entries_sorted) if e.get("resolved")), None)
            if not final or not final.get("patch"):
                continue

            # Each failed attempt before the success becomes a reflection pair
            for failed_entry in entries_sorted[:-1]:
                if failed_entry.get("resolved"):
                    continue
                if self._filter(failed_entry):
                    continue

                from agent.failure_categoriser import extract_first_error_context
                error_ctx = extract_first_error_context(failed_entry.get("test_stdout", ""))

                user_prompt = self._build_user_prompt(
                    problem_statement=failed_entry.get("problem_statement", ""),
                    localised_files=failed_entry.get("localised_files", []),
                    failed_patch=failed_entry.get("patch", ""),
                    failure_category=failed_entry.get("failure_category", ""),
                    error_context=error_ctx,
                )
                pairs.append(TrainingPair(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    assistant=final["patch"],   # correct final patch
                    metadata={
                        "instance_id": iid,
                        "pair_type": "reflection",
                        "attempt": failed_entry.get("attempt"),
                    },
                ))

        logger.info("Generated %d reflection pairs", len(pairs))
        return pairs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        problem_statement: str,
        localised_files: list[str],
        failed_patch: str = "",
        failure_category: str = "",
        error_context: str = "",
    ) -> str:
        parts = [f"## GitHub Issue\n{problem_statement[:1000]}"]

        if localised_files:
            file_list = "\n".join(f"- {fp}" for fp in localised_files[:8])
            parts.append(f"## Relevant Files\n{file_list}")

        if failed_patch and failure_category:
            parts.append(
                f"## Previous Attempt Failed\n"
                f"Failure category: **{failure_category}**\n\n"
                f"```\n{error_context[:500]}\n```\n\n"
                f"Previous patch:\n```diff\n{failed_patch[:800]}\n```"
            )

        parts.append("Generate a unified diff patch that fixes the issue.")
        return "\n\n".join(parts)

    def _load_trajectories(self) -> list[dict]:
        """Load all trajectory entries from JSONL files in trajectory_dir."""
        from agent.trajectory_logger import TrajectoryLogger
        import dataclasses

        all_entries: list[dict] = []
        if not self.trajectory_dir.exists():
            logger.warning("Trajectory directory not found: %s", self.trajectory_dir)
            return all_entries

        for jsonl_path in self.trajectory_dir.glob("*.jsonl"):
            tl = TrajectoryLogger(jsonl_path)
            for entry in tl.load_all():
                all_entries.append(dataclasses.asdict(entry))

        logger.info("Loaded %d entries from %d files", len(all_entries),
                    len(list(self.trajectory_dir.glob("*.jsonl"))))
        return all_entries

    def _export(
        self,
        pairs: list[TrainingPair],
        path: Path,
        format: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for pair in pairs:
                if format == "chatml":
                    f.write(json.dumps({"text": pair.to_chatml(), "metadata": pair.metadata}) + "\n")
                elif format == "alpaca":
                    f.write(json.dumps(pair.to_alpaca()) + "\n")
                elif format == "sharegpt":
                    f.write(json.dumps(pair.to_sharegpt()) + "\n")
                elif format == "openai":
                    f.write(json.dumps(pair.to_openai()) + "\n")
        logger.info("Exported %d %s pairs to %s", len(pairs), format, path)


# ── Token count estimator ─────────────────────────────────────────────────────

def estimate_token_counts(dataset_path: Path) -> dict:
    """
    Estimate token counts for training cost estimation.
    Uses simple word-count heuristic (1 word ≈ 1.3 tokens).
    """
    if not dataset_path.exists():
        return {}

    total_chars = 0
    n_pairs = 0
    with dataset_path.open() as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or str(obj)
            total_chars += len(text)
            n_pairs += 1

    estimated_tokens = int(total_chars / 4)  # ~4 chars per token
    return {
        "n_pairs": n_pairs,
        "estimated_tokens": estimated_tokens,
        "estimated_tokens_per_pair": estimated_tokens // max(n_pairs, 1),
        "estimated_training_cost_usd": estimated_tokens / 1e6 * 0.12,  # rough A100 estimate
    }

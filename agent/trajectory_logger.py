"""
agent/trajectory_logger.py
────────────────────────────
Trajectory logger — records every attempt as JSONL.

Each line in the trajectory file is one attempt:
{
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "attempt": 1,
    "patch": "<unified diff>",
    "test_stdout": "<pytest output>",
    "fail_to_pass_results": {"tests/test_foo.py::test_x": true},
    "pass_to_pass_results": {"tests/test_foo.py::test_y": true},
    "resolved": false,
    "failure_category": "wrong_file_edit",
    "elapsed_seconds": 12.3,
    "token_cost": {"prompt_tokens": 1200, "completion_tokens": 400},
    "localised_files": ["django/db/models/query.py"],
    "timestamp": "2025-05-01T14:23:01Z"
}

The JSONL dataset is filtered in Phase 7:
  - Keep: instances with known failure_category (not 'unknown')
  - Focus: syntax_error, hallucinated_api, wrong_file_edit — these are
    the most learnable patterns for fine-tuning
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEntry:
    instance_id: str
    repo: str
    attempt: int
    patch: str
    test_stdout: str
    fail_to_pass_results: dict[str, bool]
    pass_to_pass_results: dict[str, bool]
    resolved: bool
    failure_category: str
    elapsed_seconds: float
    token_cost: dict[str, int] = field(default_factory=dict)
    localised_files: list[str] = field(default_factory=list)
    problem_statement: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self))

    def to_instruction_pair(self) -> dict:
        """
        Format as an instruction-following pair for fine-tuning (Phase 7).

        Schema:
          system:    role description
          user:      issue + file context + failure message
          assistant: corrected unified diff
        """
        file_context = "\n\n".join(
            f"# File: {fp}" for fp in self.localised_files
        )
        failure_excerpt = self.test_stdout[-1000:] if self.test_stdout else ""

        return {
            "system": (
                "You are an expert Python software engineer. "
                "You fix bugs by generating minimal unified diffs."
            ),
            "user": (
                f"## GitHub Issue\n{self.problem_statement[:800]}\n\n"
                f"## Relevant Files\n{file_context}\n\n"
                f"## Previous Attempt Failed\n"
                f"Category: {self.failure_category}\n"
                f"Test output:\n{failure_excerpt}"
            ),
            "assistant": self.patch,
            "metadata": {
                "instance_id": self.instance_id,
                "attempt": self.attempt,
                "failure_category": self.failure_category,
                "resolved": self.resolved,
            }
        }


class TrajectoryLogger:
    """
    Appends trajectory entries to a JSONL file.
    Thread-safe for single-process use (file lock on append).
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        logger.info("TrajectoryLogger writing to %s", self.output_path)

    def log(self, entry: TrajectoryEntry) -> None:
        """Append one trajectory entry to the JSONL file."""
        with self.output_path.open("a") as f:
            f.write(entry.to_jsonl_line() + "\n")
        self._count += 1

    @property
    def total_logged(self) -> int:
        return self._count

    def load_all(self) -> list[TrajectoryEntry]:
        """Load all logged trajectories from file."""
        if not self.output_path.exists():
            return []
        entries = []
        with self.output_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(TrajectoryEntry(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed trajectory line: %s", e)
        return entries

    def stats(self) -> dict:
        """Summary statistics over all logged trajectories."""
        entries = self.load_all()
        if not entries:
            return {"total": 0}

        resolved = [e for e in entries if e.resolved]
        categories: dict[str, int] = {}
        for e in entries:
            categories[e.failure_category] = categories.get(e.failure_category, 0) + 1

        return {
            "total": len(entries),
            "resolved": len(resolved),
            "resolved_rate": len(resolved) / len(entries),
            "avg_attempts": sum(e.attempt for e in entries) / len(entries),
            "failure_categories": categories,
            "unique_instances": len({e.instance_id for e in entries}),
        }

    def export_for_finetuning(
        self,
        output_path: Path,
        filter_categories: list[str] | None = None,
        resolved_only: bool = False,
    ) -> int:
        """
        Export trajectory entries as instruction-following pairs (Phase 7).

        Args:
            output_path: where to write the fine-tuning JSONL
            filter_categories: only export entries with these categories
            resolved_only: only export successfully resolved instances

        Returns:
            Number of pairs exported
        """
        entries = self.load_all()

        if filter_categories:
            entries = [e for e in entries if e.failure_category in filter_categories]
        if resolved_only:
            entries = [e for e in entries if e.resolved]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w") as f:
            for entry in entries:
                if entry.problem_statement and entry.patch:
                    pair = entry.to_instruction_pair()
                    f.write(json.dumps(pair) + "\n")
                    count += 1

        logger.info("Exported %d fine-tuning pairs to %s", count, output_path)
        return count

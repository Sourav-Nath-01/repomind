"""
swe_bench/evaluator.py
──────────────────────
Evaluation harness for measuring agent performance on SWE-bench Lite.

Metrics tracked:
  - resolved_count  : how many issues the agent fixed (tests pass)
  - resolved_rate   : resolved_count / total_instances
  - avg_attempts    : average number of attempts taken per issue
  - token_cost      : total token usage
  - per_instance    : dict keyed by instance_id with detailed results

A result is 'resolved' if ALL fail_to_pass tests now pass AND
all pass_to_pass tests still pass (no regressions).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AttemptResult:
    """Result of a single patch attempt."""
    attempt_num: int
    patch: str                                    # unified diff generated
    test_stdout: str                              # raw pytest output
    fail_to_pass_results: dict[str, bool]         # test_id → passed
    pass_to_pass_results: dict[str, bool]         # test_id → still passing
    resolved: bool
    failure_category: Literal[
        "syntax_error",
        "hallucinated_api",
        "wrong_file_edit",
        "incomplete_patch",
        "flaky_test",
        "retrieval_miss",
        "success",
        "unknown",
    ] = "unknown"
    elapsed_seconds: float = 0.0
    token_cost: dict[str, int] = field(default_factory=dict)


@dataclass
class InstanceResult:
    """Aggregated result for one SWE-bench instance."""
    instance_id: str
    repo: str
    resolved: bool
    attempts: list[AttemptResult]
    total_attempts: int
    total_tokens: int = 0
    total_elapsed: float = 0.0
    error: str = ""          # non-empty if agent crashed entirely

    @property
    def attempts_to_fix(self) -> int:
        """Returns attempt number that resolved it, or max_attempts if not."""
        for a in self.attempts:
            if a.resolved:
                return a.attempt_num
        return self.total_attempts


@dataclass
class EvalReport:
    """Aggregate evaluation metrics over all instances."""
    total_instances: int
    resolved_count: int
    resolved_rate: float
    avg_attempts: float
    total_tokens: int
    avg_tokens_per_instance: float
    avg_elapsed_seconds: float
    failure_categories: dict[str, int]   # category → count
    per_instance: dict[str, InstanceResult]

    def to_dict(self) -> dict:
        return {
            "total_instances": self.total_instances,
            "resolved_count": self.resolved_count,
            "resolved_rate": round(self.resolved_rate, 4),
            "avg_attempts": round(self.avg_attempts, 3),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_instance": round(self.avg_tokens_per_instance, 1),
            "avg_elapsed_seconds": round(self.avg_elapsed_seconds, 2),
            "failure_categories": self.failure_categories,
        }

    def print_summary(self) -> None:
        """Pretty-print summary to stdout."""
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            console.print("\n[bold cyan]═══ SWE-bench Lite Evaluation Summary ═══[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            table.add_row("Total instances", str(self.total_instances))
            table.add_row("Resolved count", f"[green]{self.resolved_count}[/green]")
            table.add_row("Resolved rate", f"[green]{self.resolved_rate:.1%}[/green]")
            table.add_row("Avg attempts to fix", str(round(self.avg_attempts, 2)))
            table.add_row("Total tokens", f"{self.total_tokens:,}")
            table.add_row("Avg tokens / issue", f"{self.avg_tokens_per_instance:,.0f}")
            table.add_row("Avg elapsed (s)", str(round(self.avg_elapsed_seconds, 1)))
            console.print(table)
            if self.failure_categories:
                console.print("\n[bold]Failure categories:[/bold]")
                for cat, cnt in sorted(
                    self.failure_categories.items(), key=lambda x: -x[1]
                ):
                    console.print(f"  {cat}: {cnt}")
        except ImportError:
            # Fallback if rich is not installed
            print("\n=== SWE-bench Lite Evaluation Summary ===")
            print(f"Total instances  : {self.total_instances}")
            print(f"Resolved count   : {self.resolved_count}")
            print(f"Resolved rate    : {self.resolved_rate:.1%}")
            print(f"Avg attempts     : {self.avg_attempts:.2f}")
            print(f"Total tokens     : {self.total_tokens:,}")
            print(f"Failure categories: {self.failure_categories}")


# ── Aggregation helper ────────────────────────────────────────────────────────

def aggregate_results(instance_results: list[InstanceResult]) -> EvalReport:
    """Compute aggregate metrics from a list of per-instance results."""
    n = len(instance_results)
    if n == 0:
        return EvalReport(0, 0, 0.0, 0.0, 0, 0.0, 0.0, {}, {})

    resolved = [r for r in instance_results if r.resolved]
    resolved_count = len(resolved)

    attempts_list = [r.attempts_to_fix for r in instance_results]
    avg_attempts = sum(attempts_list) / n

    total_tokens = sum(r.total_tokens for r in instance_results)
    total_elapsed = sum(r.total_elapsed for r in instance_results)

    # Collect failure categories from last attempt of unresolved instances
    failure_categories: dict[str, int] = {}
    for r in instance_results:
        if not r.resolved and r.attempts:
            cat = r.attempts[-1].failure_category
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

    per_instance = {r.instance_id: r for r in instance_results}

    return EvalReport(
        total_instances=n,
        resolved_count=resolved_count,
        resolved_rate=resolved_count / n,
        avg_attempts=avg_attempts,
        total_tokens=total_tokens,
        avg_tokens_per_instance=total_tokens / n,
        avg_elapsed_seconds=total_elapsed / n,
        failure_categories=failure_categories,
        per_instance=per_instance,
    )


def save_results(report: EvalReport, output_dir: Path) -> None:
    """Persist evaluation report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Summary saved to %s", summary_path)

    details_path = output_dir / "per_instance_results.jsonl"
    with details_path.open("w") as f:
        for instance_id, r in report.per_instance.items():
            record = {
                "instance_id": instance_id,
                "repo": r.repo,
                "resolved": r.resolved,
                "total_attempts": r.total_attempts,
                "attempts_to_fix": r.attempts_to_fix,
                "total_tokens": r.total_tokens,
                "error": r.error,
            }
            f.write(json.dumps(record) + "\n")
    logger.info("Per-instance results saved to %s", details_path)

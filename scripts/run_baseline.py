"""
scripts/run_baseline.py
───────────────────────
Phase 1 evaluation script: run naive GPT-4o baseline on SWE-bench Lite.

Usage:
  python scripts/run_baseline.py --max-instances 10 --output-dir results/baseline

This script:
  1. Loads SWE-bench Lite instances
  2. Clones each repo at base_commit
  3. Generates a patch with the naive GPT-4o agent
  4. Applies the patch and runs tests in the sandbox
  5. Aggregates and logs results to MLflow
  6. Prints a rich summary table

Expected output (baseline): ~10–18% resolved on SWE-bench Lite
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from configs.settings import settings
from swe_bench.loader import load_swebench_lite, SWEInstance
from swe_bench.evaluator import (
    aggregate_results,
    save_results,
    InstanceResult,
    AttemptResult,
)
from sandbox.executor import SandboxExecutor
from agent.naive_baseline import NaiveBaselineAgent, log_baseline_attempt

console = Console()

# ── Structured logging setup ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def run_instance(
    instance: SWEInstance,
    agent: NaiveBaselineAgent,
    sandbox: SandboxExecutor,
    workspace_root: Path,
) -> InstanceResult:
    """
    Run the baseline agent on a single SWE-bench instance.

    Steps:
      1. Clone repo at base_commit
      2. Generate patch with GPT-4o
      3. Apply patch
      4. Run tests
      5. Return InstanceResult
    """
    workspace_dir = workspace_root / instance.repo_name / instance.base_commit[:8]
    workspace_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    logger.info("Processing instance", instance_id=instance.instance_id, repo=instance.repo)

    # ── Step 1: Clone repo ────────────────────────────────────────────────
    clone_result = sandbox.clone_repo(instance.repo, instance.base_commit, workspace_dir)
    if not clone_result.success:
        logger.error("Clone failed", instance_id=instance.instance_id)
        return InstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            resolved=False,
            attempts=[],
            total_attempts=1,
            error=f"Clone failed: {clone_result.stderr[:200]}",
            total_elapsed=time.monotonic() - start,
        )

    # ── Step 2: Generate patch ────────────────────────────────────────────
    try:
        patch_text, usage = agent.generate_patch(
            problem_statement=instance.problem_statement,
            repo=instance.repo,
            base_commit=instance.base_commit,
            workspace_dir=workspace_dir,
        )
    except Exception as e:
        logger.error("Patch generation failed", instance_id=instance.instance_id, error=str(e))
        return InstanceResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            resolved=False,
            attempts=[],
            total_attempts=1,
            error=f"LLM error: {str(e)[:200]}",
            total_elapsed=time.monotonic() - start,
        )

    total_tokens = usage.get("total_tokens", 0)

    # ── Step 3: Apply patch ───────────────────────────────────────────────
    apply_result = sandbox.apply_patch(patch_text, workspace_dir)
    if not apply_result.success:
        logger.warning(
            "Patch apply failed",
            instance_id=instance.instance_id,
            stderr=apply_result.stderr[:200],
        )
        # Still run tests to measure — patch may partially apply
        failure_category = "syntax_error"
    else:
        failure_category = "unknown"

    # ── Step 4: Run tests ─────────────────────────────────────────────────
    all_test_ids = instance.fail_to_pass + instance.pass_to_pass
    test_result = sandbox.run_tests(workspace_dir, all_test_ids)

    resolved, ftp_results, ptp_results = test_result.check_tests(
        instance.fail_to_pass, instance.pass_to_pass
    )

    if resolved:
        failure_category = "success"
    elif not apply_result.success:
        failure_category = "syntax_error"
    elif any(not v for v in ftp_results.values()):
        failure_category = "wrong_file_edit"

    elapsed = time.monotonic() - start
    attempt = AttemptResult(
        attempt_num=1,
        patch=patch_text,
        test_stdout=test_result.raw_output,
        fail_to_pass_results=ftp_results,
        pass_to_pass_results=ptp_results,
        resolved=resolved,
        failure_category=failure_category,
        elapsed_seconds=elapsed,
        token_cost=usage,
    )

    # ── Log to MLflow ─────────────────────────────────────────────────────
    log_baseline_attempt(
        instance_id=instance.instance_id,
        resolved=resolved,
        usage=usage,
        elapsed=elapsed,
        failure_category=failure_category,
        attempt=1,
    )

    logger.info(
        "Instance done",
        instance_id=instance.instance_id,
        resolved=resolved,
        tokens=total_tokens,
        elapsed=round(elapsed, 1),
    )

    return InstanceResult(
        instance_id=instance.instance_id,
        repo=instance.repo,
        resolved=resolved,
        attempts=[attempt],
        total_attempts=1,
        total_tokens=total_tokens,
        total_elapsed=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run naive GPT-4o baseline on SWE-bench Lite"
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Limit number of instances (default: all 300)"
    )
    parser.add_argument(
        "--instance-ids", nargs="+", default=None,
        help="Run specific instance IDs only"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/baseline"),
        help="Directory for evaluation output"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/swebench"),
        help="Local cache for SWE-bench dataset"
    )
    parser.add_argument(
        "--no-docker", action="store_true",
        help="Disable Docker, use local subprocess (for quick testing)"
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    console.print("[bold cyan]Loading SWE-bench Lite...[/bold cyan]")
    instances = load_swebench_lite(
        max_instances=args.max_instances,
        instance_ids=args.instance_ids,
        cache_dir=args.cache_dir,
    )
    console.print(f"[green]Loaded {len(instances)} instances[/green]")

    # ── Init components ───────────────────────────────────────────────────
    agent = NaiveBaselineAgent(model=args.model)
    sandbox = SandboxExecutor(use_docker=not args.no_docker)

    # ── MLflow experiment ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    results: list[InstanceResult] = []

    with tempfile.TemporaryDirectory(prefix="code-agent-workspaces-") as tmpdir:
        workspace_root = Path(tmpdir)

        with mlflow.start_run(run_name="naive_baseline"):
            mlflow.log_params({
                "model": args.model,
                "max_instances": len(instances),
                "agent_type": "naive_baseline",
            })

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Running baseline...", total=len(instances)
                )

                for instance in instances:
                    progress.update(
                        task, description=f"[{instance.instance_id}]"
                    )
                    result = run_instance(instance, agent, sandbox, workspace_root)
                    results.append(result)
                    progress.advance(task)

            # ── Aggregate ─────────────────────────────────────────────────
            report = aggregate_results(results)
            save_results(report, args.output_dir)

            # Log aggregate metrics to MLflow
            mlflow.log_metrics({
                "resolved_rate": report.resolved_rate,
                "resolved_count": report.resolved_count,
                "avg_attempts": report.avg_attempts,
                "total_tokens": report.total_tokens,
                "avg_tokens_per_instance": report.avg_tokens_per_instance,
            })

    report.print_summary()
    console.print(f"\n[bold green]Results saved to:[/bold green] {args.output_dir}")


if __name__ == "__main__":
    main()

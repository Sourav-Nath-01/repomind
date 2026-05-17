"""
tests/test_phase8_9_telemetry_benchmark.py
───────────────────────────────────────────
Tests for Phase 8 (Telemetry) and Phase 9 (Benchmarking).
All tests run without external services (Prometheus, Redis, SWE-bench).

Run with: pytest tests/test_phase8_9_telemetry_benchmark.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════
# Phase 8 — Telemetry
# ══════════════════════════════════════════════════════════════════════

class TestCostTracker:
    def test_initial_state(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        assert ct.total_tokens == 0
        assert ct.estimated_usd == 0.0

    def test_add_llm_tokens(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_llm_tokens(prompt=800, completion=200)
        assert ct.total_tokens == 1000

    def test_add_embedding_tokens(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_embedding_tokens(5000)
        assert ct.total_tokens == 5000

    def test_cost_estimation_positive(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_llm_tokens(1_000_000, 500_000)  # 1M prompt + 500K completion
        usd = ct.estimated_usd
        # 1M prompt @ $5 = $5 + 500K completion @ $15 = $7.50 → total $12.50
        assert 10.0 < usd < 15.0

    def test_embedding_cost_is_cheap(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_embedding_tokens(1_000_000)  # 1M embedding tokens
        # $0.02/M → $0.02
        assert ct.estimated_usd < 0.1

    def test_to_dict_has_expected_keys(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_llm_tokens(100, 50)
        d = ct.to_dict()
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "total_tokens" in d
        assert "estimated_usd" in d

    def test_total_tokens_sum(self):
        from telemetry.metrics import CostTracker
        ct = CostTracker()
        ct.add_llm_tokens(500, 300)
        ct.add_embedding_tokens(200)
        assert ct.total_tokens == 1000


class TestAgentMetrics:
    def test_time_phase_context_manager(self):
        from telemetry.metrics import AgentMetrics
        m = AgentMetrics()
        with m.time_phase("localisation"):
            time.sleep(0.01)
        # Should not raise

    def test_record_resolution_no_error(self):
        from telemetry.metrics import AgentMetrics
        m = AgentMetrics()
        m.record_resolution(resolved=True, attempts=2)
        m.record_resolution(resolved=False, attempts=3)

    def test_record_cache_hit_no_error(self):
        from telemetry.metrics import AgentMetrics
        m = AgentMetrics()
        m.record_cache_hit("ast", hit=True)
        m.record_cache_hit("embedding", hit=False)

    def test_prometheus_output_returns_bytes(self):
        from telemetry.metrics import AgentMetrics
        m = AgentMetrics()
        content, content_type = m.prometheus_output()
        assert isinstance(content, bytes)
        assert isinstance(content_type, str)

    def test_task_started_finished(self):
        from telemetry.metrics import AgentMetrics
        m = AgentMetrics()
        m.task_started()
        m.task_finished()  # Should not raise


class TestSlidingWindowRateLimiter:
    def test_allows_within_limit(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=5, window_seconds=60)
        for _ in range(5):
            assert lim.is_allowed("user_1")

    def test_blocks_over_limit(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=3, window_seconds=60)
        for _ in range(3):
            lim.is_allowed("user_x")
        assert not lim.is_allowed("user_x")

    def test_different_keys_independent(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=2, window_seconds=60)
        assert lim.is_allowed("alice")
        assert lim.is_allowed("alice")
        assert not lim.is_allowed("alice")
        # Bob's quota is independent
        assert lim.is_allowed("bob")

    def test_remaining_decreases(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=10, window_seconds=60)
        r0 = lim.remaining("u")
        lim.is_allowed("u")
        r1 = lim.remaining("u")
        assert r1 == r0 - 1

    def test_reset_clears_bucket(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=2, window_seconds=60)
        lim.is_allowed("u"); lim.is_allowed("u")
        assert not lim.is_allowed("u")
        lim.reset_for("u")
        assert lim.is_allowed("u")  # back to full quota

    def test_stats_returns_dict(self):
        from telemetry.rate_limiter import SlidingWindowRateLimiter
        lim = SlidingWindowRateLimiter(requests=5, window_seconds=60)
        stats = lim.stats()
        assert stats["limit"] == 5
        assert stats["window_seconds"] == 60


class TestQueueDepthMonitor:
    def test_initial_state(self):
        from telemetry.rate_limiter import QueueDepthMonitor
        m = QueueDepthMonitor(max_concurrent=3)
        snap = m.snapshot()
        assert snap["running"] == 0
        assert snap["queued"] == 0

    def test_task_accepted_under_capacity(self):
        from telemetry.rate_limiter import QueueDepthMonitor
        m = QueueDepthMonitor(max_concurrent=3)
        assert m.task_queued() is True

    def test_task_rejected_at_capacity(self):
        from telemetry.rate_limiter import QueueDepthMonitor
        m = QueueDepthMonitor(max_concurrent=2)
        m.task_queued(); m.task_started()
        m.task_queued(); m.task_started()
        assert m.is_at_capacity
        assert m.task_queued() is False

    def test_task_lifecycle(self):
        from telemetry.rate_limiter import QueueDepthMonitor
        m = QueueDepthMonitor(max_concurrent=5)
        m.task_queued()
        m.task_started()
        m.task_finished()
        snap = m.snapshot()
        assert snap["completed"] == 1
        assert snap["running"] == 0

    def test_utilisation_pct(self):
        from telemetry.rate_limiter import QueueDepthMonitor
        m = QueueDepthMonitor(max_concurrent=4)
        m.task_queued(); m.task_started()
        m.task_queued(); m.task_started()
        snap = m.snapshot()
        assert snap["utilisation_pct"] == 50.0


class TestStructuredLogging:
    def test_get_logger_returns_logger(self):
        from telemetry.structured_logging import get_logger
        log = get_logger("test.module")
        assert log is not None

    def test_configure_logging_no_error(self):
        from telemetry.structured_logging import configure_logging
        configure_logging(level="WARNING", json_output=False)

    def test_request_context_no_error(self):
        from telemetry.structured_logging import RequestContext
        with RequestContext(task_id="test-123", repo="django/django"):
            pass  # Should not raise


# ══════════════════════════════════════════════════════════════════════
# Phase 9 — Benchmarking
# ══════════════════════════════════════════════════════════════════════

class TestBenchmarkReport:
    def _make_report(self, n_resolved: int, n_total: int, variant: str = "test") -> object:
        from experiments.benchmark import BenchmarkReport
        results = []
        for i in range(n_total):
            results.append({
                "instance_id": f"inst-{i}",
                "repo": "django/django",
                "resolved": i < n_resolved,
                "attempts": 1 if i < n_resolved else 3,
                "failure_category": "success" if i < n_resolved else "assertion_error",
                "total_tokens": 2000,
                "patch": "--- a/f.py\n+++b/f.py\n",
                "variant": variant,
            })
        return BenchmarkReport(variant=variant, results=results)

    def test_pct_resolved(self):
        report = self._make_report(30, 100)
        assert abs(report.pct_resolved - 0.30) < 1e-6

    def test_avg_attempts(self):
        report = self._make_report(50, 100)
        # 50 at 1 attempt + 50 at 3 attempts = (50 + 150)/100 = 2.0
        assert abs(report.avg_attempts - 2.0) < 1e-6

    def test_avg_tokens(self):
        report = self._make_report(10, 50)
        assert report.avg_tokens == 2000.0

    def test_failure_breakdown(self):
        report = self._make_report(10, 30)
        bd = report.failure_breakdown
        assert "success" in bd
        assert bd["success"] == 10

    def test_save_and_load(self, tmp_path):
        from experiments.benchmark import BenchmarkReport
        report = self._make_report(20, 100)
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()

        loaded = BenchmarkReport.load(path)
        assert loaded.n_total == 100
        assert loaded.n_resolved == 20
        assert abs(loaded.pct_resolved - 0.20) < 1e-6

    def test_summary_dict_keys(self):
        report = self._make_report(10, 50)
        d = report.summary_dict()
        assert "variant" in d
        assert "pct_resolved" in d
        assert "avg_attempts" in d
        assert "failure_breakdown" in d

    def test_empty_report(self):
        from experiments.benchmark import BenchmarkReport
        report = BenchmarkReport(variant="empty", results=[])
        assert report.n_total == 0
        assert report.pct_resolved == 0.0
        assert report.avg_attempts == 0.0


class TestAblationTable:
    def test_build_from_results_dir(self, tmp_path):
        from experiments.benchmark import BenchmarkReport, build_ablation_table

        # Create a fake report file
        report = BenchmarkReport(variant="with_reflection", results=[
            {
                "instance_id": "i1", "repo": "r", "resolved": True,
                "attempts": 2, "failure_category": "success",
                "total_tokens": 3000, "patch": "", "variant": "with_reflection"
            }
        ])
        report.save(tmp_path / "report_with_reflection.json")

        table = build_ablation_table(tmp_path)
        assert isinstance(table, str)
        assert "Devin" in table
        assert "System Variant" in table

    def test_table_includes_published_baselines(self, tmp_path):
        from experiments.benchmark import build_ablation_table
        # Empty results dir — should still have baselines
        table = build_ablation_table(tmp_path)
        assert "Devin" in table or "SWE-agent" in table

    def test_ablation_md_file_created(self, tmp_path):
        from experiments.benchmark import build_ablation_table
        build_ablation_table(tmp_path)
        assert (tmp_path / "ablation_table.md").exists()

    def test_ablation_json_file_created(self, tmp_path):
        from experiments.benchmark import build_ablation_table
        build_ablation_table(tmp_path)
        assert (tmp_path / "ablation_table.json").exists()


class TestBenchmarkRunner:
    def _make_runner(self, tmp_path, variant="with_reflection"):
        from experiments.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(
            variant=variant,
            output_dir=tmp_path,
            max_instances=5,
        )
        return runner

    def _make_instances(self, n=3):
        return [
            {
                "instance_id": f"django__django-{i}",
                "repo": "django/django",
                "problem_statement": "Fix the bug in query filtering logic",
                "base_commit": "abc123",
                "FAIL_TO_PASS": ["tests/test_query.py::test_filter"],
                "PASS_TO_PASS": [],
            }
            for i in range(n)
        ]

    def test_runner_initialisation(self, tmp_path):
        runner = self._make_runner(tmp_path)
        assert runner.variant == "with_reflection"
        assert runner.max_instances == 5

    def test_results_path_includes_variant(self, tmp_path):
        runner = self._make_runner(tmp_path, "baseline_gpt4o")
        assert "baseline_gpt4o" in str(runner.results_path)

    def test_error_result_format(self, tmp_path):
        runner = self._make_runner(tmp_path)
        instance = {"instance_id": "test-1", "repo": "r"}
        result = runner._error_result(instance, "boom")
        assert result["resolved"] is False
        assert result["failure_category"] == "run_error"
        assert "boom" in result["error"]

    def test_summary_dict_completeness(self, tmp_path):
        from experiments.benchmark import BenchmarkReport
        results = [
            {"instance_id": "i1", "resolved": True, "attempts": 1,
             "failure_category": "success", "total_tokens": 1000, "patch": "", "repo": "r", "variant": "v"}
        ]
        report = BenchmarkReport("v", results)
        d = report.summary_dict()
        required_keys = {"variant", "n_total", "n_resolved", "pct_resolved",
                         "avg_attempts", "avg_token_cost", "failure_breakdown"}
        assert required_keys.issubset(d.keys())

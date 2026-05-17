"""
tests/test_phase1_sandbox.py
────────────────────────────
Unit tests for Phase 1: Sandbox executor, SWE-bench loader, and evaluator.
Run with: pytest tests/test_phase1_sandbox.py -v
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Sandbox Executor Tests ────────────────────────────────────────────────────

class TestSandboxExecutor:
    def test_parse_pytest_output_passed(self):
        from sandbox.executor import SandboxExecutor, ExecResult
        raw = textwrap.dedent("""
            tests/test_foo.py::test_basic PASSED                              [ 50%]
            tests/test_foo.py::test_edge PASSED                              [100%]
        """)
        result = ExecResult("pytest", 0, raw, "", 1.0)
        test_result = SandboxExecutor._parse_pytest_output(result)
        assert "tests/test_foo.py::test_basic" in test_result.passed
        assert "tests/test_foo.py::test_edge" in test_result.passed
        assert test_result.failed == []

    def test_parse_pytest_output_failed(self):
        from sandbox.executor import SandboxExecutor, ExecResult
        raw = textwrap.dedent("""
            tests/test_foo.py::test_basic PASSED
            tests/test_bar.py::test_regression FAILED
            tests/test_bar.py::test_setup ERROR
        """)
        result = ExecResult("pytest", 1, raw, "", 2.0)
        test_result = SandboxExecutor._parse_pytest_output(result)
        assert "tests/test_foo.py::test_basic" in test_result.passed
        assert "tests/test_bar.py::test_regression" in test_result.failed
        assert "tests/test_bar.py::test_setup" in test_result.errors

    def test_check_tests_resolved(self):
        from sandbox.executor import TestResult
        tr = TestResult(
            passed=["tests/test_a.py::test_x", "tests/test_b.py::test_y"],
            failed=[],
            errors=[],
        )
        resolved, ftp, ptp = tr.check_tests(
            fail_to_pass=["tests/test_a.py::test_x"],
            pass_to_pass=["tests/test_b.py::test_y"],
        )
        assert resolved is True
        assert ftp["tests/test_a.py::test_x"] is True
        assert ptp["tests/test_b.py::test_y"] is True

    def test_check_tests_not_resolved(self):
        from sandbox.executor import TestResult
        tr = TestResult(
            passed=["tests/test_b.py::test_y"],
            failed=["tests/test_a.py::test_x"],
            errors=[],
        )
        resolved, ftp, ptp = tr.check_tests(
            fail_to_pass=["tests/test_a.py::test_x"],
            pass_to_pass=["tests/test_b.py::test_y"],
        )
        assert resolved is False
        assert ftp["tests/test_a.py::test_x"] is False

    def test_command_whitelist_rejects_rm(self):
        from sandbox.executor import _validate_command
        with pytest.raises(ValueError, match="not in the allowed command whitelist"):
            _validate_command(["rm", "-rf", "/"])

    def test_command_whitelist_accepts_pytest(self):
        from sandbox.executor import _validate_command
        # Should not raise
        _validate_command(["pytest", "-v", "tests/"])

    def test_empty_patch_returns_failure(self, tmp_path):
        from sandbox.executor import SandboxExecutor
        executor = SandboxExecutor(use_docker=False)
        result = executor.apply_patch("", tmp_path)
        assert result.success is False

    def test_timeout_result(self):
        from sandbox.executor import ExecResult
        result = ExecResult("pytest", -1, "", "TIMEOUT after 60s", 60.0, timed_out=True)
        assert result.success is False
        assert result.timed_out is True


# ── SWE-bench Loader Tests ────────────────────────────────────────────────────

class TestSWEBenchLoader:
    def test_parse_list_from_string(self):
        from swe_bench.loader import _parse_list
        result = _parse_list('["test_a", "test_b"]')
        assert result == ["test_a", "test_b"]

    def test_parse_list_from_list(self):
        from swe_bench.loader import _parse_list
        result = _parse_list(["test_a", "test_b"])
        assert result == ["test_a", "test_b"]

    def test_parse_list_invalid_returns_empty(self):
        from swe_bench.loader import _parse_list
        result = _parse_list("not_json")
        assert result == []

    def test_swe_instance_repo_name(self):
        from swe_bench.loader import SWEInstance
        inst = SWEInstance(
            instance_id="django__django-12345",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix bug",
            patch="--- a\n+++ b\n",
            test_patch="",
            fail_to_pass=[],
            pass_to_pass=[],
        )
        assert inst.repo_name == "django__django"
        assert inst.org == "django"
        assert inst.project == "django"

    def test_local_cache_load(self, tmp_path):
        from swe_bench.loader import load_swebench_lite, _instance_to_dict, SWEInstance
        import json

        # Create a fake cached dataset
        fake_instance = SWEInstance(
            instance_id="test__repo-1",
            repo="test/repo",
            base_commit="deadbeef",
            problem_statement="Test issue",
            patch="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-bug\n+fix\n",
            test_patch="",
            fail_to_pass=["tests/test_foo.py::test_basic"],
            pass_to_pass=[],
        )
        cache_path = tmp_path / "swebench_lite_test.json"
        cache_path.write_text(json.dumps([_instance_to_dict(fake_instance)]))

        instances = load_swebench_lite(cache_dir=tmp_path, split="test")
        assert len(instances) == 1
        assert instances[0].instance_id == "test__repo-1"
        assert instances[0].fail_to_pass == ["tests/test_foo.py::test_basic"]


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_result(self, instance_id: str, resolved: bool, attempts: int = 1):
        from swe_bench.evaluator import InstanceResult, AttemptResult
        attempt_list = [
            AttemptResult(
                attempt_num=i + 1,
                patch="",
                test_stdout="",
                fail_to_pass_results={},
                pass_to_pass_results={},
                resolved=(i + 1 == attempts and resolved),
                failure_category="success" if (i + 1 == attempts and resolved) else "wrong_file_edit",
            )
            for i in range(attempts)
        ]
        return InstanceResult(
            instance_id=instance_id,
            repo="test/repo",
            resolved=resolved,
            attempts=attempt_list,
            total_attempts=attempts,
        )

    def test_aggregate_resolved_rate(self):
        from swe_bench.evaluator import aggregate_results
        results = [
            self._make_result("inst-1", resolved=True),
            self._make_result("inst-2", resolved=True),
            self._make_result("inst-3", resolved=False),
            self._make_result("inst-4", resolved=False),
        ]
        report = aggregate_results(results)
        assert report.resolved_count == 2
        assert report.total_instances == 4
        assert abs(report.resolved_rate - 0.5) < 1e-6

    def test_aggregate_empty(self):
        from swe_bench.evaluator import aggregate_results
        report = aggregate_results([])
        assert report.total_instances == 0
        assert report.resolved_count == 0

    def test_attempts_to_fix(self):
        from swe_bench.evaluator import aggregate_results
        # One instance resolved on attempt 2
        results = [self._make_result("inst-1", resolved=True, attempts=2)]
        report = aggregate_results(results)
        assert report.avg_attempts == 2.0

    def test_failure_categories_counted(self):
        from swe_bench.evaluator import aggregate_results
        results = [
            self._make_result("inst-1", resolved=False, attempts=1),
            self._make_result("inst-2", resolved=False, attempts=1),
        ]
        report = aggregate_results(results)
        assert sum(report.failure_categories.values()) == 2

    def test_save_and_load_results(self, tmp_path):
        from swe_bench.evaluator import aggregate_results, save_results
        results = [
            self._make_result("inst-1", resolved=True),
            self._make_result("inst-2", resolved=False),
        ]
        report = aggregate_results(results)
        save_results(report, tmp_path)

        summary = json.loads((tmp_path / "eval_summary.json").read_text())
        assert summary["resolved_count"] == 1
        assert summary["total_instances"] == 2


# ── Naive Baseline Patch Cleaning Tests ──────────────────────────────────────

class TestNaiveBaseline:
    def test_strip_code_fences(self):
        from agent.naive_baseline import _strip_code_fences
        raw = "```diff\n--- a/foo.py\n+++ b/foo.py\n```"
        cleaned = _strip_code_fences(raw)
        assert "```" not in cleaned
        assert "--- a/foo.py" in cleaned

    def test_strip_triple_backtick(self):
        from agent.naive_baseline import _strip_code_fences
        raw = "```\n--- a/foo.py\n+++ b/foo.py\n```"
        cleaned = _strip_code_fences(raw)
        assert cleaned.startswith("--- a/foo.py")

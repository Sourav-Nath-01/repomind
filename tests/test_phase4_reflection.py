"""
tests/test_phase4_reflection.py
────────────────────────────────
Unit tests for Phase 4: tools, failure categoriser, trajectory logger,
and the reflection agent loop (mocked LLM, no real API calls).

Run with: pytest tests/test_phase4_reflection.py -v
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── AgentTools ────────────────────────────────────────────────────────────────

class TestAgentTools:
    def test_read_file_success(self, tmp_path):
        from agent.tools import AgentTools
        (tmp_path / "foo.py").write_text("x = 1\ny = 2\n")
        tools = AgentTools(tmp_path)
        result = tools.read_file("foo.py")
        assert result.success
        assert "x = 1" in result.output

    def test_read_file_not_found(self, tmp_path):
        from agent.tools import AgentTools
        tools = AgentTools(tmp_path)
        result = tools.read_file("nonexistent.py")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_read_file_path_traversal_rejected(self, tmp_path):
        from agent.tools import AgentTools
        tools = AgentTools(tmp_path)
        result = tools.read_file("../../etc/passwd")
        assert not result.success
        assert "traversal" in result.error.lower()

    def test_read_file_truncation(self, tmp_path):
        from agent.tools import AgentTools
        content = "\n".join(f"line {i}" for i in range(300))
        (tmp_path / "big.py").write_text(content)
        tools = AgentTools(tmp_path)
        result = tools.read_file("big.py", max_lines=10)
        assert result.success
        assert "truncated" in result.output

    def test_write_patch_success(self, tmp_path):
        from agent.tools import AgentTools
        tools = AgentTools(tmp_path)
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = tools.write_patch(diff)
        assert result.success
        assert (tmp_path / "_agent_patch.diff").exists()

    def test_write_patch_empty_rejected(self, tmp_path):
        from agent.tools import AgentTools
        tools = AgentTools(tmp_path)
        result = tools.write_patch("")
        assert not result.success
        assert "Empty" in result.error

    def test_write_patch_invalid_format_rejected(self, tmp_path):
        from agent.tools import AgentTools
        tools = AgentTools(tmp_path)
        result = tools.write_patch("just some text without diff header")
        assert not result.success

    def test_list_files(self, tmp_path):
        from agent.tools import AgentTools
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        (tmp_path / "__pycache__").mkdir()
        tools = AgentTools(tmp_path)
        result = tools.list_files("**/*.py")
        assert result.success
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "__pycache__" not in result.output

    def test_tool_result_to_prompt_str(self):
        from agent.tools import ToolResult
        tr = ToolResult("read_file", True, "x = 1\n")
        prompt = tr.to_prompt_str()
        assert "read_file" in prompt
        assert "SUCCESS" in prompt
        assert "x = 1" in prompt

    def test_tool_result_error_in_prompt(self):
        from agent.tools import ToolResult
        tr = ToolResult("run_tests", False, "", "Timeout after 60s")
        prompt = tr.to_prompt_str()
        assert "ERROR" in prompt
        assert "Timeout" in prompt


# ── Failure Categoriser ───────────────────────────────────────────────────────

class TestFailureCategoriser:
    def _categorise(self, stdout, apply_ok=True, ftp=None, ptp=None, attempt=1, prev=None):
        from agent.failure_categoriser import categorise_failure
        return categorise_failure(
            test_stdout=stdout,
            patch_apply_success=apply_ok,
            fail_to_pass_results=ftp or {},
            pass_to_pass_results=ptp or {},
            attempt_num=attempt,
            previous_categories=prev,
        )

    def test_success(self):
        cat = self._categorise(
            "1 passed", apply_ok=True,
            ftp={"t::test_x": True},
            ptp={"t::test_y": True},
        )
        assert cat == "success"

    def test_patch_apply_failure_is_syntax_error(self):
        cat = self._categorise("", apply_ok=False)
        assert cat == "syntax_error"

    def test_syntax_error_in_output(self):
        cat = self._categorise("SyntaxError: invalid syntax (foo.py, line 5)")
        assert cat == "syntax_error"

    def test_import_error(self):
        cat = self._categorise("ModuleNotFoundError: No module named 'nonexistent'")
        assert cat == "import_error"

    def test_hallucinated_api_attribute_error(self):
        cat = self._categorise("AttributeError: 'QuerySet' object has no attribute 'bulk_filer'")
        assert cat == "hallucinated_api"

    def test_hallucinated_api_name_error(self):
        cat = self._categorise("NameError: name 'nonexistent_func' is not defined")
        assert cat == "hallucinated_api"

    def test_type_error(self):
        cat = self._categorise("TypeError: unsupported operand type(s) for +")
        assert cat == "type_error"

    def test_assertion_error(self):
        cat = self._categorise("AssertionError: expected True but got False")
        assert cat == "assertion_error"

    def test_incomplete_patch(self):
        cat = self._categorise(
            "2 failed", apply_ok=True,
            ftp={"t::a": True, "t::b": False},  # one passed, one failed
            ptp={},
        )
        assert cat == "incomplete_patch"

    def test_unknown_fallback(self):
        cat = self._categorise("some unexpected output with no pattern")
        assert cat == "unknown"

    def test_extract_first_error_context(self):
        from agent.failure_categoriser import extract_first_error_context
        output = textwrap.dedent("""
            tests/test_foo.py::test_bar FAILED
            AssertionError: expected 1, got 2
            
            tests/test_foo.py::test_baz PASSED
        """)
        context = extract_first_error_context(output)
        assert "FAILED" in context or "AssertionError" in context


# ── Trajectory Logger ─────────────────────────────────────────────────────────

class TestTrajectoryLogger:
    def test_log_and_load(self, tmp_path):
        from agent.trajectory_logger import TrajectoryLogger, TrajectoryEntry
        logger = TrajectoryLogger(tmp_path / "traj.jsonl")
        entry = TrajectoryEntry(
            instance_id="test__repo-1",
            repo="test/repo",
            attempt=1,
            patch="--- a/foo.py\n+++ b/foo.py\n",
            test_stdout="1 failed",
            fail_to_pass_results={"t::test_x": False},
            pass_to_pass_results={},
            resolved=False,
            failure_category="assertion_error",
            elapsed_seconds=5.2,
        )
        logger.log(entry)
        loaded = logger.load_all()
        assert len(loaded) == 1
        assert loaded[0].instance_id == "test__repo-1"
        assert loaded[0].failure_category == "assertion_error"

    def test_multiple_entries(self, tmp_path):
        from agent.trajectory_logger import TrajectoryLogger, TrajectoryEntry
        logger = TrajectoryLogger(tmp_path / "traj.jsonl")
        for i in range(5):
            entry = TrajectoryEntry(
                instance_id=f"inst-{i}",
                repo="test/repo",
                attempt=1,
                patch="",
                test_stdout="",
                fail_to_pass_results={},
                pass_to_pass_results={},
                resolved=(i % 2 == 0),
                failure_category="success" if i % 2 == 0 else "wrong_file_edit",
                elapsed_seconds=1.0,
            )
            logger.log(entry)
        assert logger.total_logged == 5
        loaded = logger.load_all()
        assert len(loaded) == 5

    def test_stats(self, tmp_path):
        from agent.trajectory_logger import TrajectoryLogger, TrajectoryEntry
        logger = TrajectoryLogger(tmp_path / "traj.jsonl")
        for i in range(4):
            entry = TrajectoryEntry(
                instance_id=f"inst-{i}",
                repo="r",
                attempt=1,
                patch="",
                test_stdout="",
                fail_to_pass_results={},
                pass_to_pass_results={},
                resolved=(i < 2),
                failure_category="success" if i < 2 else "assertion_error",
                elapsed_seconds=1.0,
            )
            logger.log(entry)
        stats = logger.stats()
        assert stats["total"] == 4
        assert stats["resolved"] == 2
        assert abs(stats["resolved_rate"] - 0.5) < 1e-6

    def test_export_for_finetuning(self, tmp_path):
        from agent.trajectory_logger import TrajectoryLogger, TrajectoryEntry
        logger = TrajectoryLogger(tmp_path / "traj.jsonl")
        entry = TrajectoryEntry(
            instance_id="inst-1",
            repo="r",
            attempt=1,
            patch="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-bug\n+fix\n",
            test_stdout="",
            fail_to_pass_results={},
            pass_to_pass_results={},
            resolved=True,
            failure_category="success",
            elapsed_seconds=1.0,
            problem_statement="Fix the null pointer bug",
        )
        logger.log(entry)
        out_path = tmp_path / "ft_data.jsonl"
        count = logger.export_for_finetuning(out_path)
        assert count == 1
        line = json.loads(out_path.read_text().strip())
        assert "system" in line
        assert "user" in line
        assert "assistant" in line

    def test_filter_by_category(self, tmp_path):
        from agent.trajectory_logger import TrajectoryLogger, TrajectoryEntry
        logger = TrajectoryLogger(tmp_path / "traj.jsonl")
        for cat in ["success", "assertion_error", "syntax_error", "unknown"]:
            entry = TrajectoryEntry(
                instance_id=cat,
                repo="r",
                attempt=1,
                patch="--- a/f.py\n+++ b/f.py\n",
                test_stdout="",
                fail_to_pass_results={},
                pass_to_pass_results={},
                resolved=(cat == "success"),
                failure_category=cat,
                elapsed_seconds=1.0,
                problem_statement="test issue",
            )
            logger.log(entry)
        out = tmp_path / "filtered.jsonl"
        count = logger.export_for_finetuning(
            out, filter_categories=["assertion_error", "syntax_error"]
        )
        assert count == 2

    def test_instruction_pair_format(self, tmp_path):
        from agent.trajectory_logger import TrajectoryEntry
        entry = TrajectoryEntry(
            instance_id="test-1",
            repo="r",
            attempt=2,
            patch="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n",
            test_stdout="AssertionError: expected 1, got 2",
            fail_to_pass_results={"t::test_x": False},
            pass_to_pass_results={},
            resolved=False,
            failure_category="assertion_error",
            elapsed_seconds=3.0,
            problem_statement="Fix the assertion in the filter method",
            localised_files=["models/query.py"],
        )
        pair = entry.to_instruction_pair()
        assert "Fix the assertion" in pair["user"]
        assert "assertion_error" in pair["user"]
        assert pair["assistant"] == entry.patch
        assert pair["metadata"]["attempt"] == 2


# ── Reflection Agent (mocked LLM) ─────────────────────────────────────────────

class TestReflectionAgent:
    """Tests for the agent loop — LLM calls are mocked."""

    def _make_agent(self, tmp_path, trajectory_logger=None):
        from agent.reflection_agent import ReflectionAgent
        agent = ReflectionAgent(
            model="gpt-4o",
            max_attempts=3,
            sandbox=None,
            localisation_pipeline=None,
            trajectory_logger=trajectory_logger,
        )
        return agent

    def _mock_llm_patch(self, monkeypatch, patch_text: str, tokens: int = 100):
        """Mock _call_llm to return a fixed patch without API calls."""
        import agent.reflection_agent as ra
        monkeypatch.setattr(
            ra, "_call_llm",
            lambda *args, **kwargs: (patch_text, {"total_tokens": tokens,
                                                   "prompt_tokens": 80,
                                                   "completion_tokens": 20})
        )

    def test_agent_state_initialisation(self, tmp_path):
        from agent.reflection_agent import AgentState
        state = AgentState(
            instance_id="test-1",
            repo="test/repo",
            problem_statement="Fix bug",
            base_commit="abc123",
            fail_to_pass=["tests::test_x"],
            pass_to_pass=[],
            workspace_dir=tmp_path,
        )
        assert state.current_attempt == 0
        assert state.resolved is False
        assert state.total_tokens == 0

    def test_should_retry_when_not_resolved(self):
        from agent.reflection_agent import AgentState, should_retry
        from pathlib import Path
        state = AgentState(
            instance_id="t", repo="r", problem_statement="p",
            base_commit="a", fail_to_pass=[], pass_to_pass=[],
            workspace_dir=Path("/tmp"), resolved=False, current_attempt=1
        )
        assert should_retry(state, max_attempts=3) == "retry"

    def test_should_done_when_resolved(self):
        from agent.reflection_agent import AgentState, should_retry
        from pathlib import Path
        state = AgentState(
            instance_id="t", repo="r", problem_statement="p",
            base_commit="a", fail_to_pass=[], pass_to_pass=[],
            workspace_dir=Path("/tmp"), resolved=True, current_attempt=1
        )
        assert should_retry(state, max_attempts=3) == "done"

    def test_should_done_when_max_attempts_reached(self):
        from agent.reflection_agent import AgentState, should_retry
        from pathlib import Path
        state = AgentState(
            instance_id="t", repo="r", problem_statement="p",
            base_commit="a", fail_to_pass=[], pass_to_pass=[],
            workspace_dir=Path("/tmp"), resolved=False, current_attempt=3
        )
        assert should_retry(state, max_attempts=3) == "done"

    def test_node_generate_patch_increments_attempt(self, tmp_path, monkeypatch):
        from agent.reflection_agent import AgentState, node_generate_patch
        self._mock_llm_patch(monkeypatch, "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n")
        state = AgentState(
            instance_id="t", repo="r", problem_statement="fix the bug please",
            base_commit="abc", fail_to_pass=[], pass_to_pass=[],
            workspace_dir=tmp_path,
        )
        state = node_generate_patch(state)
        assert state.current_attempt == 1
        assert "--- a/foo.py" in state.last_patch

    def test_node_generate_patch_uses_reflection_on_retry(self, tmp_path, monkeypatch):
        from agent.reflection_agent import AgentState, node_generate_patch
        prompts_seen = []

        def mock_call_llm(user_prompt, *args, **kwargs):
            prompts_seen.append(user_prompt)
            return ("--- a/f.py\n+++ b/f.py\n", {"total_tokens": 50, "prompt_tokens": 40, "completion_tokens": 10})

        import agent.reflection_agent as ra
        monkeypatch.setattr(ra, "_call_llm", mock_call_llm)

        state = AgentState(
            instance_id="t", repo="r",
            problem_statement="fix the long detailed issue description here",
            base_commit="abc", fail_to_pass=[], pass_to_pass=[],
            workspace_dir=tmp_path,
            current_attempt=1,                         # simulate already one attempt
            last_test_stdout="AssertionError: expected 1",
            last_failure_category="assertion_error",
            last_patch="--- a/wrong.py\n+++ b/wrong.py\n",
            attempts=[{"attempt_num": 1}],
        )
        state = node_generate_patch(state)
        # Should use reflection prompt (contains "Previous Attempt")
        assert "Previous Attempt" in prompts_seen[-1]

    def test_agent_logs_trajectories(self, tmp_path, monkeypatch):
        from agent.reflection_agent import AgentState, node_generate_patch
        from agent.trajectory_logger import TrajectoryLogger
        traj_path = tmp_path / "traj.jsonl"
        traj_logger = TrajectoryLogger(traj_path)

        # Mock node_apply_and_test to mark as resolved immediately
        import agent.reflection_agent as ra
        def mock_apply(state, sandbox=None):
            state.resolved = True
            state.last_test_stdout = "1 passed"
            state.last_failure_category = "success"
            state.attempts.append({
                "attempt_num": state.current_attempt,
                "patch": state.last_patch,
                "test_stdout": "1 passed",
                "fail_to_pass_results": {},
                "pass_to_pass_results": {},
                "resolved": True,
                "failure_category": "success",
            })
            return state

        monkeypatch.setattr(ra, "node_apply_and_test", mock_apply)
        monkeypatch.setattr(ra, "_call_llm",
                            lambda *a, **kw: ("--- a/f.py\n+++ b/f.py\n", {"total_tokens": 10, "prompt_tokens": 8, "completion_tokens": 2}))

        agent = self._make_agent(tmp_path, trajectory_logger=traj_logger)
        state = agent.run(
            instance_id="test-1",
            repo="test/repo",
            problem_statement="fix the bug",
            base_commit="abc",
            fail_to_pass=[],
            pass_to_pass=[],
            workspace_dir=tmp_path,
        )
        assert state.resolved
        assert traj_logger.total_logged >= 1

    def test_strip_code_fences(self):
        from agent.reflection_agent import _strip_code_fences
        raw = "```diff\n--- a/f.py\n+++ b/f.py\n```"
        cleaned = _strip_code_fences(raw)
        assert "```" not in cleaned
        assert "--- a/f.py" in cleaned

    def test_build_file_context(self):
        from agent.reflection_agent import _build_file_context
        contents = {
            "a.py": "def foo(): pass",
            "b.py": "class Bar: pass",
        }
        ctx = _build_file_context(contents)
        assert "a.py" in ctx
        assert "b.py" in ctx
        assert "def foo" in ctx

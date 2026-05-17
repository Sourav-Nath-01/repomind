"""
agent/tools.py
───────────────
Tool definitions for the reflection agent.

Tools available to the agent:
  read_file(path)          — read a file from the workspace
  write_patch(diff)        — write a unified diff to the workspace
  run_tests(test_ids)      — run pytest and return structured output
  git_diff()               — show current diff vs base commit
  list_files(pattern)      — list files matching a glob

Each tool returns a structured ToolResult with success/error.
The agent's LLM sees ToolResult.to_prompt_str() in its context.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Tool result ───────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def to_prompt_str(self) -> str:
        """Format result for inclusion in LLM prompt."""
        status = "SUCCESS" if self.success else "ERROR"
        parts = [f"[TOOL: {self.tool_name} | {status}]"]
        if self.output:
            parts.append(self.output[:3000])  # truncate for token budget
        if self.error:
            parts.append(f"ERROR: {self.error[:500]}")
        return "\n".join(parts)


# ── Individual tools ──────────────────────────────────────────────────────────

class AgentTools:
    """
    Collection of tools available to the reflection agent.
    All file operations are scoped to workspace_dir (sandbox root).
    """

    def __init__(self, workspace_dir: Path, sandbox=None):
        self.workspace_dir = Path(workspace_dir)
        self.sandbox = sandbox  # SandboxExecutor instance (optional)

    def read_file(self, path: str, max_lines: int = 200) -> ToolResult:
        """
        Read the contents of a file relative to workspace_dir.

        Args:
            path: relative file path within the workspace
            max_lines: truncate to this many lines (token budget control)
        """
        full_path = self.workspace_dir / path
        # Prevent path traversal
        try:
            full_path.resolve().relative_to(self.workspace_dir.resolve())
        except ValueError:
            return ToolResult("read_file", False, "", f"Path traversal rejected: {path}")

        if not full_path.exists():
            return ToolResult("read_file", False, "", f"File not found: {path}")

        try:
            content = full_path.read_text(errors="replace")
            lines = content.splitlines()
            truncated = len(lines) > max_lines
            visible = "\n".join(lines[:max_lines])
            if truncated:
                visible += f"\n... [{len(lines) - max_lines} more lines truncated]"
            return ToolResult(
                "read_file", True, visible,
                metadata={"total_lines": len(lines), "truncated": truncated}
            )
        except Exception as e:
            return ToolResult("read_file", False, "", str(e))

    def write_patch(self, diff_text: str) -> ToolResult:
        """
        Write a unified diff to a staging file for git apply.
        Does NOT apply the patch — call the sandbox apply_patch() separately.

        Args:
            diff_text: unified diff text (git format)
        """
        if not diff_text.strip():
            return ToolResult("write_patch", False, "", "Empty patch text")

        # Basic validation: must start with --- or diff --git
        stripped = diff_text.strip()
        if not (stripped.startswith("---") or stripped.startswith("diff --git")):
            return ToolResult(
                "write_patch", False, "",
                "Patch must start with '---' or 'diff --git'"
            )

        patch_file = self.workspace_dir / "_agent_patch.diff"
        try:
            patch_file.write_text(diff_text)
            return ToolResult(
                "write_patch", True,
                f"Patch written to {patch_file.name} ({len(diff_text)} chars)",
                metadata={"patch_path": str(patch_file)}
            )
        except Exception as e:
            return ToolResult("write_patch", False, "", str(e))

    def run_tests(self, test_ids: list[str], timeout: int = 60) -> ToolResult:
        """
        Run pytest on specific test IDs.

        Returns structured output including PASSED/FAILED counts and
        the first failing test's traceback (for reflection context).
        """
        if not test_ids:
            return ToolResult("run_tests", False, "", "No test IDs provided")

        if self.sandbox:
            test_result = self.sandbox.run_tests(self.workspace_dir, test_ids)
            output = test_result.raw_output
            success = test_result.all_passed
        else:
            # Local subprocess fallback
            cmd = ["python", "-m", "pytest", "-v", "--tb=short", "--no-header", "-rN"] + test_ids
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, cwd=str(self.workspace_dir)
                )
                output = proc.stdout + proc.stderr
                success = proc.returncode == 0
            except subprocess.TimeoutExpired:
                return ToolResult("run_tests", False, "", f"Tests timed out after {timeout}s")
            except Exception as e:
                return ToolResult("run_tests", False, "", str(e))

        # Extract key info for the agent
        summary = _extract_test_summary(output)
        return ToolResult(
            "run_tests", success,
            summary,
            metadata={"full_output": output[:5000]}
        )

    def git_diff(self) -> ToolResult:
        """Show the current diff vs HEAD (to review what the agent has changed)."""
        try:
            result = subprocess.run(
                ["git", "diff"], capture_output=True, text=True,
                cwd=str(self.workspace_dir), timeout=10
            )
            diff = result.stdout or "(no changes)"
            return ToolResult("git_diff", True, diff[:3000])
        except Exception as e:
            return ToolResult("git_diff", False, "", str(e))

    def list_files(self, pattern: str = "**/*.py", max_results: int = 50) -> ToolResult:
        """List files in the workspace matching a glob pattern."""
        try:
            files = sorted(self.workspace_dir.glob(pattern))
            rel_files = [
                str(f.relative_to(self.workspace_dir))
                for f in files
                if "__pycache__" not in str(f) and ".git" not in str(f)
            ][:max_results]
            output = "\n".join(rel_files) or "(no files found)"
            return ToolResult("list_files", True, output,
                              metadata={"count": len(rel_files)})
        except Exception as e:
            return ToolResult("list_files", False, "", str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_test_summary(pytest_output: str) -> str:
    """
    Extract a concise test summary from raw pytest output.
    Includes: pass/fail counts + first failure traceback.
    """
    lines = pytest_output.splitlines()
    summary_lines = []
    in_failure_section = False
    failure_lines: list[str] = []

    for line in lines:
        # Capture summary line
        if re.search(r"\d+ (passed|failed|error)", line):
            summary_lines.append(line)
        # Capture short failure tracebacks
        if line.startswith("FAILED") or "AssertionError" in line or "Error" in line:
            failure_lines.append(line)
        # Short traceback block
        if line.startswith("_ " * 3) or "FAILURES" in line:
            in_failure_section = True
        if in_failure_section:
            failure_lines.append(line)
            if len(failure_lines) > 40:  # cap failure context
                break

    parts = summary_lines + ["---"] + failure_lines[:40] if failure_lines else summary_lines
    return "\n".join(parts) or pytest_output[:1000]

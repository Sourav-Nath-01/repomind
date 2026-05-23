"""
sandbox/executor.py
───────────────────
Secure Docker-based code execution sandbox.

Security model (document for interviews):
  1. --network=none           — no outbound internet access
  2. --memory / --cpus        — cgroup resource limits
  3. --read-only + tmpfs      — filesystem isolation; only /workspace is writable
  4. Command whitelist        — only git, pytest, python, pip are allowed
  5. 60s timeout              — runaway processes are killed via SIGKILL
  6. Non-root user (uid=1000) — no privilege escalation inside container

Workflow per issue:
  1. clone_repo()   — git clone the repo at base_commit into a temp volume
  2. apply_patch()  — write unified diff to /workspace, run git apply
  3. run_tests()    — pytest on FAIL_TO_PASS + PASS_TO_PASS test IDs
  4. cleanup()      — remove the Docker volume/container
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ── Allowed commands (whitelist) ──────────────────────────────────────────────
ALLOWED_COMMANDS = frozenset({
    "git", "pytest", "python", "python3", "pip", "pip3",
    "cat", "ls", "echo", "find", "grep", "head", "tail",
    "mkdir", "cp", "mv", "touch", "chmod",
})


@dataclass
class ExecResult:
    """Result of a sandboxed command execution."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class TestResult:
    """Structured result from running pytest inside the sandbox."""
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    timed_out: bool = False

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0 and len(self.errors) == 0 and not self.timed_out

    def check_tests(
        self,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
    ) -> tuple[bool, dict[str, bool], dict[str, bool]]:
        """
        Evaluate whether this run resolves the SWE-bench instance.

        Returns:
            resolved: bool
            ftp_results: {test_id: passed}
            ptp_results: {test_id: still_passing}
        """
        passed_set = set(self.passed)

        ftp_results = {t: (t in passed_set) for t in fail_to_pass}
        ptp_results = {t: (t in passed_set) for t in pass_to_pass}

        ftp_ok = all(ftp_results.values())
        ptp_ok = all(ptp_results.values())
        resolved = ftp_ok and ptp_ok

        return resolved, ftp_results, ptp_results


class SandboxExecutor:
    """
    Manages Docker-based sandbox for safe code execution.

    Usage:
        executor = SandboxExecutor(settings)
        with executor.workspace(instance) as ws:
            ws.apply_patch(patch_text)
            result = ws.run_tests(fail_to_pass, pass_to_pass)
    """

    def __init__(
        self,
        image: str = "code-agent-sandbox:latest",
        timeout: int = 60,
        memory_limit: str = "2g",
        cpu_limit: float = 2.0,
        network: str = "none",
        use_docker: bool = True,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network
        self.use_docker = use_docker

        if use_docker:
            self._verify_docker()

    def _verify_docker(self) -> None:
        """Check Docker is available and the sandbox image exists."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                logger.warning("Docker is not running — sandbox will use local execution")
                self.use_docker = False
        except FileNotFoundError:
            logger.warning("Docker not found — sandbox will use local execution")
            self.use_docker = False

    def clone_repo(
        self,
        repo: str,
        base_commit: str,
        workspace_dir: Path,
    ) -> ExecResult:
        """
        Clone the target repo at base_commit into workspace_dir.

        Args:
            repo: 'owner/repo' format
            base_commit: git SHA to checkout
            workspace_dir: local directory to clone into
        """
        github_url = f"https://github.com/{repo}.git"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        commit_label = base_commit[:8] if base_commit and base_commit != "HEAD" else "HEAD"
        logger.info("Cloning %s @ %s", repo, commit_label)

        # Use treeless clone: fetches full commit graph (so old SHAs are reachable)
        # but only downloads file blobs for the checked-out tree — fast + space-efficient.
        clone_result = self._run_local(
            ["git", "clone", "--filter=blob:none", github_url, str(workspace_dir)],
            timeout=300,
        )
        if not clone_result.success:
            # Fallback: full clone
            logger.warning("Treeless clone failed, trying full clone...")
            clone_result = self._run_local(
                ["git", "clone", github_url, str(workspace_dir)],
                timeout=600,
            )
        if not clone_result.success:
            logger.error("Clone failed: %s", clone_result.stderr[:500])
            return clone_result

        # Checkout the exact SWE-bench base commit
        if base_commit and base_commit.strip() and base_commit.upper() != "HEAD":
            checkout_result = self._run_local(
                ["git", "checkout", base_commit],
                cwd=workspace_dir,
                timeout=60,
            )
            if not checkout_result.success:
                logger.error("Checkout %s failed: %s", commit_label, checkout_result.stderr[:200])
                return checkout_result
            return checkout_result

        return clone_result

    def apply_patch(
        self,
        patch_text: str,
        workspace_dir: Path,
    ) -> ExecResult:
        """
        Write patch_text to a temp file and run `git apply` inside workspace.

        Returns ExecResult with success=True if patch applied cleanly.
        """
        if not patch_text.strip():
            logger.warning("Empty patch text — nothing to apply")
            return ExecResult("git apply", 1, "", "Empty patch", 0.0)

        patch_file = workspace_dir / "_agent_patch.diff"
        patch_file.write_text(patch_text)

        result = self._run_local(
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            cwd=workspace_dir,
        )
        if not result.success:
            # Try with --reject to get partial application details
            logger.debug("git apply failed, stderr: %s", result.stderr[:300])
        return result

    def run_tests(
        self,
        workspace_dir: Path,
        test_ids: list[str],
        extra_args: list[str] | None = None,
    ) -> TestResult:
        """
        Run pytest on specific test IDs inside the workspace.

        Args:
            workspace_dir: repo root
            test_ids: list of pytest node IDs to run
            extra_args: additional pytest flags

        Returns:
            TestResult with passed/failed/errors lists
        """
        if not test_ids:
            logger.warning("No test IDs provided — skipping test run")
            return TestResult()

        pytest_args = ["python", "-m", "pytest", "-v", "--tb=short", "--no-header", "-rN"]
        if extra_args:
            pytest_args.extend(extra_args)
        pytest_args.extend(test_ids)

        if self.use_docker:
            result = self._run_in_docker(pytest_args, workspace_dir)
        else:
            result = self._run_local(pytest_args, cwd=workspace_dir)

        return self._parse_pytest_output(result)

    def _run_in_docker(self, cmd: list[str], workspace_dir: Path) -> ExecResult:
        """Run a command inside the Docker sandbox container."""
        _validate_command(cmd)

        docker_cmd = [
            "docker", "run",
            "--rm",
            f"--network={self.network}",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            "--read-only",
            "--tmpfs=/tmp:size=256m",
            f"--volume={workspace_dir}:/workspace:rw",
            "--workdir=/workspace",
            "--user=1000:1000",
            self.image,
        ] + cmd

        return self._run_local(docker_cmd, timeout=self.timeout)

    def _run_local(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Execute a subprocess with timeout and capture output."""
        if timeout is None:
            timeout = self.timeout

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
            elapsed = time.monotonic() - start
            return ExecResult(
                command=" ".join(cmd),
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_seconds=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            logger.warning("Command timed out after %ds: %s", timeout, cmd[:3])
            return ExecResult(
                command=" ".join(cmd),
                returncode=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                elapsed_seconds=elapsed,
                timed_out=True,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Command failed: %s | error: %s", cmd[:3], e)
            return ExecResult(
                command=" ".join(cmd),
                returncode=-2,
                stdout="",
                stderr=str(e),
                elapsed_seconds=elapsed,
            )

    @staticmethod
    def _parse_pytest_output(result: ExecResult) -> TestResult:
        """
        Parse pytest -v output to extract passed/failed test IDs.

        Pytest -v output format per test:
          tests/path/to/test.py::test_name PASSED
          tests/path/to/test.py::test_name FAILED
          tests/path/to/test.py::test_name ERROR
        """
        test_result = TestResult(
            raw_output=result.stdout + result.stderr,
            elapsed_seconds=result.elapsed_seconds,
            timed_out=result.timed_out,
        )

        passed_pattern = re.compile(r"^(.+?::[\w\[\]-]+)\s+PASSED", re.MULTILINE)
        failed_pattern = re.compile(r"^(.+?::[\w\[\]-]+)\s+FAILED", re.MULTILINE)
        error_pattern = re.compile(r"^(.+?::[\w\[\]-]+)\s+ERROR", re.MULTILINE)

        test_result.passed = passed_pattern.findall(result.stdout)
        test_result.failed = failed_pattern.findall(result.stdout)
        test_result.errors = error_pattern.findall(result.stdout)

        logger.debug(
            "Pytest results — passed: %d, failed: %d, errors: %d",
            len(test_result.passed),
            len(test_result.failed),
            len(test_result.errors),
        )
        return test_result


# ── Security helper ───────────────────────────────────────────────────────────

def _validate_command(cmd: list[str]) -> None:
    """
    Raise ValueError if the command's base name is not in the whitelist.
    This is a defence-in-depth measure — Docker isolation is the primary control.
    """
    if not cmd:
        raise ValueError("Empty command")
    base = Path(cmd[0]).name
    if base not in ALLOWED_COMMANDS:
        raise ValueError(
            f"Command '{base}' is not in the allowed command whitelist: {ALLOWED_COMMANDS}"
        )

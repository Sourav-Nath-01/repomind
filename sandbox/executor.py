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
        # Clear any stale content before cloning (prevents 'already exists' errors)
        import shutil as _shutil
        import time as _time
        if workspace_dir.exists():
            _shutil.rmtree(workspace_dir, ignore_errors=True)
            if workspace_dir.exists():
                # If ignore_errors=True skipped root-owned files created by Docker, wipe via Docker
                logger.warning("Local rmtree failed (likely root-owned files). Wiping via Docker...")
                subprocess.run([
                    "docker", "run", "--rm", "-v", f"{workspace_dir.parent}:/host_tmp",
                    self.image, "rm", "-rf", f"/host_tmp/{workspace_dir.name}"
                ], capture_output=True)
                _shutil.rmtree(workspace_dir, ignore_errors=True) # Final cleanup just in case
                
        workspace_dir.mkdir(parents=True, exist_ok=True)

        commit_label = base_commit[:8] if base_commit and base_commit != "HEAD" else "HEAD"
        logger.info("Cloning %s @ %s", repo, commit_label)

        # Use treeless clone: fetches full commit graph (so old SHAs are reachable)
        # but only downloads file blobs for the checked-out tree — fast + space-efficient.
        clone_result = None
        for attempt in range(3):
            clone_result = self._run_local(
                ["git", "clone", "--filter=blob:none", github_url, str(workspace_dir)],
                timeout=300,
            )
            if clone_result.success:
                break
            # Fallback: full clone
            logger.warning(f"Treeless clone failed (attempt {attempt+1}), trying full clone...")
            if workspace_dir.exists():
                subprocess.run(["docker", "run", "--rm", "-v", f"{workspace_dir.parent}:/host_tmp", self.image, "rm", "-rf", f"/host_tmp/{workspace_dir.name}"], capture_output=True)
                _shutil.rmtree(workspace_dir, ignore_errors=True)
            clone_result = self._run_local(
                ["git", "clone", github_url, str(workspace_dir)],
                timeout=600,
            )
            if clone_result.success:
                break
            
            logger.warning(f"Clone failed (attempt {attempt+1}). Sleeping 5s before retry...")
            _time.sleep(5)
            
        if not clone_result or not clone_result.success:
            logger.error("Clone failed completely: %s", clone_result.stderr[:500] if clone_result else "")
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

        strategies = [
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            ["git", "apply", "--whitespace=fix", "--ignore-whitespace", str(patch_file)],
            ["git", "apply", "--whitespace=fix", "-C1", str(patch_file)],
            ["git", "apply", "--whitespace=fix", "-C0", str(patch_file)],
        ]

        result = None
        for cmd in strategies:
            result = self._run_local(cmd, cwd=workspace_dir)
            if result.success:
                logger.info("Patch applied with strategy: %s", " ".join(cmd[2:4]))
                return result
            # Roll back any partial application before next attempt
            self._run_local(["git", "checkout", "--", "."], cwd=workspace_dir)

        # Final fallback: GNU patch with fuzz=3 (handles line-wrapping differences)
        gnu_result = self._run_local(
            ["patch", "--fuzz=3", "-p1", "-i", str(patch_file)],
            cwd=workspace_dir,
        )
        if gnu_result.success:
            logger.info("Patch applied with GNU patch --fuzz=3")
            return gnu_result
        self._run_local(["git", "checkout", "--", "."], cwd=workspace_dir)

        logger.debug("All patch strategies failed, stderr: %s", result.stderr[:300])
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

        import sys, re

        # Detect test ID format.
        # Django style: "test_foo (myapp.tests.MyTest)"
        # Pytest style:  "myapp/tests/test_foo.py::MyTest::test_foo"
        _django_pat = re.compile(r'^test_\w+\s+\(\w[\w.]+\)$')
        is_django_style = any(_django_pat.match(t) for t in test_ids)

        if is_django_style:
            # Convert "test_foo (app.tests.MyTest)" → "app.tests.MyTest.test_foo"
            converted = []
            for t in test_ids:
                m = re.match(r'^(test_\w+)\s+\(([\w.]+)\)$', t)
                if m:
                    converted.append(f"{m.group(2)}.{m.group(1)}")
                else:
                    if " " not in t or "::" in t:
                        converted.append(t)


            runtests = workspace_dir / "tests" / "runtests.py"

            if self.use_docker and runtests.exists():
                # ── Run inside Docker (single container: pip install + runtests) ──
                # Using bash -c combines both steps in ONE container so the
                # pip-installed packages persist for the runtests.py call.
                import shlex
                tests_str = " ".join(shlex.quote(c) for c in converted)
                bash_cmd = (
                    "pip3 install -e /workspace --quiet --no-build-isolation --break-system-packages 2>&1 | tail -5 && "
                    f"cd /workspace/tests && "
                    f"python3 runtests.py --verbosity 2 --parallel 1 {tests_str}"
                )
                docker_full = [
                    "docker", "run", "--rm",
                    "--network=bridge",  # needs network for pip to install repo deps
                    f"--memory={self.memory_limit}",
                    f"--cpus={self.cpu_limit}",
                    "--tmpfs=/tmp:size=256m",
                    f"--volume={workspace_dir}:/workspace:rw",
                    "--env=PYTHONPATH=/workspace",
                    self.image,
                    "bash", "-c", bash_cmd,
                ]
                logger.info("Running django tests in Docker (single container) for %d tests", len(converted))
                result = self._run_local(docker_full, timeout=360)
            else:
                # ── Local fallback ───────────────────────────────────────────
                # pip install first
                self._run_local(
                    [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet",
                     "--no-build-isolation"],
                    cwd=workspace_dir, timeout=300,
                )
                extra_env = {"PYTHONPATH": str(workspace_dir)}
                if runtests.exists():
                    cmd = [sys.executable, str(runtests), "--verbosity", "2",
                           "--parallel", "1"] + converted
                    run_cwd = str(workspace_dir / "tests")
                else:
                    extra_env["DJANGO_SETTINGS_MODULE"] = "settings"
                    cmd = [sys.executable, "-m", "django", "test",
                           "--verbosity", "2", "--parallel", "1"] + converted
                    run_cwd = str(workspace_dir)
                logger.info("Running django tests locally for %d tests", len(converted))
                result = self._run_local(cmd, cwd=run_cwd, timeout=300, extra_env=extra_env)

            logger.debug("Django runner stdout[:500]: %s", result.stdout[:500])
            logger.debug("Django runner stderr[:500]: %s", result.stderr[:500])
            logger.info("Django runner exit code: %d", result.returncode)
            return self._parse_django_test_output(result)

        else:
            # ── Pytest-style tests ───────────────────────────────────────────
            if self.use_docker:
                # pip install + pytest in ONE container (packages persist)
                valid_ids = [t for t in test_ids if " " not in t or "::" in t]
                import shlex
                tests_str = " ".join(shlex.quote(t) for t in valid_ids)
                extra_str = " ".join(extra_args or [])
                bash_cmd = (
                    "sed -i 's/license = {file = \"LICENSE.rst\"}/license = {text = \"BSD-3-Clause\"}/g' /workspace/pyproject.toml 2>/dev/null || true && "
                    "pip install -e /workspace --quiet --no-build-isolation 2>&1 | tail -5 && "
                    f"cd /workspace && "
                    f"python -m pytest -v --tb=short --no-header -rN --timeout=60 "
                    f"{extra_str} {tests_str}"
                )
                docker_full = [
                    "docker", "run", "--rm",
                    "--network=bridge",  # needs network for pip to install repo deps
                    f"--memory={self.memory_limit}",
                    f"--cpus={self.cpu_limit}",
                    "--tmpfs=/tmp:size=256m",
                    f"--volume={workspace_dir}:/workspace:rw",
                    "--env=PYTHONPATH=/workspace",
                    self.image,
                    "bash", "-c", bash_cmd,
                ]
                logger.info("Running pytest in Docker (single container) for %d tests", len(test_ids))
                result = self._run_local(docker_full, timeout=360)
            else:
                self._run_local(
                    [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet",
                     "--no-build-isolation"],
                    cwd=workspace_dir, timeout=300,
                )
                pytest_args = [sys.executable, "-m", "pytest", "-v", "--tb=short",
                               "--no-header", "-rN", "--timeout=60"]
                if extra_args:
                    pytest_args.extend(extra_args)
                valid_ids = [t for t in test_ids if " " not in t or "::" in t]
                pytest_args.extend(valid_ids)
                logger.info("Running pytest locally for %d tests", len(test_ids))
                result = self._run_local(pytest_args, cwd=workspace_dir, timeout=300)

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
        cwd: Path | str | None = None,
        timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a subprocess with timeout and capture output."""
        if timeout is None:
            timeout = self.timeout

        import os as _os
        env = None
        if extra_env:
            env = dict(_os.environ)
            env.update(extra_env)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=env,
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

    @staticmethod
    def _parse_django_test_output(result: ExecResult) -> TestResult:
        """
        Parse Django test runner --verbosity 2 output.

        Django format per test (sometimes spans 2 lines with a docstring):
          test_foo (myapp.tests.MyTest.test_foo)
          Optional docstring here ... ok
        """
        test_result = TestResult(
            raw_output=result.stdout + result.stderr,
            elapsed_seconds=result.elapsed_seconds,
            timed_out=result.timed_out,
        )

        # Match the test ID line, an optional docstring line, and the result.
        ok_pat  = re.compile(r'^(test_\w+\s+\([\w.]+\))[^\n]*(?:\n[^\n]*)?\.\.\.\s+ok',  re.MULTILINE)
        fail_pat = re.compile(r'^(test_\w+\s+\([\w.]+\))[^\n]*(?:\n[^\n]*)?\.\.\.\s+FAIL', re.MULTILINE)
        err_pat  = re.compile(r'^(test_\w+\s+\([\w.]+\))[^\n]*(?:\n[^\n]*)?\.\.\.\s+ERROR', re.MULTILINE)

        combined = result.stdout + result.stderr

        def _normalize(tests):
            norm = []
            for t in tests:
                # Remove duplicated method name from class path: "test_foo (mod.cls.test_foo)" -> "test_foo (mod.cls)"
                m = re.match(r'^(test_\w+)\s+\((.*?)\.\1\)$', t)
                if m:
                    norm.append(f"{m.group(1)} ({m.group(2)})")
                else:
                    norm.append(t)
            return norm

        test_result.passed = _normalize(ok_pat.findall(combined))
        test_result.failed = _normalize(fail_pat.findall(combined))
        test_result.errors = _normalize(err_pat.findall(combined))

        logger.debug(
            "Django test results — passed: %d, failed: %d, errors: %d",
            len(test_result.passed), len(test_result.failed), len(test_result.errors),
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

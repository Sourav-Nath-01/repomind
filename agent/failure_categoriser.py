"""
agent/failure_categoriser.py
──────────────────────────────
Rule-based + regex failure categoriser.

After each failed attempt, the agent parses pytest output and classifies
the failure into one of these categories:

  syntax_error        — the patch introduced a SyntaxError
  hallucinated_api    — agent called a function/attribute that doesn't exist
  wrong_file_edit     — agent edited the wrong file (tests in different module fail)
  incomplete_patch    — partial fix: some tests pass but not all FAIL_TO_PASS
  flaky_test          — test is non-deterministic (passes on retry)
  import_error        — missing import or circular import introduced
  type_error          — wrong argument type passed
  assertion_error     — logic bug remains, assertion fails with unexpected value
  unknown             — can't categorise

The category is logged to MLflow and stored in trajectory JSONL.
This taxonomy directly drives which trajectories we select for fine-tuning
(Phase 7 filters on known-category failures).
"""
from __future__ import annotations

import re
from typing import Literal

FailureCategory = Literal[
    "syntax_error",
    "hallucinated_api",
    "wrong_file_edit",
    "incomplete_patch",
    "flaky_test",
    "import_error",
    "type_error",
    "assertion_error",
    "success",
    "unknown",
]

# ── Regex patterns ────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[FailureCategory, re.Pattern]] = [
    ("syntax_error",     re.compile(r"SyntaxError|IndentationError|TabError", re.I)),
    ("import_error",     re.compile(r"ImportError|ModuleNotFoundError|cannot import name", re.I)),
    ("hallucinated_api", re.compile(
        r"AttributeError: .+ object has no attribute|"
        r"TypeError: .+ takes \d+ positional argument|"
        r"NameError: name .+ is not defined",
        re.I
    )),
    ("type_error",       re.compile(r"TypeError:", re.I)),
    ("assertion_error",  re.compile(r"AssertionError", re.I)),
]

_FLAKY_PATTERNS = re.compile(
    r"ResourceWarning|"
    r"random|"
    r"race condition|"
    r"flaky|"
    r"connection refused|"
    r"socket\.timeout",
    re.I
)


def categorise_failure(
    test_stdout: str,
    patch_apply_success: bool,
    fail_to_pass_results: dict[str, bool],
    pass_to_pass_results: dict[str, bool],
    attempt_num: int = 1,
    previous_categories: list[FailureCategory] | None = None,
) -> FailureCategory:
    """
    Classify a failed attempt into a FailureCategory.

    Decision flow:
      1. Patch didn't apply → syntax_error
      2. All FAIL_TO_PASS pass → success
      3. Scan error messages in stdout for pattern matches
      4. If same test failed differently across attempts → flaky_test
      5. If some FTP pass but not all → incomplete_patch
      6. Fallback: unknown

    Args:
        test_stdout:           raw pytest output
        patch_apply_success:   whether `git apply` succeeded
        fail_to_pass_results:  {test_id: passed} for FAIL_TO_PASS tests
        pass_to_pass_results:  {test_id: still_passing} for PASS_TO_PASS tests
        attempt_num:           current attempt number (1-indexed)
        previous_categories:   categories from earlier attempts (flaky detection)

    Returns:
        FailureCategory string
    """
    # 1. Patch apply failed → likely syntax_error in diff
    if not patch_apply_success:
        return "syntax_error"

    # 2. All tests pass → success
    ftp_ok = all(fail_to_pass_results.values()) if fail_to_pass_results else False
    ptp_ok = all(pass_to_pass_results.values()) if pass_to_pass_results else True
    if ftp_ok and ptp_ok:
        return "success"

    # 3. Scan pytest output for error patterns
    for category, pattern in _PATTERNS:
        if pattern.search(test_stdout):
            return category

    # 4. Flaky test detection: if we've seen different failures across attempts
    if previous_categories and len(set(previous_categories)) > 1:
        if _FLAKY_PATTERNS.search(test_stdout):
            return "flaky_test"

    # 5. Partial success — some FTP tests pass but not all
    ftp_passed = sum(1 for v in fail_to_pass_results.values() if v)
    ftp_total = len(fail_to_pass_results)
    if ftp_passed > 0 and ftp_passed < ftp_total:
        return "incomplete_patch"

    # 6. PASS_TO_PASS regression only (our patch broke existing tests)
    ptp_failed = sum(1 for v in pass_to_pass_results.values() if not v)
    if ptp_failed > 0 and ftp_passed == ftp_total:
        return "wrong_file_edit"

    return "unknown"


def extract_first_error_context(test_stdout: str, max_lines: int = 20) -> str:
    """
    Extract the most relevant error lines from pytest output.
    Used to build the reflection prompt — give the LLM targeted failure info.
    """
    lines = test_stdout.splitlines()

    # Find first FAILED line and return context around it
    for i, line in enumerate(lines):
        if "FAILED" in line or "ERROR" in line or "assert" in line.lower():
            start = max(0, i - 2)
            end = min(len(lines), i + max_lines)
            return "\n".join(lines[start:end])

    # Fallback: last N lines (pytest puts summary at end)
    return "\n".join(lines[-max_lines:])

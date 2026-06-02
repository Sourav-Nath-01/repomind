"""
agent/reflection_agent.py
──────────────────────────
Agentic Reflection Loop — self-correcting bug-fix agent.

Loop (max 3 attempts):
  1. Localise relevant files (from Phase 3 pipeline)
  2. Build prompt: issue + file contents + (on retry) error context
  3. Call LLM → get unified diff
  4. Apply patch (git apply)
  5. Run tests (sandbox)
  6. If PASS → done ✅
  7. If FAIL → categorise failure, update prompt with error context → goto 2

On each iteration the agent:
  - Reads the exact pytest error output
  - Appends it to the prompt with a targeted correction request
  - The LLM sees the code it wrote AND the test failure it caused

This is the "genuinely ML hard" part:
  - Each trajectory is logged as JSONL (for Phase 7 fine-tuning)
  - Failure categories are tracked in MLflow
  - Token cost is metered per attempt

LangGraph is used to model the state machine: each node is one step,
edges have conditional routing based on test outcome.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state passed between LangGraph nodes."""
    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    workspace_dir: Path

    # Filled during execution
    localised_files: list[str] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)  # path → content
    attempts: list[dict] = field(default_factory=list)           # attempt records
    current_attempt: int = 0
    last_patch: str = ""
    last_test_stdout: str = ""
    last_failure_category: str = "unknown"
    resolved: bool = False
    error: str = ""  # non-empty if agent crashed

    # Token tracking
    total_tokens: int = 0


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python software engineer specialising in bug fixes.
Your task is to fix a bug in a Python repository by generating a minimal unified diff.

CRITICAL RULES — READ CAREFULLY:
1. Output ONLY the raw unified diff. No explanations, no markdown, no code fences.
2. The diff MUST be in standard `git diff` format:
   --- a/<filepath>
   +++ b/<filepath>
   @@ -<start>,<count> +<start>,<count> @@ <optional context>
   <unchanged context lines starting with a space>
   -<removed line>
   +<added line>
   <unchanged context lines starting with a space>
3. Context lines (unchanged lines) MUST be COPIED EXACTLY character-for-character from
   the file content shown to you below. Do NOT paraphrase or rewrite them.
4. Include 3 unchanged context lines before and after each change.
5. The line numbers in @@ headers MUST match the actual file line numbers shown to you.
6. Be minimal: only change what is necessary. Do not refactor unrelated code.
7. NEVER reformat, re-indent, or split long lines in context. If a line in the file is
   100 characters long, it must appear as ONE line in the diff — never as two wrapped lines.
   Wrapping a context line breaks the patch and makes it impossible to apply.
8. File contents below are shown as `NNN | code`. DO NOT copy the `NNN | ` prefix
   into your diff — write only the raw source code lines in context/added/removed lines.
9. NEVER add comments to `+` lines (e.g., `# This line...`). Only write the exact replacement
   code, nothing else. Comments in added lines change the program and break tests.
10. The diff MUST end with a trailing newline character.
"""

INITIAL_PROMPT_TEMPLATE = """\
## GitHub Issue
{problem_statement}

## Relevant File Contents (with line numbers)
{file_context}

Generate a unified diff patch that fixes this issue.
Remember: copy context lines EXACTLY from the file contents above.
"""

REFLECTION_PROMPT_TEMPLATE = """\
## GitHub Issue
{problem_statement}

## Relevant Files
{file_context}

## Previous Attempt #{attempt_num} FAILED
Failure category: {failure_category}

### Test Output (showing failures)
{error_context}

### Your Previous Patch
{previous_patch}

The patch above did not fully fix the issue. Carefully analyse the test failures
and generate a CORRECTED unified diff. Focus specifically on the error shown above.
"""


# ── LangGraph node functions ──────────────────────────────────────────────────

def node_localise(state: AgentState, pipeline=None) -> AgentState:
    """
    Node: run the localisation pipeline to find relevant files.
    If pipeline is None, reads file_contents from state (already provided).
    """
    if pipeline and not state.file_contents:
        result = pipeline.localise(state.problem_statement, top_k=5)
        state.localised_files = result.top_k_paths
        logger.info(
            "Localised %d files for %s", len(state.localised_files), state.instance_id
        )

    # If still no files, scan the workspace for relevant Python files
    if not state.localised_files and state.workspace_dir and state.workspace_dir.exists():
        import re

        # Common English words unlikely to be filenames
        _STOPWORDS = {
            'database', 'support', 'current', 'statement', 'consider', 'changes',
            'because', 'without', 'function', 'argument', 'instance', 'however',
            'correct', 'instead', 'related', 'problem', 'setting', 'between',
            'version', 'through', 'default', 'returns', 'contains', 'existing',
            'objects', 'methods', 'classes', 'calling', 'running', 'working',
            'example', 'feature', 'another', 'getting', 'testing', 'possible',
            'message', 'missing', 'provide', 'request', 'response', 'created',
            'updated', 'deleted', 'whether', 'against', 'already', 'nothing',
            'further', 'minimum', 'maximum', 'integer', 'boolean', 'section',
            'initial', 'expected', 'following', 'executor', 'changing', 'features',
            'rollback', 'connection', 'migration', 'assignment', 'transactional',
            'consideration', 'attribute', 'parameter', 'exception', 'wrapping',
            'optional', 'iterable', 'callable', 'readable', 'writable',
        }

        file_hints = set()

        # S1a: quoted .py filenames
        file_hints.update(re.findall(r'[`\'"](\S+?\.py)[`\'\"\s]', state.problem_statement))

        # S1b: backtick-quoted identifiers (e.g. `sqlmigrate`, `can_rollback_ddl`)
        for word in re.findall(r'`([a-z][a-z0-9_]{3,})`', state.problem_statement.lower()):
            if word not in _STOPWORDS:
                file_hints.add(word + '.py')

        # S1c: dotted module paths (e.g. connection.features.can_rollback_ddl)
        for m in re.findall(r'\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){2,})\b',
                            state.problem_statement.lower()):
            file_hints.add(m.replace('.', '/') + '.py')
            file_hints.add(m.split('.')[-1] + '.py')

        # S2a: underscore_names (e.g. output_transaction, can_rollback_ddl)
        for word in re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b',
                               state.problem_statement.lower()):
            if len(word) > 5 and word not in _STOPWORDS:
                file_hints.add(word + '.py')

        # S2b: long plain words 7+ chars — catches "sqlmigrate", "separable", etc.
        for word in re.findall(r'\b([a-z]{7,})\b', state.problem_statement.lower()):
            if word not in _STOPWORDS:
                file_hints.add(word + '.py')

        # S2c: CamelCase -> snake_case (e.g. MigrateTests -> migrate_tests)
        for word in re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', state.problem_statement):
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower()
            if snake not in _STOPWORDS:
                file_hints.add(snake + '.py')

        # S3: instance_id parts — skip first 2 generic project-name parts
        # e.g. "django__django-11039" -> skip "django", only use specific parts
        id_parts = re.split(r'[-_]', state.instance_id.lower())
        generic_names = set(id_parts[:2])
        for p in id_parts[2:]:
            if len(p) > 3 and not p.isdigit() and p not in _STOPWORDS:
                file_hints.add(p + '.py')
        # Remove generic project names to avoid false positives like "django.py"
        file_hints -= {n + '.py' for n in generic_names}

        # Score Python files in workspace
        candidates = []
        for py_file in state.workspace_dir.rglob('*.py'):
            rel = str(py_file.relative_to(state.workspace_dir))
            rel_lower = rel.lower()
            filename = py_file.stem.lower()
            if any(seg in rel_lower for seg in ('test', '__pycache__', '.egg', 'doc', 'setup')):
                continue
            score = 0
            for hint in file_hints:
                hint_stem = hint.replace('.py', '').lower()
                if hint_stem == filename:
                    score += 3   # exact filename match
                elif hint_stem in rel_lower:
                    score += 1   # path fragment match
            if score > 0:
                candidates.append((score, rel))

        candidates.sort(key=lambda x: -x[0])
        state.localised_files = [rel for _, rel in candidates[:5]]

        if state.localised_files:
            logger.info(
                "Smart-localised %d files for %s: %s",
                len(state.localised_files), state.instance_id, state.localised_files
            )
        else:
            all_py = sorted(state.workspace_dir.rglob('*.py'))
            state.localised_files = [
                str(f.relative_to(state.workspace_dir))
                for f in all_py
                if not any(p.startswith('.') or p in ('tests', '__pycache__', 'docs')
                           for p in f.parts)
            ][:5]
            logger.info(
                "Fallback-localised %d files for %s",
                len(state.localised_files), state.instance_id
            )

    # Read file contents from workspace
    from agent.tools import AgentTools
    tools = AgentTools(state.workspace_dir)
    for fp in state.localised_files:
        read_result = tools.read_file(fp, max_lines=2000)
        if read_result.success:
            state.file_contents[fp] = read_result.output
        else:
            logger.debug("Could not read %s: %s", fp, read_result.error)

    return state


def node_generate_patch(state: AgentState, llm_client=None, model: str = "gpt-4o") -> AgentState:
    """
    Node: call LLM to generate a patch.
    First attempt uses initial prompt; subsequent attempts use reflection prompt.
    """
    state.current_attempt += 1

    file_context = _build_file_context(state.file_contents)

    if state.current_attempt == 1:
        user_prompt = INITIAL_PROMPT_TEMPLATE.format(
            problem_statement=state.problem_statement[:2000],
            file_context=file_context,
        )
    else:
        from agent.failure_categoriser import extract_first_error_context
        error_context = extract_first_error_context(state.last_test_stdout)

        user_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            problem_statement=state.problem_statement[:1500],
            file_context=file_context,
            attempt_num=state.current_attempt - 1,
            failure_category=state.last_failure_category,
            error_context=error_context[:800],
            previous_patch=state.last_patch[:1000],
        )

    logger.info(
        "Generating patch for %s (attempt %d/%d)",
        state.instance_id, state.current_attempt, 3
    )

    patch_text, usage = _call_llm(user_prompt, llm_client, model)
    # Normalize CRLF → LF: git apply rejects patches with Windows line endings
    raw = _strip_code_fences(patch_text).replace('\r\n', '\n').replace('\r', '\n')
    # Recompute @@ hunk counts: LLMs often produce wrong counts when wrapping long lines
    fixed = _fix_patch_hunk_counts(raw)
    # Ensure patch ends with a trailing newline (required by git apply)
    state.last_patch = fixed if fixed.endswith('\n') else fixed + '\n'
    state.total_tokens += usage.get("total_tokens", 0)
    return state


def node_apply_and_test(state: AgentState, sandbox=None) -> AgentState:
    """
    Node: apply the patch and run tests.
    Populates state.resolved and state.last_test_stdout.
    """
    import subprocess as _sp
    from agent.tools import AgentTools
    tools = AgentTools(state.workspace_dir, sandbox)

    # ── Reset workspace to base_commit before every attempt ───────────────
    # This is critical: if a prior attempt partially modified files, the next
    # patch (generated against the original base) will fail to apply.
    if state.workspace_dir.exists():
        _sp.run(
            ["git", "reset", "--hard", state.base_commit],
            capture_output=True, cwd=str(state.workspace_dir)
        )
        _sp.run(
            ["git", "clean", "-fd"],
            capture_output=True, cwd=str(state.workspace_dir)
        )

    # Write and apply patch
    write_result = tools.write_patch(state.last_patch)
    patch_apply_success = False

    if write_result.success:
        if sandbox:
            from sandbox.executor import SandboxExecutor
            apply_result = sandbox.apply_patch(state.last_patch, state.workspace_dir)
            patch_apply_success = apply_result.success
        else:
            try:
                proc = _sp.run(
                    ["git", "apply", "--whitespace=fix", "_agent_patch.diff"],
                    capture_output=True, text=True, cwd=str(state.workspace_dir), timeout=10
                )
                patch_apply_success = proc.returncode == 0
            except Exception:
                patch_apply_success = False

    # Run tests
    all_test_ids = state.fail_to_pass + state.pass_to_pass
    test_result_obj = tools.run_tests(all_test_ids)
    state.last_test_stdout = test_result_obj.metadata.get("full_output", test_result_obj.output)

    # Parse results
    if sandbox:
        from sandbox.executor import SandboxExecutor
        test_result = sandbox.run_tests(state.workspace_dir, all_test_ids)
        resolved, ftp_results, ptp_results = test_result.check_tests(
            state.fail_to_pass, state.pass_to_pass
        )
        state.last_test_stdout = test_result.raw_output
    else:
        # Minimal local parse
        ftp_results = _parse_local_test_results(
            state.last_test_stdout, state.fail_to_pass
        )
        ptp_results = _parse_local_test_results(
            state.last_test_stdout, state.pass_to_pass
        )
        resolved = all(ftp_results.values()) and all(ptp_results.values())

    state.resolved = resolved

    # Categorise failure
    from agent.failure_categoriser import categorise_failure
    prev_cats = [a.get("failure_category", "unknown") for a in state.attempts]
    state.last_failure_category = categorise_failure(
        test_stdout=state.last_test_stdout,
        patch_apply_success=patch_apply_success,
        fail_to_pass_results=ftp_results,
        pass_to_pass_results=ptp_results,
        attempt_num=state.current_attempt,
        previous_categories=prev_cats,
    )

    # Record attempt
    state.attempts.append({
        "attempt_num": state.current_attempt,
        "patch": state.last_patch,
        "test_stdout": state.last_test_stdout[:3000],
        "fail_to_pass_results": ftp_results,
        "pass_to_pass_results": ptp_results,
        "resolved": resolved,
        "failure_category": state.last_failure_category,
    })

    logger.info(
        "Attempt %d: resolved=%s category=%s",
        state.current_attempt, resolved, state.last_failure_category
    )
    return state


def should_retry(state: AgentState, max_attempts: int = 3) -> Literal["retry", "done"]:
    """LangGraph conditional edge: retry if not resolved and budget remains."""
    if state.resolved:
        return "done"
    if state.current_attempt >= max_attempts:
        return "done"
    return "retry"


# ── Full agent ────────────────────────────────────────────────────────────────

class ReflectionAgent:
    """
    Self-correcting bug-fix agent with configurable retry budget.

    Uses LangGraph for state machine management if available,
    falls back to a simple Python loop otherwise.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_attempts: int = 3,
        sandbox=None,
        localisation_pipeline=None,
        trajectory_logger=None,
    ):
        self.model = model
        self.max_attempts = max_attempts
        self.sandbox = sandbox
        self.pipeline = localisation_pipeline
        self.traj_logger = trajectory_logger
        self._use_langgraph = self._check_langgraph()

    def _check_langgraph(self) -> bool:
        try:
            import langgraph  # noqa: F401
            return True
        except ImportError:
            logger.debug("LangGraph not installed — using simple loop")
            return False

    def run(
        self,
        instance_id: str,
        repo: str,
        problem_statement: str,
        base_commit: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        workspace_dir: Path,
        localised_files: list[str] | None = None,
    ) -> AgentState:
        """
        Run the full reflection loop on one SWE-bench instance.

        Returns final AgentState (resolved/not, all attempts recorded).
        """
        state = AgentState(
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem_statement,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            workspace_dir=Path(workspace_dir),
            localised_files=localised_files or [],
        )

        if self._use_langgraph:
            state = self._run_with_langgraph(state)
        else:
            state = self._run_simple_loop(state)

        # Log trajectories
        if self.traj_logger:
            self._log_trajectories(state)

        return state

    def _run_simple_loop(self, state: AgentState) -> AgentState:
        """Fallback: plain Python loop (no LangGraph dependency)."""
        # Localise files
        state = node_localise(state, self.pipeline)

        for _ in range(self.max_attempts):
            # Generate patch
            state = node_generate_patch(state, model=self.model)
            # Apply and test
            state = node_apply_and_test(state, self.sandbox)
            # Check outcome
            if should_retry(state, self.max_attempts) == "done":
                break

        return state

    def _run_with_langgraph(self, state: AgentState) -> AgentState:
        """LangGraph state machine — same logic, better observability."""
        try:
            from langgraph.graph import StateGraph, END

            pipeline = self.pipeline
            sandbox = self.sandbox
            model = self.model
            max_attempts = self.max_attempts

            graph = StateGraph(AgentState)

            graph.add_node("localise", lambda s: node_localise(s, pipeline))
            graph.add_node("generate", lambda s: node_generate_patch(s, model=model))
            graph.add_node("test",     lambda s: node_apply_and_test(s, sandbox))

            graph.set_entry_point("localise")
            graph.add_edge("localise", "generate")
            graph.add_edge("generate", "test")
            graph.add_conditional_edges(
                "test",
                lambda s: should_retry(s, max_attempts),
                {"retry": "generate", "done": END},
            )

            app = graph.compile()
            final = app.invoke(state)

            # LangGraph may return a plain dict instead of AgentState.
            # Normalise back to the dataclass so downstream code is consistent.
            if isinstance(final, dict):
                final = AgentState(**{
                    k: final[k] for k in AgentState.__dataclass_fields__
                    if k in final
                })

            return final

        except Exception as e:
            logger.warning("LangGraph failed (%s) — falling back to simple loop", e)
            return self._run_simple_loop(state)

    def _log_trajectories(self, state: AgentState) -> None:
        """Write all attempt records to the trajectory logger."""
        from agent.trajectory_logger import TrajectoryEntry

        # Handle both AgentState dataclass and plain dict (LangGraph compat)
        if isinstance(state, dict):
            attempts    = state.get("attempts", [])
            instance_id = state.get("instance_id", "")
            repo        = state.get("repo", "")
            localised   = state.get("localised_files", [])
            problem     = state.get("problem_statement", "")
        else:
            attempts    = state.attempts
            instance_id = state.instance_id
            repo        = state.repo
            localised   = state.localised_files
            problem     = state.problem_statement

        for attempt_data in attempts:
            entry = TrajectoryEntry(
                instance_id=instance_id,
                repo=repo,
                attempt=attempt_data["attempt_num"],
                patch=attempt_data["patch"],
                test_stdout=attempt_data["test_stdout"],
                fail_to_pass_results=attempt_data["fail_to_pass_results"],
                pass_to_pass_results=attempt_data["pass_to_pass_results"],
                resolved=attempt_data["resolved"],
                failure_category=attempt_data["failure_category"],
                elapsed_seconds=0.0,
                localised_files=localised,
                problem_statement=problem,
                token_cost={},
            )
            self.traj_logger.log(entry)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_file_context(file_contents: dict[str, str], max_files: int = 4) -> str:
    """Build file contents for the LLM prompt with accurate 1-based line numbers.

    Each line is shown as:
        NNN | <source line>
    so the model can generate correct @@ hunk headers in unified diffs.
    Multi-line docstrings are collapsed to a placeholder to reduce token usage.
    """
    import re

    _NUM_RE = re.compile(r'^\s*\d+\s*\|\s?', re.MULTILINE)

    def strip_existing_line_numbers(text: str) -> str:
        """Remove any pre-existing NNN | prefixes added by read_file()."""
        return _NUM_RE.sub('', text)

    def strip_docstrings(source: str) -> str:
        """Collapse multi-line docstrings to a single placeholder."""
        def replace_doc(m: re.Match) -> str:
            body = m.group(0)
            if '\n' not in body:
                return body
            indent = len(body) - len(body.lstrip())
            return ' ' * indent + '"""..."""'
        return re.sub(r'(\'\'\'|""")[\s\S]*?\1', replace_doc, source)

    def add_line_numbers(source: str) -> str:
        """Add 1-based line numbers in `NNN | code` format."""
        lines = source.splitlines()
        width = max(len(str(len(lines))), 3)
        return '\n'.join(f'{i+1:{width}d} | {line}' for i, line in enumerate(lines))

    parts = []
    for fp, content in list(file_contents.items())[:max_files]:
        # 1. Strip any NNN | prefixes that read_file() may have added
        raw = strip_existing_line_numbers(content[:6500])
        # 2. Collapse multi-line docstrings
        cleaned = strip_docstrings(raw)
        # 3. Re-number cleanly so gpt-4o knows exact line positions
        numbered = add_line_numbers(cleaned)
        parts.append(f'### {fp}\n```python\n{numbered}\n```')
    return '\n\n'.join(parts)


def _strip_code_fences(text: str) -> str:
    """Remove ```diff``` / ``` fences from LLM output."""
    import re
    text = re.sub(r"```(?:diff|patch)?\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _fix_patch_hunk_counts(patch: str) -> str:
    """
    Recompute the @@ -old,count +new,count @@ hunk headers from the actual diff lines.

    LLMs frequently produce wrong line counts in hunk headers (e.g. wrapping one long
    context line into two, causing a count mismatch). This function ignores the LLM's
    counts and recalculates them from the actual +/-/space lines in each hunk.
    """
    import re

    result_lines: list[str] = []
    i = 0
    lines = patch.split("\n")

    while i < len(lines):
        line = lines[i]

        # File headers — pass through unchanged
        if line.startswith("--- ") or line.startswith("+++ "):
            result_lines.append(line)
            i += 1
            continue

        # Hunk header
        m = re.match(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@(.*)", line)
        if m:
            old_start = int(m.group(1))
            new_start = int(m.group(2))
            suffix = m.group(3)  # e.g. " def func_name():"

            # Collect all lines belonging to this hunk
            hunk_body: list[str] = []
            i += 1
            while i < len(lines):
                l = lines[i]
                if re.match(r"^@@\s+-\d+", l) or l.startswith("--- ") or l.startswith("+++ "):
                    break
                hunk_body.append(l)
                i += 1

            # Count old and new lines
            old_count = sum(1 for l in hunk_body if not l.startswith("+"))
            new_count = sum(1 for l in hunk_body if not l.startswith("-"))

            # Rebuild header with correct counts
            if old_count == 1:
                old_part = f"-{old_start}"
            else:
                old_part = f"-{old_start},{old_count}"
            if new_count == 1:
                new_part = f"+{new_start}"
            else:
                new_part = f"+{new_start},{new_count}"

            result_lines.append(f"@@ {old_part} {new_part} @@{suffix}")
            result_lines.extend(hunk_body)
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


import tenacity

class _OllamaError(Exception):
    """Raised on Ollama API errors — not retried like rate limits."""

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=60, max=300),
    stop=tenacity.stop_after_attempt(6),
    retry=tenacity.retry_if_exception_type(RuntimeError),
    before_sleep=lambda rs: logger.warning("Rate limited. Retrying LLM call... Attempt %s", rs.attempt_number)
)
def _call_llm(
    user_prompt: str,
    client=None,
    model: str = "gpt-4o",
) -> tuple[str, dict]:
    """
    Call the configured LLM provider via httpx (Groq) or OpenAI SDK.
    Uses httpx directly for Groq to avoid SDK connection issues in HF Spaces.
    Returns (patch_text, usage_dict).
    """
    import os
    import time
    from configs.settings import settings

    # Sleep for 4.5 seconds to guarantee staying under GitHub's 15 RPM limit
    provider = settings.llm_provider.lower()
    effective_model = settings.llm_model

    # Rate-limit sleep: GitHub gpt-4o = 10 RPM → 6s; others = 4.5s
    _sleep = 6.5 if provider == "github" else 4.5
    time.sleep(_sleep)

    # ── Groq via httpx directly (most reliable in containerised envs) ──────
    if client is None and provider == "groq":
        import httpx
        api_key = (os.environ.get("GROQ_API_KEY") or settings.groq_api_key).strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it as an env var or HF Space secret.")

        logger.info("Calling Groq API: model=%s", effective_model)
        try:
            with httpx.Client(timeout=120.0) as http:
                resp = http.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": effective_model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        "max_tokens": settings.llm_max_tokens,
                        "temperature": settings.llm_temperature,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            patch_text = data["choices"][0]["message"]["content"] or ""
            usage_raw  = data.get("usage", {})
            return patch_text, {
                "prompt_tokens":     usage_raw.get("prompt_tokens", 0),
                "completion_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens":      usage_raw.get("total_tokens", 0),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RuntimeError(f"Rate limited (429): {e.response.text[:300]}") from e
            raise RuntimeError(f"Groq API error {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot reach Groq API — check network / GROQ_API_KEY: {e}") from e

    # ── GitHub Models via httpx (OpenAI-compatible, free with PAT) ─────────
    if client is None and provider == "github":
        import httpx
        api_key = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("OPENAI_API_KEY")
            or getattr(settings, "github_token", "")
            or getattr(settings, "openai_api_key", "")
            or ""
        ).strip()
        if not api_key:
            raise ValueError("GITHUB_TOKEN is not set. Add it as an env var.")

        model_name = effective_model or "gpt-4o"
        logger.info("Calling GitHub Models API: model=%s", model_name)
        try:
            with httpx.Client(timeout=120.0) as http:
                resp = http.post(
                    "https://models.inference.ai.azure.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        "max_tokens": settings.llm_max_tokens,
                        "temperature": settings.llm_temperature,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            patch_text = data["choices"][0]["message"]["content"] or ""
            usage_raw  = data.get("usage", {})
            return patch_text, {
                "prompt_tokens":     usage_raw.get("prompt_tokens", 0),
                "completion_tokens": usage_raw.get("completion_tokens", 0),
                "total_tokens":      usage_raw.get("total_tokens", 0),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RuntimeError(f"Rate limited (429): {e.response.text[:300]}") from e
            raise RuntimeError(f"GitHub Models error {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot reach GitHub Models API: {e}") from e

    # ── Google Gemini via httpx REST (no SDK needed) ───────────────────────
    if client is None and provider == "gemini":
        import httpx as _httpx

        api_key = (os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key).strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it as an HF Space secret.")

        model_name = effective_model or settings.llm_model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
            },
        }
        logger.info("Calling Gemini API: model=%s", model_name)
        try:
            with _httpx.Client(timeout=120.0) as http:
                resp = http.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            patch_text = data["candidates"][0]["content"]["parts"][0]["text"] or ""
            meta = data.get("usageMetadata", {})
            return patch_text, {
                "prompt_tokens":     meta.get("promptTokenCount", 0),
                "completion_tokens": meta.get("candidatesTokenCount", 0),
                "total_tokens":      meta.get("totalTokenCount", 0),
            }
        except _httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RuntimeError(f"Rate limited (429): {e.response.text[:300]}") from e
            raise RuntimeError(f"Gemini API error {e.response.status_code}: {e.response.text[:300]}") from e
    # ── Ollama via httpx ───────────────────────────────────────────────────
    if client is None and provider == "ollama":
        import httpx as _httpx
        base_url = settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        payload["stream"] = True
        headers = {
            "bypass-tunnel-reminder": "true",
            "User-Agent": "python-requests/2.31.0",
            "Content-Type": "application/json"
        }
        try:
            import json
            with _httpx.Client(timeout=600.0) as http:
                with http.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    text = ""
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    for line in resp.iter_lines():
                        if line.strip():
                            chunk = json.loads(line)
                            text += chunk.get("message", {}).get("content", "")
                            if chunk.get("done"):
                                usage["prompt_tokens"] = chunk.get("prompt_eval_count", 0)
                                usage["completion_tokens"] = chunk.get("eval_count", 0)
                                usage["total_tokens"] = chunk.get("eval_count", 0) + chunk.get("prompt_eval_count", 0)
                    return text, usage
        except Exception as e:
            logger.error("Ollama API error: %s: %s", type(e).__name__, e)
            raise _OllamaError(f"Ollama API error: {e}") from e

    # ── OpenAI SDK fallback ────────────────────────────────────────────────
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or None,
                base_url=settings.openai_base_url or None
            )
        except ImportError as e:
            raise ImportError(
                "No LLM client available. Set LLM_PROVIDER=groq + GROQ_API_KEY, "
                "or install openai: pip install openai"
            ) from e

    response = client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    patch_text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
    return patch_text, usage


def _parse_local_test_results(test_stdout: str, test_ids: list[str]) -> dict[str, bool]:
    """Parse local pytest output to get pass/fail per test ID."""
    import re
    passed = set(re.findall(r"^(.+?::[\w\[\]-]+)\s+PASSED", test_stdout, re.MULTILINE))
    return {tid: tid in passed for tid in test_ids}

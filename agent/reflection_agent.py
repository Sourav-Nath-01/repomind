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

Rules:
- Output ONLY the unified diff. No explanations, no markdown code fences.
- Start with '--- a/<file>' and use proper unified diff format.
- Be minimal: only change what is necessary to fix the bug.
- If multiple files need changes, include all in one diff.
- Do not remove or modify unrelated code.
- Ensure your Python syntax is valid.
"""

INITIAL_PROMPT_TEMPLATE = """\
## GitHub Issue
{problem_statement}

## Relevant Files
{file_context}

Generate a unified diff patch that fixes this issue.
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

    # Read file contents from workspace
    from agent.tools import AgentTools
    tools = AgentTools(state.workspace_dir)
    for fp in state.localised_files:
        read_result = tools.read_file(fp, max_lines=150)
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
    state.last_patch = _strip_code_fences(patch_text)
    state.total_tokens += usage.get("total_tokens", 0)
    return state


def node_apply_and_test(state: AgentState, sandbox=None) -> AgentState:
    """
    Node: apply the patch and run tests.
    Populates state.resolved and state.last_test_stdout.
    """
    from agent.tools import AgentTools
    tools = AgentTools(state.workspace_dir, sandbox)

    # Write and apply patch
    write_result = tools.write_patch(state.last_patch)
    patch_apply_success = False

    if write_result.success:
        if sandbox:
            from sandbox.executor import SandboxExecutor
            apply_result = sandbox.apply_patch(state.last_patch, state.workspace_dir)
            patch_apply_success = apply_result.success
        else:
            import subprocess
            try:
                proc = subprocess.run(
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

def _build_file_context(file_contents: dict[str, str], max_files: int = 5) -> str:
    """Build a formatted string of file contents for the LLM prompt."""
    parts = []
    for fp, content in list(file_contents.items())[:max_files]:
        parts.append(f"### {fp}\n```python\n{content[:1500]}\n```")
    return "\n\n".join(parts)


def _strip_code_fences(text: str) -> str:
    """Remove ```diff``` / ``` fences from LLM output."""
    import re
    text = re.sub(r"```(?:diff|patch)?\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


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
    from configs.settings import settings

    provider = (os.environ.get("LLM_PROVIDER") or settings.llm_provider).lower()
    effective_model = os.environ.get("LLM_MODEL") or settings.llm_model

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
            raise RuntimeError(f"Groq API error {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.ConnectError as e:
            raise RuntimeError(f"Cannot reach Groq API — check network / GROQ_API_KEY: {e}") from e

    # ── Google Gemini via httpx REST (no SDK needed) ───────────────────────
    if client is None and provider == "gemini":
        import httpx as _httpx

        api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it as an HF Space secret.")

        model_name = effective_model or "gemini-2.0-flash"
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
            raise RuntimeError(f"Gemini API error {e.response.status_code}: {e.response.text[:300]}") from e

    # ── OpenAI SDK fallback ────────────────────────────────────────────────
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or None)
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

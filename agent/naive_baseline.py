"""
agent/naive_baseline.py
───────────────────────
Phase 1 Naive Baseline:
  Issue text → GPT-4o (single-shot) → unified diff → apply → run tests

This establishes the baseline % resolved we need to beat in later phases.
Expected performance: ~10–18% on SWE-bench Lite.

The agent:
  1. Loads the issue text and top-level file listing of the repo
  2. Sends a single prompt to GPT-4o asking for a unified diff patch
  3. Applies the patch via git apply
  4. Runs fail_to_pass + pass_to_pass tests
  5. Logs attempt result to MLflow
"""
from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Python software engineer. Your task is to fix a bug in a Python repository.

You will be given:
1. The GitHub issue describing the bug
2. A list of files in the repository

Your response MUST be a valid unified diff (git diff format) that:
- Fixes the described bug
- Is minimal — only change what is necessary
- Uses correct Python syntax
- Does not introduce new bugs

Output ONLY the unified diff. Start with '---' and end with the diff.
Do not include any explanation, markdown code blocks, or other text.
"""

USER_PROMPT_TEMPLATE = """\
## GitHub Issue

{problem_statement}

## Repository: {repo}
Commit: {base_commit}

## Repository File Structure (top-level)
{file_listing}

Generate a unified diff patch to fix this issue.
"""


class NaiveBaselineAgent:
    """
    Single-shot GPT-4o baseline agent.
    No retrieval, no reflection — just raw issue text → patch.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError as e:
                raise ImportError("Install openai: pip install openai") from e
        return self._client

    def generate_patch(
        self,
        problem_statement: str,
        repo: str,
        base_commit: str,
        workspace_dir: Path | None = None,
    ) -> tuple[str, dict]:
        """
        Generate a patch for the given issue.

        Returns:
            patch_text: unified diff string
            usage: token usage dict {prompt_tokens, completion_tokens, total_tokens}
        """
        file_listing = self._get_file_listing(workspace_dir) if workspace_dir else "(unavailable)"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement[:3000],  # truncate to stay under budget
            repo=repo,
            base_commit=base_commit[:12],
            file_listing=file_listing,
        )

        logger.info("Calling %s for patch generation...", self.model)
        start = time.monotonic()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        elapsed = time.monotonic() - start
        patch_text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.info(
            "Patch generated in %.1fs | tokens: %d prompt + %d completion",
            elapsed, usage["prompt_tokens"], usage["completion_tokens"]
        )

        # Clean up patch text — remove markdown code fences if present
        patch_text = _strip_code_fences(patch_text)
        return patch_text, usage

    @staticmethod
    def _get_file_listing(workspace_dir: Path, max_files: int = 100) -> str:
        """Get a truncated file listing for context."""
        try:
            files = sorted(
                p.relative_to(workspace_dir)
                for p in workspace_dir.rglob("*.py")
                if not any(part.startswith(".") for part in p.parts)
                and "__pycache__" not in str(p)
            )
            listing = "\n".join(str(f) for f in files[:max_files])
            if len(files) > max_files:
                listing += f"\n... and {len(files) - max_files} more files"
            return listing
        except Exception:
            return "(could not list files)"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    # Remove ```diff ... ``` or ``` ... ```
    text = re.sub(r"```(?:diff|patch)?\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ── MLflow helpers ────────────────────────────────────────────────────────────

def log_baseline_attempt(
    instance_id: str,
    resolved: bool,
    usage: dict,
    elapsed: float,
    failure_category: str = "unknown",
    attempt: int = 1,
) -> None:
    """Log a single attempt to MLflow."""
    import mlflow  # lazy import — not needed in tests without mlflow
    with mlflow.start_run(run_name=f"{instance_id}_attempt_{attempt}", nested=True):

        mlflow.log_params({
            "instance_id": instance_id,
            "attempt": attempt,
            "failure_category": failure_category,
        })
        mlflow.log_metrics({
            "resolved": int(resolved),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "elapsed_seconds": elapsed,
        })

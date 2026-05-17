"""
api/models.py
──────────────
Pydantic request/response models for the FastAPI backend.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


class SolveRequest(BaseModel):
    repo: str = Field(..., description="GitHub repo in 'owner/repo' format")
    issue_url: str = Field("", description="GitHub issue URL (optional)")
    problem_statement: str = Field(..., description="Issue description text")
    instance_id: str = Field("", description="SWE-bench instance ID (optional)")
    base_commit: str = Field("", description="Git commit SHA to checkout")
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    max_attempts: int = Field(3, ge=1, le=5)
    top_k_files: int = Field(5, ge=1, le=20)


class SolveResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "done", "error"]
    message: str = ""


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["queued", "running", "done", "error"]
    resolved: bool = False
    attempts: int = 0
    localised_files: list[str] = Field(default_factory=list)
    patch: str = ""
    failure_category: str = ""
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


# ── WebSocket event types ─────────────────────────────────────────────────────

class WSEvent(BaseModel):
    """Streaming event sent over WebSocket."""
    event: Literal[
        "status",           # overall task status
        "log",              # log message
        "localised_files",  # files retrieved
        "patch",            # generated patch
        "test_result",      # pytest result
        "reflection",       # retry with reflection context
        "done",             # final result
        "error",            # fatal error
    ]
    data: dict = Field(default_factory=dict)
    timestamp: str = ""

    def to_json(self) -> str:
        import json
        return json.dumps(self.model_dump())


class MetricsSnapshot(BaseModel):
    """Live metrics for the dashboard."""
    total_issues_solved: int = 0
    avg_elapsed_seconds: float = 0.0
    avg_attempts: float = 0.0
    recall_at_5: float = 0.0
    total_token_cost: int = 0
    avg_token_cost_per_issue: float = 0.0
    failure_category_counts: dict[str, int] = Field(default_factory=dict)

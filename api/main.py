"""
api/main.py
────────────
FastAPI application — REST + WebSocket API for the Code Review Agent.

Endpoints:
  POST /api/solve          — submit a new solve request → returns task_id
  GET  /api/task/{task_id} — get task status + results
  WS   /ws/{task_id}       — stream execution events in real time
  GET  /api/metrics        — live metrics for the dashboard
  GET  /api/health         — health check

WebSocket event stream format:
  {"event": "log",             "data": {"step": 2, "message": "Cloning..."}}
  {"event": "localised_files", "data": {"files": [...], "graph_nodes": 450}}
  {"event": "patch",           "data": {"attempt": 1, "patch": "--- a/..."}}
  {"event": "test_result",     "data": {"resolved": false, "failure_category": "..."}}
  {"event": "reflection",      "data": {"attempt": 2, "message": "Retrying..."}}
  {"event": "done",            "data": {"resolved": true, "attempts": 2, ...}}
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    MetricsSnapshot,
    SolveRequest,
    SolveResponse,
    TaskStatus,
)
from api.tasks import create_task_id, get_task_status, run_agent_task_async, update_task_status
from api.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Code Review Agent API starting up...")
    yield
    logger.info("Code Review Agent API shutting down...")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Code Review & Bug-Fix Agent",
    description=(
        "API for the autonomous code review agent. "
        "Submit a GitHub issue + repo, stream agent execution, get a patch."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


@app.post("/api/solve", response_model=SolveResponse)
async def solve(request: SolveRequest, background_tasks=None):
    """
    Submit a bug-fix request. Returns a task_id immediately.
    Connect to /ws/{task_id} to stream execution progress.
    """
    task_id = create_task_id()
    update_task_status(task_id, status="queued",
                       repo=request.repo,
                       created_at=datetime.now(timezone.utc).isoformat())

    # Store request for the WS handler to pick up
    update_task_status(task_id, request_data=request.model_dump())

    logger.info("Task created: %s | repo=%s", task_id, request.repo)
    return SolveResponse(task_id=task_id, status="queued",
                         message=f"Task queued. Connect to /ws/{task_id}")


@app.get("/api/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    """Poll task status (alternative to WebSocket streaming)."""
    status = get_task_status(task_id)
    if status.get("status") == "unknown":
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskStatus(
        task_id=task_id,
        status=status.get("status", "unknown"),
        resolved=status.get("resolved", False),
        attempts=status.get("attempts", 0),
        localised_files=status.get("localised_files", []),
        patch=status.get("patch", ""),
        failure_category=status.get("failure_category", ""),
        total_tokens=status.get("total_tokens", 0),
        elapsed_seconds=status.get("elapsed_seconds", 0.0),
        error=status.get("error", ""),
    )


@app.get("/api/metrics", response_model=MetricsSnapshot)
async def get_metrics():
    """Aggregate metrics for the live dashboard."""
    from pathlib import Path
    from agent.trajectory_logger import TrajectoryLogger

    traj_dir = Path("results/trajectories")
    if not traj_dir.exists():
        return MetricsSnapshot()

    all_entries = []
    for jsonl_file in traj_dir.glob("*.jsonl"):
        tl = TrajectoryLogger(jsonl_file)
        all_entries.extend(tl.load_all())

    if not all_entries:
        return MetricsSnapshot()

    resolved = [e for e in all_entries if e.resolved]
    categories: dict[str, int] = {}
    for e in all_entries:
        categories[e.failure_category] = categories.get(e.failure_category, 0) + 1

    return MetricsSnapshot(
        total_issues_solved=len(resolved),
        avg_elapsed_seconds=sum(e.elapsed_seconds for e in all_entries) / len(all_entries),
        avg_attempts=sum(e.attempt for e in all_entries) / len(all_entries),
        total_token_cost=sum(e.token_cost.get("total_tokens", 0) for e in all_entries),
        avg_token_cost_per_issue=(
            sum(e.token_cost.get("total_tokens", 0) for e in all_entries) / len(all_entries)
        ),
        failure_category_counts=categories,
    )


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    Stream real-time execution events for task_id.

    Event flow:
      Client connects → server starts agent task → events streamed → connection closes
    """
    await ws_manager.connect(task_id, websocket)

    try:
        # Retrieve queued request
        task_info = get_task_status(task_id)
        if task_info.get("status") == "unknown":
            await websocket.send_text('{"event":"error","data":{"message":"Task not found"}}')
            return

        request_data = task_info.get("request_data", {})
        if not request_data:
            await websocket.send_text('{"event":"error","data":{"message":"No request data"}}')
            return

        # Define streaming emitter
        async def emit(event_type: str, data: dict):
            await ws_manager.emit(task_id, event_type, data)

        # Run agent pipeline (async, streaming events)
        await run_agent_task_async(task_id, request_data, emit)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: task=%s", task_id)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_text(
                f'{{"event":"error","data":{{"message":"{str(e)[:200]}"}}}}'
            )
        except Exception:
            pass
    finally:
        ws_manager.disconnect(task_id, websocket)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from configs.settings import settings
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )

"""
api/tasks.py
─────────────
Celery tasks for async agent execution.

Each /solve request spawns a Celery task that:
  1. Clones the repo (or uses cache)
  2. Parses AST + builds dependency graph (or cache hit)
  3. Runs localisation pipeline
  4. Runs reflection agent (up to max_attempts)
  5. Publishes streaming events to Redis → WebSocket

The Celery task publishes structured events during execution so the
frontend gets real-time updates without polling.

Event stream:
  [1/5] status: "Cloning repository..."
  [2/5] localised_files: ["django/db/models/query.py", ...]
  [3/5] patch: "<unified diff>"
  [4/5] test_result: {passed: [...], failed: [...]}
  [5/5] done: {resolved: true, attempts: 2, ...}
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def get_celery_app():
    """Lazy-init Celery to avoid import errors when broker is unavailable."""
    try:
        from celery import Celery
        from configs.settings import settings
        app = Celery(
            "code_agent",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend if hasattr(settings, "celery_result_backend") else settings.redis_url,
        )
        app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
        )
        return app
    except Exception as e:
        logger.warning("Celery not available: %s", e)
        return None


# In-memory task store (dev fallback when Celery/Redis not running)
_task_store: dict[str, dict] = {}


def create_task_id() -> str:
    return str(uuid.uuid4())


def get_task_status(task_id: str) -> dict:
    """Get task status from Redis or in-memory store."""
    status = _task_store.get(task_id, {"status": "unknown", "task_id": task_id})
    return status


def update_task_status(task_id: str, **kwargs) -> None:
    """Update task status in the in-memory store."""
    if task_id not in _task_store:
        _task_store[task_id] = {"task_id": task_id, "status": "queued"}
    _task_store[task_id].update(kwargs)


async def run_agent_task_async(
    task_id: str,
    request_data: dict,
    emit_fn,   # async callable(event_type: str, data: dict)
) -> dict:
    """
    Run the full agent pipeline asynchronously with streaming events.
    Used directly by FastAPI when Celery is unavailable (dev mode).

    Args:
        task_id:      unique task identifier
        request_data: SolveRequest dict
        emit_fn:      async callable to push events to WebSocket

    Returns:
        Final result dict
    """
    import asyncio
    import tempfile

    start = time.monotonic()
    update_task_status(task_id, status="running")

    try:
        # ── Step 1: Setup ─────────────────────────────────────────────────
        await emit_fn("log", {"step": 1, "total": 5, "message": "Setting up workspace..."})
        await emit_fn("status", {"status": "running", "step": "setup"})

        repo = request_data["repo"]
        problem_statement = request_data["problem_statement"]
        base_commit = request_data.get("base_commit") or "HEAD"
        fail_to_pass = request_data.get("fail_to_pass", [])
        pass_to_pass = request_data.get("pass_to_pass", [])
        max_attempts = request_data.get("max_attempts", 3)
        top_k_files = request_data.get("top_k_files", 5)

        # ── Step 2: Clone & Parse ─────────────────────────────────────────
        await emit_fn("log", {"step": 2, "total": 5, "message": f"Cloning {repo}..."})

        workspace_dir = Path(tempfile.mkdtemp(prefix=f"agent_{task_id[:8]}_"))

        from sandbox.executor import SandboxExecutor
        sandbox = SandboxExecutor(use_docker=False)
        clone_result = sandbox.clone_repo(repo, base_commit, workspace_dir)

        if not clone_result.success:
            await emit_fn("error", {"message": f"Clone failed: {clone_result.stderr[:200]}"})
            update_task_status(task_id, status="error", error="clone_failed")
            return {"status": "error", "error": "clone_failed"}

        # ── Step 3: AST Parse + Localise ──────────────────────────────────
        await emit_fn("log", {"step": 3, "total": 5, "message": "Parsing AST & building dependency graph..."})

        from ast_parser.cache import ASTCache
        from configs.settings import settings
        cache = ASTCache(settings.diskcache_dir)
        repo_key = f"{repo.replace('/', '__')}_{base_commit[:8]}"
        symbols, graph = cache.get_or_parse_repo(workspace_dir, repo_key)

        await emit_fn("log", {
            "step": 3, "total": 5,
            "message": f"Parsed {len(symbols)} files, {graph.graph.number_of_nodes()} graph nodes"
        })

        from localisation.pipeline import LocalisationPipeline
        pipeline = LocalisationPipeline(
            use_embeddings=False,   # skip OpenAI embeddings for speed in demo
            use_colbert=False,
            use_ppr=True,
        )
        pipeline.index_repo(symbols, graph)
        loc_result = pipeline.localise(problem_statement, top_k=top_k_files)
        localised_files = loc_result.top_k_paths

        await emit_fn("localised_files", {
            "files": localised_files,
            "graph_nodes": graph.graph.number_of_nodes(),
            "graph_edges": graph.graph.number_of_edges(),
            "recall_at_5": loc_result.recall_at_5,
        })

        # ── Step 4: Reflection Agent ──────────────────────────────────────
        await emit_fn("log", {"step": 4, "total": 5, "message": "Generating patch..."})

        from agent.trajectory_logger import TrajectoryLogger
        traj_path = Path(f"results/trajectories/{task_id}.jsonl")
        traj_logger = TrajectoryLogger(traj_path)

        from configs.settings import settings
        from agent.reflection_agent import ReflectionAgent
        agent = ReflectionAgent(
            model=settings.llm_model,   # reads LLM_MODEL from env (e.g. deepseek-r1-distill-llama-70b)
            max_attempts=max_attempts,
            sandbox=sandbox,
            trajectory_logger=traj_logger,
        )

        # Wrap agent to emit events during execution (monkey-patch for streaming)
        original_generate = agent._run_simple_loop

        async def streaming_run(state):
            # Can't make _run_simple_loop truly async here without refactor
            # Run in thread pool to avoid blocking event loop
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result_state = await loop.run_in_executor(pool, original_generate, state)
            return result_state

        # Emit progress after each attempt
        agent_state = agent.run(
            instance_id=request_data.get("instance_id", task_id),
            repo=repo,
            problem_statement=problem_statement,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            workspace_dir=workspace_dir,
            localised_files=localised_files,
        )

        # Emit attempt results
        for attempt_data in agent_state.attempts:
            if attempt_data["attempt_num"] > 1:
                await emit_fn("reflection", {
                    "attempt": attempt_data["attempt_num"],
                    "failure_category": attempt_data.get("failure_category", "unknown"),
                    "message": f"Attempt {attempt_data['attempt_num']}: reflecting on failure...",
                })
            await emit_fn("patch", {
                "attempt": attempt_data["attempt_num"],
                "patch": attempt_data["patch"][:3000],
                "resolved": attempt_data["resolved"],
            })
            await emit_fn("test_result", {
                "attempt": attempt_data["attempt_num"],
                "resolved": attempt_data["resolved"],
                "failure_category": attempt_data.get("failure_category", "unknown"),
                "fail_to_pass_results": attempt_data.get("fail_to_pass_results", {}),
            })

        # ── Step 5: Done ──────────────────────────────────────────────────
        elapsed = time.monotonic() - start
        result = {
            "task_id": task_id,
            "status": "done",
            "resolved": agent_state.resolved,
            "attempts": agent_state.current_attempt,
            "localised_files": localised_files,
            "patch": agent_state.last_patch,
            "failure_category": agent_state.last_failure_category,
            "total_tokens": agent_state.total_tokens,
            "elapsed_seconds": round(elapsed, 2),
        }

        update_task_status(task_id, **{k: v for k, v in result.items() if k != "task_id"})
        await emit_fn("done", result)
        await emit_fn("log", {
            "step": 5, "total": 5,
            "message": f"{'✅ Resolved!' if agent_state.resolved else '❌ Not resolved'} "
                       f"({agent_state.current_attempt} attempt(s), {elapsed:.1f}s)"
        })

        return result

    except Exception as e:
        logger.exception("Agent task failed: %s", e)
        await emit_fn("error", {"message": str(e)[:300]})
        update_task_status(task_id, status="error", error=str(e)[:200])
        return {"status": "error", "error": str(e)}

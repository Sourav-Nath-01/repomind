"""
telemetry/rate_limiter.py
──────────────────────────
Rate limiting + request queue depth monitoring for the FastAPI API.

Uses slowapi (Starlette-compatible limiter) with Redis backend.
Falls back to in-memory storage when Redis is unavailable.

Limits:
  POST /api/solve:        5 requests/minute per IP
  GET  /api/task/{id}:   60 requests/minute per IP
  WS   /ws/{id}:         10 connections/minute per IP
  GET  /api/metrics:      30 requests/minute per IP

Why rate limiting?
  - GPT-4o costs $5/1M tokens; unconstrained API = runaway costs
  - Sandbox execution uses CPU + memory; need to bound concurrency
  - Public demo must be resilient to abuse/crawlers
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


# ── Sliding window rate limiter (in-memory fallback) ──────────────────────────

class SlidingWindowRateLimiter:
    """
    Token-bucket / sliding window rate limiter.
    Thread-safe for single-process deployments.
    For multi-process, use Redis-backed slowapi instead.
    """

    def __init__(self, requests: int, window_seconds: int):
        self.limit = requests
        self.window = window_seconds
        self._buckets: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.monotonic()
        bucket = self._buckets[key]

        # Remove expired timestamps
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self.limit:
            return False

        bucket.append(now)
        return True

    def remaining(self, key: str) -> int:
        """Return how many requests remain in the current window."""
        now = time.monotonic()
        bucket = self._buckets[key]
        cutoff = now - self.window
        active = sum(1 for t in bucket if t > cutoff)
        return max(0, self.limit - active)

    def reset_for(self, key: str) -> None:
        """Clear rate limit for a key (admin use)."""
        self._buckets.pop(key, None)

    def stats(self) -> dict:
        return {
            "limit": self.limit,
            "window_seconds": self.window,
            "tracked_keys": len(self._buckets),
        }


# ── Shared limiters ───────────────────────────────────────────────────────────

# In-memory fallback limiters (used when Redis/slowapi not available)
SOLVE_LIMITER   = SlidingWindowRateLimiter(requests=5,  window_seconds=60)
QUERY_LIMITER   = SlidingWindowRateLimiter(requests=60, window_seconds=60)
WS_LIMITER      = SlidingWindowRateLimiter(requests=10, window_seconds=60)
METRICS_LIMITER = SlidingWindowRateLimiter(requests=30, window_seconds=60)


# ── SlowAPI integration helper ─────────────────────────────────────────────────

def setup_slowapi(app, redis_url: str = "redis://localhost:6379/2") -> Optional[object]:
    """
    Attach slowapi rate limiter to a FastAPI app.
    Returns the limiter instance, or None if slowapi is unavailable.
    """
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded

        storage_uri = redis_url if redis_url else "memory://"
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"],
            storage_uri=storage_uri,
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("SlowAPI rate limiter attached (storage: %s)", storage_uri)
        return limiter

    except ImportError:
        logger.debug("slowapi not installed — using in-memory rate limiter")
        return None


# ── Queue depth monitor ───────────────────────────────────────────────────────

class QueueDepthMonitor:
    """
    Tracks running and queued task counts.
    Exposes metrics for the Grafana dashboard.
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._running: int = 0
        self._queued: int = 0
        self._completed: int = 0
        self._rejected: int = 0

    def task_queued(self) -> bool:
        """Returns True if task was accepted, False if queue is full."""
        if self._running >= self.max_concurrent:
            self._rejected += 1
            return False
        self._queued += 1
        return True

    def task_started(self) -> None:
        self._queued = max(0, self._queued - 1)
        self._running += 1

    def task_finished(self) -> None:
        self._running = max(0, self._running - 1)
        self._completed += 1

    @property
    def is_at_capacity(self) -> bool:
        return self._running >= self.max_concurrent

    def snapshot(self) -> dict:
        return {
            "running": self._running,
            "queued": self._queued,
            "completed": self._completed,
            "rejected": self._rejected,
            "capacity": self.max_concurrent,
            "utilisation_pct": round(self._running / self.max_concurrent * 100, 1),
        }


# Singleton
QUEUE_MONITOR = QueueDepthMonitor(max_concurrent=5)

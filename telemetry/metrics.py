"""
telemetry/metrics.py
─────────────────────
Prometheus metrics for the Code Review Agent API.

Metrics tracked:
  - code_agent_requests_total          Counter: API requests by endpoint + status
  - code_agent_latency_seconds         Histogram: end-to-end latency per phase
  - code_agent_token_cost_total        Counter: OpenAI tokens consumed
  - code_agent_resolved_total          Counter: issues resolved vs failed
  - code_agent_attempts_histogram      Histogram: attempts per resolved issue
  - code_agent_localisation_recall     Gauge: rolling recall@5 average
  - code_agent_cache_hits_total        Counter: AST + embedding cache hits/misses
  - code_agent_active_tasks            Gauge: currently running tasks
  - code_agent_failure_category_total  Counter: failure categories breakdown

Usage:
    from telemetry.metrics import METRICS
    METRICS.record_request("solve", 200, elapsed=12.3)
    METRICS.record_token_cost(prompt_tokens=800, completion_tokens=200)
    METRICS.record_resolution(resolved=True, attempts=2)
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ── Prometheus (graceful no-op if not installed) ──────────────────────────────

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.debug("prometheus_client not installed — metrics disabled")


class _NoOpMetric:
    """Fallback metric that silently ignores all calls."""
    def labels(self, **kwargs): return self
    def inc(self, n=1): pass
    def dec(self, n=1): pass
    def set(self, v): pass
    def observe(self, v): pass


def _make_counter(name, doc, labels=()):
    if _PROM_AVAILABLE:
        return Counter(name, doc, labels)
    return _NoOpMetric()


def _make_histogram(name, doc, labels=(), buckets=None):
    if _PROM_AVAILABLE:
        kwargs = {"labelnames": labels}
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, doc, **kwargs)
    return _NoOpMetric()


def _make_gauge(name, doc, labels=()):
    if _PROM_AVAILABLE:
        return Gauge(name, doc, labels)
    return _NoOpMetric()


# ── Metric definitions ─────────────────────────────────────────────────────────

_requests_total = _make_counter(
    "code_agent_requests_total",
    "Total API requests", ["endpoint", "status"]
)

_latency_seconds = _make_histogram(
    "code_agent_latency_seconds",
    "Request latency in seconds", ["phase"],
    buckets=[1, 5, 15, 30, 60, 120, 300]
)

_token_cost_total = _make_counter(
    "code_agent_token_cost_total",
    "Total OpenAI tokens consumed", ["token_type"]
)

_resolved_total = _make_counter(
    "code_agent_resolved_total",
    "Issues resolved vs failed", ["outcome"]
)

_attempts_histogram = _make_histogram(
    "code_agent_attempts_histogram",
    "Attempts per issue", [],
    buckets=[1, 2, 3, 4, 5]
)

_localisation_recall = _make_gauge(
    "code_agent_localisation_recall",
    "Rolling recall@5 average", ["k"]
)

_cache_hits_total = _make_counter(
    "code_agent_cache_hits_total",
    "Cache hits and misses", ["cache_type", "result"]
)

_active_tasks = _make_gauge(
    "code_agent_active_tasks",
    "Currently running agent tasks", []
)

_failure_category_total = _make_counter(
    "code_agent_failure_category_total",
    "Failure categories", ["category"]
)


# ── High-level metrics interface ───────────────────────────────────────────────

class AgentMetrics:
    """
    High-level metrics interface — wraps raw Prometheus metrics with
    domain-friendly methods. Can be used as a context manager for timing.
    """

    def record_request(self, endpoint: str, status_code: int, elapsed: float) -> None:
        status = "2xx" if 200 <= status_code < 300 else f"{status_code // 100}xx"
        _requests_total.labels(endpoint=endpoint, status=status).inc()
        _latency_seconds.labels(phase="request").observe(elapsed)

    def record_phase_latency(self, phase: str, elapsed: float) -> None:
        """Record latency for a specific pipeline phase."""
        _latency_seconds.labels(phase=phase).observe(elapsed)

    def record_token_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        _token_cost_total.labels(token_type="prompt").inc(prompt_tokens)
        _token_cost_total.labels(token_type="completion").inc(completion_tokens)

    def record_resolution(self, resolved: bool, attempts: int) -> None:
        outcome = "resolved" if resolved else "failed"
        _resolved_total.labels(outcome=outcome).inc()
        _attempts_histogram.observe(attempts)

    def record_localisation_recall(self, recall_at_5: float, recall_at_10: float) -> None:
        _localisation_recall.labels(k="5").set(recall_at_5)
        _localisation_recall.labels(k="10").set(recall_at_10)

    def record_cache_hit(self, cache_type: Literal["ast", "embedding", "repo"], hit: bool) -> None:
        result = "hit" if hit else "miss"
        _cache_hits_total.labels(cache_type=cache_type, result=result).inc()

    def record_failure_category(self, category: str) -> None:
        _failure_category_total.labels(category=category).inc()

    def task_started(self) -> None:
        _active_tasks.inc()

    def task_finished(self) -> None:
        _active_tasks.dec()

    @contextmanager
    def time_phase(self, phase: str):
        """Context manager: time a pipeline phase."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.record_phase_latency(phase, time.monotonic() - start)

    def prometheus_output(self) -> tuple[bytes, str]:
        """Return (metrics_bytes, content_type) for the /metrics endpoint."""
        if _PROM_AVAILABLE:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest(), CONTENT_TYPE_LATEST
        return b"# prometheus_client not installed\n", "text/plain"


# Singleton
METRICS = AgentMetrics()


# ── Cost tracker ───────────────────────────────────────────────────────────────

@dataclass
class CostTracker:
    """
    Per-issue cost tracker.
    Estimates USD cost from token usage.

    Pricing (May 2025 approximate):
        GPT-4o:            $5.00/M input, $15.00/M output
        text-embedding-3s: $0.02/M tokens
        DeepSeek-7B:       ~$0.14/M tokens (self-hosted on RunPod)
    """
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _embedding_tokens: int = 0

    # USD per 1M tokens
    PROMPT_COST_PER_M: float = 5.00
    COMPLETION_COST_PER_M: float = 15.00
    EMBEDDING_COST_PER_M: float = 0.02

    def add_llm_tokens(self, prompt: int, completion: int) -> None:
        self._prompt_tokens += prompt
        self._completion_tokens += completion

    def add_embedding_tokens(self, n: int) -> None:
        self._embedding_tokens += n

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens + self._embedding_tokens

    @property
    def estimated_usd(self) -> float:
        prompt_cost  = self._prompt_tokens    / 1e6 * self.PROMPT_COST_PER_M
        comp_cost    = self._completion_tokens / 1e6 * self.COMPLETION_COST_PER_M
        embed_cost   = self._embedding_tokens  / 1e6 * self.EMBEDDING_COST_PER_M
        return round(prompt_cost + comp_cost + embed_cost, 6)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "embedding_tokens": self._embedding_tokens,
            "total_tokens": self.total_tokens,
            "estimated_usd": self.estimated_usd,
        }

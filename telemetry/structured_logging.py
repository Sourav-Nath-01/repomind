"""
telemetry/structured_logging.py
─────────────────────────────────
Structured JSON logging via structlog.

Every log event emitted through this module includes:
  - timestamp (ISO 8601 UTC)
  - level
  - logger name
  - event message
  - structured key-value context

Usage:
    from telemetry.structured_logging import get_logger
    log = get_logger(__name__)
    log.info("task_started", task_id="abc123", repo="django/django")
    log.error("patch_failed", failure_category="syntax_error", attempt=2)

The structured format makes logs queryable in tools like:
  - CloudWatch Logs Insights: fields @timestamp, @message | filter level="ERROR"
  - Grafana Loki: {app="code-agent"} | json | failure_category="syntax_error"
  - PostHog: track custom events from log stream

Fallback: if structlog is not installed, returns a standard logging.Logger
with a JSON formatter.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_caller_info: bool = False,
) -> None:
    """
    Configure structured logging for the application.
    Call once at startup (e.g. in FastAPI lifespan or main()).
    """
    if _STRUCTLOG_AVAILABLE:
        _configure_structlog(level, json_output, include_caller_info)
    else:
        _configure_stdlib(level, json_output)


def _configure_structlog(level: str, json_output: bool, caller_info: bool) -> None:
    import structlog

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if caller_info:
        processors.append(structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.LINENO]
        ))
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def _configure_stdlib(level: str, json_output: bool) -> None:
    """Fallback when structlog is not available."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "event": record.getMessage(),
            }
            if hasattr(record, "extra"):
                data.update(record.extra)
            return json.dumps(data)

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger for the given name.
    Returns a structlog BoundLogger if available, else stdlib Logger.
    """
    if _STRUCTLOG_AVAILABLE:
        import structlog
        return structlog.get_logger(name)
    return logging.getLogger(name)


# ── Request context binder ─────────────────────────────────────────────────────

class RequestContext:
    """
    Bind per-request context to all log lines within a request/task.

    Usage:
        with RequestContext(task_id="abc", repo="django/django"):
            log.info("processing")   # automatically includes task_id, repo
    """
    def __init__(self, **kwargs):
        self._ctx = kwargs

    def __enter__(self):
        if _STRUCTLOG_AVAILABLE:
            import structlog
            structlog.contextvars.bind_contextvars(**self._ctx)
        return self

    def __exit__(self, *args):
        if _STRUCTLOG_AVAILABLE:
            import structlog
            structlog.contextvars.unbind_contextvars(*self._ctx.keys())

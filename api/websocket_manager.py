"""
api/websocket_manager.py
──────────────────────────
WebSocket connection manager for streaming execution logs.

Each task_id has a list of connected WebSocket clients.
When the Celery worker emits an event, it's broadcast to all
connected clients watching that task.

Pattern: pub/sub via Redis — worker publishes to Redis channel,
FastAPI subscribes and forwards to WebSocket clients.
Fallback: in-memory queue (single-process mode for development).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from fastapi import WebSocket

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages active WebSocket connections per task_id.

    Usage:
        manager = WebSocketManager()
        
        # In WebSocket endpoint:
        await manager.connect(task_id, websocket)
        
        # In Celery task (via Redis pub/sub):
        await manager.broadcast(task_id, event_dict)
    """

    def __init__(self):
        # task_id → list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        # task_id → event queue (for in-memory fallback)
        self._queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[task_id].append(websocket)
        logger.info("WS connected: task=%s | total=%d",
                    task_id, len(self._connections[task_id]))

    def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(task_id, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.info("WS disconnected: task=%s | remaining=%d", task_id, len(conns))

    async def broadcast(self, task_id: str, event: dict) -> None:
        """Send an event to all WebSocket clients watching task_id."""
        message = json.dumps(event)
        dead = []
        for ws in self._connections.get(task_id, []):
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.debug("WS send failed: %s", e)
                dead.append(ws)
        for ws in dead:
            self.disconnect(task_id, ws)

    async def emit(self, task_id: str, event_type: str, data: dict) -> None:
        """Convenience: wrap data in event envelope and broadcast."""
        from datetime import datetime, timezone
        event = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.broadcast(task_id, event)

    def enqueue(self, task_id: str, event: dict) -> None:
        """
        Non-async version for Celery workers.
        Events are stored in an asyncio.Queue and drained by the WS listener.
        """
        try:
            self._queues[task_id].put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full for task %s — dropping event", task_id)

    async def drain_queue(self, task_id: str, websocket: WebSocket) -> None:
        """
        Drain events from the in-memory queue and forward to WebSocket.
        Called by the WebSocket endpoint's receive loop.
        """
        queue = self._queues[task_id]
        while True:
            try:
                event = queue.get_nowait()
                await websocket.send_text(json.dumps(event))
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)
            except Exception:
                break

    def active_tasks(self) -> list[str]:
        return [tid for tid, conns in self._connections.items() if conns]


# Singleton used across the app
ws_manager = WebSocketManager()

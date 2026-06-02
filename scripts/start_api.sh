#!/usr/bin/env bash
# scripts/start_api.sh
# ─────────────────────
# Start the FastAPI development server

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Code Review Agent API..."
echo "   Docs: http://localhost:8000/docs"
echo "   WS:   ws://localhost:8000/ws/{task_id}"
echo ""

# Install dependencies if not in venv
if [ ! -d ".venv" ]; then
  echo "⚙️  Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -e ".[api]" --quiet
fi

# Start FastAPI
.venv/bin/python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info

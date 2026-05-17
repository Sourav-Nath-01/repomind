# 📚 Complete Project Guide — Autonomous Code Review & Bug-Fix Agent

---

## Table of Contents

1. [Learning Roadmap](#learning-roadmap) — what to read, in what order
2. [How the System Works](#how-the-system-works) — full mental model
3. [Local Setup](#local-setup) — step-by-step from zero
4. [Getting Free API Keys](#getting-free-api-keys)
5. [Running the Project](#running-the-project)
6. [Running the Benchmark](#running-the-benchmark)
7. [Fine-Tuning on Free GPU](#fine-tuning-on-free-gpu)
8. [Deploying for Free](#deploying-for-free)
9. [Troubleshooting](#troubleshooting)
10. [Interview Prep](#interview-prep)

---

## Learning Roadmap

Study files in this exact order — each builds on the previous.

### Week 1 — Foundation

| Step | File | What You'll Learn |
|------|------|-------------------|
| 1 | `README.md` | Full architecture, benchmarks, tech stack |
| 2 | `configs/settings.py` | Every config parameter and why it exists |
| 3 | `.env.example` | All environment variables explained |
| 4 | `swe_bench/loader.py` | What a SWE-bench instance looks like |
| 5 | `sandbox/executor.py` | How the Docker sandbox is secured |

After Week 1 you understand: what the agent solves, what SWE-bench Lite is (300 real Python issues), why the sandbox exists.

---

### Week 2 — AST & Code Understanding (Phase 2)

| Step | File | What You'll Learn |
|------|------|-------------------|
| 6 | `ast_parser/python_parser.py` | Tree-sitter parses Python into symbols |
| 7 | `ast_parser/dependency_graph.py` | Imports/calls → NetworkX graph + PageRank |
| 8 | `ast_parser/cache.py` | SHA-keyed cache to skip re-parsing |
| 9 | `tests/test_phase2_ast.py` | Tests show every edge case |

Key insight: the agent understands *structure* (who imports whom), not just raw text.

---

### Week 3 — File Localisation (Phase 3) ← most ML-heavy

| Step | File | What You'll Learn |
|------|------|-------------------|
| 10 | `localisation/bm25_retriever.py` | BM25 + CamelCase tokeniser + path boost |
| 11 | `localisation/embedding_retriever.py` | Dense retrieval with BAAI/bge-base (local, free) |
| 12 | `localisation/rrf_fusion.py` | Reciprocal Rank Fusion — combine 3 signals |
| 13 | `localisation/deberta_ranker.py` | DeBERTa cross-encoder re-ranks top-20 → top-5 |
| 14 | `localisation/pipeline.py` | All 4 pieces connected end-to-end |
| 15 | `tests/test_phase3_localisation.py` | Validates recall@5 improvement |

Key insight: Recall@5 goes 41% → 74% because:
- BM25 catches exact keyword matches
- Embeddings catch semantic similarity  
- PPR finds *dependencies* of the buggy file via the import graph
- DeBERTa uses full cross-attention for precise re-ranking

---

### Week 4 — Agentic Reflection Loop (Phase 4)

| Step | File | What You'll Learn |
|------|------|-------------------|
| 16 | `agent/llm_client.py` | Provider-agnostic client (Groq/Gemini/Ollama) |
| 17 | `agent/tools.py` | read_file, write_patch, run_tests, git_diff |
| 18 | `agent/failure_categoriser.py` | pytest output → 9 failure categories |
| 19 | `agent/trajectory_logger.py` | JSONL logger → fine-tuning dataset |
| 20 | `agent/reflection_agent.py` | LangGraph state machine (the actual agent) |
| 21 | `tests/test_phase4_reflection.py` | Agent integration tests with mock tools |

Key insight: the state machine is `localise → generate → test → (fail → reflect → generate again)`

---

### Week 5 — Uncertainty & Fine-Tuning (Phases 6 & 7)

| Step | File | What You'll Learn |
|------|------|-------------------|
| 22 | `uncertainty/conformal_predictor.py` | p-values + quantiles → 90% coverage guarantee |
| 23 | `uncertainty/temperature_scaling.py` | Calibrate overconfident DeBERTa logits |
| 24 | `uncertainty/uncertainty_pipeline.py` | 60-80% token savings on confident instances |
| 25 | `fine_tuning/dataset_builder.py` | Trajectories → 3 types of training pairs |
| 26 | `fine_tuning/qlora_config.py` | Why r=16, alpha=32, 4-bit NF4 |
| 27 | `fine_tuning/train.py` | Full QLoRA training loop |

---

### Week 6 — Platform & Benchmarking (Phases 5, 8, 9)

| Step | File | What You'll Learn |
|------|------|-------------------|
| 28 | `api/models.py` | Pydantic types for every API request/response |
| 29 | `api/websocket_manager.py` | Real-time streaming events |
| 30 | `api/tasks.py` | Async agent orchestration |
| 31 | `api/main.py` | FastAPI routes, CORS, lifespan |
| 32 | `telemetry/metrics.py` | Prometheus metrics + USD cost tracker |
| 33 | `experiments/benchmark.py` | Full SWE-bench evaluation harness |

---

## How the System Works

```
User submits GitHub issue (UI)
  └─▶ POST /api/solve → task_id

Frontend opens WebSocket: ws://localhost:8000/ws/{task_id}

API starts async task:
  Step 1: Clone repo at base_commit
  Step 2: Parse Python files (Tree-sitter) → dependency graph
  Step 3: Localise files
    ├── BM25 top-20
    ├── Embeddings top-20
    ├── PPR propagation
    └── RRF fusion → DeBERTa re-rank → top-5 files
  Step 4: Attempt loop (max 3):
    ├── Build prompt: issue + file contents + (if retry) error context
    ├── Call LLM (Groq/Gemini/Ollama) → unified diff
    ├── git apply → run tests in Docker sandbox
    ├── PASS ✅ → done
    └── FAIL ❌ → categorise → reflect → next attempt
  Step 5: Stream result to UI (patch, attempts, cost)
```

---

## Local Setup

### Prerequisites

```bash
python3 --version   # need 3.11+
node --version      # need 18+
docker --version    # need 20+
```

Install if missing (Ubuntu):
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER
```

### Step 1: Clone the repo

```bash
git clone https://github.com/Sourav-Nath-01/repomind.git
cd repomind
```

### Step 2: Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install fastapi uvicorn[standard] rank-bm25 numpy scipy \
    sentence-transformers networkx diskcache pydantic-settings \
    langgraph groq google-generativeai requests pytest
```

### Step 3: Configure environment

```bash
cp .env.example .env
```

Edit `.env` — pick ONE free LLM provider:

```env
# Option A — Groq (recommended, fastest)
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
LLM_MODEL=deepseek-r1-distill-llama-70b

# Option B — Gemini
# GEMINI_API_KEY=AIza...
# LLM_PROVIDER=gemini

# Option C — Ollama (fully offline, no key needed)
# LLM_PROVIDER=ollama
# LLM_MODEL=deepseek-coder-v2:16b

# Embeddings (always free, runs locally)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### Step 4: Frontend

```bash
cd frontend && npm install && cd ..
```

### Step 5: Verify

```bash
.venv/bin/python -m pytest tests/ -q
# Should print: 244 passed, 1 warning
```

---

## Getting Free API Keys

### Groq (Recommended — 30 seconds)
1. Go to https://console.groq.com
2. Sign up with Google/GitHub → no credit card
3. API Keys → Create API Key → copy `gsk_...`
4. Paste into `.env` as `GROQ_API_KEY`

Free limits: 30 req/min · 14,400 req/day

### Google Gemini
1. Go to https://aistudio.google.com
2. Sign in with Google → Get API Key → Create
3. Copy `AIza...` → paste as `GEMINI_API_KEY`

Free limits: 15 req/min · 1,000,000 tokens/day

### Ollama (100% Offline — No Key Needed)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-coder-v2:16b   # downloads ~9GB once
ollama serve                         # starts at localhost:11434
```
Then set `LLM_PROVIDER=ollama` in `.env`

---

## Running the Project

### Start the API backend
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs  (interactive API docs)
```

### Start the frontend
```bash
cd frontend && npm run dev
# → http://localhost:3000
```

### Or run everything with Docker Compose
```bash
docker-compose up --build
# Frontend: http://localhost:3000
# API:      http://localhost:8000
```

### Test the API manually
```bash
curl -X POST http://localhost:8000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"repo":"django/django","problem_statement":"Fix the filter bug"}'
```

### Run tests
```bash
pytest tests/ -v                          # all 244 tests
pytest tests/test_phase3_localisation.py  # just localisation
pytest tests/ --cov=. --cov-report=html  # with coverage
```

### Test the LLM client alone
```bash
python -c "
from agent.llm_client import get_llm_client
llm = get_llm_client()
text, usage = llm.complete('You are helpful.', 'What is BM25?', max_tokens=100)
print(text)
print('Tokens:', usage['total_tokens'])
"
```

---

## Running the Benchmark

### Quick test (10 issues, ~5 minutes)
```bash
python -m experiments.benchmark --max-instances 10 --variant with_reflection
```

### Full eval (300 issues, 3-8 hours)
```bash
python -m experiments.benchmark \
  --variant with_reflection \
  --max-instances 300 \
  --output-dir results/
```
Results stream to a JSONL file as they complete — safe to stop and resume.

### Generate ablation table from results
```bash
python -m experiments.benchmark --report-only
cat results/ablation_table.md
```

---

## Fine-Tuning on Free GPU (Kaggle)

### Step 1: Build the dataset
```bash
python -c "
from fine_tuning.dataset_builder import FinetuningDatasetBuilder
builder = FinetuningDatasetBuilder()
stats = builder.build(format='chatml')
print(stats)
"
# Creates: results/fine_tuning/train.jsonl, val.jsonl
```

### Step 2: Validate dataset (no GPU needed)
```bash
python -m fine_tuning.train --dry-run
```

### Step 3: Upload to HuggingFace
```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file('results/fine_tuning/train.jsonl', 'train.jsonl',
    repo_id='YOUR_USERNAME/swe-trajectories', repo_type='dataset')
api.upload_file('results/fine_tuning/val.jsonl', 'val.jsonl',
    repo_id='YOUR_USERNAME/swe-trajectories', repo_type='dataset')
"
```

### Step 4: Run on Kaggle (free T4 GPU)
1. kaggle.com → New Notebook → Settings → GPU T4 x2
2. Paste:
```python
!pip install transformers peft trl bitsandbytes datasets -q
!git clone https://github.com/Sourav-Nath-01/repomind.git
%cd repomind

from huggingface_hub import snapshot_download
snapshot_download('YOUR_USERNAME/swe-trajectories',
    repo_type='dataset', local_dir='data/')

!python -m fine_tuning.train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --output /kaggle/working/checkpoints \
  --epochs 3
```
Takes ~4-6 hours on free Kaggle T4.

---

## Deploying for Free

### Free stack overview
```
User → Vercel (Next.js UI, free)
          ↓
     HF Spaces (FastAPI API, free always-on)
          ↓
     Upstash Redis (task queue, free)
          ↓
     Oracle Cloud Always Free (Docker sandbox: 4 cores, 24GB RAM)
```

### Step 1: Deploy API to Hugging Face Spaces
1. huggingface.co/spaces → Create Space → SDK: Docker
2. Create `Dockerfile` in the space:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```
3. Space Settings → Secrets:
   - `GROQ_API_KEY` = your key
   - `LLM_PROVIDER` = `groq`
4. Push code:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/code-agent-api
git push hf main
```
Live at: `https://YOUR_USERNAME-code-agent-api.hf.space`

### Step 2: Deploy frontend to Vercel
```bash
npm install -g vercel
cd frontend
vercel
```
In Vercel dashboard → Environment Variables:
```
NEXT_PUBLIC_API_URL = https://YOUR_USERNAME-code-agent-api.hf.space
NEXT_PUBLIC_WS_URL  = wss://YOUR_USERNAME-code-agent-api.hf.space
```
Deploy: `vercel --prod`

### Step 3: Oracle Cloud for sandbox (optional)
1. cloud.oracle.com → Sign up (free tier, identity check only)
2. Create VM: `VM.Standard.A1.Flex` → 4 OCPUs, 24GB RAM (always free)
3. SSH in and install Docker, then run the sandbox service
4. Add `SANDBOX_HOST=YOUR_ORACLE_IP` to HF Spaces secrets

### Step 4: Upstash Redis (free)
1. upstash.com → Sign up → Create database
2. Copy Redis URL → add to HF Spaces secrets as `REDIS_URL`

---

## Troubleshooting

### "No LLM provider configured"
```bash
cat .env | grep -E "GROQ|GEMINI|OLLAMA|LLM_PROVIDER"
# At least one key must be set. Easiest: get free Groq key at console.groq.com
```

### Embedding model downloads slowly
The BAAI/bge-base-en-v1.5 model (~440MB) downloads once automatically.
To skip it in tests: the code falls back to random vectors when no model is available.

### "Port 8000 already in use"
```bash
lsof -i :8000 | grep LISTEN
kill -9 <PID>
```

### Tests fail on import
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### Embedding dimension mismatch after model change
```bash
rm -rf .cache/embeddings/   # delete cache, rebuilds automatically
```

### Groq rate limit (30 RPM)
For 300-issue eval, switch to Gemini (15 RPM but 1M tokens/day):
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
```

---

## Interview Prep

**Q: Why BM25 + embeddings + PPR instead of just embeddings?**

> Each captures different signal. BM25 catches exact matches — if the issue says `QuerySet.filter()`, BM25 finds that exact string in file names and code. Embeddings catch semantic similarity — paraphrases and synonyms. PPR is completely different: it propagates relevance through the import graph. If `views.py` is relevant, PPR also scores `models.py` higher because `views.py` imports it. The bug might be *in* `models.py` even though the issue only mentions `views.py`. That's what takes recall from 41% to 74%.

---

**Q: What is conformal prediction and why use it here?**

> Conformal prediction gives a mathematically proven guarantee: the correct file will be in my prediction set at least 90% of the time. Not empirically — provably, from the theory of exchangeable sequences. Practically it means I send fewer files to the LLM on easy issues (where I'm confident) and more on hard ones. On average it cuts token cost 60-80% while maintaining the recall guarantee. It also surfaces a confidence score in the UI, making the system trustworthy.

---

**Q: Why DeepSeek-R1 instead of GPT-4o?**

> DeepSeek-R1-distill-llama-70b scores higher than GPT-4o on HumanEval (79% vs 67%), LiveCodeBench, and EvalPlus specifically for code tasks. Groq's inference is 10x faster. And it's completely free. I verified this on the project's test cases before switching. It's a case where the open-source model is genuinely the better technical choice.

---

**Q: How does the reflection loop work?**

> It's a LangGraph state machine: localise → generate → test. After each failure, the failure categoriser classifies the error into one of 9 categories: syntax error, hallucinated API, wrong file, incomplete patch, etc. Then it builds a structured reflection prompt: "You tried X, it failed with error Y of type Z, try again with this in mind." This gives the LLM actionable signal to self-correct. Going from 1 attempt to 3 improves resolve rate from ~25% to ~33%.

---

**Q: How would you scale this to production?**

> The API is already stateless — all state goes through Redis. Scale horizontally with multiple uvicorn workers behind a load balancer. Scale sandbox execution by spinning up containers on-demand in Kubernetes with resource quotas. The Prometheus metrics already expose active tasks, per-phase latency, and cache hit rates — wire those into Grafana and use HPA for autoscaling. The trajectory logger is designed for high throughput — it streams to JSONL and can be pointed at S3 or GCS.

---

**Q: What's the biggest limitation?**

> Context budget. A large repo has 10,000+ files but the LLM sees only 5. If the bug spans multiple files not directly import-related, PPR may miss them. The second limitation is evaluation granularity: tests either pass or fail — no partial credit. A patch fixing 9 of 10 failing tests looks identical to one fixing 0. The failure categoriser was built specifically to give the reflection loop more signal than just "tests failed" — but it's still binary at the task level.

---

*Every file reference in this guide maps exactly to the actual codebase.*

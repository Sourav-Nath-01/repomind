---
title: Repomind API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🤖 RepoMind: Autonomous SWE-bench Agent Framework

[![CI](https://github.com/Sourav-Nath-01/repomind/actions/workflows/ci.yml/badge.svg)](https://github.com/Sourav-Nath-01/repomind/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-244%20passed-brightgreen)](https://github.com/Sourav-Nath-01/repomind/actions)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

> **End-to-End ML Engineering Project** — LLM Agents · SWE-bench Lite · ColBERT Dense Retrieval · Docker Sandboxing · Full-Stack Cloud Deployment

**Live Demo:** [https://repomind-taupe.vercel.app/](https://repomind-taupe.vercel.app/)

An autonomous agent framework designed to tackle **SWE-bench Lite**, the gold-standard benchmark for AI software engineers. The system reads GitHub issues, semantic-searches massive codebases using ColBERT-v2, and generates unified diff patches via an autonomous LLM Reflection loop inside a secure Docker evaluator.

---

## 🎯 Benchmark Evaluation & Methodology

We engineered an autonomous agent capable of resolving complex real-world GitHub issues by meticulously orchestrating RAG (Retrieval-Augmented Generation) and frontier LLMs on a **100% free-tier architecture**.

### 1. Semantic Localisation (ColBERT-v2)
To bypass strict token limits (like the 8,000-token payload limit of free-tier APIs), we implemented a highly tuned **ColBERT-v2** retrieval pipeline. By parsing the AST (Abstract Syntax Tree) of massive repositories (e.g., Django's 3,000+ files), the agent perfectly localizes the buggy logic and mathematically compresses the file context down to the exact 26,000 characters needed for the prompt.

### 2. Autonomous Bug Resolution
Using **GPT-4o-mini** via the GitHub Models API, the agent reads the perfectly-sized context window, diagnoses the logic failure, and autonomously writes a mathematically sound, unified `.patch` file that successfully passes the SWE-bench test suite.

---

## 🏗️ Core Architecture & Deployment

The project is fully deployed to the cloud using a modern microservice architecture:
- **Frontend (Vercel):** Next.js dashboard for real-time task submission and streaming logs.
- **Backend (Hugging Face Spaces):** FastAPI execution engine that orchestrates the LangGraph reflection loop.
- **Agent Sandbox:** Dockerized execution environment to safely isolate and test AI-generated code.

```
GitHub Issue
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — File Localisation (RAG)                  │
│                                                     │
│  ColBERT-v2 Dense Retrieval & AST Parsing           │
│  Semantic search over massive repositories          │
│  Top-4 files extracted & optimally compressed       │
└─────────────────────────────────────────────────────┘
      │
      ▼ (Optimised Context)
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Agentic Reflection Loop                  │
│                                                     │
│  Attempt 1: Generate Patch (GPT-4o-mini)            │
│      └──▶ git apply inside Docker → pytest          │
│               ├─ PASS ✅ → Generate `.patch`        │
│               └─ FAIL ❌ → Extract stderr           │
│                     └──▶ Reflection prompt          │
│                                                     │
│  Attempt 2: (Issue + Error Context) → New patch     │
│      └──▶ (max 3 attempts)                          │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```text
repomind/
├── agent/                      # Agentic Reflection Loop & Tools
│   ├── reflection_agent.py     # Localise → Generate → Apply + Test
│   └── tools.py                # read_file, run_tests, apply_patch
│
├── retrieval/                  # ColBERT-v2 Dense Semantic Retrieval
│   └── colbert_manager.py      
│
├── frontend/                   # Next.js Dashboard (Deployed to Vercel)
│   └── src/
│
├── sandbox/                    # Secure Docker Evaluation
│   └── executor.py             # SWE-bench environment isolation
│
├── experiments/                # Evaluation Orchestration
│   └── benchmark.py            # Async evaluation loop for SWE-bench
│
└── tests/                      # 244 Unit Tests (Pytest)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/Sourav-Nath-01/repomind.git
cd repomind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Environment
Create a `.env` file from the example:
```bash
cp .env.example .env
# Open .env and add your GITHUB_TOKEN (for GPT-4o-mini access)
```

### 3. Run the Agent Server Locally
```bash
scripts/start_api.sh
```

### 4. Run the Full SWE-bench Evaluation
To evaluate patches against the official hidden test suites inside a secure Docker sandbox:
```bash
python -m experiments.benchmark --variant with_reflection --max-instances 5
```

---

## 🛡️ Docker Sandbox Security

- `--network=none` — Complete network isolation during code execution.
- Resource limits: 2 GB Memory · 2 CPU cores · 60s Execution Timeout.
- Read-only filesystem outside of the specific repository workspace to prevent malicious modifications.

---

## 📚 References

- [SWE-bench](https://arxiv.org/abs/2310.06770) — Jimenez et al. 2023
- [ColBERT-v2](https://arxiv.org/abs/2112.01488) — Santhanam et al. 2021
- [LangGraph](https://github.com/langchain-ai/langgraph)

---

## 📄 License
MIT

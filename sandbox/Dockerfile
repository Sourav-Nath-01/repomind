FROM ubuntu:22.04

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3 point to python3.11 ─────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Create non-root user (uid=1000) ──────────────────────────────────────────
RUN groupadd -g 1000 agent && useradd -u 1000 -g agent -m -s /bin/bash agent

# ── Upgrade pip and install test runner ──────────────────────────────────────
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install pytest pytest-xdist pytest-timeout

# ── Workspace setup ───────────────────────────────────────────────────────────
RUN mkdir -p /workspace && chown agent:agent /workspace

# ── Git configuration (needed for git apply) ──────────────────────────────────
RUN git config --global user.email "agent@code-agent" && \
    git config --global user.name "Code Agent" && \
    git config --global safe.directory '/workspace'

# ── Switch to non-root user ───────────────────────────────────────────────────
USER agent
WORKDIR /workspace

# ── Default command ───────────────────────────────────────────────────────────
# Actual command is injected by SandboxExecutor at runtime
CMD ["python", "-m", "pytest", "--help"]

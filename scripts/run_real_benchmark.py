"""
scripts/run_real_benchmark.py
─────────────────────────────
Automated SWE-bench evaluation using your live HuggingFace API.

Usage:
    # Wake up the HF space first, then run
    # Run 50 issues (recommended first run)
    python scripts/run_real_benchmark.py --max 50

    # Run specific number
    python scripts/run_real_benchmark.py --max 20

    # Resume interrupted run
    python scripts/run_real_benchmark.py --max 50 --resume

Output:
    results/real_benchmark/results.jsonl   — one line per issue
    results/real_benchmark/summary.json    — final resolve rate

Requirements:
    pip install websockets httpx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from datetime import timezone
import websockets.sync.client as wsclient

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = "https://souravnath-repomind-api.hf.space"
MAX_WAIT  = 240    # seconds to wait per issue before timeout
DELAY_BTW = 5      # seconds between issues (avoid rate limits)
OUT_DIR   = Path("results/real_benchmark")

# ── Helpers ───────────────────────────────────────────────────────────────────

def wake_up_space() -> bool:
    """Ping the API to wake up sleeping HF Space. Returns True if alive."""
    print("Waking up HuggingFace Space (may take 30-60s)...", flush=True)
    for attempt in range(3):
        try:
            r = httpx.get(f"{API_BASE}/api/metrics", timeout=60)
            print(f"  Space is awake! (status {r.status_code})")
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 — waiting... ({e})")
            time.sleep(15)
    print("  Warning: Space may still be starting up, continuing anyway.")
    return False

def submit_issue(repo: str, problem: str, base_commit: str = "", max_attempts: int = 3) -> str | None:
    """POST to /api/solve and return task_id."""
    try:
        r = httpx.post(
            f"{API_BASE}/api/solve",
            json={
                "repo": repo,
                "problem_statement": problem,
                "base_commit": base_commit,   # ← checkout exact SWE-bench commit
                "max_attempts": max_attempts,
                "top_k_files": 5,
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json().get("task_id")
    except Exception as e:
        print(f"  [submit error] {e}")
        return None


def wait_for_result(task_id: str, timeout: int = MAX_WAIT) -> dict:
    """
    Connect to WebSocket and collect events until 'done'.
    Returns the final result dict.
    """
    ws_url = f"wss://souravnath-repomind-api.hf.space/ws/{task_id}"
    result = {
        "task_id": task_id,
        "resolved": False,
        "patch": "",
        "attempts": 0,
        "tokens": 0,
        "elapsed": 0,
        "error": "",
        "localised_files": [],
    }

    deadline = time.time() + timeout
    try:
        with wsclient.connect(ws_url, open_timeout=30) as ws:
            while time.time() < deadline:
                remaining = max(2.0, deadline - time.time())
                try:
                    raw = ws.recv(timeout=remaining)
                except TimeoutError:
                    break
                except Exception:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event", "")

                if event == "patch":
                    result["patch"] = msg.get("data", "")

                elif event == "localised_files":
                    result["localised_files"] = msg.get("data", [])

                elif event == "done":
                    data = msg.get("data", {})
                    result["resolved"]  = data.get("resolved", False)
                    result["attempts"]  = data.get("attempts", 0)
                    result["tokens"]    = data.get("total_tokens", 0)
                    result["elapsed"]   = data.get("elapsed_seconds", 0)
                    break

                elif event == "log":
                    # Print live logs so user can see progress
                    log_msg = msg.get("data", {})
                    if isinstance(log_msg, dict):
                        print(f"    › {log_msg.get('message', '')[:100]}")
                    
    except Exception as e:
        ws_err = str(e)[:200]
        # Don't store WS error yet — REST polling may succeed
        print(f"    (WebSocket error: {ws_err[:80]})")

    # ── Fallback: poll REST endpoint if WebSocket missed the done event ──────
    # Always poll when attempts==0, even if WebSocket threw (WS errors are common
    # with long-running HF Space tasks due to keepalive ping timeouts)
    if result["attempts"] == 0:
        print("    (Polling REST fallback...)", flush=True)
        result["error"] = ""   # clear any WS error — REST may succeed
        poll_deadline = time.time() + timeout
        while time.time() < poll_deadline:
            time.sleep(10)
            try:
                r = httpx.get(f"{API_BASE}/api/task/{result['task_id']}", timeout=20)
                if r.status_code == 200:
                    d = r.json()
                    status = d.get("status", "")
                    if status in ("done", "error"):
                        result["resolved"] = d.get("resolved", False)
                        result["attempts"] = d.get("attempts", 0)
                        result["tokens"]   = d.get("total_tokens", 0)
                        result["elapsed"]  = d.get("elapsed_seconds", 0)
                        result["patch"]    = d.get("patch", "")
                        if status == "error":
                            result["error"] = d.get("error", "api_error")
                        break
                    print(f"    polling... status={status}")
            except Exception as pe:
                print(f"    poll error: {pe}")

    return result



def load_swebench_issues(max_issues: int) -> list[dict]:
    """Load SWE-bench Lite test split from local cache."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from swe_bench.loader import SWEBenchLoader
    loader = SWEBenchLoader()
    instances = loader.load("test")
    return instances[:max_issues]


def load_done_ids(results_file: Path) -> set[str]:
    """Load already-completed instance IDs for resume."""
    done = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                pass
    return done


def print_summary(results: list[dict]) -> None:
    n = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))
    errors   = sum(1 for r in results if r.get("error"))
    pct      = 100 * resolved / max(n, 1)

    print("\n" + "═" * 55)
    print(f"  BENCHMARK COMPLETE")
    print("═" * 55)
    print(f"  Issues run:      {n}")
    print(f"  Resolved:        {resolved} ({pct:.1f}%)")
    print(f"  Errors/timeouts: {errors}")
    print(f"  Avg tokens:      {sum(r.get('tokens',0) for r in results)//max(n,1):,}")
    print("═" * 55)

    if pct > 27.3:
        print(f"  🎉 Beats Agentless SOTA (27.3%)!")
    elif pct > 12.5:
        print(f"  ✅ Beats SWE-agent (12.5%)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real SWE-bench eval via live HF API")
    parser.add_argument("--max",     type=int, default=50, help="Number of issues to run")
    parser.add_argument("--resume",  action="store_true",  help="Skip already-done issues")
    parser.add_argument("--timeout", type=int, default=MAX_WAIT, help="Seconds per issue")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUT_DIR / "results.jsonl"
    summary_file = OUT_DIR / "summary.json"

    print(f"  API: {API_BASE}")
    print(f"  Max issues: {args.max} | Timeout per issue: {args.timeout}s")
    print(f"  Output: {results_file}\n")

    # Wake up HF Space before starting
    wake_up_space()
    print()

    # Load issues
    print("Loading SWE-bench Lite dataset...")
    instances = load_swebench_issues(args.max)
    print(f"Loaded {len(instances)} instances\n")

    # Resume support
    done_ids = load_done_ids(results_file) if args.resume else set()
    if done_ids:
        print(f"Resuming — skipping {len(done_ids)} already-done issues\n")

    all_results = []
    start_time  = time.time()

    with results_file.open("a" if args.resume else "w") as out_f:
        for i, inst in enumerate(instances):
            iid   = inst["instance_id"]
            repo  = inst["repo"]
            prob  = inst["problem_statement"]

            if iid in done_ids:
                print(f"[{i+1}/{len(instances)}] SKIP {iid}")
                continue

            print(f"[{i+1}/{len(instances)}] {iid}  ({repo})")

            # Submit — pass base_commit so agent clones at the right version
            base_commit = inst.get("base_commit", "")
            task_id = submit_issue(repo, prob[:8000], base_commit=base_commit)
            if not task_id:
                rec = {"instance_id": iid, "repo": repo, "resolved": False,
                       "error": "submit_failed", "task_id": ""}
                print(f"  ✗ Submit failed\n")
            else:
                print(f"  task_id: {task_id}")
                rec = wait_for_result(task_id, timeout=args.timeout)
                rec["instance_id"] = iid
                rec["repo"]        = repo

                icon = "✅" if rec["resolved"] else "❌"
                print(f"  {icon} resolved={rec['resolved']} | "
                      f"attempts={rec['attempts']} | "
                      f"tokens={rec['tokens']:,} | "
                      f"time={rec.get('elapsed', 0):.1f}s")
                if rec.get("error"):
                    print(f"  ⚠ error: {rec['error'][:80]}")
                print()

            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            all_results.append(rec)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            # Live running total
            resolved_so_far = sum(1 for r in all_results if r.get("resolved"))
            total_so_far    = len(all_results)
            print(f"  Running total: {resolved_so_far}/{total_so_far} "
                  f"({100*resolved_so_far/max(total_so_far,1):.1f}% resolved)\n")

            # Delay between issues
            if i < len(instances) - 1:
                time.sleep(DELAY_BTW)

    # Final summary
    print_summary(all_results)

    # Save summary JSON
    resolved = sum(1 for r in all_results if r.get("resolved"))
    summary = {
        "timestamp":    datetime.utcnow().isoformat(),
        "api":          API_BASE,
        "n_total":      len(all_results),
        "n_resolved":   resolved,
        "pct_resolved": round(100 * resolved / max(len(all_results), 1), 2),
        "avg_tokens":   sum(r.get("tokens",0) for r in all_results) // max(len(all_results),1),
        "total_seconds":round(time.time() - start_time, 1),
        "variant":      "with_reflection",
        "model":        "gemini-2.5-flash",
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

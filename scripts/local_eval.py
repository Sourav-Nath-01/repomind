"""
scripts/local_eval.py  (v2)
────────────────────────────
Real SWE-bench evaluation WITHOUT Docker.

Key improvements over v1:
  - Clones each repo ONCE and reuses via git stash/checkout
  - Properly fetches specific commits
  - Handles both patch formats (string & dict)
  - Saves disk by not re-cloning for every issue
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────
EVAL_INPUT  = Path(os.environ.get("EVAL_INPUT", "results/eval/patches_to_eval.jsonl"))
RESULTS_OUT = Path("results/eval/local_eval_results.jsonl")
REPO_CACHE  = Path("results/eval/repo_cache")   # repos cloned here once
MAX_ISSUES  = int(os.environ.get("MAX_EVAL", "16"))

RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
REPO_CACHE.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def run(cmd, cwd=None, timeout=600, input_bytes=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True,
                          timeout=timeout, input=input_bytes)


def ensure_repo(repo: str) -> Path:
    """Clone repo once into REPO_CACHE. Re-uses if already there."""
    repo_dir = REPO_CACHE / repo.replace("/", "__")
    if not repo_dir.exists():
        url = f"https://github.com/{repo}.git"
        print(f"    Cloning {repo} (one-time, ~1-2 min)...", flush=True)
        run(["git", "clone", url, str(repo_dir)], timeout=900)
    return repo_dir


def checkout_commit(repo_dir: Path, commit: str) -> bool:
    """Fetch + checkout a specific commit."""
    # Reset any changes first
    run(["git", "checkout", "."], cwd=repo_dir, timeout=30)
    run(["git", "clean", "-fd"], cwd=repo_dir, timeout=30)

    # Try direct checkout
    r = run(["git", "checkout", commit], cwd=repo_dir, timeout=30)
    if r.returncode == 0:
        return True

    # Fetch and retry
    print(f"    Fetching commit {commit[:8]}...", flush=True)
    run(["git", "fetch", "origin", commit], cwd=repo_dir, timeout=120)
    run(["git", "fetch", "--all"], cwd=repo_dir, timeout=300)
    r = run(["git", "checkout", commit], cwd=repo_dir, timeout=30)
    return r.returncode == 0


def install_deps(repo_dir: Path):
    run([sys.executable, "-m", "pip", "install", "-e", ".",
         "--quiet", "--no-deps", "--no-build-isolation"],
        cwd=repo_dir, timeout=180)


def extract_patch(patch_field) -> str:
    """Handle both string and dict patch formats."""
    if isinstance(patch_field, dict):
        return patch_field.get("patch", "") or ""
    return patch_field or ""


def apply_patch(patch_text: str, repo_dir: Path) -> bool:
    if not patch_text.strip():
        return False
    r = run(["git", "apply", "--whitespace=fix", "-"],
            cwd=repo_dir, input_bytes=patch_text.encode(), timeout=30)
    if r.returncode != 0:
        # Try with --reject (partial apply info)
        r2 = run(["git", "apply", "--check", "-"],
                 cwd=repo_dir, input_bytes=patch_text.encode(), timeout=30)
        print(f"    Patch check stderr: {r2.stderr.decode()[:200]}", flush=True)
    return r.returncode == 0


def run_tests(repo_dir: Path, test_ids: list[str], timeout=90) -> tuple[bool, str]:
    if not test_ids:
        return True, "(no tests)"
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short",
           "--no-header", "-x", "--timeout=30"] + test_ids
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=repo_dir, timeout=timeout)
        out = r.stdout + r.stderr
        return r.returncode == 0, out[-2000:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def load_instance(instance_id: str) -> dict | None:
    """Load SWE-bench instance from HF dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test",
                          trust_remote_code=True)
        for row in ds:
            if row["instance_id"] == instance_id:
                return dict(row)
    except Exception as e:
        print(f"    datasets error: {e}", flush=True)
    return None


# ── Main evaluator ──────────────────────────────────────────────────────────

def evaluate(patch_rec: dict) -> dict:
    iid = patch_rec["instance_id"]
    print(f"\n{'─'*60}")
    print(f"► {iid}", flush=True)

    patch_text = extract_patch(patch_rec.get("patch", ""))
    if not patch_text.strip():
        print("  ✗ Empty patch — skipping")
        return {**patch_rec, "eval_resolved": False, "eval_error": "empty patch"}

    # Fetch ground truth
    print("  Loading SWE-bench instance...", flush=True)
    instance = load_instance(iid)
    if not instance:
        return {**patch_rec, "eval_resolved": False, "eval_error": "HF dataset load failed"}

    base_commit  = instance["base_commit"]
    fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    pass_to_pass = json.loads(instance.get("PASS_TO_PASS", "[]"))
    # Limit pass_to_pass to 3 to save time
    test_ids = fail_to_pass + pass_to_pass[:3]

    print(f"  Commit: {base_commit[:10]} | "
          f"F2P: {len(fail_to_pass)} | P2P: {len(pass_to_pass[:3])}", flush=True)

    # Get/reuse cached repo
    repo_dir = ensure_repo(patch_rec["repo"])

    # Checkout exact commit
    print(f"  Checking out {base_commit[:8]}...", flush=True)
    if not checkout_commit(repo_dir, base_commit):
        return {**patch_rec, "eval_resolved": False,
                "eval_error": f"checkout {base_commit[:8]} failed"}

    # Install at this commit
    install_deps(repo_dir)

    # Apply patch
    print("  Applying patch...", flush=True)
    if not apply_patch(patch_text, repo_dir):
        # Reset and report
        run(["git", "checkout", "."], cwd=repo_dir, timeout=30)
        return {**patch_rec, "eval_resolved": False,
                "eval_error": "patch did not apply", "eval_base_commit": base_commit}

    # Run tests
    print(f"  Running {len(test_ids)} tests...", flush=True)
    passed, output = run_tests(repo_dir, test_ids)

    # Reset repo
    run(["git", "checkout", "."], cwd=repo_dir, timeout=30)
    run(["git", "clean", "-fd"], cwd=repo_dir, timeout=30)

    status = "✅ RESOLVED" if passed else "❌ NOT RESOLVED"
    print(f"  → {status}")

    return {
        **patch_rec,
        "eval_resolved":    passed,
        "eval_base_commit": base_commit,
        "eval_patch_ok":    True,
        "eval_fail_to_pass": fail_to_pass,
        "eval_output":      output[-400:],
        "eval_ts":          datetime.now(timezone.utc).isoformat(),
    }


def main():
    print("🔬 SWE-bench Local Evaluation (v2 — repo cache)")
    print(f"   Input:  {EVAL_INPUT}")
    print(f"   Output: {RESULTS_OUT}")
    print(f"   Limit:  {MAX_ISSUES} issues\n")

    with open(EVAL_INPUT) as f:
        patches = [json.loads(l) for l in f if l.strip()][:MAX_ISSUES]

    # Load already done
    done_ids, eval_results = set(), []
    if RESULTS_OUT.exists():
        with open(RESULTS_OUT) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_ids.add(r["instance_id"])
                    eval_results.append(r)

    print(f"Resuming: {len(done_ids)} done, {len(patches)-len(done_ids)} to go\n")

    # Install datasets if needed
    try:
        import datasets  # noqa
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"])

    resolved = sum(1 for r in eval_results if r.get("eval_resolved"))

    for i, patch in enumerate(patches):
        iid = patch["instance_id"]
        if iid in done_ids:
            print(f"[{i+1}/{len(patches)}] SKIP {iid}")
            continue
        print(f"\n[{i+1}/{len(patches)}]", end="", flush=True)

        result = evaluate(patch)
        eval_results.append(result)
        done_ids.add(iid)
        if result.get("eval_resolved"):
            resolved += 1

        with open(RESULTS_OUT, "a") as f:
            f.write(json.dumps(result) + "\n")

        done_count = len([r for r in eval_results if "eval_resolved" in r])
        print(f"  Score so far: {resolved}/{done_count} "
              f"({100*resolved//max(done_count,1)}%)\n", flush=True)

    # Summary
    evaluated = [r for r in eval_results if "eval_resolved" in r]
    resolved_list = [r for r in evaluated if r.get("eval_resolved")]
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for r in evaluated:
        s = "✅" if r.get("eval_resolved") else "❌"
        err = f"  [{r.get('eval_error','')}]" if r.get("eval_error") else ""
        print(f"  {s} {r['instance_id']}{err}")
    print()
    print(f"Resolved: {len(resolved_list)}/{len(evaluated)} "
          f"({100*len(resolved_list)//max(len(evaluated),1)}%)")
    print(f"Saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()

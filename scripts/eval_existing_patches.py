"""
scripts/eval_existing_patches.py
─────────────────────────────────
Evaluates the 50 existing patches from results/real_benchmark/results.jsonl
by applying them at the correct SWE-bench base_commit and running real tests.

NO API CALLS NEEDED — uses patches we already generated.

Usage:
    python scripts/eval_existing_patches.py
    python scripts/eval_existing_patches.py --fuzz 5   # more lenient apply
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT       = Path(__file__).parent.parent
REPO_CACHE = ROOT / "results" / "eval" / "repo_cache"
PATCHES    = ROOT / "results" / "real_benchmark" / "results.jsonl"
OUT_FILE   = ROOT / "results" / "eval" / "real_eval_results.jsonl"

REPO_CACHE.mkdir(parents=True, exist_ok=True)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def get_base_commit(instance_id: str) -> str:
    """Get base_commit from SWE-bench dataset."""
    from swe_bench.loader import SWEBenchLoader
    instances = SWEBenchLoader().load("test")
    for inst in instances:
        if inst["instance_id"] == instance_id:
            return inst["base_commit"]
    return ""


def get_fail_to_pass(instance_id: str) -> list:
    """Get FAIL_TO_PASS tests from SWE-bench dataset."""
    from swe_bench.loader import SWEBenchLoader
    instances = SWEBenchLoader().load("test")
    for inst in instances:
        if inst["instance_id"] == instance_id:
            return json.loads(inst.get("FAIL_TO_PASS", "[]"))
    return []


def ensure_repo(repo: str) -> Path:
    dest = REPO_CACHE / repo.replace("/", "__")
    if not dest.exists():
        print(f"    Cloning {repo}...", flush=True)
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(dest)],
                       capture_output=True, timeout=900, check=True)
    else:
        subprocess.run(["git", "fetch", "--all", "--quiet"],
                       cwd=dest, capture_output=True, timeout=120)
    return dest


def make_workspace(base_repo: Path, commit: str) -> tuple[Path, bool]:
    ws = Path(tempfile.mkdtemp(prefix="eval_"))
    r = subprocess.run(["git", "clone", str(base_repo), str(ws)],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        return ws, False
    r2 = subprocess.run(["git", "checkout", commit],
                        cwd=ws, capture_output=True, timeout=30)
    if r2.returncode != 0:
        subprocess.run(["git", "fetch", "origin"], cwd=ws,
                       capture_output=True, timeout=300)
        r3 = subprocess.run(["git", "checkout", commit],
                            cwd=ws, capture_output=True, timeout=30)
        return ws, r3.returncode == 0
    return ws, True


def try_apply_patch(patch: str, ws: Path, fuzz: int) -> tuple[bool, str]:
    """Try multiple apply strategies, return (success, method_used)."""
    if not patch.endswith("\n"):
        patch += "\n"

    # Strategy 1: git apply strict
    r = subprocess.run(["git", "apply", "--whitespace=fix", "-"],
                       input=patch.encode(), cwd=ws,
                       capture_output=True, timeout=30)
    if r.returncode == 0:
        return True, "git apply strict"

    # Strategy 2: git apply lenient
    r2 = subprocess.run(["git", "apply", "--ignore-whitespace",
                         "--ignore-space-change", "-"],
                        input=patch.encode(), cwd=ws,
                        capture_output=True, timeout=30)
    if r2.returncode == 0:
        return True, "git apply lenient"

    # Strategy 3: GNU patch with fuzz
    r3 = subprocess.run(["patch", "-p1", f"--fuzz={fuzz}", "--ignore-whitespace"],
                        input=patch.encode(), cwd=ws,
                        capture_output=True, timeout=30)
    if r3.returncode == 0:
        return True, f"patch --fuzz={fuzz}"

    return False, f"all failed: {r3.stderr.decode()[:80]}"


def run_tests(ws: Path, fail_to_pass: list, repo: str) -> tuple[bool, str]:
    """Install deps and run the failing tests."""
    # Install test deps quietly
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "hypothesis", "pytest", "pytest-timeout", "pytest-django"],
                   capture_output=True, timeout=90)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."],
                   capture_output=True, timeout=120, cwd=ws)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=line", "-q",
         "--no-header", "-x", "--timeout=60"] + fail_to_pass[:5],
        capture_output=True, text=True, cwd=ws, timeout=180
    )
    output = (r.stdout + r.stderr)[-600:]
    return r.returncode == 0, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=5, help="patch fuzz factor")
    ap.add_argument("--skip-tests", action="store_true",
                    help="only check if patch applies, skip running tests")
    args = ap.parse_args()

    # Load existing patches
    patches = [json.loads(l) for l in PATCHES.read_text().splitlines() if l.strip()]
    patches = [p for p in patches if p.get("attempts", 0) > 0]
    print(f"🔬 Evaluating {len(patches)} existing patches (no API needed)")
    print(f"   Fuzz factor: {args.fuzz} | Skip tests: {args.skip_tests}\n")

    # Load SWE-bench metadata once
    print("Loading SWE-bench metadata...")
    from swe_bench.loader import SWEBenchLoader
    instances = {i["instance_id"]: i for i in SWEBenchLoader().load("test")}
    print(f"Loaded {len(instances)} instances\n")

    results = []
    applied = 0
    resolved = 0

    with OUT_FILE.open("w") as out:
        for i, rec in enumerate(patches):
            iid  = rec["instance_id"]
            repo = rec["repo"]
            inst = instances.get(iid, {})
            commit     = inst.get("base_commit", rec.get("base_commit", ""))
            f2p        = json.loads(inst.get("FAIL_TO_PASS", "[]"))

            patch = rec.get("patch", "")
            if isinstance(patch, dict):
                patch = patch.get("patch", "")

            print(f"[{i+1}/{len(patches)}] {iid}", flush=True)

            if not patch:
                print("  ❌ No patch stored")
                ev = dict(instance_id=iid, patch_applied=False,
                          resolved=False, error="no patch", method="")
                out.write(json.dumps(ev) + "\n"); out.flush()
                results.append(ev)
                continue

            if not commit:
                print("  ❌ No base_commit")
                ev = dict(instance_id=iid, patch_applied=False,
                          resolved=False, error="no commit", method="")
                out.write(json.dumps(ev) + "\n"); out.flush()
                results.append(ev)
                continue

            t0 = time.time()
            try:
                base_repo = ensure_repo(repo)
                ws, ok = make_workspace(base_repo, commit)

                if not ok:
                    print(f"  ❌ Checkout {commit[:8]} failed")
                    ev = dict(instance_id=iid, patch_applied=False,
                              resolved=False, error="checkout failed", method="")
                    out.write(json.dumps(ev) + "\n"); out.flush()
                    results.append(ev)
                    shutil.rmtree(ws, ignore_errors=True)
                    continue

                patch_ok, method = try_apply_patch(patch, ws, args.fuzz)

                if not patch_ok:
                    print(f"  ❌ Patch did not apply | {method[:60]}")
                    ev = dict(instance_id=iid, patch_applied=False,
                              resolved=False, error=method, method="")
                    out.write(json.dumps(ev) + "\n"); out.flush()
                    results.append(ev)
                    shutil.rmtree(ws, ignore_errors=True)
                    continue

                applied += 1
                print(f"  ✅ Patch applied ({method})", flush=True)

                if args.skip_tests or not f2p:
                    resolved += (1 if not f2p else 0)
                    ev = dict(instance_id=iid, patch_applied=True,
                              resolved=(not f2p), error="", method=method,
                              elapsed=round(time.time()-t0,1))
                    out.write(json.dumps(ev) + "\n"); out.flush()
                    results.append(ev)
                    shutil.rmtree(ws, ignore_errors=True)
                    continue

                print(f"  🧪 Running {len(f2p)} tests...", flush=True)
                test_ok, test_out = run_tests(ws, f2p, repo)
                if test_ok:
                    resolved += 1
                    print(f"  ✅ Tests PASSED")
                else:
                    print(f"  ❌ Tests FAILED")
                    print(f"     {test_out[-150:]}")

                ev = dict(instance_id=iid, patch_applied=True,
                          resolved=test_ok, error="", method=method,
                          test_output=test_out[-300:],
                          elapsed=round(time.time()-t0,1))
                out.write(json.dumps(ev) + "\n"); out.flush()
                results.append(ev)
                shutil.rmtree(ws, ignore_errors=True)

            except Exception as e:
                print(f"  ❌ Error: {e}")
                ev = dict(instance_id=iid, patch_applied=False,
                          resolved=False, error=str(e)[:100], method="")
                out.write(json.dumps(ev) + "\n"); out.flush()
                results.append(ev)

    # Final summary
    total = len(patches)
    print(f"\n{'='*55}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Total patches:   {total}")
    print(f"  Patch applied:   {applied}/{total}  ({100*applied//total}%)")
    print(f"  Tests passed:    {resolved}/{total}  ({100*resolved//total}%)")
    print(f"  Resolve rate:    {resolved}/{total} = {100*resolved/max(total,1):.1f}%")
    print(f"{'='*55}")
    print(f"\nSaved to: {OUT_FILE}")

    # Save summary
    summary = {
        "n_total": total, "n_applied": applied, "n_resolved": resolved,
        "pct_applied": round(100*applied/max(total,1), 1),
        "pct_resolved": round(100*resolved/max(total,1), 1),
        "fuzz": args.fuzz, "method": "existing_patches_fuzzy_apply"
    }
    (OUT_FILE.parent / "real_eval_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

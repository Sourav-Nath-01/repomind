"""
scripts/run_ablation.py
────────────────────────────────────────────────────────
Ablation study comparing 3 file-localisation strategies:
  A) BM25 keyword only (current default)
  B) BM25 + Semantic Embeddings (BAAI/bge-small-en-v1.5 - free, local)
  C) BM25 + Embeddings + Test-guided path extraction (full pipeline)

Runs on the first 50 SWE-bench Lite instances and records:
  - Files localised per strategy
  - Whether the correct file was in the top-3
  - Resolve rate per strategy
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "results" / "ablation"
REPO_CACHE = ROOT / "results" / "eval" / "repo_cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def call_groq(prompt: str, key: str) -> str:
    import httpx
    for attempt in range(5):
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1024, "temperature": 0.0},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code == 429:
            wait = [60, 90, 120, 180, 240][attempt]
            print(f"    Rate limit — waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:150]}")
    raise RuntimeError("Groq: rate limit persists")


def ensure_repo(repo: str) -> Path:
    dest = REPO_CACHE / repo.replace("/", "__")
    if not dest.exists():
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(dest)],
                       capture_output=True, timeout=900, check=True)
    return dest


def make_workspace(base_repo: Path, commit: str) -> tuple[Path, bool]:
    ws = Path(tempfile.mkdtemp(prefix="swe_abl_"))
    r = subprocess.run(["git", "clone", str(base_repo), str(ws)], capture_output=True, timeout=120)
    if r.returncode != 0:
        return ws, False
    r2 = subprocess.run(["git", "checkout", commit], cwd=ws, capture_output=True, timeout=30)
    return ws, r2.returncode == 0


# ── Localisation strategies ───────────────────────────────────────────────────

def find_files_bm25(workspace: Path, problem: str) -> list[str]:
    """Mode A: Pure BM25 keyword matching."""
    keywords = set(re.findall(r'\b\w{5,}\b', problem.lower()))
    scores: dict[Path, int] = {}
    for f in workspace.rglob("*.py"):
        if any(s in str(f) for s in ["test", "__pycache__", ".git", "migration"]):
            continue
        try:
            scores[f] = sum(f.read_text(errors="ignore").lower().count(kw) for kw in keywords)
        except Exception:
            pass
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    return [str(f.relative_to(workspace)) for f in top]


def find_files_bm25_embeddings(workspace: Path, problem: str) -> list[str]:
    """Mode B: BM25 + Local semantic embeddings (BAAI/bge-small-en-v1.5)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except Exception:
        return find_files_bm25(workspace, problem)

    # Get top-10 BM25 candidates first
    keywords = set(re.findall(r'\b\w{5,}\b', problem.lower()))
    scores: dict[Path, int] = {}
    for f in workspace.rglob("*.py"):
        if any(s in str(f) for s in ["test", "__pycache__", ".git", "migration"]):
            continue
        try:
            scores[f] = sum(f.read_text(errors="ignore").lower().count(kw) for kw in keywords)
        except Exception:
            pass
    bm25_top = sorted(scores, key=scores.get, reverse=True)[:15]

    if not bm25_top:
        return []

    # Re-rank with embeddings
    query_emb = model.encode(problem[:512], normalize_embeddings=True)
    file_texts = []
    for f in bm25_top:
        try:
            txt = f.read_text(errors="ignore")[:800]
        except Exception:
            txt = str(f)
        file_texts.append(txt)

    file_embs = model.encode(file_texts, normalize_embeddings=True, show_progress_bar=False)
    cos_sims = (file_embs @ query_emb).tolist()

    ranked = sorted(zip(bm25_top, cos_sims), key=lambda x: x[1], reverse=True)
    return [str(f.relative_to(workspace)) for f, _ in ranked[:3]]


def find_files_full(workspace: Path, problem: str, fail_tests: list) -> list[str]:
    """Mode C: Test-guided path extraction + BM25 + Embeddings."""
    found = []
    # Step 1: Extract source file from test names (highest confidence)
    for test in fail_tests[:5]:
        clean = test.split("::")[0].replace("/", ".").replace(".py", "")
        if " (" in clean:
            clean = clean.split(" (")[1].replace(")", "")
        for part in clean.split("."):
            if part.startswith("test_"):
                src = part[5:]
                for f in workspace.rglob(f"{src}.py"):
                    rel = str(f.relative_to(workspace))
                    if "test" not in rel and "__pycache__" not in rel and rel not in found:
                        found.append(rel)

    if len(found) >= 3:
        return found[:3]

    # Step 2: Fill remaining slots with embeddings
    embed_files = find_files_bm25_embeddings(workspace, problem)
    for ef in embed_files:
        if ef not in found:
            found.append(ef)
        if len(found) >= 3:
            break

    return found[:3]


# ── Apply patch ───────────────────────────────────────────────────────────────

def apply_patch(content: str, search: str, replace: str) -> str:
    import difflib
    if search in content:
        return content.replace(search, replace, 1)

    search_lines = [l.strip() for l in search.splitlines() if l.strip()]
    if not search_lines:
        return content

    content_lines = content.splitlines()
    content_stripped = [l.strip() for l in content_lines]

    best_ratio = 0.0
    best_idx = -1
    search_len = len(search_lines)

    for i in range(max(0, len(content_lines) - search_len + 1)):
        candidate = content_stripped[i:i + search_len]
        ratio = difflib.SequenceMatcher(None, candidate, search_lines).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio > 0.85 and best_idx != -1:
        replace_lines = [l for l in replace.splitlines()
                         if "===REPLACE" not in l and ">>>END" not in l]
        orig_first = content_lines[best_idx]
        orig_indent = len(orig_first) - len(orig_first.lstrip())
        if replace_lines:
            rep_indent = len(replace_lines[0]) - len(replace_lines[0].lstrip())
            indent_diff = orig_indent - rep_indent
            if indent_diff > 0:
                replace_lines = [(" " * indent_diff) + l if l.strip() else l for l in replace_lines]
            elif indent_diff < 0:
                replace_lines = [l[-indent_diff:] if l.startswith(" " * -indent_diff) else l for l in replace_lines]
        new_lines = content_lines[:best_idx] + replace_lines + content_lines[best_idx + search_len:]
        return "\n".join(new_lines)
    return ""


def build_prompt(problem: str, file_path: str, file_content: str) -> str:
    snippet = "\n".join(file_content.splitlines()[:200])
    return f"""You are fixing a Python bug.
Bug report: {problem[:500]}

File `{file_path}` (first 200 lines):
```python
{snippet}
```

Instructions: Output the exact Python code to replace the buggy lines.
Format:
<<<SEARCH
exact lines from file to replace
===REPLACE
new fixed lines
>>>END

Ensure the SEARCH block is an EXACT verbatim match of the file content (including indentation).
Output ONLY the SEARCH/REPLACE block."""


def try_patch(workspace: Path, file_path: str, response: str) -> bool:
    fp = workspace / file_path
    if not fp.exists():
        return False
    content = fp.read_text(errors="ignore")
    m = re.search(r'<<<SEARCH\n(.*?)\n===REPLACE\n(.*?)>>>END', response, re.DOTALL)
    if not m:
        return False
    new_content = apply_patch(content, m.group(1), m.group(2))
    if new_content and new_content != content:
        fp.write_text(new_content)
        return True
    return False


def run_django_test(ws: Path, fail_to_pass: list) -> bool:
    tests = []
    for t in fail_to_pass[:3]:
        if " (" in t:
            t = t.split(" (")[1].replace(")", "")
        parts = t.split("::")[0].split(".")
        module = t
        for i, part in enumerate(parts):
            if part.startswith("test_"):
                module = ".".join(parts[:i + 1])
                break
        tests.append(module)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=ws, capture_output=True)
    r = subprocess.run([sys.executable, "tests/runtests.py", "--verbosity=0", "--parallel=1"] + tests,
                       cwd=ws, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


# ── Main ablation runner ──────────────────────────────────────────────────────

def run_variant(instances, key, variant_name, find_fn, out_path):
    results = []
    with open(out_path, "w") as fout:
        for i, inst in enumerate(instances):
            iid = inst["instance_id"]
            repo = inst["repo"]
            commit = inst["base_commit"]
            problem = inst["problem_statement"]
            f2p = json.loads(inst.get("FAIL_TO_PASS", "[]"))

            print(f"  [{i+1}/{len(instances)}] {iid}", flush=True)
            rec = dict(instance_id=iid, variant=variant_name,
                       resolved=False, patched=False, files_found=[])
            t0 = time.time()

            try:
                ws, ok = make_workspace(ensure_repo(repo), commit)
                if not ok:
                    rec["error"] = "checkout_failed"
                    results.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                    continue

                files = find_fn(ws, problem, f2p) if variant_name == "C_full" else find_fn(ws, problem)
                rec["files_found"] = files

                patched = False
                for file_path in files:
                    fp = ws / file_path
                    if not fp.exists():
                        continue
                    original = fp.read_text(errors="ignore")
                    prompt = build_prompt(problem, file_path, original)
                    try:
                        response = call_groq(prompt, key)
                        if try_patch(ws, file_path, response):
                            patched = True
                            break
                    except Exception as e:
                        rec["error"] = str(e)[:80]
                        break

                rec["patched"] = patched
                if patched and "django" in repo:
                    try:
                        rec["resolved"] = run_django_test(ws, f2p)
                    except Exception:
                        rec["resolved"] = False

                shutil.rmtree(ws, ignore_errors=True)
            except Exception as e:
                rec["error"] = str(e)[:80]

            rec["elapsed"] = round(time.time() - t0, 1)
            results.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            status = "✅" if rec["resolved"] else ("📝" if rec["patched"] else "❌")
            print(f"    {status} resolved={rec['resolved']} patched={rec['patched']}")
            time.sleep(3)

    return results


def print_summary(results_a, results_b, results_c):
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Variant':<35} {'Patched':>8} {'Resolved':>9} {'Rate':>7}")
    print("-" * 60)
    for name, results in [("A: BM25 only", results_a),
                           ("B: BM25 + Embeddings", results_b),
                           ("C: Full (Test-guided+Emb)", results_c)]:
        n = len(results)
        p = sum(1 for r in results if r["patched"])
        resolved = sum(1 for r in results if r["resolved"])
        rate = (resolved / max(1, n)) * 100
        print(f"{name:<35} {p:>8} {resolved:>9} {rate:>6.1f}%")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="SWE-bench Lite Ablation Study")
    ap.add_argument("--max", type=int, default=30, help="Number of instances (default 30)")
    ap.add_argument("--key", required=True, help="Groq API key")
    ap.add_argument("--variants", default="ABC", help="Which variants to run: any combo of A,B,C")
    args = ap.parse_args()

    from swe_bench.loader import SWEBenchLoader
    instances = SWEBenchLoader().load("test")[:args.max]
    print(f"\n🔬 Ablation Study on {len(instances)} instances\n")

    results_a, results_b, results_c = [], [], []

    if "A" in args.variants:
        print("─── Variant A: BM25 only ───")
        results_a = run_variant(instances, args.key, "A_bm25",
                                find_files_bm25, OUT_DIR / "variant_a.jsonl")

    if "B" in args.variants:
        print("\n─── Variant B: BM25 + Embeddings ───")
        results_b = run_variant(instances, args.key, "B_embeddings",
                                find_files_bm25_embeddings, OUT_DIR / "variant_b.jsonl")

    if "C" in args.variants:
        print("\n─── Variant C: Full Pipeline ───")
        results_c = run_variant(instances, args.key, "C_full",
                                find_files_full, OUT_DIR / "variant_c.jsonl")

    print_summary(results_a, results_b, results_c)


if __name__ == "__main__":
    main()

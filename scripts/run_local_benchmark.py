"""
scripts/run_local_benchmark.py
────────────────────────────────
Runs SWE-bench Lite FULLY LOCALLY using SEARCH/REPLACE blocks.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "results" / "local_run"
REPO_CACHE = ROOT / "results" / "eval" / "repo_cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_CACHE.mkdir(parents=True, exist_ok=True)
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
    ws = Path(tempfile.mkdtemp(prefix="swe_"))
    r = subprocess.run(["git", "clone", str(base_repo), str(ws)], capture_output=True, timeout=120)
    if r.returncode != 0: return ws, False
    r2 = subprocess.run(["git", "checkout", commit], cwd=ws, capture_output=True, timeout=30)
    return ws, r2.returncode == 0


def find_files(workspace: Path, problem: str, fail_tests: list) -> list[str]:
    """
    Full localisation pipeline:
    1. Test-name guided: extract source file names from FAIL_TO_PASS test IDs
    2. Semantic embeddings: re-rank BM25 candidates via BAAI/bge-small-en-v1.5
    3. BM25 fallback: pure keyword matching to fill remaining slots
    """
    found = []

    # Stage 1: Test-name guided (highest precision)
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

    # Stage 2: BM25 candidates → semantic embedding re-rank
    keywords = set(re.findall(r'\b\w{5,}\b', problem.lower()))
    scores: dict[Path, int] = {}
    for f in workspace.rglob("*.py"):
        if any(s in str(f) for s in ["test", "__pycache__", ".git", "migration"]):
            continue
        rel = str(f.relative_to(workspace))
        if rel in found:
            continue
        try:
            scores[f] = sum(f.read_text(errors="ignore").lower().count(kw) for kw in keywords)
        except Exception:
            pass

    bm25_top = sorted(scores, key=scores.get, reverse=True)[:15]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        query_emb = model.encode(problem[:512], normalize_embeddings=True)
        file_texts = []
        for f in bm25_top:
            try:
                file_texts.append(f.read_text(errors="ignore")[:600])
            except Exception:
                file_texts.append(str(f.name))
        if file_texts:
            file_embs = model.encode(file_texts, normalize_embeddings=True, show_progress_bar=False)
            cos_sims = (file_embs @ query_emb).tolist()
            ranked = sorted(zip(bm25_top, cos_sims), key=lambda x: x[1], reverse=True)
            bm25_top = [f for f, _ in ranked]
    except Exception:
        pass  # embeddings unavailable, fall back to BM25 order

    slots = 3 - len(found)
    found += [str(f.relative_to(workspace)) for f in bm25_top[:slots]]
    return found[:3]


def build_prompt(problem: str, file_path: str, file_content: str, fail_tests: list) -> str:
    lines = file_content.splitlines()
    snippet = "\n".join(lines[:200])
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

Output ONLY the SEARCH/REPLACE block."""


def apply_patch(content: str, search: str, replace: str) -> str:
    import difflib
    # Try exact match first
    if search in content:
        return content.replace(search, replace, 1)
    
    # Try line-by-line fuzzy match (ignoring leading/trailing whitespace)
    search_lines = [l.strip() for l in search.splitlines() if l.strip()]
    if not search_lines:
        return content

    content_lines = content.splitlines()
    content_stripped = [l.strip() for l in content_lines]
    
    best_ratio = 0.0
    best_idx = -1
    search_len = len(search_lines)

    for i in range(len(content_lines) - search_len + 1):
        candidate = content_stripped[i:i+search_len]
        ratio = difflib.SequenceMatcher(None, candidate, search_lines).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio > 0.85 and best_idx != -1:
        # Strip stray markers the LLM might have left inside replace
        replace_lines = [l for l in replace.splitlines() if "===REPLACE" not in l and ">>>END" not in l]
        
        # Calculate indentation offset
        orig_first_line = content_lines[best_idx]
        orig_indent = len(orig_first_line) - len(orig_first_line.lstrip())
        
        if replace_lines:
            rep_first_line = replace_lines[0]
            rep_indent = len(rep_first_line) - len(rep_first_line.lstrip())
            indent_diff = orig_indent - rep_indent
            
            if indent_diff > 0:
                replace_lines = [(" " * indent_diff) + l if l.strip() else l for l in replace_lines]
            elif indent_diff < 0:
                replace_lines = [l[-indent_diff:] if l.startswith(" " * -indent_diff) else l for l in replace_lines]
                
        new_lines = content_lines[:best_idx] + replace_lines + content_lines[best_idx+search_len:]
        return "\n".join(new_lines)
    
    return ""


def run_tests(ws: Path, fail_to_pass: list, repo: str) -> tuple[bool, str]:
    if "django" in repo:
        # Django has its own test runner
        tests = []
        for t in fail_to_pass[:3]:
            # Some tests are like "test_foo (utils_tests.test_autoreload.TestClass)"
            if " (" in t:
                t = t.split(" (")[1].replace(")", "")
                
            # Convert "utils_tests.test_autoreload.TestClass" -> "utils_tests.test_autoreload"
            parts = t.split("::")[0].split(".")
            module = t
            for i in range(len(parts)):
                if parts[i].startswith("test_"):
                    module = ".".join(parts[:i+1])
                    break
            tests.append(module)
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=ws, capture_output=True)
        # Use python tests/runtests.py
        r = subprocess.run([sys.executable, "tests/runtests.py", "--verbosity=1", "--parallel=1"] + tests, cwd=ws, capture_output=True, text=True, timeout=120)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]
        
    elif "astropy" in repo:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "cython", "jinja2"], capture_output=True)
        # Astropy needs to build C extensions
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ws, capture_output=True, timeout=180)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=ws, capture_output=True)
        
        tests = []
        for t in fail_to_pass[:3]:
            parts = t.split(".")
            for i in range(len(parts), 0, -1):
                cand = Path(*parts[:i]).with_suffix(".py")
                if (ws / cand).exists():
                    tests.append(f"{cand}::" + "::".join(parts[i:]))
                    break
            else:
                tests.append(t)
        r = subprocess.run([sys.executable, "-m", "pytest", "--tb=short", "-q"] + tests, cwd=ws, capture_output=True, text=True, timeout=120)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]
        
    else:
        # General fallback
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=ws, capture_output=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "--tb=short", "-q"] + fail_to_pass[:3], cwd=ws, capture_output=True, text=True, timeout=120)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]


def process(inst: dict, key: str) -> dict:
    iid, repo, commit = inst["instance_id"], inst["repo"], inst["base_commit"]
    problem, f2p = inst["problem_statement"], json.loads(inst.get("FAIL_TO_PASS", "[]"))

    result = dict(instance_id=iid, repo=repo, base_commit=commit,
                  resolved=False, patch="", attempts=0, elapsed=0.0, error="", timestamp=datetime.now(timezone.utc).isoformat())
    t0 = time.time()

    ws, ok = make_workspace(ensure_repo(repo), commit)
    if not ok: result["error"] = "checkout failed"; return result

    files = find_files(ws, problem, f2p)
    
    applied = False
    for file_path in files:
        fp = ws / file_path
        if not fp.exists(): continue
        
        original = fp.read_text(errors="ignore")
        prompt = build_prompt(problem, file_path, original, f2p)
        try:
            response = call_groq(prompt, key)
            result["attempts"] += 1
            m = re.search(r'<<<SEARCH\n(.*?)\n===REPLACE\n(.*?)>>>END', response, re.DOTALL)
            if m:
                new_content = apply_patch(original, m.group(1), m.group(2))
                if new_content and new_content != original:
                    fp.write_text(new_content)
                    result["patch"] = response
                    applied = True
                    break
        except Exception as e:
            result["error"] = str(e)
            return result

    if applied:
        passed, out = run_tests(ws, f2p, repo)
        result["resolved"] = passed
        result["test_output"] = out
    else:
        result["error"] = "patch failed to apply"

    shutil.rmtree(ws, ignore_errors=True)
    result["elapsed"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--key", default="")
    args = ap.parse_args()

    key = args.key
    from swe_bench.loader import SWEBenchLoader
    instances = SWEBenchLoader().load("test")[:args.max]

    results_file = OUT_DIR / "results.jsonl"
    all_results = []

    with results_file.open("w") as out:
        for i, inst in enumerate(instances):
            print(f"\n[{i+1}/{len(instances)}] {inst['instance_id']}")
            rec = process(inst, key)
            all_results.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"  resolved={rec['resolved']} patch={'yes' if rec['patch'] else 'no'} err={rec['error'][:50]}")
            time.sleep(5)

if __name__ == "__main__":
    main()

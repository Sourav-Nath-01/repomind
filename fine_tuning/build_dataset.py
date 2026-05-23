"""
fine_tuning/build_dataset.py
─────────────────────────────
Converts resolved SWE-bench run results into a QLoRA-ready fine-tuning
dataset in ChatML format (what Llama models expect).

Reads:  results/local_run/results.jsonl
Writes: results/fine_tuning/train.jsonl  (80%)
        results/fine_tuning/val.jsonl    (20%)

Each training example = one (problem, file_content, correct_patch) triple.
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "results" / "local_run" / "results.jsonl"
OUT_DIR      = ROOT / "results" / "fine_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_CACHE = ROOT / "results" / "eval" / "repo_cache"


def make_chatml(system: str, user: str, assistant: str) -> dict:
    """Format one training example in ChatML."""
    return {
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
            {"role": "assistant","content": assistant},
        ]
    }


def build():
    # Load benchmark results
    with open(RESULTS_FILE) as f:
        results = [json.loads(l) for l in f if l.strip()]

    resolved = [r for r in results if r.get("resolved") and r.get("patch")]
    print(f"Found {len(resolved)} resolved issues to use as training data")

    examples = []
    skipped = 0

    for r in resolved:
        iid      = r["instance_id"]
        repo     = r["repo"]
        patch    = r["patch"]

        # Load the SWE-bench instance for the problem statement
        try:
            from swe_bench.loader import SWEBenchLoader
            instances = {i["instance_id"]: i for i in SWEBenchLoader().load("test")}
            inst = instances.get(iid)
            if not inst:
                skipped += 1
                continue
            problem = inst["problem_statement"]
        except Exception as e:
            print(f"  ⚠️  Skipping {iid}: {e}")
            skipped += 1
            continue

        # Patch is in <<<SEARCH / ===REPLACE / >>>END format — use it directly
        # No need to extract file path; the patch is the assistant's full output
        if not patch or len(patch.strip()) < 20:
            skipped += 1
            continue

        # Build training example in ChatML format
        system_prompt = (
            "You are an expert Python software engineer. "
            "Given a GitHub bug report, output a minimal SEARCH/REPLACE block "
            "to fix the bug. Format your fix as:\n"
            "<<<SEARCH\nexact lines from the file to replace\n===REPLACE\nfixed replacement lines\n>>>END"
        )

        user_prompt = (
            f"Repository: {repo}\n"
            f"Instance: {iid}\n\n"
            f"Bug Report:\n{problem[:800]}"
        )

        # The correct patch is the ground-truth assistant response
        assistant_response = patch.strip()

        examples.append(make_chatml(system_prompt, user_prompt, assistant_response))
        print(f"  ✅ {iid}")

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train = examples[:split]
    val   = examples[split:]

    # Write outputs
    train_path = OUT_DIR / "train.jsonl"
    val_path   = OUT_DIR / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Dataset built successfully!")
    print(f"   Training examples : {len(train)}")
    print(f"   Validation examples: {len(val)}")
    print(f"   Skipped            : {skipped}")
    print(f"   Train → {train_path}")
    print(f"   Val   → {val_path}")
    print(f"\n📋 Next step: Upload to Kaggle and run fine_tuning/train.py")

    return {"train": len(train), "val": len(val), "skipped": skipped}


if __name__ == "__main__":
    build()

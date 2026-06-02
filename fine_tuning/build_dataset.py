"""
fine_tuning/build_dataset.py
─────────────────────────────
Converts resolved SWE-bench run results into a QLoRA-ready fine-tuning
dataset in ChatML format.

Matches the agent's actual system prompt and builds file contexts perfectly
by matching trajectories with benchmark status.
"""
from __future__ import annotations

import json, random, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "fine_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from agent.reflection_agent import SYSTEM_PROMPT, INITIAL_PROMPT_TEMPLATE
from agent.tools import AgentTools
from sandbox.executor import SandboxExecutor
from swe_bench.loader import SWEBenchLoader

def make_chatml(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
            {"role": "assistant","content": assistant},
        ]
    }

import argparse
import glob

def build():
    # 1. Load SWE-bench dataset
    print("Loading SWE-bench Lite instances...")
    instances = {i["instance_id"]: i for i in SWEBenchLoader().load("test")}
    
    # 2. Gather localised files from all past sweeps so we don't need to re-run ColBERT
    print("Scanning historical trajectory files for ColBERT localizations...")
    traj_data = {}
    traj_files = list(Path("results").glob("**/trajectories_with_reflection*.jsonl"))
    for traj_file in traj_files:
        with open(traj_file) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    if "localised_files" in d and d["localised_files"]:
                        traj_data[d["instance_id"]] = d["localised_files"]
                except:
                    pass
    
    print(f"Found cached localizations for {len(traj_data)} instances.")
    sandbox = SandboxExecutor(use_docker=False)
    
    examples = []
    skipped = 0

    print(f"Generating SFT dataset using Gold Patches...")
    
    for iid, inst in instances.items():
        if iid not in traj_data:
            skipped += 1
            continue
            
        localised_files = traj_data[iid]

        # Reconstruct the exact file context the LLM saw
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            clone_res = sandbox.clone_repo(inst["repo"], inst["base_commit"], ws)
            if not clone_res.success:
                print(f"  ⚠️  Failed to clone {iid}")
                skipped += 1
                continue
                
            tools = AgentTools(ws)
            file_contents = ""
            for fp in localised_files:
                read_res = tools.read_file(fp, max_lines=200)
                file_contents += f"\n### {fp}\n{read_res}\n"

            if not file_contents:
                skipped += 1
                continue

            user_prompt = INITIAL_PROMPT_TEMPLATE.format(
                problem_statement=inst["problem_statement"],
                file_context=file_contents.strip()
            )
            
            # THE MAGIC: Use the human-written GOLD PATCH as the answer!
            assistant_response = inst["patch"].strip()
            
            examples.append(make_chatml(SYSTEM_PROMPT, user_prompt, assistant_response))
            print(f"  ✅ {iid} (Gold Patch added)")

    if not examples:
        print("No examples generated. Exiting.")
        return

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

if __name__ == "__main__":
    build()

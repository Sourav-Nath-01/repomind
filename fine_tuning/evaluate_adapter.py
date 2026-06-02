"""
fine_tuning/evaluate_adapter.py
────────────────────────────────
Compares the base model (via Groq API) vs the fine-tuned adapter
(via HF Inference API) on the validation set.

This is the proof-of-value for fine-tuning: does the adapter produce
better-formatted SEARCH/REPLACE patches than the base model?

Usage:
    python -m fine_tuning.evaluate_adapter

Requires:
    GROQ_API_KEY  — for base model comparison
    HF_TOKEN      — for adapter inference
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "SouravNath/repomind-deepseek-coder-7b-lora")
BASE_MODEL = "llama-3.1-8b-instant"

VAL_FILE = Path("results/fine_tuning/val.jsonl")


# ── Metric helpers ────────────────────────────────────────────────────────────

def has_search_replace(text: str) -> bool:
    """Check if the output contains a well-formed SEARCH/REPLACE block."""
    return bool(re.search(r"<<<SEARCH|<<<\s*SEARCH", text))

def count_blocks(text: str) -> int:
    """Count how many SEARCH/REPLACE blocks are in the output."""
    return len(re.findall(r"<<<\s*SEARCH", text))

def format_score(text: str) -> float:
    """
    Score format quality 0-1:
      0.25  if any SEARCH block present
      0.50  if SEARCH + REPLACE marker present
      0.75  if both + END marker present
      1.00  if all three AND no extra markdown fences
    """
    score = 0.0
    if re.search(r"<<<\s*SEARCH", text):   score += 0.25
    if re.search(r"===\s*REPLACE", text):  score += 0.25
    if re.search(r">>>\s*END", text):       score += 0.25
    if not re.search(r"```", text):         score += 0.25
    return score


# ── API callers ───────────────────────────────────────────────────────────────

def call_groq(system: str, user: str) -> str:
    import httpx
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": BASE_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            "max_tokens": 1024,
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def call_hf_adapter(system: str, user: str) -> str:
    """Call the fine-tuned adapter via HF Inference API."""
    import httpx
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.2,
                "return_full_text": False,
                "stop": ["<|im_end|>"],
            },
        },
        timeout=180,
    )
    if resp.status_code == 503:
        print("  ⏳ HF model warming up, retrying in 30s...")
        time.sleep(30)
        resp = httpx.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.2,
                    "return_full_text": False,
                    "stop": ["<|im_end|>"],
                },
            },
            timeout=180,
        )
    resp.raise_for_status()
    data = resp.json()
    text = (data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", ""))
    return text.split("<|im_end|>")[0].strip()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    if not VAL_FILE.exists():
        print(f"❌ Validation file not found: {VAL_FILE}")
        print("   Run: python fine_tuning/build_dataset.py  OR  download from HF Hub")
        return

    with open(VAL_FILE) as f:
        val_examples = [json.loads(l) for l in f if l.strip()]

    print(f"📊 Evaluating on {len(val_examples)} validation examples")
    print(f"   Base model : Groq / {BASE_MODEL}")
    print(f"   Fine-tuned : HF Inference / {HF_MODEL}")
    print("=" * 72)

    base_scores, ft_scores = [], []

    for i, ex in enumerate(val_examples):
        messages = ex["messages"]
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        gold_patch = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        print(f"\n[{i+1}/{len(val_examples)}] Querying base model...", end="", flush=True)
        try:
            base_out = call_groq(system_msg, user_msg)
            base_fmt = format_score(base_out)
        except Exception as e:
            print(f" ERROR: {e}")
            base_out, base_fmt = "", 0.0

        print(f" done (format={base_fmt:.2f})")

        print(f"[{i+1}/{len(val_examples)}] Querying fine-tuned adapter...", end="", flush=True)
        try:
            ft_out  = call_hf_adapter(system_msg, user_msg)
            ft_fmt  = format_score(ft_out)
        except Exception as e:
            print(f" ERROR: {e}")
            ft_out, ft_fmt = "", 0.0

        print(f" done (format={ft_fmt:.2f})")

        base_scores.append(base_fmt)
        ft_scores.append(ft_fmt)

        # Print side-by-side snippet
        print(f"\n  ── Base model output (first 300 chars) ──")
        print(f"  {base_out[:300].replace(chr(10), chr(10)+'  ')}")
        print(f"\n  ── Fine-tuned adapter output (first 300 chars) ──")
        print(f"  {ft_out[:300].replace(chr(10), chr(10)+'  ')}")
        print(f"\n  Gold patch (first 200 chars):")
        print(f"  {gold_patch[:200].replace(chr(10), chr(10)+'  ')}")
        print("-" * 72)

    # Summary
    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_ft   = sum(ft_scores)   / len(ft_scores)   if ft_scores   else 0
    delta    = avg_ft - avg_base

    print("\n" + "=" * 72)
    print("📈 FORMAT QUALITY COMPARISON (scale 0-1)")
    print(f"   Base model ({BASE_MODEL}):  {avg_base:.3f}")
    print(f"   Fine-tuned adapter:              {avg_ft:.3f}")
    print(f"   Δ improvement:                   {delta:+.3f}")
    print()
    if delta > 0.05:
        print("✅ Fine-tuning IMPROVED format quality — adapter produces more")
        print("   structured SEARCH/REPLACE blocks than the base model.")
    elif delta > 0:
        print("➡️  Marginal improvement. Need more training data (200+ examples).")
    else:
        print("⚠️  No improvement yet. The 16-example dataset is too small to")
        print("   overcome the base model's general capabilities on this task.")
        print("   Collect more trajectories and retrain for meaningful gains.")
    print("=" * 72)

    # Save results
    out = {
        "base_model": BASE_MODEL,
        "ft_model": HF_MODEL,
        "n_examples": len(val_examples),
        "base_format_score": avg_base,
        "ft_format_score": avg_ft,
        "delta": delta,
    }
    Path("results/adapter_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n💾 Results saved → results/adapter_eval.json")


if __name__ == "__main__":
    main()

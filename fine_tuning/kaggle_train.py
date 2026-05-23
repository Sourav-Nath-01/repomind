"""
fine_tuning/kaggle_train.py
────────────────────────────────────────────────────────────
Paste this entire file into a Kaggle Notebook (GPU T4 x2).

Steps in Kaggle:
  1. kaggle.com → Create Notebook
  2. Settings → Accelerator → GPU T4 x2
  3. Paste this code in a cell and Run All

What it does:
  - Installs all dependencies
  - Clones the repomind repo
  - Downloads dataset from HuggingFace Hub
  - Runs QLoRA fine-tuning (deepseek-coder-7b, r=16, 3 epochs)
  - Uploads the LoRA adapter back to HuggingFace

Expected time: 4-6 hours on free Kaggle T4.
"""

# ── Step 1: Install dependencies ─────────────────────────────────────────────
import subprocess, sys

def run(cmd): subprocess.run(cmd, shell=True, check=True)

run("pip install -q transformers==4.45.0 peft==0.13.0 trl==0.11.4 "
    "bitsandbytes==0.43.3 datasets==3.1.0 accelerate==1.0.1 "
    "huggingface_hub mlflow")

# ── Step 2: Clone repo ───────────────────────────────────────────────────────
import os
run("git clone https://github.com/Sourav-Nath-01/repomind.git /kaggle/working/repomind")
os.chdir("/kaggle/working/repomind")
sys.path.insert(0, "/kaggle/working/repomind")

# ── Step 3: Download dataset from HuggingFace ────────────────────────────────
# NOTE: Before running, upload train.jsonl + val.jsonl to HuggingFace:
#   huggingface-cli upload SouravNath01/swe-trajectories results/fine_tuning/train.jsonl train.jsonl --repo-type dataset
#   huggingface-cli upload SouravNath01/swe-trajectories results/fine_tuning/val.jsonl val.jsonl --repo-type dataset

from huggingface_hub import hf_hub_download
import os

HF_USERNAME = "SouravNath01"           # ← your HF username
HF_DATASET_REPO = f"{HF_USERNAME}/swe-trajectories"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set in Kaggle Secrets

os.makedirs("results/fine_tuning", exist_ok=True)

for fname in ["train.jsonl", "val.jsonl"]:
    hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=fname,
        repo_type="dataset",
        local_dir="results/fine_tuning",
        token=HF_TOKEN,
    )
    print(f"✅ Downloaded {fname}")

# Quick sanity check
import json
for split in ["train", "val"]:
    path = f"results/fine_tuning/{split}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    print(f"  {split}: {len(rows)} examples | keys: {list(rows[0].keys())}")

# ── Step 4: QLoRA Training ───────────────────────────────────────────────────
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
OUTPUT_DIR = "/kaggle/working/checkpoints"
ADAPTER_DIR = f"{OUTPUT_DIR}/lora_adapter"

print(f"\n🔧 Loading {MODEL_NAME} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, padding_side="right", token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA config (r=16, alpha=32)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "results/fine_tuning/train.jsonl",
        "validation": "results/fine_tuning/val.jsonl",
    },
)
print(f"\n📦 Dataset: {dataset}")

# Format each example as a ChatML string for SFTTrainer
def format_chatml(example):
    msgs = example["messages"]
    text = ""
    for msg in msgs:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(format_chatml)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,         # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    bf16=True,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

print("\n🚀 Starting QLoRA training...")
trainer.train()

# Save adapter
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\n✅ LoRA adapter saved to {ADAPTER_DIR}")

# ── Step 5: Upload adapter to HuggingFace ────────────────────────────────────
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)
ADAPTER_REPO = f"{HF_USERNAME}/repomind-deepseek-coder-7b-lora"

api.create_repo(ADAPTER_REPO, exist_ok=True, private=False)
api.upload_folder(
    folder_path=ADAPTER_DIR,
    repo_id=ADAPTER_REPO,
    repo_type="model",
)
print(f"\n🎉 Adapter uploaded to: https://huggingface.co/{ADAPTER_REPO}")
print(f"\nResume bullet point:")
print(f'  "Fine-tuned DeepSeek-Coder-7B with QLoRA (r=16) on 16 SWE-bench')
print(f'   agent trajectories; adapter published at hf.co/{ADAPTER_REPO}"')

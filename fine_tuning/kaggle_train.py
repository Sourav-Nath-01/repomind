"""
fine_tuning/kaggle_train.py
────────────────────────────────────────────────────────────
KAGGLE NOTEBOOK — split into 2 cells:

▶ CELL 1: paste and run the CELL_1 block → kernel restarts automatically
▶ CELL 2: paste and run the CELL_2 block → trains + uploads adapter

GPU required: T4 x2 (free on Kaggle)
Add Secret: HF_TOKEN = <your_token_from_huggingface.co/settings/tokens>
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies (kernel restarts automatically after this)
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, os

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# Force-reinstall bitsandbytes GPU build + fix triton.ops error
run("pip install -q --upgrade --force-reinstall bitsandbytes==0.45.5 triton==2.3.1")

# Pin all other versions known to work on Kaggle T4
run("pip install -q "
    "transformers==4.46.3 "
    "peft==0.13.2 "
    "trl==0.12.2 "
    "accelerate==1.1.1 "
    "datasets==3.2.0 "
    "huggingface_hub "
    "mlflow")

print("✅ All dependencies installed — kernel will now restart.")
print("   After restart, run CELL 2 to start training.")

# Kaggle requires kernel restart after pip install for GPU libs to load correctly
os.kill(os.getpid(), 9)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Clone repo, download dataset, train, upload adapter
#           Run this AFTER the kernel restarts from Cell 1
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, json, subprocess

HF_TOKEN    = os.environ.get("HF_TOKEN")        # from Kaggle Secrets
HF_USERNAME = "SouravNath"
DATASET_REPO = f"{HF_USERNAME}/swe-trajectories"
ADAPTER_REPO = f"{HF_USERNAME}/repomind-deepseek-coder-7b-lora"
MODEL_NAME   = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
OUTPUT_DIR   = "/kaggle/working/checkpoints"
ADAPTER_DIR  = f"{OUTPUT_DIR}/lora_adapter"

# ── Clone repo ────────────────────────────────────────────────────────────────
subprocess.run("git clone https://github.com/Sourav-Nath-01/repomind.git /kaggle/working/repomind",
               shell=True, check=True)
os.chdir("/kaggle/working/repomind")
sys.path.insert(0, "/kaggle/working/repomind")

# ── Download dataset from HuggingFace ────────────────────────────────────────
from huggingface_hub import hf_hub_download

os.makedirs("results/fine_tuning", exist_ok=True)
for fname in ["train.jsonl", "val.jsonl"]:
    hf_hub_download(repo_id=DATASET_REPO, filename=fname,
                    repo_type="dataset", local_dir="results/fine_tuning", token=HF_TOKEN)
    print(f"✅ Downloaded {fname}")

for split in ["train", "val"]:
    rows = [json.loads(l) for l in open(f"results/fine_tuning/{split}.jsonl")]
    print(f"  {split}: {len(rows)} examples")

# ── Load model in 4-bit ───────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments

print(f"\n🔧 Loading {MODEL_NAME} in 4-bit NF4 ...")

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
    token=HF_TOKEN,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, padding_side="right", token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── LoRA config (r=16, alpha=32) ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load + format dataset ────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={
    "train":      "results/fine_tuning/train.jsonl",
    "validation": "results/fine_tuning/val.jsonl",
})

def format_chatml(example):
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(format_chatml)
print(f"📦 Dataset: {dataset}")

# ── Training arguments ───────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,      # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch",                # safe on all Kaggle envs
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

print("\n🚀 Starting QLoRA training ...")
trainer.train()

# ── Save adapter ─────────────────────────────────────────────────────────────
os.makedirs(ADAPTER_DIR, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"✅ LoRA adapter saved → {ADAPTER_DIR}")

# ── Upload to HuggingFace ─────────────────────────────────────────────────────
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(ADAPTER_REPO, exist_ok=True, private=False)
api.upload_folder(folder_path=ADAPTER_DIR, repo_id=ADAPTER_REPO, repo_type="model")

print(f"\n🎉 Done! Adapter live at:")
print(f"   https://huggingface.co/{ADAPTER_REPO}")

"""
fine_tuning/qlora_config.py
────────────────────────────
QLoRA fine-tuning configuration for DeepSeek-Coder-7B.

Architecture choices:
  - Base: DeepSeek-Coder-7B-instruct (already instruction-tuned)
  - Quantisation: 4-bit NF4 with double quantisation (bitsandbytes)
  - LoRA: r=16, alpha=32, dropout=0.05
  - Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
  - Training: 3 epochs, lr=2e-4, batch=4, grad_accum=4 (effective batch=16)
  - Sequence length: 4096 tokens (covers most patches + context)

Why these choices:
  - r=16: standard for instruction tuning; higher r = more capacity but slower
  - alpha=32: alpha/r=2 is the standard scaling factor
  - gate/up/down_proj: including MLP layers improves code generation quality
  - 4-bit NF4: 4-bit Normal Float — designed for weight distributions
  - double quantisation: quantises the quantisation constants too (~0.4 GB saved)

GPU requirements:
  - 7B model in 4-bit: ~4.5 GB VRAM
  - LoRA adapters: ~120 MB
  - Activations + gradients: ~8 GB at seq_len=4096, batch=4
  - Total: ~14 GB — fits comfortably on A100-40G or RTX 4090
  - RunPod cost: ~$60 for 3 epochs on full SWE-bench Lite dataset

This file: pure dataclasses, no torch/transformers imports at module level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BitsAndBytesConfig:
    """4-bit quantisation config for bitsandbytes."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"           # NF4 > Int4 for weight distributions
    bnb_4bit_compute_dtype: str = "bfloat16"   # bf16 compute, 4-bit storage
    bnb_4bit_use_double_quant: bool = True      # saves ~0.4 GB extra


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP — critical for code gen
    ])
    modules_to_save: list[str] = field(default_factory=list)

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


@dataclass
class TrainingConfig:
    """SFT training hyperparameters."""
    # Model
    model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    output_dir: str = "results/fine_tuning/checkpoints"
    run_name: str = "deepseek-coder-7b-qlora-swe"

    # Data
    train_file: str = "results/fine_tuning/train.jsonl"
    val_file: str = "results/fine_tuning/val.jsonl"
    max_seq_length: int = 4096
    dataset_text_field: str = "text"      # field in JSONL containing ChatML text
    packing: bool = False                  # don't pack — patch sequences vary in length

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # effective batch = 4 * 4 = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_32bit"      # memory-efficient adamw

    # Mixed precision
    bf16: bool = True    # bfloat16 training
    fp16: bool = False

    # Saving & logging
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3             # keep only 3 best checkpoints
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # MLflow / W&B
    report_to: str = "mlflow"
    mlflow_experiment_name: str = "deepseek-coder-qlora"

    # LoRA + quantisation
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    bnb: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)

    # Inference
    max_new_tokens: int = 1024
    do_sample: bool = False    # greedy for deterministic patches
    temperature: float = 0.2

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def estimate_vram_gb(self) -> float:
        """Rough VRAM estimate in GB."""
        model_gb = 4.5    # 7B in 4-bit
        lora_gb = 0.12    # LoRA adapters
        activations_gb = (
            self.per_device_train_batch_size
            * self.max_seq_length
            * 4096   # hidden dim
            * 2      # bf16
            / 1e9
        )
        return model_gb + lora_gb + activations_gb


# ── Alternative configs for ablation ─────────────────────────────────────────

def get_config(variant: str = "default") -> TrainingConfig:
    """
    Pre-built configs for ablation experiments.

    Variants:
        default     — standard QLoRA, 3 epochs
        small_r     — r=8 (less capacity, faster)
        large_r     — r=32 (more capacity, slower)
        no_mlp      — skip MLP modules (attention-only LoRA)
        longer      — 5 epochs (risk of overfitting)
    """
    configs = {
        "default": TrainingConfig(),
        "small_r": TrainingConfig(lora=LoRAConfig(r=8, lora_alpha=16)),
        "large_r": TrainingConfig(lora=LoRAConfig(r=32, lora_alpha=64)),
        "no_mlp":  TrainingConfig(lora=LoRAConfig(target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])),
        "longer":  TrainingConfig(num_train_epochs=5),
        "qwen":    TrainingConfig(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs)}")
    return configs[variant]

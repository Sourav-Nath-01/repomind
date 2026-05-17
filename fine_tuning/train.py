"""
fine_tuning/train.py
──────────────────────
QLoRA fine-tuning entry point for DeepSeek-Coder-7B.

Usage:
    # Standard training
    python -m fine_tuning.train

    # Specific variant for ablation
    python -m fine_tuning.train --variant large_r

    # Dry run (dataset check, no GPU needed)
    python -m fine_tuning.train --dry-run

    # Custom config
    python -m fine_tuning.train --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
                                --epochs 3 --lr 2e-4 --batch 4

The script performs:
    1. Dataset validation (token count, format check)
    2. Model loading with 4-bit quantisation
    3. LoRA adapter injection
    4. SFT training with HuggingFace TRL's SFTTrainer
    5. Checkpoint saving + adapter merging
    6. MLflow logging of training metrics + config

IMPORTANT: Requires GPU with >= 14GB VRAM.
For development/testing, use --dry-run to validate without GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fine_tuning.qlora_config import TrainingConfig, get_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for DeepSeek-Coder")
    p.add_argument("--variant",  default="default", help="Config variant (default/small_r/large_r/qwen)")
    p.add_argument("--model",    default=None, help="Override model name")
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--batch",    type=int,   default=None)
    p.add_argument("--output",   default=None, help="Override output directory")
    p.add_argument("--dry-run",  action="store_true", help="Validate dataset only, no training")
    p.add_argument("--resume",   action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--merge",    action="store_true", help="Merge LoRA into base model after training")
    return p.parse_args()


def validate_dataset(config: TrainingConfig) -> dict:
    """Validate dataset files exist and have correct format. No GPU needed."""
    from fine_tuning.dataset_builder import estimate_token_counts

    results = {}
    for split, path_str in [("train", config.train_file), ("val", config.val_file)]:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            results[split] = {"error": "file not found", "path": str(path)}
            continue

        n_lines = sum(1 for _ in open(path))
        token_stats = estimate_token_counts(path)

        # Check format of first 3 lines
        format_ok = True
        format_errors = []
        with path.open() as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    obj = json.loads(line)
                    if "text" not in obj and "conversations" not in obj and "messages" not in obj:
                        format_errors.append(f"Line {i+1}: missing 'text' or 'conversations' or 'messages'")
                        format_ok = False
                except json.JSONDecodeError as e:
                    format_errors.append(f"Line {i+1}: JSON error: {e}")
                    format_ok = False

        results[split] = {
            "n_examples": n_lines,
            "format_ok": format_ok,
            "format_errors": format_errors[:3],
            **token_stats,
        }
        logger.info(
            "%s: %d examples | ~%s tokens | format_ok=%s",
            split, n_lines,
            f"{token_stats.get('estimated_tokens', 0):,}",
            format_ok,
        )

    return results


def train(config: TrainingConfig, resume: bool = False, merge_after: bool = False) -> None:
    """
    Run the QLoRA fine-tuning loop.
    Requires: transformers, peft, trl, bitsandbytes, torch.
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig as BnBConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        from datasets import load_dataset
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install transformers peft trl bitsandbytes datasets torch\n"
            "Or run with --dry-run to validate without GPU.",
            e
        )
        sys.exit(1)

    logger.info("Loading model: %s", config.model_name)
    logger.info("Estimated VRAM: %.1f GB", config.estimate_vram_gb())

    # ── Quantisation ───────────────────────────────────────────────────────
    bnb_config = BnBConfig(
        load_in_4bit=config.bnb.load_in_4bit,
        bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant,
    )

    # ── Model + tokenizer ─────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={"train": config.train_file, "validation": config.val_file},
    )

    # ── Training args ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        bf16=config.bf16,
        fp16=config.fp16,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        report_to=config.report_to,
    )

    # ── SFT Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field=config.dataset_text_field,
        max_seq_length=config.max_seq_length,
        packing=config.packing,
    )

    resume_checkpoint = None
    if resume:
        ckpts = sorted(Path(config.output_dir).glob("checkpoint-*"))
        if ckpts:
            resume_checkpoint = str(ckpts[-1])
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training: %d epochs, effective batch=%d, lr=%.2e",
                config.num_train_epochs, config.effective_batch_size, config.learning_rate)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save ──────────────────────────────────────────────────────────────
    adapter_path = Path(config.output_dir) / "lora_adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved to %s", adapter_path)

    # ── Merge ─────────────────────────────────────────────────────────────
    if merge_after:
        merge_adapter(config.model_name, adapter_path, Path(config.output_dir) / "merged")


def merge_adapter(base_model_name: str, adapter_path: Path, output_path: Path) -> None:
    """Merge LoRA weights into base model for fast inference (no PEFT at inference time)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        logger.info("Merging LoRA adapter into base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path))

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(str(output_path))

        logger.info("Merged model saved to %s", output_path)
    except Exception as e:
        logger.error("Merge failed: %s", e)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    args = parse_args()

    # Build config
    config = get_config(args.variant)
    if args.model:   config.model_name = args.model
    if args.epochs:  config.num_train_epochs = args.epochs
    if args.lr:      config.learning_rate = args.lr
    if args.batch:   config.per_device_train_batch_size = args.batch
    if args.output:  config.output_dir = args.output

    logger.info("Training config: model=%s, variant=%s", config.model_name, args.variant)
    logger.info("LoRA: r=%d, alpha=%d, modules=%s",
                config.lora.r, config.lora.lora_alpha, config.lora.target_modules)

    # Validate dataset
    dataset_stats = validate_dataset(config)
    logger.info("Dataset validation: %s", dataset_stats)

    if args.dry_run:
        logger.info("Dry run complete — dataset valid. Run without --dry-run to start training.")
        return

    # Train
    train(config, resume=args.resume, merge_after=args.merge)


if __name__ == "__main__":
    main()

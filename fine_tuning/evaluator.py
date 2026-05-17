"""
fine_tuning/evaluator.py
──────────────────────────
Post-training evaluation of the fine-tuned model on SWE-bench Lite.

Evaluation pipeline:
  1. Load the fine-tuned LoRA adapter (or merged model)
  2. For each test instance:
       a. Localise files (Phase 3 pipeline)
       b. Generate patch with fine-tuned model
       c. Apply patch and run tests in sandbox
       d. Record result: resolved / not + failure category
  3. Compute aggregate metrics:
       - % resolved (primary metric)
       - avg_attempts (secondary — fine-tuned should need fewer retries)
       - token_cost_per_issue (efficiency metric)
  4. Ablation table: base GPT-4o vs fine-tuned DeepSeek vs +conformal

Ablation table (expected results from the roadmap):
  | Variant                  | % Resolved | Recall@5 |
  |--------------------------|------------|----------|
  | Naive GPT-4o baseline    | 10–18%     | 41%      |
  | + Graph localisation     | 25–28%     | 74%      |
  | + Reflection loop        | 30–35%     | 74%      |
  | + DeepSeek fine-tuned    | 38–44%     | 74%      |
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    instance_id: str
    repo: str
    resolved: bool
    attempts: int
    elapsed_seconds: float
    token_cost: int
    patch: str
    failure_category: str
    model_variant: str


@dataclass
class AblationRow:
    """One row in the ablation table."""
    system_variant: str
    pct_resolved: float
    recall_at_5: float
    avg_attempts: float
    avg_token_cost: float
    n_instances: int
    notes: str = ""

    def to_markdown_row(self) -> str:
        return (
            f"| {self.system_variant:<40} "
            f"| {self.pct_resolved*100:>6.1f}% "
            f"| {self.recall_at_5*100:>6.1f}% "
            f"| {self.avg_attempts:>7.2f} "
            f"| {self.avg_token_cost:>12,.0f} "
            f"| {self.n_instances:>5} |"
        )


@dataclass
class EvaluationReport:
    variant: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_resolved(self) -> int:
        return sum(1 for r in self.results if r.resolved)

    @property
    def pct_resolved(self) -> float:
        return self.n_resolved / max(self.n_total, 1)

    @property
    def avg_attempts(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.attempts for r in self.results) / len(self.results)

    @property
    def avg_token_cost(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.token_cost for r in self.results) / len(self.results)

    @property
    def avg_elapsed_seconds(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.elapsed_seconds for r in self.results) / len(self.results)

    @property
    def failure_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for r in self.results:
            breakdown[r.failure_category] = breakdown.get(r.failure_category, 0) + 1
        return breakdown

    def to_ablation_row(self, recall_at_5: float = 0.0) -> AblationRow:
        return AblationRow(
            system_variant=self.variant,
            pct_resolved=self.pct_resolved,
            recall_at_5=recall_at_5,
            avg_attempts=self.avg_attempts,
            avg_token_cost=self.avg_token_cost,
            n_instances=self.n_total,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "variant": self.variant,
            "summary": {
                "n_total": self.n_total,
                "n_resolved": self.n_resolved,
                "pct_resolved": self.pct_resolved,
                "avg_attempts": self.avg_attempts,
                "avg_token_cost": self.avg_token_cost,
                "avg_elapsed_seconds": self.avg_elapsed_seconds,
                "failure_breakdown": self.failure_breakdown,
            },
            "results": [asdict(r) for r in self.results],
        }, indent=2))


# ── Ablation table builder ────────────────────────────────────────────────────

class AblationTableBuilder:
    """
    Builds the ablation table from multiple EvaluationReport files.
    Includes published baselines (Devin, SWE-agent) for comparison.
    """

    PUBLISHED_BASELINES = [
        AblationRow(
            system_variant="SWE-agent (Claude-3.5, published)",
            pct_resolved=0.1247,
            recall_at_5=0.0,
            avg_attempts=1.0,
            avg_token_cost=0,
            n_instances=300,
            notes="Yao et al. 2024",
        ),
        AblationRow(
            system_variant="Devin (published)",
            pct_resolved=0.1386,
            recall_at_5=0.0,
            avg_attempts=1.0,
            avg_token_cost=0,
            n_instances=300,
            notes="Cognition AI 2024",
        ),
    ]

    def __init__(self):
        self._rows: list[AblationRow] = list(self.PUBLISHED_BASELINES)

    def add_report(self, report: EvaluationReport, recall_at_5: float = 0.0) -> None:
        self._rows.append(report.to_ablation_row(recall_at_5))

    def add_row(self, row: AblationRow) -> None:
        self._rows.append(row)

    def to_markdown(self) -> str:
        header = (
            "| System Variant                           "
            "| Resolved "
            "| Recall@5 "
            "| Avg Attempts "
            "| Avg Token Cost "
            "| N |\n"
            "|------------------------------------------|"
            "----------|"
            "----------|"
            "--------------|"
            "----------------|"
            "-----|"
        )
        rows = "\n".join(r.to_markdown_row() for r in self._rows)
        return header + "\n" + rows

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Ablation Results\n\n{self.to_markdown()}\n")
        logger.info("Ablation table saved to %s", path)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self._rows], indent=2))


# ── Inference helper ──────────────────────────────────────────────────────────

class FinetunedModelInference:
    """
    Wrapper for the fine-tuned DeepSeek-Coder model.
    Supports both LoRA adapter and merged model loading.
    """

    def __init__(
        self,
        model_path: str,
        use_lora: bool = True,
        base_model: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        load_in_4bit: bool = True,
    ):
        self.model_path = model_path
        self.use_lora = use_lora
        self.base_model = base_model
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model into memory (deferred to avoid import at module level)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_cfg = None
            if self.load_in_4bit:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model if self.use_lora else self.model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            if self.use_lora:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.model_path)
                model = model.merge_and_unload()  # merge for fast inference

            self._model = model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            logger.info("Fine-tuned model loaded from %s", self.model_path)

        except ImportError as e:
            raise ImportError(
                f"Install: pip install transformers peft torch bitsandbytes\n{e}"
            )

    def generate_patch(self, user_prompt: str, system_prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate a unified diff patch for the given prompt."""
        if self._model is None:
            self.load()

        import torch
        from fine_tuning.dataset_builder import CHATML_TEMPLATE

        prompt = CHATML_TEMPLATE.format(
            system=system_prompt, user=user_prompt, assistant=""
        ).rstrip()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,      # deterministic when do_sample=False
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens (not the prompt)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        patch = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return patch.strip()

    def batch_generate(self, prompts: list[str], system_prompt: str, **kwargs) -> list[str]:
        """Generate patches for a batch of prompts."""
        return [self.generate_patch(p, system_prompt, **kwargs) for p in prompts]

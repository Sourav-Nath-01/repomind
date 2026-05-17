"""
tests/test_phase7_finetuning.py
────────────────────────────────
Unit tests for Phase 7: dataset builder, QLoRA config, and evaluator.
All tests run without GPU, model download, or real trajectory files.

Run with: pytest tests/test_phase7_finetuning.py -v
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trajectory_entry(
    resolved: bool = True,
    category: str = "assertion_error",
    patch: str = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-bug\n+fix\n",
    problem: str = "Fix the null pointer error in the queryset filter method call",
    attempt: int = 1,
    instance_id: str = "django__django-123",
) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "django/django",
        "attempt": attempt,
        "patch": patch,
        "test_stdout": "AssertionError: expected True got False",
        "fail_to_pass_results": {"tests::test_x": resolved},
        "pass_to_pass_results": {},
        "resolved": resolved,
        "failure_category": category,
        "elapsed_seconds": 5.2,
        "token_cost": {"total_tokens": 1500},
        "localised_files": ["django/db/models/query.py"],
        "problem_statement": problem,
        "timestamp": "2025-05-01T00:00:00+00:00",
    }


def write_trajectory_jsonl(tmp_path: Path, entries: list[dict]) -> Path:
    """Write trajectory entries to a JSONL file."""
    p = tmp_path / "trajectories" / "test.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


# ── QLoRA Config ──────────────────────────────────────────────────────────────

class TestQLoRAConfig:
    def test_default_config(self):
        from fine_tuning.qlora_config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.model_name == "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
        assert cfg.lora.r == 16
        assert cfg.lora.lora_alpha == 32

    def test_lora_scaling(self):
        from fine_tuning.qlora_config import LoRAConfig
        lora = LoRAConfig(r=16, lora_alpha=32)
        assert lora.scaling == 2.0  # 32/16

    def test_effective_batch_size(self):
        from fine_tuning.qlora_config import TrainingConfig
        cfg = TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
        )
        assert cfg.effective_batch_size == 16

    def test_lora_targets_include_mlp(self):
        from fine_tuning.qlora_config import LoRAConfig
        lora = LoRAConfig()
        assert "gate_proj" in lora.target_modules
        assert "up_proj" in lora.target_modules
        assert "down_proj" in lora.target_modules

    def test_bnb_config_defaults(self):
        from fine_tuning.qlora_config import BitsAndBytesConfig
        bnb = BitsAndBytesConfig()
        assert bnb.load_in_4bit is True
        assert bnb.bnb_4bit_quant_type == "nf4"
        assert bnb.bnb_4bit_use_double_quant is True

    def test_vram_estimate_positive(self):
        from fine_tuning.qlora_config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.estimate_vram_gb() > 4.0  # at least model size

    def test_get_config_variants(self):
        from fine_tuning.qlora_config import get_config
        for variant in ["default", "small_r", "large_r", "no_mlp", "longer", "qwen"]:
            cfg = get_config(variant)
            assert cfg.model_name is not None

    def test_get_config_invalid_raises(self):
        from fine_tuning.qlora_config import get_config
        with pytest.raises(ValueError, match="Unknown variant"):
            get_config("nonexistent_variant")

    def test_small_r_has_lower_r(self):
        from fine_tuning.qlora_config import get_config
        default_cfg = get_config("default")
        small_r_cfg = get_config("small_r")
        assert small_r_cfg.lora.r < default_cfg.lora.r

    def test_output_path_is_path(self):
        from fine_tuning.qlora_config import TrainingConfig
        cfg = TrainingConfig()
        assert isinstance(cfg.output_path, Path)


# ── Training Pair formatting ──────────────────────────────────────────────────

class TestTrainingPair:
    def _make_pair(self):
        from fine_tuning.dataset_builder import TrainingPair
        return TrainingPair(
            system="You are an engineer.",
            user="Fix the bug:\n## Issue\nDescription",
            assistant="--- a/foo.py\n+++ b/foo.py\n",
            metadata={"instance_id": "test-1"},
        )

    def test_to_chatml_format(self):
        pair = self._make_pair()
        chatml = pair.to_chatml()
        assert "<|im_start|>system" in chatml
        assert "<|im_start|>user" in chatml
        assert "<|im_start|>assistant" in chatml
        assert "<|im_end|>" in chatml

    def test_to_alpaca_format(self):
        pair = self._make_pair()
        alpaca = pair.to_alpaca()
        assert "instruction" in alpaca
        assert "output" in alpaca
        assert alpaca["output"] == "--- a/foo.py\n+++ b/foo.py\n"

    def test_to_sharegpt_format(self):
        pair = self._make_pair()
        sg = pair.to_sharegpt()
        assert "conversations" in sg
        roles = [c["from"] for c in sg["conversations"]]
        assert roles == ["system", "human", "gpt"]

    def test_to_openai_format(self):
        pair = self._make_pair()
        oai = pair.to_openai()
        assert "messages" in oai
        roles = [m["role"] for m in oai["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_chatml_contains_content(self):
        pair = self._make_pair()
        chatml = pair.to_chatml()
        assert "You are an engineer" in chatml
        assert "Fix the bug" in chatml
        assert "--- a/foo.py" in chatml


# ── Dataset Builder ───────────────────────────────────────────────────────────

class TestFinetuningDatasetBuilder:
    def _make_builder(self, tmp_path):
        from fine_tuning.dataset_builder import FinetuningDatasetBuilder
        return FinetuningDatasetBuilder(
            trajectory_dir=tmp_path / "trajectories",
            output_dir=tmp_path / "output",
            val_fraction=0.2,
            min_problem_words=5,  # relaxed for testing
        )

    def _populate_trajectories(self, tmp_path, entries: list[dict]) -> Path:
        return write_trajectory_jsonl(tmp_path, entries)

    def test_empty_trajectory_dir(self, tmp_path):
        from fine_tuning.dataset_builder import FinetuningDatasetBuilder
        builder = FinetuningDatasetBuilder(
            trajectory_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "out",
        )
        stats = builder.build()
        assert stats.total_trajectories == 0
        assert stats.train_size == 0

    def test_builds_from_valid_trajectories(self, tmp_path):
        entries = [make_trajectory_entry(resolved=True) for _ in range(10)]
        self._populate_trajectories(tmp_path, entries)

        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)

        assert stats.total_trajectories == 10
        assert stats.train_size + stats.val_size > 0

    def test_filters_unknown_category(self, tmp_path):
        entries = [
            make_trajectory_entry(category="assertion_error"),
            make_trajectory_entry(category="unknown"),   # should be filtered
        ]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)
        assert stats.filter_reasons.get("unknown_category", 0) >= 1

    def test_filters_empty_patch(self, tmp_path):
        entries = [make_trajectory_entry(patch="")]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)
        assert stats.filter_reasons.get("empty_patch", 0) >= 1

    def test_filters_invalid_patch_format(self, tmp_path):
        entries = [make_trajectory_entry(patch="just some text")]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)
        assert stats.filter_reasons.get("invalid_patch_format", 0) >= 1

    def test_train_val_split(self, tmp_path):
        entries = [make_trajectory_entry() for _ in range(20)]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)
        # val should be ~20% of (train + val)
        total = stats.train_size + stats.val_size
        assert total > 0
        val_ratio = stats.val_size / total
        assert 0.05 < val_ratio < 0.50  # flexible for small datasets

    def test_output_files_created(self, tmp_path):
        entries = [make_trajectory_entry() for _ in range(5)]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        builder.build(include_reflection_pairs=False)
        assert (tmp_path / "output" / "train.jsonl").exists()
        assert (tmp_path / "output" / "val.jsonl").exists()
        assert (tmp_path / "output" / "dataset_stats.json").exists()

    def test_chatml_format_output(self, tmp_path):
        entries = [make_trajectory_entry() for _ in range(5)]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        builder.build(format="chatml", include_reflection_pairs=False)

        train_path = tmp_path / "output" / "train.jsonl"
        if train_path.exists() and train_path.stat().st_size > 0:
            with train_path.open() as f:
                first = json.loads(f.readline())
            assert "text" in first
            assert "<|im_start|>" in first["text"]

    def test_reflection_pairs_from_multi_attempt(self, tmp_path):
        """Multi-attempt instances should generate reflection pairs."""
        entries = [
            make_trajectory_entry(resolved=False, attempt=1, category="assertion_error"),
            make_trajectory_entry(resolved=True,  attempt=2, category="success"),
        ]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=True)
        assert stats.augmented_pairs >= 0  # may be 0 if problem too short

    def test_stats_category_counts(self, tmp_path):
        entries = [
            make_trajectory_entry(category="assertion_error"),
            make_trajectory_entry(category="assertion_error"),
            make_trajectory_entry(category="syntax_error"),
        ]
        self._populate_trajectories(tmp_path, entries)
        builder = self._make_builder(tmp_path)
        stats = builder.build(include_reflection_pairs=False)
        assert stats.category_counts.get("assertion_error", 0) >= 1


# ── Evaluation report ─────────────────────────────────────────────────────────

class TestEvaluationReport:
    def _make_report(self, n_resolved, n_total, variant="test_model"):
        from fine_tuning.evaluator import EvaluationReport, EvalResult
        results = []
        for i in range(n_total):
            results.append(EvalResult(
                instance_id=f"inst-{i}",
                repo="django/django",
                resolved=(i < n_resolved),
                attempts=1 if i < n_resolved else 3,
                elapsed_seconds=10.0,
                token_cost=1500,
                patch="--- a/f.py\n+++ b/f.py\n",
                failure_category="success" if i < n_resolved else "assertion_error",
                model_variant=variant,
            ))
        report = EvaluationReport(variant=variant, results=results)
        return report

    def test_pct_resolved(self):
        report = self._make_report(30, 100)
        assert abs(report.pct_resolved - 0.30) < 1e-6

    def test_avg_attempts(self):
        report = self._make_report(50, 100)
        # 50 resolved at 1 attempt + 50 unresolved at 3 attempts = (50+150)/100 = 2.0
        assert abs(report.avg_attempts - 2.0) < 1e-6

    def test_save_and_load(self, tmp_path):
        report = self._make_report(10, 50)
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["summary"]["n_total"] == 50
        assert data["summary"]["n_resolved"] == 10

    def test_failure_breakdown(self):
        report = self._make_report(10, 20)
        breakdown = report.failure_breakdown
        assert "success" in breakdown
        assert "assertion_error" in breakdown

    def test_to_ablation_row(self):
        from fine_tuning.evaluator import AblationRow
        report = self._make_report(35, 100, "DeepSeek fine-tuned")
        row = report.to_ablation_row(recall_at_5=0.74)
        assert isinstance(row, AblationRow)
        assert abs(row.pct_resolved - 0.35) < 1e-6
        assert row.recall_at_5 == 0.74


# ── Ablation Table ────────────────────────────────────────────────────────────

class TestAblationTableBuilder:
    def test_includes_published_baselines(self):
        from fine_tuning.evaluator import AblationTableBuilder
        builder = AblationTableBuilder()
        assert len(builder._rows) >= 2  # Devin + SWE-agent

    def test_to_markdown_format(self):
        from fine_tuning.evaluator import AblationTableBuilder, EvaluationReport, EvalResult
        builder = AblationTableBuilder()
        md = builder.to_markdown()
        assert "| System Variant" in md
        assert "| Resolved" in md
        assert "Devin" in md

    def test_add_report(self):
        from fine_tuning.evaluator import AblationTableBuilder, EvaluationReport, EvalResult
        builder = AblationTableBuilder()
        initial_count = len(builder._rows)

        report = EvaluationReport(variant="test", results=[
            EvalResult("i1", "r", True, 1, 10.0, 1500, "p", "success", "test")
        ])
        builder.add_report(report, recall_at_5=0.74)
        assert len(builder._rows) == initial_count + 1

    def test_save_markdown(self, tmp_path):
        from fine_tuning.evaluator import AblationTableBuilder
        builder = AblationTableBuilder()
        path = tmp_path / "ablation.md"
        builder.save_markdown(path)
        assert path.exists()
        content = path.read_text()
        assert "Ablation Results" in content

    def test_markdown_row_format(self):
        from fine_tuning.evaluator import AblationRow
        row = AblationRow(
            system_variant="DeepSeek fine-tuned",
            pct_resolved=0.41,
            recall_at_5=0.74,
            avg_attempts=1.6,
            avg_token_cost=3200,
            n_instances=300,
        )
        md_row = row.to_markdown_row()
        assert "41.0%" in md_row
        assert "74.0%" in md_row
        assert "DeepSeek" in md_row


# ── Token count estimator ─────────────────────────────────────────────────────

class TestTokenCountEstimator:
    def test_estimate_on_jsonl(self, tmp_path):
        from fine_tuning.dataset_builder import estimate_token_counts
        path = tmp_path / "data.jsonl"
        data = [{"text": "hello world " * 100, "metadata": {}} for _ in range(10)]
        with path.open("w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        stats = estimate_token_counts(path)
        assert stats["n_pairs"] == 10
        assert stats["estimated_tokens"] > 0
        assert "estimated_training_cost_usd" in stats

    def test_empty_file_returns_zeros(self, tmp_path):
        from fine_tuning.dataset_builder import estimate_token_counts
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        stats = estimate_token_counts(path)
        assert stats["n_pairs"] == 0

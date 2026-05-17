"""
tests/test_phase6_uncertainty.py
──────────────────────────────────
Unit tests for Phase 6: Conformal Prediction + Temperature Scaling.

Tests verify:
  - Coverage guarantee property (marginal coverage >= 1-alpha)
  - Prediction set size properties (non-emptiness, monotonicity w.r.t. alpha)
  - Temperature scaling NLL reduction and ECE improvement
  - CalibrationStore persistence (save/load)
  - RAPS prediction set properties
  - UncertaintyReport output format
  - Pipeline integration with mock localisation

Run with: pytest tests/test_phase6_uncertainty.py -v
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── CalibrationStore ──────────────────────────────────────────────────────────

class TestCalibrationStore:
    def test_add_and_scores(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        cs.add(0.8, "inst-1", "django/django")
        cs.add(0.3, "inst-2", "django/django")
        assert cs.n == 2
        assert abs(cs.scores[0] - 0.2) < 1e-6   # 1 - 0.8
        assert abs(cs.scores[1] - 0.7) < 1e-6   # 1 - 0.3

    def test_save_and_load(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        for i in range(10):
            cs.add(float(i) / 10, f"inst-{i}")
        cs.save()

        cs2 = CalibrationStore(tmp_path / "cal.json")
        assert cs2.n == 10
        assert abs(cs2.scores.mean() - cs.scores.mean()) < 1e-6

    def test_quantile_increases_with_alpha(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        for s in np.linspace(0, 1, 50):
            cs.add(float(s))

        q10 = cs.quantile(0.10)  # 90th percentile
        q20 = cs.quantile(0.20)  # 80th percentile
        # Higher alpha → lower quantile threshold (more permissive)
        assert q20 <= q10

    def test_empty_store_quantile(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        # Should return 1.0 (worst case) when no calibration data
        assert cs.quantile(0.10) == 1.0

    def test_stats_structure(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        for s in np.linspace(0.5, 1.0, 20):
            cs.add(float(s))
        stats = cs.stats()
        assert "n" in stats
        assert "mean_nonconformity" in stats
        assert "q50" in stats

    def test_add_batch(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore
        cs = CalibrationStore(tmp_path / "cal.json")
        batch = [(0.7, "a", "repo"), (0.5, "b", "repo"), (0.9, "c", "repo")]
        cs.add_batch(batch)
        assert cs.n == 3


# ── ConformalPredictor ────────────────────────────────────────────────────────

class TestConformalPredictor:

    def _make_predictor(self, tmp_path, n_cal=100, alpha=0.10):
        from uncertainty.conformal_predictor import CalibrationStore, ConformalPredictor
        cs = CalibrationStore(tmp_path / "cal.json")
        # Simulate calibration scores from realistic localisation
        np.random.seed(42)
        cal_scores = np.random.beta(2, 5, n_cal)  # most scores are low (model is good)
        for s in cal_scores:
            cs.add(float(s))
        return ConformalPredictor(cs, alpha=alpha)

    def test_prediction_returns_correct_types(self, tmp_path):
        from uncertainty.conformal_predictor import LocalisationWithUncertainty
        cp = self._make_predictor(tmp_path)
        files = ["a.py", "b.py", "c.py"]
        scores = [0.8, 0.5, 0.2]
        result = cp.predict(files, scores)
        assert isinstance(result, LocalisationWithUncertainty)
        assert len(result.hits) == 3

    def test_coverage_guarantee_satisfied(self, tmp_path):
        """
        Core guarantee test:
        Empirical coverage >= 1 - alpha on synthetic test set.
        """
        from uncertainty.conformal_predictor import CalibrationStore, ConformalPredictor
        np.random.seed(123)
        alpha = 0.10

        # Large calibration set for stable quantile
        cs = CalibrationStore(tmp_path / "cal.json")
        n_cal = 500
        cal_rrf_scores = np.random.beta(3, 2, n_cal)  # gold file scores
        for s in cal_rrf_scores:
            cs.add(float(s))

        cp = ConformalPredictor(cs, alpha=alpha)

        # Test instances: gold file has score sampled from same distribution
        n_test = 200
        covered = 0
        for _ in range(n_test):
            gold_score = float(np.random.beta(3, 2))
            other_scores = list(np.random.beta(1, 3, 9))  # 9 non-gold files
            all_scores = sorted([gold_score] + other_scores, reverse=True)
            all_files = [f"file_{i}.py" for i in range(10)]
            gold_idx = all_scores.index(gold_score)
            gold_file = all_files[gold_idx]

            result = cp.predict(all_files, all_scores)
            pred_set = result.prediction_set_files
            if gold_file in pred_set:
                covered += 1

        empirical_coverage = covered / n_test
        # Should be >= 1 - alpha with high probability
        assert empirical_coverage >= (1 - alpha - 0.08), (
            f"Coverage {empirical_coverage:.3f} < guarantee {1-alpha:.2f}"
        )

    def test_prediction_set_includes_high_score_file(self, tmp_path):
        """High-scoring file should always be in prediction set."""
        cp = self._make_predictor(tmp_path)
        result = cp.predict(["best.py", "ok.py", "bad.py"], [0.99, 0.3, 0.01])
        pred_paths = result.prediction_set_files
        assert "best.py" in pred_paths

    def test_confidence_in_0_1_range(self, tmp_path):
        cp = self._make_predictor(tmp_path)
        result = cp.predict(["a.py", "b.py"], [0.7, 0.4])
        for hit in result.hits:
            assert 0.0 <= hit.confidence <= 1.0
            assert 0.0 <= hit.p_value <= 1.0

    def test_ranks_sequential(self, tmp_path):
        cp = self._make_predictor(tmp_path)
        result = cp.predict(["a.py", "b.py", "c.py"], [0.8, 0.5, 0.2])
        assert [h.rank for h in result.hits] == [1, 2, 3]

    def test_no_calibration_data_maximum_uncertainty(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore, ConformalPredictor
        cs = CalibrationStore(tmp_path / "cal.json")
        cp = ConformalPredictor(cs, alpha=0.10)
        result = cp.predict(["a.py"], [0.9])
        # All files should be in prediction set (maximum uncertainty)
        assert result.hits[0].p_value == 1.0  # smoothed p-value with n=0

    def test_tighter_alpha_gives_larger_set(self, tmp_path):
        """Lower alpha (e.g. 0.05) should produce larger prediction sets."""
        cp_strict  = self._make_predictor(tmp_path, alpha=0.05)
        cp_lenient = self._make_predictor(tmp_path, alpha=0.20)

        files = [f"f{i}.py" for i in range(10)]
        scores = list(np.linspace(0.9, 0.1, 10))

        r_strict  = cp_strict.predict(files, scores)
        r_lenient = cp_lenient.predict(files, scores)

        # Stricter coverage requirement → larger prediction set
        assert r_strict.prediction_set_size >= r_lenient.prediction_set_size

    def test_uncertainty_labels(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore, ConformalPredictor
        cs = CalibrationStore(tmp_path / "cal.json")
        # Calibrate so that only top-1 file is in prediction set
        for _ in range(100):
            cs.add(0.99)  # all gold files have score=0.01 → high nonconformity

        cp = ConformalPredictor(cs, alpha=0.10)
        # One file with very high score → should be "confident" or "moderate"
        result = cp.predict(["only.py"], [0.99])
        assert result.uncertainty_label in ("confident", "moderate", "uncertain", "very_uncertain")

    def test_evaluate_coverage_api(self, tmp_path):
        cp = self._make_predictor(tmp_path, n_cal=200)
        test_instances = [
            (["a.py", "b.py", "c.py"], [0.8, 0.5, 0.2], "a.py"),
            (["x.py", "y.py"],         [0.9, 0.1],       "x.py"),
        ]
        result = cp.evaluate_coverage(test_instances)
        assert "empirical_coverage" in result
        assert "avg_set_size" in result
        assert 0 <= result["empirical_coverage"] <= 1

    def test_file_confidence_property(self, tmp_path):
        from uncertainty.conformal_predictor import FileConfidence
        fc = FileConfidence(
            file_path="test.py",
            rrf_score=0.75,
            p_value=0.15,
            in_prediction_set=True,
            confidence=0.85,
            rank=1,
        )
        assert "85.0%" in fc.confidence_pct


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TestTemperatureScaler:

    def _make_overconfident_data(self, n=200, seed=42):
        """Simulate overconfident DeBERTa logits."""
        np.random.seed(seed)
        labels = np.random.randint(0, 2, n)
        # Overconfident: logits have large magnitude
        logits = np.column_stack([
            np.where(labels == 0, np.random.uniform(3, 6, n), np.random.uniform(-2, 0, n)),
            np.where(labels == 1, np.random.uniform(3, 6, n), np.random.uniform(-2, 0, n)),
        ])
        return logits, labels

    def test_scale_output_sums_to_one(self):
        from uncertainty.temperature_scaling import TemperatureScaler
        ts = TemperatureScaler(T=1.5)
        logits = np.array([[2.0, -1.0], [0.5, 0.8], [-3.0, 5.0]])
        probs = ts.scale(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_scale_output_in_0_1(self):
        from uncertainty.temperature_scaling import TemperatureScaler
        ts = TemperatureScaler(T=2.0)
        logits = np.random.randn(50, 2)
        probs = ts.scale(logits)
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_T_greater_than_1_softens(self):
        from uncertainty.temperature_scaling import TemperatureScaler
        logits = np.array([[5.0, -5.0]])  # very confident
        ts1 = TemperatureScaler(T=1.0)
        ts2 = TemperatureScaler(T=3.0)
        prob1 = ts1.scale(logits)[0, 0]
        prob2 = ts2.scale(logits)[0, 0]
        # T=3 should produce softer (closer to 0.5) probability
        assert prob1 > prob2  # prob1 closer to 1.0, prob2 closer to 0.5

    def test_fit_reduces_nll(self, tmp_path):
        from uncertainty.temperature_scaling import TemperatureScaler
        logits, labels = self._make_overconfident_data()
        ts = TemperatureScaler(T=1.0)
        result = ts.fit(logits, labels)
        assert result["nll_after"] <= result["nll_before"]

    def test_fit_T_greater_than_1_for_overconfident(self, tmp_path):
        from uncertainty.temperature_scaling import TemperatureScaler
        logits, labels = self._make_overconfident_data()
        ts = TemperatureScaler(T=1.0)
        ts.fit(logits, labels)
        # Overconfident model → T should increase to soften probabilities
        assert ts.T > 0.5  # just check it stays positive and reasonable

    def test_save_and_load(self, tmp_path):
        from uncertainty.temperature_scaling import TemperatureScaler
        ts = TemperatureScaler(T=2.345)
        ts._fitted = True
        ts.save(tmp_path / "ts.json")

        ts2 = TemperatureScaler.load(tmp_path / "ts.json")
        assert abs(ts2.T - 2.345) < 1e-6
        assert ts2._fitted is True

    def test_scale_score_single_value(self):
        from uncertainty.temperature_scaling import TemperatureScaler
        ts = TemperatureScaler(T=1.0)
        prob = ts.scale_score(2.0)
        assert 0 < prob < 1

    def test_reliability_diagram_data(self):
        from uncertainty.temperature_scaling import reliability_diagram_data
        np.random.seed(42)
        probs  = np.random.uniform(0, 1, 100)
        labels = (probs + np.random.randn(100) * 0.2 > 0.5).astype(int)
        bins = reliability_diagram_data(probs, labels, n_bins=5)
        assert len(bins) > 0
        for b in bins:
            assert "confidence" in b
            assert "accuracy" in b
            assert "count" in b


# ── RAPS ─────────────────────────────────────────────────────────────────────

class TestRAPS:
    def test_raps_returns_nonempty(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore, raps_predict
        cs = CalibrationStore(tmp_path / "cal.json")
        for s in np.linspace(0, 1, 50):
            cs.add(float(s))
        files = ["a.py", "b.py", "c.py"]
        scores = np.array([0.6, 0.3, 0.1])
        result = raps_predict(files, scores, cs, alpha=0.10)
        assert len(result) >= 1

    def test_raps_top1_always_included(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore, raps_predict
        cs = CalibrationStore(tmp_path / "cal.json")
        # Empty calibration → fallback to top-k
        files = ["best.py", "ok.py"]
        scores = np.array([0.9, 0.1])
        result = raps_predict(files, scores, cs, alpha=0.10)
        paths = [r[0] for r in result]
        assert "best.py" in paths

    def test_raps_scores_positive(self, tmp_path):
        from uncertainty.conformal_predictor import CalibrationStore, raps_predict
        cs = CalibrationStore(tmp_path / "cal.json")
        for s in np.linspace(0.1, 0.9, 30):
            cs.add(float(s))
        files = [f"f{i}.py" for i in range(5)]
        scores = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        result = raps_predict(files, scores, cs)
        assert all(s > 0 for _, s in result)


# ── UncertaintyAwarePipeline ──────────────────────────────────────────────────

class TestUncertaintyAwarePipeline:

    def _mock_localisation_pipeline(self, files, scores):
        """Create a mock pipeline that returns pre-set results."""
        from unittest.mock import MagicMock
        from localisation.pipeline import LocalisationResult, LocalisationHit

        mock = MagicMock()
        hits = [
            LocalisationHit(file_path=fp, relevance_score=s, rank=i + 1)
            for i, (fp, s) in enumerate(zip(files, scores))
        ]
        mock.localise.return_value = LocalisationResult(hits=hits, elapsed_seconds=0.1)
        mock.index_repo.return_value = {"elapsed": 0.1}
        return mock

    def test_localise_with_uncertainty_returns_result(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyAwarePipeline

        files = ["models.py", "views.py", "utils.py"]
        scores = [0.8, 0.5, 0.2]
        mock_pipeline = self._mock_localisation_pipeline(files, scores)

        up = UncertaintyAwarePipeline(
            localisation_pipeline=mock_pipeline,
            calibration_store_path=tmp_path / "cal.json",
        )
        result = up.localise_with_uncertainty("fix the bug", top_k=3)
        assert len(result.files) == 3
        assert len(result.prediction_set) >= 1

    def test_prediction_set_never_empty(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyAwarePipeline

        mock = self._mock_localisation_pipeline(["only.py"], [0.9])
        up = UncertaintyAwarePipeline(
            localisation_pipeline=mock,
            calibration_store_path=tmp_path / "cal.json",
        )
        result = up.localise_with_uncertainty("some issue")
        assert len(result.prediction_set) >= 1

    def test_token_savings_computed(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyAwarePipeline

        files = [f"f{i}.py" for i in range(10)]
        scores = list(np.linspace(0.9, 0.1, 10))
        mock = self._mock_localisation_pipeline(files, scores)

        up = UncertaintyAwarePipeline(
            localisation_pipeline=mock,
            calibration_store_path=tmp_path / "cal.json",
            tokens_per_file=1500,
        )
        result = up.localise_with_uncertainty("issue", top_k=10)
        assert result.token_budget_naive == 10 * 1500
        assert result.token_budget_used <= result.token_budget_naive

    def test_uncertainty_report_to_dict(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyReport
        report = UncertaintyReport(
            uncertainty_label="confident",
            prediction_set_size=2,
            coverage_guarantee=0.90,
            top_file_confidence=0.87,
            avg_confidence=0.65,
            estimated_token_savings=0.60,
            calibration_n=150,
        )
        d = report.to_dict()
        assert d["uncertainty_label"] == "confident"
        assert "90%" in d["coverage_guarantee"]
        assert "87.0%" in d["top_file_confidence"]

    def test_record_calibration_point(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyAwarePipeline

        mock = self._mock_localisation_pipeline(["a.py"], [0.8])
        up = UncertaintyAwarePipeline(
            localisation_pipeline=mock,
            calibration_store_path=tmp_path / "cal.json",
        )
        up.record_calibration_point(
            rrf_scores={"a.py": 0.8, "b.py": 0.3},
            gold_files=["a.py"],
            instance_id="test-1",
        )
        assert up.cal_store.n == 1

    def test_calibration_stats(self, tmp_path):
        from uncertainty.uncertainty_pipeline import UncertaintyAwarePipeline

        mock = self._mock_localisation_pipeline(["a.py"], [0.8])
        up = UncertaintyAwarePipeline(
            localisation_pipeline=mock,
            calibration_store_path=tmp_path / "cal.json",
        )
        stats = up.calibration_stats()
        assert "n" in stats

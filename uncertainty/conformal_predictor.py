"""
uncertainty/conformal_predictor.py
─────────────────────────────────────
Conformal Prediction for file localisation.

Standard Conformal Prediction framework (Venn-Abers / RAPS variant):

1. Calibration phase (run once on held-out SWE-bench val set):
   - For each (issue, gold_file) pair, record the localisation score
     of the gold file in the ranked list (its "non-conformity score").
   - Store the empirical distribution of these scores as the calibration set.

2. Inference phase (run per new issue):
   - Score each candidate file (BM25 + embed + PPR → RRF fused score).
   - Compute a p-value: what fraction of calibration non-conformity scores
     are >= this file's score?
   - Files with p-value >= (1 - alpha) are included in the prediction set.
   - The prediction set is guaranteed to contain the true file with
     probability >= 1 - alpha (marginal coverage guarantee).

Non-conformity score used here:
    s(x, y) = 1 - rank_score(y | x)
             = 1 - (RRF_score of gold file)
Higher score = less conforming (more surprising = file is suspicious).

Coverage guarantee:
    P(gold_file ∈ prediction_set) >= 1 - alpha

With alpha = 0.10: prediction set covers gold file >=90% of the time.
The set size (how many files needed to achieve coverage) is a measure of
localisation difficulty — small set = confident, large set = uncertain.

References:
    Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
    Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
    Jin & Candès (2023) "Selection by Prediction with Conformal P-values"
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class FileConfidence:
    """Conformal prediction result for one file."""
    file_path: str
    rrf_score: float            # raw RRF fusion score
    p_value: float              # conformal p-value ∈ [0, 1]
    in_prediction_set: bool     # whether included at alpha threshold
    confidence: float           # 1 - p_value (intuitive confidence %)
    rank: int                   # rank in the full localisation list

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"


@dataclass
class LocalisationWithUncertainty:
    """Augmented localisation result with conformal coverage guarantees."""
    hits: list[FileConfidence]
    alpha: float                    # target miscoverage rate
    prediction_set_size: int        # |C(x)| at this alpha
    coverage_guarantee: float       # 1 - alpha
    calibration_n: int              # size of calibration set
    uncertainty_label: str          # 'confident' / 'uncertain' / 'very_uncertain'
    avg_confidence: float

    @property
    def prediction_set_files(self) -> list[str]:
        return [h.file_path for h in self.hits if h.in_prediction_set]

    @property
    def top_file(self) -> Optional[FileConfidence]:
        return self.hits[0] if self.hits else None


# ── Calibration store ─────────────────────────────────────────────────────────

class CalibrationStore:
    """
    Stores non-conformity scores from the validation set.
    Persisted as a JSON file — survives restarts.

    Non-conformity score for instance (x, y):
        s = 1 - rrf_score(y | x)   if y was in localisation candidates
            1.0                     if y was NOT in candidates (worst case)
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._scores: list[float] = []
        self._metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._scores = data.get("scores", [])
                self._metadata = data.get("metadata", [])
                logger.info("Calibration store loaded: %d scores from %s", len(self._scores), self.path)
            except Exception as e:
                logger.warning("Failed to load calibration store: %s", e)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "scores": self._scores,
            "metadata": self._metadata,
            "n": len(self._scores),
        }, indent=2))

    def add(self, rrf_score_of_gold_file: float, instance_id: str = "", repo: str = "") -> None:
        """
        Record one calibration point.

        Args:
            rrf_score_of_gold_file: RRF score of the true file (0 if not in candidates)
            instance_id: for diagnostics
            repo: repository name
        """
        nonconformity = 1.0 - rrf_score_of_gold_file  # higher = more surprising
        self._scores.append(nonconformity)
        self._metadata.append({"instance_id": instance_id, "repo": repo, "s": nonconformity})

    def add_batch(self, scores: list[tuple[float, str, str]]) -> None:
        """Add multiple calibration points: [(rrf_score, instance_id, repo), ...]"""
        for rrf_score, instance_id, repo in scores:
            self.add(rrf_score, instance_id, repo)

    @property
    def n(self) -> int:
        return len(self._scores)

    @property
    def scores(self) -> np.ndarray:
        return np.array(self._scores, dtype=float)

    def quantile(self, alpha: float) -> float:
        """
        Compute the (1-alpha) quantile of non-conformity scores.
        Uses the finite-sample corrected quantile:
            q_hat = ceil((n+1)(1-alpha)) / n
        to achieve marginal coverage guarantee.
        """
        if self.n == 0:
            return 1.0  # worst case: no calibration data

        scores = self.scores
        n = len(scores)
        level = math.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(scores, level))

    def stats(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        s = self.scores
        return {
            "n": self.n,
            "mean_nonconformity": float(s.mean()),
            "std_nonconformity": float(s.std()),
            "q10": float(np.quantile(s, 0.10)),
            "q50": float(np.quantile(s, 0.50)),
            "q90": float(np.quantile(s, 0.90)),
        }


# ── Conformal predictor ────────────────────────────────────────────────────────

class ConformalPredictor:
    """
    Wraps the localisation pipeline with conformal prediction.

    Computes:
      - p-value per candidate file (probability that the file is non-conforming)
      - Prediction set at alpha = 0.10 (90% coverage guarantee)
      - Confidence label: 'confident' / 'uncertain' / 'very_uncertain'

    Usage:
        cp = ConformalPredictor(calibration_store, alpha=0.10)
        result = cp.predict(localisation_hits, raw_scores)
    """

    def __init__(
        self,
        calibration_store: CalibrationStore,
        alpha: float = 0.10,
    ):
        self.cal = calibration_store
        self.alpha = alpha

    def predict(
        self,
        file_paths: list[str],
        rrf_scores: list[float],
        alpha: Optional[float] = None,
    ) -> LocalisationWithUncertainty:
        """
        Generate conformal prediction set from localisation results.

        Args:
            file_paths:  ordered list of file paths (rank 1 first)
            rrf_scores:  RRF fused scores for each file (same order)
            alpha:       target miscoverage rate (default: self.alpha)

        Returns:
            LocalisationWithUncertainty with per-file confidence scores
        """
        alpha = alpha if alpha is not None else self.alpha

        # Compute quantile threshold
        q_hat = self.cal.quantile(alpha)

        hits: list[FileConfidence] = []
        for rank, (fp, score) in enumerate(zip(file_paths, rrf_scores), start=1):
            # Non-conformity of this file
            s = 1.0 - score
            # p-value: fraction of cal scores >= s (empirical tail prob)
            p_value = self._p_value(s)
            # File is in prediction set if its non-conformity is low enough
            in_set = s <= q_hat

            hits.append(FileConfidence(
                file_path=fp,
                rrf_score=score,
                p_value=p_value,
                in_prediction_set=in_set,
                confidence=1.0 - p_value,
                rank=rank,
            ))

        pred_set_size = sum(1 for h in hits if h.in_prediction_set)
        avg_conf = float(np.mean([h.confidence for h in hits])) if hits else 0.0

        uncertainty_label = self._uncertainty_label(pred_set_size, len(file_paths))

        return LocalisationWithUncertainty(
            hits=hits,
            alpha=alpha,
            prediction_set_size=pred_set_size,
            coverage_guarantee=1.0 - alpha,
            calibration_n=self.cal.n,
            uncertainty_label=uncertainty_label,
            avg_confidence=avg_conf,
        )

    def _p_value(self, nonconformity: float) -> float:
        """
        Compute empirical p-value: P(S_cal >= s) over calibration scores.
        Laplace-smoothed with 1/(n+1) to avoid p-value = 0.
        """
        if self.cal.n == 0:
            return 1.0  # maximum uncertainty when no calibration data

        cal_scores = self.cal.scores
        n = len(cal_scores)
        # Count calibration scores >= nonconformity
        count = int(np.sum(cal_scores >= nonconformity))
        # Smoothed p-value (Venn-Abers style)
        return (count + 1) / (n + 1)

    def _uncertainty_label(self, set_size: int, total_candidates: int) -> str:
        """Classify uncertainty level based on prediction set size."""
        if set_size == 0:
            return "very_uncertain"    # nothing meets the threshold
        if set_size == 1:
            return "confident"         # exactly one file — high certainty
        if set_size <= 3:
            return "moderate"
        if set_size <= total_candidates // 2:
            return "uncertain"
        return "very_uncertain"

    def evaluate_coverage(
        self,
        test_instances: list[tuple[list[str], list[float], str]],
        alpha: Optional[float] = None,
    ) -> dict:
        """
        Evaluate empirical coverage on a test set.
        Tests that P(gold_file ∈ prediction_set) >= 1 - alpha.

        Args:
            test_instances: list of (file_paths, rrf_scores, gold_file)
            alpha: miscoverage rate to test

        Returns:
            {empirical_coverage, avg_set_size, coverage_guarantee, alpha}
        """
        alpha = alpha if alpha is not None else self.alpha
        covered = 0
        set_sizes = []

        for file_paths, rrf_scores, gold_file in test_instances:
            result = self.predict(file_paths, rrf_scores, alpha)
            if gold_file in result.prediction_set_files:
                covered += 1
            set_sizes.append(result.prediction_set_size)

        n = len(test_instances)
        empirical_cov = covered / n if n > 0 else 0.0

        return {
            "empirical_coverage": empirical_cov,
            "coverage_guarantee": 1.0 - alpha,
            "coverage_satisfied": empirical_cov >= (1.0 - alpha),
            "avg_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
            "n_test": n,
            "alpha": alpha,
        }


# ── Adaptive prediction set (RAPS variant) ────────────────────────────────────

def raps_predict(
    file_paths: list[str],
    softmax_scores: np.ndarray,
    calibration_store: CalibrationStore,
    alpha: float = 0.10,
    k_reg: int = 5,
    lambda_reg: float = 0.01,
) -> list[tuple[str, float]]:
    """
    RAPS: Regularized Adaptive Prediction Sets.

    Extends conformal prediction with a regularisation term that penalises
    large prediction sets. This is the state-of-the-art method from:
        Angelopoulos et al. (2021) "Uncertainty Sets for Image Classifiers"

    The regularisation term discourages including low-ranked files
    (rank > k_reg) by adding lambda_reg per extra file.

    Args:
        file_paths:         ranked candidate files (most relevant first)
        softmax_scores:     softmax probabilities (sums to ~1)
        calibration_store:  fitted calibration distribution
        alpha:              target miscoverage rate
        k_reg:              regularisation start rank
        lambda_reg:         penalty per file beyond k_reg

    Returns:
        List of (file_path, adjusted_score) in the prediction set
    """
    n_cal = calibration_store.n
    if n_cal == 0:
        # No calibration — return top-k as fallback
        return [(fp, float(s)) for fp, s in zip(file_paths, softmax_scores)][:5]

    # Regularised non-conformity score
    reg_scores = []
    cumsum = 0.0
    for i, (fp, s) in enumerate(zip(file_paths, softmax_scores)):
        cumsum += float(s)
        # Penalise files ranked beyond k_reg
        penalty = lambda_reg * max(0, i + 1 - k_reg)
        reg_score = cumsum - float(s) + penalty
        reg_scores.append((fp, float(s), reg_score))

    # Calibration threshold
    q_hat = calibration_store.quantile(alpha)

    # Include files up to threshold
    prediction_set = []
    for fp, score, reg_s in reg_scores:
        if reg_s <= q_hat:
            prediction_set.append((fp, score))

    # Always include at least top-1 (avoids empty prediction sets)
    if not prediction_set and reg_scores:
        prediction_set = [(reg_scores[0][0], reg_scores[0][1])]

    return prediction_set


# ── Calibration utilities ──────────────────────────────────────────────────────

def calibrate_from_trajectories(
    trajectory_path: Path,
    localisation_results: dict[str, list[tuple[str, float]]],
    cal_store: CalibrationStore,
) -> int:
    """
    Build calibration set from saved trajectory JSONL.

    For each trajectory entry:
      - Look up localisation results for that instance
      - Find the RRF score of the gold file(s) in the results
      - Add to calibration store

    Args:
        trajectory_path:       path to trajectory JSONL
        localisation_results:  {instance_id: [(file_path, rrf_score), ...]}
        cal_store:             CalibrationStore to append to

    Returns:
        Number of calibration points added
    """
    from agent.trajectory_logger import TrajectoryLogger
    from localisation.deberta_ranker import _extract_files_from_patch

    tl = TrajectoryLogger(trajectory_path)
    entries = tl.load_all()

    added = 0
    for entry in entries:
        instance_results = localisation_results.get(entry.instance_id, [])
        if not instance_results:
            continue

        # Extract gold files from the patch
        gold_files = set(_extract_files_from_patch(entry.patch))
        if not gold_files:
            continue

        # For each gold file, find its RRF score
        score_map = {fp: score for fp, score in instance_results}
        for gold_fp in gold_files:
            # Score = 0 if not localised (worst case non-conformity = 1)
            rrf_score = score_map.get(gold_fp, 0.0)
            cal_store.add(rrf_score, entry.instance_id, entry.repo)
            added += 1

    cal_store.save()
    logger.info("Added %d calibration points from %s", added, trajectory_path)
    return added

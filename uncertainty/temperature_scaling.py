"""
uncertainty/temperature_scaling.py
────────────────────────────────────
Temperature scaling for DeBERTa classifier logits.

After fine-tuning, DeBERTa's raw logits are often overconfident.
Temperature scaling is the simplest, most effective calibration method
(Guo et al., 2017 — "On Calibration of Modern Neural Networks").

Method:
    calibrated_prob = softmax(logits / T)
    T is learned by minimising NLL on a held-out calibration set.

For our use case, T is fit on the SWE-bench validation split:
    - True positives: (issue, gold_file) pairs → label=1
    - True negatives: (issue, non-gold_file) pairs → label=0
    - T is scalar, so only one parameter to fit (no overfitting risk)

After calibration:
    - ECE (Expected Calibration Error) < 0.05 target
    - Reliability diagram should be close to diagonal

Integration:
    DeBERTa ranker outputs raw logits → temperature_scale() → calibrated prob
    Calibrated prob replaces raw relevance_score in RankedFile
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """
    Learns a single temperature parameter T by minimising NLL on validation data.

    T > 1: softer probabilities (reduces overconfidence)
    T < 1: harder probabilities (makes model more confident)
    T = 1: uncalibrated (no change)
    """

    def __init__(self, T: float = 1.0):
        self.T = T
        self._fitted = False

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: shape (n, 2) — binary classification logits
        Returns:
            probs: shape (n, 2) — calibrated probabilities
        """
        scaled = logits / self.T
        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def scale_score(self, logit_positive: float) -> float:
        """Scale a single logit for the positive class → calibrated probability."""
        # Convert single value to 2-class logit pair
        logits = np.array([[0.0, logit_positive]])
        probs = self.scale(logits)
        return float(probs[0, 1])

    def fit(
        self,
        logits: np.ndarray,   # shape (n, 2)
        labels: np.ndarray,   # shape (n,) — 0 or 1
        n_iter: int = 100,
        lr: float = 0.01,
        tol: float = 1e-6,
    ) -> dict:
        """
        Fit temperature by minimising NLL using gradient descent.

        Returns:
            stats dict: {T_before, T_after, nll_before, nll_after, ece_before, ece_after}
        """
        T_init = self.T
        nll_before = self._nll(logits, labels, T_init)
        ece_before = self._ece(logits, labels, T_init)

        # Simple gradient descent over scalar T
        T = float(T_init)
        for i in range(n_iter):
            grad = self._nll_gradient(logits, labels, T)
            T_new = T - lr * grad
            T_new = max(T_new, 0.01)  # T must be positive
            if abs(T_new - T) < tol:
                logger.debug("Temperature scaling converged at iteration %d", i)
                break
            T = T_new

        self.T = T
        self._fitted = True

        nll_after = self._nll(logits, labels, T)
        ece_after = self._ece(logits, labels, T)

        logger.info(
            "Temperature scaling: T=%.3f→%.3f | NLL: %.4f→%.4f | ECE: %.4f→%.4f",
            T_init, T, nll_before, nll_after, ece_before, ece_after
        )
        return {
            "T_before": T_init, "T_after": T,
            "nll_before": nll_before, "nll_after": nll_after,
            "ece_before": ece_before, "ece_after": ece_after,
            "fitted": True,
        }

    def _nll(self, logits: np.ndarray, labels: np.ndarray, T: float) -> float:
        """Negative log-likelihood at temperature T."""
        probs = self._softmax(logits / T)
        eps = 1e-8
        correct_probs = probs[np.arange(len(labels)), labels.astype(int)]
        return float(-np.mean(np.log(correct_probs + eps)))

    def _nll_gradient(self, logits: np.ndarray, labels: np.ndarray, T: float) -> float:
        """Numerical gradient of NLL w.r.t. T."""
        eps = 1e-4
        return (self._nll(logits, labels, T + eps) - self._nll(logits, labels, T - eps)) / (2 * eps)

    def _ece(self, logits: np.ndarray, labels: np.ndarray, T: float, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        probs = self._softmax(logits / T)
        max_probs = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels.astype(int))

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (max_probs > bins[i]) & (max_probs <= bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            conf = max_probs[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        return float(ece)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"T": self.T, "fitted": self._fitted}))
        logger.info("Temperature scaler saved: T=%.4f → %s", self.T, path)

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        data = json.loads(Path(path).read_text())
        ts = cls(T=data["T"])
        ts._fitted = data.get("fitted", False)
        logger.info("Temperature scaler loaded: T=%.4f from %s", ts.T, path)
        return ts


# ── ECE visualisation helper ──────────────────────────────────────────────────

def reliability_diagram_data(
    probs: np.ndarray,        # shape (n,) — predicted positive probabilities
    labels: np.ndarray,       # shape (n,) — true binary labels
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute data for a reliability diagram.

    Returns list of bins:
        [{"confidence": 0.15, "accuracy": 0.12, "count": 45}, ...]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        result.append({
            "confidence": float((bins[i] + bins[i + 1]) / 2),
            "accuracy": float(labels[mask].mean()),
            "count": int(mask.sum()),
        })
    return result

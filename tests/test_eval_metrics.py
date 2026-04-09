"""Tests for evaluation metrics computation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from eval_metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        pred = np.array([0, 1, 2, 3])
        true = np.array([0, 1, 2, 3])
        metrics = compute_metrics(pred, pred, pred[:2].repeat(2), true, true, true[:2].repeat(2))
        assert metrics["exact_match"] == 1.0
        assert metrics["F1_T"] == 1.0
        assert metrics["F1_N"] == 1.0
        assert metrics["F1_M"] == 1.0
        assert metrics["Macro-F1"] == pytest.approx(1.0)
        assert metrics["Micro-F1"] == pytest.approx(1.0)

    def test_zero_exact_match(self):
        pred_t = np.array([1, 2, 3, 0])
        true_t = np.array([0, 1, 2, 3])
        pred_n = np.array([0, 0, 0, 0])
        true_n = np.array([0, 0, 0, 0])
        pred_m = np.array([0, 1, 0, 1])
        true_m = np.array([0, 1, 0, 1])
        metrics = compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m)
        assert metrics["exact_match"] == 0.0

    def test_partial_match(self):
        pred_t = np.array([0, 1, 0, 1])
        true_t = np.array([0, 1, 2, 3])
        pred_n = np.array([0, 0, 0, 0])
        true_n = np.array([0, 0, 0, 0])
        pred_m = np.array([0, 0, 0, 0])
        true_m = np.array([0, 0, 0, 0])
        metrics = compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m)
        assert metrics["exact_match"] == 0.5

    def test_all_metrics_present(self):
        pred = np.array([0, 1, 2, 3])
        true = np.array([0, 1, 2, 3])
        m_pred = np.array([0, 1, 0, 1])
        m_true = np.array([0, 1, 0, 1])
        metrics = compute_metrics(pred, pred, m_pred, true, true, m_true)
        for key in ["F1_T", "F1_N", "F1_M", "Micro-F1", "Micro-Precision",
                     "Micro-Recall", "Macro-F1", "Macro-Precision", "Macro-Recall",
                     "exact_match"]:
            assert key in metrics, f"Missing key: {key}"

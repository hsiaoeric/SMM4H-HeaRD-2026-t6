"""
Evaluation: per-head and aggregated F1/precision/recall, AUROC, exact-match.
Usage: python eval_metrics.py <predictions.csv> <ground_truth.csv>
Both CSVs must have identifier column and t/n/m columns.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def extract_tnm_indices(df):
    """Return (t, n, m) as 0-based integer arrays.

    Accepts:
      - Numeric format: t/n/m where t is 1-indexed (1-4), n/m are 0-indexed
      - String format: T_label/N_label/M_label (e.g. "T2", "N1", "M0")
    """
    if {"t", "n", "m"}.issubset(df.columns):
        t = df["t"].values - 1  # t is 1-indexed (1-4) -> 0-based (0-3)
        n = df["n"].values
        m = df["m"].values
    elif {"T_label", "N_label", "M_label"}.issubset(df.columns):
        from constants import LABEL_TO_IDX
        t = df["T_label"].map(LABEL_TO_IDX).values
        n = df["N_label"].map(LABEL_TO_IDX).values
        m = df["M_label"].map(LABEL_TO_IDX).values
    else:
        raise ValueError(
            f"CSV must have t/n/m or T_label/N_label/M_label columns. Found: {list(df.columns)}"
        )
    return t, n, m


def compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m,
                    probs_t=None, probs_n=None, probs_m=None):
    def _f1(y_true, y_pred, avg):
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    def _prec(y_true, y_pred, avg):
        return float(precision_score(y_true, y_pred, average=avg, zero_division=0))

    def _rec(y_true, y_pred, avg):
        return float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    # Per-head macro metrics
    metrics = {
        "F1_T": _f1(true_t, pred_t, "macro"),
        "F1_N": _f1(true_n, pred_n, "macro"),
        "F1_M": _f1(true_m, pred_m, "macro"),
    }

    # Per-head micro metrics
    mi_f1_t = _f1(true_t, pred_t, "micro")
    mi_f1_n = _f1(true_n, pred_n, "micro")
    mi_f1_m = _f1(true_m, pred_m, "micro")
    mi_pr_t = _prec(true_t, pred_t, "micro")
    mi_pr_n = _prec(true_n, pred_n, "micro")
    mi_pr_m = _prec(true_m, pred_m, "micro")
    mi_re_t = _rec(true_t, pred_t, "micro")
    mi_re_n = _rec(true_n, pred_n, "micro")
    mi_re_m = _rec(true_m, pred_m, "micro")

    # Per-head macro precision/recall
    ma_pr_t = _prec(true_t, pred_t, "macro")
    ma_pr_n = _prec(true_n, pred_n, "macro")
    ma_pr_m = _prec(true_m, pred_m, "macro")
    ma_re_t = _rec(true_t, pred_t, "macro")
    ma_re_n = _rec(true_n, pred_n, "macro")
    ma_re_m = _rec(true_m, pred_m, "macro")

    # Aggregated (mean across heads)
    metrics["Micro-F1"] = (mi_f1_t + mi_f1_n + mi_f1_m) / 3
    metrics["Micro-Precision"] = (mi_pr_t + mi_pr_n + mi_pr_m) / 3
    metrics["Micro-Recall"] = (mi_re_t + mi_re_n + mi_re_m) / 3
    metrics["Macro-F1"] = (metrics["F1_T"] + metrics["F1_N"] + metrics["F1_M"]) / 3
    metrics["Macro-Precision"] = (ma_pr_t + ma_pr_n + ma_pr_m) / 3
    metrics["Macro-Recall"] = (ma_re_t + ma_re_n + ma_re_m) / 3

    metrics["exact_match"] = float(np.mean(
        (np.array(pred_t) == np.array(true_t))
        & (np.array(pred_n) == np.array(true_n))
        & (np.array(pred_m) == np.array(true_m))
    ))

    # AUROC (optional)
    if probs_m is not None and len(np.unique(true_m)) > 1:
        try:
            metrics["AUROC_M"] = float(roc_auc_score(true_m, probs_m[:, 1]))
        except ValueError as e:
            logger.warning("Could not compute AUROC_M: %s", e)
            metrics["AUROC_M"] = float("nan")
    if probs_t is not None and len(np.unique(true_t)) > 1:
        try:
            metrics["AUROC_T"] = float(
                roc_auc_score(true_t, probs_t, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            logger.warning("Could not compute AUROC_T: %s", e)
    if probs_n is not None and len(np.unique(true_n)) > 1:
        try:
            metrics["AUROC_N"] = float(
                roc_auc_score(true_n, probs_n, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            logger.warning("Could not compute AUROC_N: %s", e)

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_csv")
    p.add_argument("ground_truth_csv")
    p.add_argument("--id-col", default="patient_filename")
    p.add_argument("--output-metrics", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pred_df = pd.read_csv(args.predictions_csv)
    gt_df = pd.read_csv(args.ground_truth_csv)
    id_col = args.id_col
    if id_col not in pred_df.columns:
        sys.exit(f"Missing id column '{id_col}' in predictions. Available: {list(pred_df.columns)}")
    if id_col not in gt_df.columns:
        sys.exit(f"Missing id column '{id_col}' in ground_truth. Available: {list(gt_df.columns)}")

    try:
        pred_t, pred_n, pred_m = extract_tnm_indices(pred_df)
        true_t, true_n, true_m = extract_tnm_indices(gt_df)
    except ValueError as e:
        sys.exit(str(e))

    merged_ids = pred_df[[id_col]].merge(gt_df[[id_col]], on=id_col, how="inner")
    if merged_ids.empty:
        logger.error("No matching rows between predictions and ground truth.")
        return 1

    pred_mask = pred_df[id_col].isin(merged_ids[id_col])
    gt_mask = gt_df[id_col].isin(merged_ids[id_col])
    metrics = compute_metrics(
        pred_t[pred_mask.values], pred_n[pred_mask.values], pred_m[pred_mask.values],
        true_t[gt_mask.values], true_n[gt_mask.values], true_m[gt_mask.values],
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    if args.output_metrics:
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

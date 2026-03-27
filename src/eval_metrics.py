"""
Evaluation: F1 per label (macro), AUROC (ovr), exact-match.
Usage: python eval_metrics.py <predictions.csv> <ground_truth.csv>
Both CSVs must have identifier column and T_label, N_label, M_label.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from constants import LABEL_TO_IDX

logger = logging.getLogger(__name__)


def compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m,
                    probs_t=None, probs_n=None, probs_m=None):
    metrics = {
        "F1_T": float(f1_score(true_t, pred_t, average="macro", zero_division=0)),
        "F1_N": float(f1_score(true_n, pred_n, average="macro", zero_division=0)),
        "F1_M": float(f1_score(true_m, pred_m, average="macro", zero_division=0)),
        "exact_match": float(np.mean(
            (np.array(pred_t) == np.array(true_t))
            & (np.array(pred_n) == np.array(true_n))
            & (np.array(pred_m) == np.array(true_m))
        )),
    }
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
            metrics["AUROC_T"] = float("nan")
    if probs_n is not None and len(np.unique(true_n)) > 1:
        try:
            metrics["AUROC_N"] = float(
                roc_auc_score(true_n, probs_n, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            logger.warning("Could not compute AUROC_N: %s", e)
            metrics["AUROC_N"] = float("nan")
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
    # Require T_label, N_label, M_label in both
    cols = [id_col, "T_label", "N_label", "M_label"]
    for df, name in [(pred_df, "predictions"), (gt_df, "ground_truth")]:
        for c in cols:
            if c not in df.columns:
                sys.exit(f"Missing column '{c}' in {name} CSV. Available: {list(df.columns)}")
    merged = pred_df[cols].merge(
        gt_df[cols],
        on=id_col,
        how="inner",
        suffixes=("_pred", "_true"),
    )
    if merged.empty:
        logger.error("No matching rows between predictions and ground truth.")
        return 1
    pred_t = merged["T_label_pred"].map(LABEL_TO_IDX).values
    pred_n = merged["N_label_pred"].map(LABEL_TO_IDX).values
    pred_m = merged["M_label_pred"].map(LABEL_TO_IDX).values
    true_t = merged["T_label_true"].map(LABEL_TO_IDX).values
    true_n = merged["N_label_true"].map(LABEL_TO_IDX).values
    true_m = merged["M_label_true"].map(LABEL_TO_IDX).values
    metrics = compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    if args.output_metrics:
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

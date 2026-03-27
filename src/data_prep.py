"""
Data preparation for TNM staging: join reports with T/N/M metadata,
map labels to T14/N03/M01, and create stratified train/val/test splits.
"""
import argparse
import logging
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import T_IDX_TO_LABEL, N_IDX_TO_LABEL, M_IDX_TO_LABEL

logger = logging.getLogger(__name__)


# Label mapping: raw AJCC -> T14 (0-3), N03 (0-3), M01 (0-1)
def map_t_to_t14(raw_t: str) -> int | None:
    """Map ajcc_pathologic_t to T14 class (0=T1, 1=T2, 2=T3, 3=T4). Returns None for T0/TX to drop."""
    if raw_t in ("T0", "TX") or pd.isna(raw_t):
        return None
    raw_t = str(raw_t).strip().upper()
    if "1" in raw_t:
        return 0
    if "2" in raw_t:
        return 1
    if "3" in raw_t:
        return 2
    if "4" in raw_t:
        return 3
    return None


def map_n_to_n03(raw_n: str) -> int | None:
    """Map ajcc_pathologic_n to N03 class (0=N0, 1=N1, 2=N2, 3=N3). Returns None for NX/variants to drop."""
    if raw_n == "NX" or pd.isna(raw_n):
        return None
    raw_n = str(raw_n).strip()
    # Exclude N0 variants per baseline (case-sensitive in source data)
    if raw_n in ("N0 (i+)", "N0 (i-)", "N0 (mol+)"):
        return None
    raw_n_upper = raw_n.upper()
    if raw_n_upper.startswith("N0"):
        return 0
    if "1" in raw_n_upper:
        return 1
    if "2" in raw_n_upper:
        return 2
    if "3" in raw_n_upper:
        return 3
    return None


def map_m_to_m01(raw_m: str) -> int | None:
    """Map ajcc_pathologic_m to M01 class (0=M0, 1=M1). Returns None for MX to drop."""
    if raw_m == "MX" or pd.isna(raw_m):
        return None
    raw_m = str(raw_m).strip().upper()
    if raw_m.startswith("M0"):
        return 0
    if raw_m.startswith("M1"):
        return 1
    return None


def normalize_text(text: str) -> str:
    """Normalize whitespace and encoding."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Prepare TNM dataset: join, map labels, split.")
    parser.add_argument("--reports", default="TCGA_Reports.csv", help="Path to TCGA_Reports.csv")
    parser.add_argument("--meta-dir", default="TCGA_Metadata", help="Directory with T14/N03/M01 CSVs")
    parser.add_argument("--out-dir", default="data", help="Output directory for train/val/test CSVs")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for test set")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of train for validation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    reports_path = args.reports
    meta_dir = args.meta_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load reports
    logger.info("Loading reports...")
    reports = pd.read_csv(reports_path)
    if "patient_filename" not in reports.columns or "text" not in reports.columns:
        raise ValueError("Reports CSV must have columns: patient_filename, text")
    reports["case_submitter_id"] = reports["patient_filename"].str.split(".").str[0]
    reports["text"] = reports["text"].apply(normalize_text)

    # Load metadata
    t_df = pd.read_csv(os.path.join(meta_dir, "TCGA_T14_patients.csv"))
    n_df = pd.read_csv(os.path.join(meta_dir, "TCGA_N03_patients.csv"))
    m_df = pd.read_csv(os.path.join(meta_dir, "TCGA_M01_patients.csv"))

    # Map and drop invalid
    t_df["T"] = t_df["ajcc_pathologic_t"].apply(map_t_to_t14)
    t_df = t_df.dropna(subset=["T"])
    t_df["T"] = t_df["T"].astype(int)
    t_df = t_df[["case_submitter_id", "T"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    n_df["N"] = n_df["ajcc_pathologic_n"].apply(map_n_to_n03)
    n_df = n_df.dropna(subset=["N"])
    n_df["N"] = n_df["N"].astype(int)
    n_df = n_df[["case_submitter_id", "N"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    m_df["M"] = m_df["ajcc_pathologic_m"].apply(map_m_to_m01)
    m_df = m_df.dropna(subset=["M"])
    m_df["M"] = m_df["M"].astype(int)
    m_df = m_df[["case_submitter_id", "M"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    # Join: reports -> T -> N -> M (inner joins so only rows with all three labels)
    df = reports.merge(t_df, on="case_submitter_id", how="inner")
    df = df.merge(n_df, on="case_submitter_id", how="inner")
    df = df.merge(m_df, on="case_submitter_id", how="inner")

    # Add string labels for submission
    df["T_label"] = df["T"].map(T_IDX_TO_LABEL)
    df["N_label"] = df["N"].map(N_IDX_TO_LABEL)
    df["M_label"] = df["M"].map(M_IDX_TO_LABEL)

    logger.info("Unified dataset: %d rows with all T, N, M labels.", len(df))

    # Stratified split
    stratify_col = "T"
    try:
        train_val, test = train_test_split(
            df, test_size=args.test_size, stratify=df[stratify_col], random_state=args.seed
        )
    except ValueError as e:
        logger.warning("Stratification on T failed (%s), falling back to N.", e)
        stratify_col = "N"
        train_val, test = train_test_split(
            df, test_size=args.test_size, stratify=df[stratify_col], random_state=args.seed
        )
    val_ratio = args.val_size / (1 - args.test_size)
    try:
        train, val = train_test_split(
            train_val, test_size=val_ratio, stratify=train_val[stratify_col], random_state=args.seed
        )
    except ValueError as e:
        logger.warning("Stratification on %s failed for val split (%s), falling back to N.", stratify_col, e)
        train, val = train_test_split(
            train_val, test_size=val_ratio, stratify=train_val["N"], random_state=args.seed
        )

    # Save
    out_cols = ["patient_filename", "case_submitter_id", "text", "T", "N", "M", "T_label", "N_label", "M_label"]
    train[out_cols].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val[out_cols].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test[out_cols].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # Sanity: label distributions
    for name, part in [("train", train), ("val", val), ("test", test)]:
        logger.info("%s: n=%d", name, len(part))
        logger.info("  T: %s", part["T"].value_counts().sort_index().to_dict())
        logger.info("  N: %s", part["N"].value_counts().sort_index().to_dict())
        logger.info("  M: %s", part["M"].value_counts().sort_index().to_dict())

    logger.info("Saved train/val/test to %s/", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

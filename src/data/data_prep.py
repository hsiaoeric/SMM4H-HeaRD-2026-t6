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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def load_metadata(meta_dir):
    """Load and map TCGA metadata into T/N/M DataFrames keyed by case_submitter_id."""
    t_df = pd.read_csv(os.path.join(meta_dir, "TCGA_T14_patients.csv"))
    n_df = pd.read_csv(os.path.join(meta_dir, "TCGA_N03_patients.csv"))
    m_df = pd.read_csv(os.path.join(meta_dir, "TCGA_M01_patients.csv"))

    t_df["t"] = t_df["ajcc_pathologic_t"].apply(map_t_to_t14)
    t_df = t_df.dropna(subset=["t"]).astype({"t": int})
    t_df = t_df[["case_submitter_id", "t"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    n_df["n"] = n_df["ajcc_pathologic_n"].apply(map_n_to_n03)
    n_df = n_df.dropna(subset=["n"]).astype({"n": int})
    n_df = n_df[["case_submitter_id", "n"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    m_df["m"] = m_df["ajcc_pathologic_m"].apply(map_m_to_m01)
    m_df = m_df.dropna(subset=["m"]).astype({"m": int})
    m_df = m_df[["case_submitter_id", "m"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

    return t_df, n_df, m_df


def enrich_with_metadata(df, meta_dir):
    """Join TCGA_Metadata labels into a DataFrame using patient_filename."""
    t_df, n_df, m_df = load_metadata(meta_dir)

    df = df.copy()
    df["case_submitter_id"] = df["patient_filename"].str.split(".").str[0]

    df = df.merge(t_df, on="case_submitter_id", how="left")
    df = df.merge(n_df, on="case_submitter_id", how="left")
    df = df.merge(m_df, on="case_submitter_id", how="left")

    # T from metadata is 0-indexed (0-3) but train.csv uses 1-indexed (1-4).
    # Convert to 1-indexed so _normalize() in train.py handles both uniformly.
    df["t"] = df["t"].where(df["t"].isna(), df["t"] + 1)
    for col in ("t", "n", "m"):
        df[col] = df[col].fillna(-1).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare TNM dataset: join reports with metadata, or enrich val.csv.")
    parser.add_argument("--meta-dir", default="TCGA_Metadata", help="Directory with T14/N03/M01 CSVs")
    parser.add_argument("--out-dir", default="data", help="Output directory")

    sub = parser.add_subparsers(dest="command")

    # Subcommand: build train.csv from reports + metadata
    build_p = sub.add_parser("build", help="Build labeled dataset from TCGA_Reports.csv + metadata")
    build_p.add_argument("--reports", default="TCGA_Reports.csv", help="Path to TCGA_Reports.csv")
    build_p.add_argument("--partial-labels", action="store_true",
                         help="Keep samples with partial TNM labels (left join). Missing labels stored as -1.")

    # Subcommand: enrich val.csv with metadata labels
    enrich_p = sub.add_parser("enrich-val", help="Enrich unlabeled val.csv with TCGA_Metadata labels")
    enrich_p.add_argument("--val-csv", default="data/val.csv", help="Path to unlabeled val.csv")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "enrich-val":
        df = pd.read_csv(args.val_csv)
        df = enrich_with_metadata(df, args.meta_dir)
        out_path = os.path.join(args.out_dir, "val.csv")
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(
            "Enriched val: %d samples (t=%d, n=%d, m=%d valid). Saved to %s",
            len(df), (df["t"] >= 0).sum(), (df["n"] >= 0).sum(), (df["m"] >= 0).sum(), out_path,
        )
        return 0

    # Default / build: create labeled dataset from reports
    if args.command is None:
        args.command = "build"
        args.reports = "TCGA_Reports.csv"
        args.partial_labels = False

    reports_path = args.reports
    meta_dir = args.meta_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading reports...")
    reports = pd.read_csv(reports_path)
    if "patient_filename" not in reports.columns or "text" not in reports.columns:
        raise ValueError("Reports CSV must have columns: patient_filename, text")
    reports["case_submitter_id"] = reports["patient_filename"].str.split(".").str[0]
    reports["text"] = reports["text"].apply(normalize_text)

    t_df, n_df, m_df = load_metadata(meta_dir)

    if args.partial_labels:
        df = reports.merge(t_df, on="case_submitter_id", how="left")
        df = df.merge(n_df, on="case_submitter_id", how="left")
        df = df.merge(m_df, on="case_submitter_id", how="left")
        has_any = df["t"].notna() | df["n"].notna() | df["m"].notna()
        df = df[has_any].copy()
        for col in ("t", "n", "m"):
            df[col] = df[col].fillna(-1).astype(int)
        logger.info("Partial-label dataset: %d rows (t=%d, n=%d, m=%d valid).",
                     len(df), (df["t"] >= 0).sum(), (df["n"] >= 0).sum(), (df["m"] >= 0).sum())
    else:
        df = reports.merge(t_df, on="case_submitter_id", how="inner")
        df = df.merge(n_df, on="case_submitter_id", how="inner")
        df = df.merge(m_df, on="case_submitter_id", how="inner")
        logger.info("Unified dataset: %d rows with all t, n, m labels.", len(df))

    # Save (no split — train/val are already separate files)
    out_cols = ["patient_filename", "case_submitter_id", "text", "t", "n", "m"]
    df[out_cols].to_csv(os.path.join(out_dir, "train.csv"), index=False)

    # Sanity: label distributions
    for col in ("t", "n", "m"):
        valid = df[df[col] >= 0][col]
        missing = (df[col] < 0).sum()
        dist = valid.value_counts().sort_index().to_dict()
        if missing > 0:
            logger.info("  %s: %s (missing=%d)", col, dist, missing)
        else:
            logger.info("  %s: %s", col, dist)

    logger.info("Saved train.csv to %s/", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

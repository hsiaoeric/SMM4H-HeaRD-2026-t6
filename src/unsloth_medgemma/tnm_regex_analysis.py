"""
Analyze coverage and accuracy of rule-based TNM extraction from pathology reports.

Usage:
    uv run python src/tnm_regex_analysis.py
    uv run python src/tnm_regex_analysis.py --split all   # analyze all splits
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from tnm_regex import extract_tnm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_split(df: pd.DataFrame, split_name: str) -> dict:
    """Run extraction on one split and return stats."""
    n = len(df)
    has_ground_truth = "T_label" in df.columns

    rows = []
    for _, row in df.iterrows():
        extracted = extract_tnm(str(row["text"]))
        entry = {
            "patient_filename": row.get("patient_filename", ""),
            "ext_t": extracted["t"],
            "ext_n": extracted["n"],
            "ext_m": extracted["m"],
        }
        if has_ground_truth:
            entry["true_t"] = row.get("T_label", "")
            entry["true_n"] = row.get("N_label", "")
            entry["true_m"] = row.get("M_label", "")
        rows.append(entry)

    results = pd.DataFrame(rows)

    # Coverage
    cov_t = results["ext_t"].notna().sum()
    cov_n = results["ext_n"].notna().sum()
    cov_m = results["ext_m"].notna().sum()
    cov_all = (results["ext_t"].notna() & results["ext_n"].notna() & results["ext_m"].notna()).sum()
    cov_any = (results["ext_t"].notna() | results["ext_n"].notna() | results["ext_m"].notna()).sum()

    logger.info("=" * 60)
    logger.info(f"Split: {split_name}  (n={n})")
    logger.info("-" * 60)
    logger.info(f"  Coverage — any stage extracted : {cov_any:5d} / {n} ({100*cov_any/n:.1f}%)")
    logger.info(f"  Coverage — all 3 stages found  : {cov_all:5d} / {n} ({100*cov_all/n:.1f}%)")
    logger.info(f"  T extracted                    : {cov_t:5d} / {n} ({100*cov_t/n:.1f}%)")
    logger.info(f"  N extracted                    : {cov_n:5d} / {n} ({100*cov_n/n:.1f}%)")
    logger.info(f"  M extracted                    : {cov_m:5d} / {n} ({100*cov_m/n:.1f}%)")

    stats = {
        "split": split_name,
        "n": n,
        "cov_any": cov_any,
        "cov_all": cov_all,
        "cov_t": cov_t,
        "cov_n": cov_n,
        "cov_m": cov_m,
    }

    # Accuracy (only where ground truth exists AND extraction succeeded)
    if has_ground_truth:
        logger.info("-" * 60)

        def acc(col_ext, col_true):
            mask = results[col_ext].notna() & (results[col_true] != "")
            if mask.sum() == 0:
                return 0.0, 0
            correct = (results.loc[mask, col_ext] == results.loc[mask, col_true]).sum()
            return correct / mask.sum(), mask.sum()

        acc_t, n_t = acc("ext_t", "true_t")
        acc_n, n_n = acc("ext_n", "true_n")
        acc_m, n_m = acc("ext_m", "true_m")

        # Exact match: all three correct where all three extracted
        full_mask = (
            results["ext_t"].notna() & results["ext_n"].notna() & results["ext_m"].notna()
            & (results["true_t"] != "") & (results["true_n"] != "") & (results["true_m"] != "")
        )
        if full_mask.sum() > 0:
            exact = (
                (results.loc[full_mask, "ext_t"] == results.loc[full_mask, "true_t"])
                & (results.loc[full_mask, "ext_n"] == results.loc[full_mask, "true_n"])
                & (results.loc[full_mask, "ext_m"] == results.loc[full_mask, "true_m"])
            ).sum()
            exact_acc = exact / full_mask.sum()
        else:
            exact, exact_acc = 0, 0.0

        logger.info(f"  Accuracy T (n={n_t:4d})              : {100*acc_t:.1f}%")
        logger.info(f"  Accuracy N (n={n_n:4d})              : {100*acc_n:.1f}%")
        logger.info(f"  Accuracy M (n={n_m:4d})              : {100*acc_m:.1f}%")
        logger.info(f"  Exact match (n={full_mask.sum():4d})           : {100*exact_acc:.1f}%")

        stats.update({
            "acc_t": acc_t, "acc_n": acc_n, "acc_m": acc_m,
            "exact_match": exact_acc,
            "n_exact_eligible": int(full_mask.sum()),
        })

        # Show a few error examples
        if full_mask.sum() > 0:
            errors = results[full_mask & ~(
                (results["ext_t"] == results["true_t"])
                & (results["ext_n"] == results["true_n"])
                & (results["ext_m"] == results["true_m"])
            )].head(5)
            if len(errors):
                logger.info("\n  Sample mismatches (ext → true):")
                for _, r in errors.iterrows():
                    logger.info(
                        f"    T:{r['ext_t']}→{r['true_t']}  "
                        f"N:{r['ext_n']}→{r['true_n']}  "
                        f"M:{r['ext_m']}→{r['true_m']}  "
                        f"  [{r['patient_filename'][:40]}]"
                    )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory with train/val/test CSVs")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to analyse",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_stats = []
    for split in splits:
        path = data_dir / f"{split}.csv"
        if not path.exists():
            logger.warning(f"  {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        stats = analyse_split(df, split)
        all_stats.append(stats)

    # Aggregate across splits
    if len(all_stats) > 1:
        total_n = sum(s["n"] for s in all_stats)
        total_any = sum(s["cov_any"] for s in all_stats)
        total_all = sum(s["cov_all"] for s in all_stats)
        logger.info("=" * 60)
        logger.info(f"TOTAL across splits (n={total_n})")
        logger.info(f"  Any stage coverage : {total_any} / {total_n} ({100*total_any/total_n:.1f}%)")
        logger.info(f"  All-3 coverage     : {total_all} / {total_n} ({100*total_all/total_n:.1f}%)")

        if "exact_match" in all_stats[0]:
            n_elig = sum(s["n_exact_eligible"] for s in all_stats)
            wt_exact = sum(s["exact_match"] * s["n_exact_eligible"] for s in all_stats)
            logger.info(f"  Exact match (weighted) : {100*wt_exact/n_elig:.1f}%  (n={n_elig})")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

# TNM Staging Classification

## Project Overview
Multi-label classification of cancer TNM staging (T1-T4, N0-N3, M0-M1) from TCGA free-text pathology reports using a fine-tuned BioClinical-ModernBERT-base encoder (8192 token context) with three classification heads.

## Tech Stack
- **Python 3.12**, managed with **uv**
- PyTorch, HuggingFace Transformers, scikit-learn
- pytest for testing

## Key Commands
```bash
uv sync --extra dev          # Install all deps including dev
uv run pytest                # Run tests

# Data prep (default: inner join, all-three-labels required)
uv run python src/data_prep.py --reports TCGA_Reports.csv --meta-dir TCGA_Metadata --out-dir data

# Data prep (partial labels: left join, missing labels stored as -1)
uv run python src/data_prep.py --reports TCGA_Reports.csv --meta-dir TCGA_Metadata --out-dir data --partial-labels

# Training (standard)
uv run python src/train.py --data-dir data --output-dir outputs

# Training (two-phase: 2 epochs T+N only, then all heads)
uv run python src/train.py --data-dir data --output-dir outputs --two-phase 2

uv run python src/predict.py --checkpoint outputs/best.pt --input-csv data/val.csv --output-csv submission.csv
uv run python src/eval_metrics.py submission.csv data/val.csv
```

## Architecture
- `src/constants.py` — Shared label mappings and defaults (single source of truth)
- `src/model.py` — TNMClassifier: shared BERT encoder + 3 linear heads
- `src/dataset.py` — PyTorch Dataset with validity masks for partial labels
- `src/data_prep.py` — Data joining, label mapping, stratified splits (supports `--partial-labels`)
- `src/train.py` — Training loop with masked loss per head and optional `--two-phase` training
- `src/predict.py` — Inference from checkpoint
- `src/eval_metrics.py` — F1, AUROC, exact-match evaluation
- `src/explain.py` — Attention-based evidence extraction
- `configs/default.yaml` — Reference training config

## Conventions
- All source code lives in `src/`, tests in `tests/`
- Label mappings are defined once in `src/constants.py` — do not duplicate
- Use Python `logging` module, not `print()`, in all source files
- Catch specific exceptions (ValueError, RuntimeError), not bare `Exception`
- Class weights for M stage are enabled by default (disable with `--no-class-weights-m`)
- Missing labels use sentinel value -1; the training loop masks these out of per-head loss
- Two-phase training (`--two-phase N`): first N epochs train T+N heads only, remaining epochs train all three

## Data (not tracked in git)
- `TCGA_Reports.csv` — Raw pathology reports (~9,500 rows)
- `TCGA_Metadata/` — T/N/M label CSVs
- `data/` — Processed train/val splits (80/20, no test set)
- `outputs/` — Model checkpoints and configs
- `docs/data_analysis.md` — Dataset analysis: data funnel, token distribution, TNM combinations

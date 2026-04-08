# TNM Staging Classification

## Project Overview
Multi-label classification of cancer TNM staging (T1-T4, N0-N3, M0-M1) from TCGA free-text pathology reports. Supports both encoder (BioClinical-ModernBERT) and decoder-only (MedGemma 4B + LoRA) backbones with configurable classification heads (CE or CORAL ordinal).

## Tech Stack
- **Python 3.12**, managed with **uv**
- PyTorch, HuggingFace Transformers, PEFT (LoRA), scikit-learn
- pytest for testing

## Project Structure
```
src/
  models/
    classifier.py     # TNMClassifier: backbone + 3 heads (CE or CORAL)
  data/
    dataset.py        # TNMDataset with validity masks for partial labels
    data_prep.py      # Data joining, label mapping, stratified splits
  constants.py        # Label mappings, defaults (single source of truth)
  train.py            # Training loop (--head-type ce|coral)
  predict.py          # Inference from checkpoint
  eval_metrics.py     # F1, precision, recall, exact-match evaluation
  explain.py          # Attention-based evidence extraction
  tnm_regex.py        # Rule-based TNM extraction from text
  tnm_regex_analysis.py  # Regex coverage/accuracy analysis
configs/
  default.yaml        # Reference training config
docs/
  experiments.md      # Experiment log (approaches, configs, results)
  data_analysis.md    # Dataset analysis
tests/                # pytest test suite
archive/              # Old model versions (baseline v1, CORAL v1)
```

## Key Commands
```bash
uv sync --extra dev          # Install all deps including dev
uv run pytest                # Run tests

# Training — BioClinical-ModernBERT + CE heads (default)
uv run python src/train.py --data-dir data --output-dir outputs --head-type ce

# Training — BioClinical-ModernBERT + CORAL ordinal heads
uv run python src/train.py --data-dir data --output-dir outputs_coral --head-type coral

# Training — MedGemma 4B + LoRA + CE heads
uv run python src/train.py --data-dir data --output-dir outputs_medgemma \
  --encoder google/medgemma-1.5-4b-it --head-type ce \
  --lora-r 16 --lora-alpha 32 --head-lr 1e-3 --lr 2e-4 \
  --batch-size 2 --grad-accum-steps 8

# Training with W&B logging
uv run python src/train.py --data-dir data --output-dir outputs --wandb

# Prediction
uv run python src/predict.py --checkpoint outputs/best.pt --input-csv data/val.csv --output-csv submission.csv

# Evaluation
uv run python src/eval_metrics.py submission.csv data/train.csv
```

## Architecture
- **TNMClassifier** (`src/models/classifier.py`): Unified model supporting:
  - Encoder models (BioClinical-ModernBERT): CLS/mean pooling
  - Decoder-only models (MedGemma): last-token pooling, auto-detected
  - LoRA: optional, enabled with `--lora-r > 0`
  - Head types: `ce` (3x nn.Linear + CrossEntropyLoss) or `coral` (ordinal T/N + binary M)

## Model Defaults
- **Encoder**: `thomas-sounack/BioClinical-ModernBERT-base` (encoder, 149M params, 8192 context)
- **Max length**: 4096 (covers 99%+ of samples)
- **Head type**: CE (cross-entropy)
- **Optimizer**: AdamW, lr=5e-5, warmup + cosine decay

## Data Format
- **train.csv**: `patient_filename, text, t, n, m` — t is 1-indexed (1-4), n/m are 0-indexed, NaN for missing
- **val.csv**: `patient_filename, text` — no labels in raw file; enriched from `TCGA_Metadata/` at load time
- **TCGA_Metadata/**: `TCGA_T14_patients.csv`, `TCGA_N03_patients.csv`, `TCGA_M01_patients.csv` — ground-truth labels joined by `case_submitter_id`
- To pre-enrich val.csv: `uv run python src/data/data_prep.py enrich-val --val-csv data/val.csv`

## Conventions
- All source code lives in `src/`, tests in `tests/`
- Label mappings defined once in `src/constants.py`
- Use Python `logging` module, not `print()`, in all source files
- Missing labels use sentinel value -1; training loop masks these out
- Experiment approaches documented in `docs/experiments.md`
- Old model versions archived in `archive/`

**git commit it and remember to edit related docs if needed**

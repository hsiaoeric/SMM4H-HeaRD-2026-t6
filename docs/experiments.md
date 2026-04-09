# Experiment Log

Tracking all approaches tried for TNM staging classification.

## Dataset

- **Train**: 6,774 samples (with partial labels — NaN for missing T/N/M)
- **Val (submission)**: 2,279 samples (no labels, for final evaluation)
- **Local val**: 15% split from train (stratified on T)
- **Labels**: T (1-4, 0-indexed internally as 0-3), N (0-3), M (0-1)
- **Class imbalance**: M1 is ~6.4% of M labels; T4 is ~10% of T labels

---

## Experiment 1: BioClinical-ModernBERT + CE Heads (baseline v1)

| Field | Value |
|-------|-------|
| **Backbone** | thomas-sounack/BioClinical-ModernBERT-base (encoder, 149M params) |
| **Heads** | 3x nn.Linear (T=4, N=4, M=2) + CrossEntropyLoss |
| **Max length** | 512 |
| **Dataset** | Old format (3,898 rows, all three labels required) |
| **Optimizer** | AdamW, lr=5e-5 |
| **Status** | Completed (old dataset, results invalid for current eval) |
| **Notes** | 54.7% of samples truncated at 512 tokens. Used regex hint features. |

**Results**: N/A (run on old dataset with different splits — not comparable)

**Takeaways**:
- Encoder model is natural fit for classification
- 512 max_length too short — significant truncation loss
- Regex hints provided marginal improvement

---

## Experiment 2: MedGemma 4B + LoRA + CORAL Ordinal Heads (v2)

| Field | Value |
|-------|-------|
| **Backbone** | google/medgemma-1.5-4b-it (decoder-only, 4B params, bf16) |
| **LoRA** | r=16, alpha=32, targets=q_proj+v_proj (~0.3% trainable) |
| **Heads** | CORAL ordinal for T/N (K-1 thresholds), binary for M |
| **Max length** | 2048 |
| **Dataset** | Old format (3,898 rows) |
| **Optimizer** | AdamW, backbone_lr=2e-4, head_lr=1e-3, cosine schedule |
| **Status** | Completed (old dataset, results invalid) |

**Results**: N/A (old dataset)

**Takeaways**:
- Decoder-only model is wasteful for classification (4B params, only use last-token pooling)
- CORAL adds complexity without clear benefit over CE for this task
- Very slow training (4B params even with LoRA)
- Unclear if poor results were from decoder-only architecture, CORAL, or both

---

## Experiment 3: BioClinical-ModernBERT + CE Heads (v3 — current)

| Field | Value |
|-------|-------|
| **Backbone** | thomas-sounack/BioClinical-ModernBERT-base (encoder, 149M params) |
| **Heads** | 3x nn.Linear (T=4, N=4, M=2) + CrossEntropyLoss |
| **Max length** | 4096 (covers 99%+ of samples; model supports 8192) |
| **Dataset** | New format (6,774 rows, partial labels with NaN/-1 masking) |
| **Optimizer** | AdamW, lr=5e-5, warmup 10%, cosine decay |
| **Status** | **TODO** |

**Config**:
```bash
uv run python src/train.py --data-dir data --output-dir outputs --head-type ce --max-length 4096
```

---

## Experiment 4: BioClinical-ModernBERT + CORAL Heads (v3 variant)

| Field | Value |
|-------|-------|
| **Backbone** | thomas-sounack/BioClinical-ModernBERT-base |
| **Heads** | CORAL ordinal for T/N, binary for M |
| **Max length** | 4096 |
| **Dataset** | New format (6,774 rows) |
| **Status** | **TODO** |

**Config**:
```bash
uv run python src/train.py --data-dir data --output-dir outputs_coral --head-type coral --max-length 4096
```

---

## Experiment 5: MedGemma 4B + LoRA + CE Heads (v3 variant)

| Field | Value |
|-------|-------|
| **Backbone** | google/medgemma-1.5-4b-it + LoRA (r=16) |
| **Heads** | 3x nn.Linear + CE |
| **Max length** | 2048 |
| **Dataset** | New format (6,774 rows) |
| **Status** | **TODO** |

**Config**:
```bash
uv run python src/train.py --data-dir data --output-dir outputs_medgemma_ce \
  --encoder google/medgemma-1.5-4b-it --head-type ce --max-length 2048 \
  --lora-r 16 --lora-alpha 32 --head-lr 1e-3 --lr 2e-4 \
  --batch-size 2 --grad-accum-steps 8
```

---

## Planned Experiments

- [ ] Exp 3: ModernBERT + CE (baseline on new data)
- [ ] Exp 4: ModernBERT + CORAL (test if ordinal helps with encoder)
- [ ] Exp 5: MedGemma + CE (isolate decoder-only vs CORAL effect)
- [ ] Class-weighted CE for T/N (address T4, N3 imbalance)
- [ ] Two-phase training (T+N first, then all)
- [ ] Ensemble: regex extraction + model predictions

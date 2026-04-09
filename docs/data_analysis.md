# TNM Dataset Analysis

Analysis date: 2026-03-28

## Data Funnel

| Stage | Count | Lost |
|-------|-------|------|
| Original reports | 9,523 | - |
| Have T metadata | 6,966 | 2,557 |
| Have N metadata | 5,678 | 3,845 |
| Have M metadata | 4,608 | 4,915 |
| Valid T (drop T0/TX) | 6,887 | 79 dropped |
| Valid N (drop NX etc) | 5,678 | 0 dropped |
| Valid M (drop MX) | 4,608 | 0 dropped |
| **All three valid TNM** | **3,898** | **5,625 (59.1%)** |

The biggest bottleneck is M metadata availability: only 4,608 of 9,523 patients
have M-stage labels. Requiring all three valid labels drops 59.1% of original reports.

## Token Length Distribution (Bio_ClinicalBERT tokenizer)

| Stat | Tokens |
|------|--------|
| Mean | 864 |
| Median | 608 |
| Min / Max | 29 / 5,282 |
| 25th / 75th pct | 249 / 1,209 |

| Bucket | Count |
|--------|-------|
| 0-128 | 289 |
| 129-256 | 714 |
| 257-512 | 762 |
| 513-1024 | 936 |
| 1025-2048 | 845 |
| 2048+ | 352 |

**54.7% of samples exceed the 512-token model limit** and are truncated.

## Individual Label Distributions

| T | Count | N | Count | M | Count |
|---|-------|---|-------|---|-------|
| T1 | 876 | N0 | 2,261 | M0 | 3,648 |
| T2 | 1,373 | N1 | 1,006 | M1 | 250 |
| T3 | 1,196 | N2 | 472 | | |
| T4 | 453 | N3 | 159 | | |

## TNM Combination Counts (30 of 32 possible)

| Combo | Count | | Combo | Count |
|-------|-------|-|-------|-------|
| T1_N0_M0 | 696 | | T3_N0_M0 | 545 |
| T1_N0_M1 | 7 | | T3_N0_M1 | 37 |
| T1_N1_M0 | 133 | | T3_N1_M0 | 332 |
| T1_N1_M1 | 5 | | T3_N1_M1 | 47 |
| T1_N2_M0 | 32 | | T3_N2_M0 | 143 |
| T1_N2_M1 | 0 | | T3_N2_M1 | 32 |
| T1_N3_M0 | 3 | | T3_N3_M0 | 49 |
| T1_N3_M1 | 0 | | T3_N3_M1 | 11 |
| T2_N0_M0 | 774 | | T4_N0_M0 | 169 |
| T2_N0_M1 | 21 | | T4_N0_M1 | 12 |
| T2_N1_M0 | 371 | | T4_N1_M0 | 82 |
| T2_N1_M1 | 12 | | T4_N1_M1 | 24 |
| T2_N2_M0 | 143 | | T4_N2_M0 | 91 |
| T2_N2_M1 | 7 | | T4_N2_M1 | 24 |
| T2_N3_M0 | 44 | | T4_N3_M0 | 41 |
| T2_N3_M1 | 1 | | T4_N3_M1 | 10 |

Missing combinations: T1_N2_M1, T1_N3_M1.
Several combos have fewer than 10 samples (T1_N3_M0: 3, T2_N3_M1: 1, T1_N1_M1: 5).

## Key Observations

1. **M1 is very rare** (250/3,898 = 6.4%) -- heavy class imbalance justifies class weights for M.
2. **59.1% data loss** from requiring all three labels -- addressed by partial-label training.
3. **54.7% truncation** at 512 tokens -- resolved by switching to BioClinical-ModernBERT-base (8192 token context). Only 352 samples (9%) exceed 2048 tokens; none exceed the 8192 limit.

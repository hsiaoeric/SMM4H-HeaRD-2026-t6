"""
Benchmark training throughput across batch sizes.

Runs a few steps of real training for each (batch_size, grad_accum) combo,
measures wall-clock time, and reports samples/sec. Auto-skips OOM configs.

Usage:
    uv run python src/bench_batch_size.py
    uv run python src/bench_batch_size.py --encoder google/medgemma-1.5-4b-it --lora-r 16
"""
import argparse
import gc
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from constants import (
    DEFAULT_ENCODER, DEFAULT_MAX_LENGTH,
    T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS,
    DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
)
from data.dataset import TNMDataset
from models.classifier import TNMClassifier
from train import masked_ce_loss, coral_loss, binary_loss

logger = logging.getLogger(__name__)

# Warmup steps discarded from timing; bench steps measured
WARMUP_STEPS = 2
BENCH_STEPS = 5


def make_synthetic_data(n, max_length, tokenizer):
    """Create synthetic dataset that matches real tokenizer output shapes."""
    vocab_size = tokenizer.vocab_size
    input_ids = np.random.randint(1, vocab_size, (n, max_length))
    attention_mask = np.ones((n, max_length), dtype=int)
    # Random labels
    labels_t = np.random.randint(0, T_NUM_LABELS, n)
    labels_n = np.random.randint(0, N_NUM_LABELS, n)
    labels_m = np.random.randint(0, M_NUM_LABELS, n)
    encodings = {"input_ids": input_ids, "attention_mask": attention_mask}
    return TNMDataset(encodings, labels_t, labels_n, labels_m)


def bench_one(model, loader, device, head_type, grad_accum, max_steps):
    """Run training steps and return (samples_per_sec, peak_mem_mb) or None on OOM."""
    criterion_t = nn.CrossEntropyLoss()
    criterion_n = nn.CrossEntropyLoss()
    criterion_m = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )
    model.train()
    optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    total_steps = WARMUP_STEPS + max_steps
    step = 0
    total_samples = 0
    start_time = None

    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels_t"].to(device)
            labels_n = batch["labels_n"].to(device)
            labels_m = batch["labels_m"].to(device)
            mask_t = batch["mask_t"].to(device)
            mask_n = batch["mask_n"].to(device)
            mask_m = batch["mask_m"].to(device)

            logits_t, logits_n, logits_m = model(
                input_ids=input_ids, attention_mask=attention_mask,
            )

            if head_type == "coral":
                loss = (coral_loss(logits_t, labels_t, mask_t)
                        + coral_loss(logits_n, labels_n, mask_n)
                        + binary_loss(logits_m, labels_m, mask_m))
            else:
                loss = (masked_ce_loss(criterion_t, logits_t, labels_t, mask_t)
                        + masked_ce_loss(criterion_n, logits_n, labels_n, mask_n)
                        + masked_ce_loss(criterion_m, logits_m, labels_m, mask_m))

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step == WARMUP_STEPS:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
            elif step > WARMUP_STEPS:
                total_samples += input_ids.size(0)

            if step >= total_steps:
                break

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        peak_mem = 0.0
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6

        return total_samples / elapsed, peak_mem

    except torch.cuda.OutOfMemoryError:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--head-type", choices=["ce", "coral"], default="ce")
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-targets", nargs="+", default=None)
    parser.add_argument("--effective-batch", type=int, default=16,
                        help="Target effective batch size (batch * grad_accum)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[2, 4, 8, 16, 32],
                        help="Batch sizes to try")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.9,
        help="Optional CUDA per-process memory fraction (0,1], e.g. 0.9",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        if args.cuda_memory_fraction is not None:
            if not (0.0 < args.cuda_memory_fraction <= 1.0):
                raise ValueError("--cuda-memory-fraction must be in (0, 1].")
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
            logger.info("Set CUDA per-process memory fraction: %.2f", args.cuda_memory_fraction)
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Synthetic data — enough for all configs
    max_bs = max(args.batch_sizes)
    n_samples = max_bs * (WARMUP_STEPS + BENCH_STEPS + 2)
    ds = make_synthetic_data(n_samples, args.max_length, tokenizer)

    results = []
    for bs in args.batch_sizes:
        grad_accum = max(1, args.effective_batch // bs)
        eff_bs = bs * grad_accum

        # Build fresh model each time to avoid leftover state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        torch_dtype = torch.bfloat16 if args.lora_r > 0 else torch.float32
        model = TNMClassifier(
            encoder_name=args.encoder,
            t_num_labels=T_NUM_LABELS, n_num_labels=N_NUM_LABELS, m_num_labels=M_NUM_LABELS,
            head_type=args.head_type, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, lora_targets=args.lora_targets,
            torch_dtype=torch_dtype,
        ).to(device)

        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

        logger.info("--- batch_size=%d  grad_accum=%d  effective=%d ---", bs, grad_accum, eff_bs)
        throughput, peak_mem = bench_one(
            model, loader, device, args.head_type, grad_accum, BENCH_STEPS,
        )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if throughput is None:
            logger.info("  OOM — skipped")
            results.append({"batch_size": bs, "grad_accum": grad_accum,
                            "effective_batch": eff_bs, "status": "OOM"})
        else:
            logger.info("  %.1f samples/sec  |  peak GPU: %.0f MB", throughput, peak_mem)
            results.append({"batch_size": bs, "grad_accum": grad_accum,
                            "effective_batch": eff_bs, "status": "OK",
                            "samples_per_sec": round(throughput, 1),
                            "peak_mem_mb": round(peak_mem)})

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'BS':>4}  {'Accum':>5}  {'Eff BS':>6}  {'Status':>6}  {'Samp/s':>8}  {'GPU MB':>8}")
    print("-" * 70)
    for r in results:
        if r["status"] == "OOM":
            print(f"{r['batch_size']:>4}  {r['grad_accum']:>5}  {r['effective_batch']:>6}  {'OOM':>6}  {'---':>8}  {'---':>8}")
        else:
            print(f"{r['batch_size']:>4}  {r['grad_accum']:>5}  {r['effective_batch']:>6}  {'OK':>6}  {r['samples_per_sec']:>8.1f}  {r['peak_mem_mb']:>8}")
    print("=" * 70)

    # Recommendation
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        best = max(ok_results, key=lambda r: r["samples_per_sec"])
        print(f"\nRecommendation: --batch-size {best['batch_size']} --grad-accum-steps {best['grad_accum']}")
        print(f"  ({best['samples_per_sec']:.1f} samples/sec, {best['peak_mem_mb']} MB peak)")


if __name__ == "__main__":
    main()

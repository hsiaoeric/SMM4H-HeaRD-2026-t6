"""
Train TNM staging model: single encoder + three heads.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from constants import DEFAULT_ENCODER, DEFAULT_MAX_LENGTH, T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS
from dataset import TNMDataset
from model import TNMClassifier
from tnm_regex import encode_hints

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(criterion, logits, labels, msk):
    """Compute loss only on samples where the label is valid (mask=True).

    Returns 0.0 if no valid samples in the batch.
    """
    if mask.any():
        return criterion(logits[mask], labels[mask])
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m,
                    mask_t=None, mask_n=None, mask_m=None):
    """F1 (macro) per component, exact-match. Respects validity masks."""
    pred_t, pred_n, pred_m = np.array(pred_t), np.array(pred_n), np.array(pred_m)
    true_t, true_n, true_m = np.array(true_t), np.array(true_n), np.array(true_m)

    if mask_t is None:
        mask_t = np.ones(len(true_t), dtype=bool)
    if mask_n is None:
        mask_n = np.ones(len(true_n), dtype=bool)
    if mask_m is None:
        mask_m = np.ones(len(true_m), dtype=bool)

    f1_t = f1_score(true_t[mask_t], pred_t[mask_t], average="macro", zero_division=0) if mask_t.any() else 0.0
    f1_n = f1_score(true_n[mask_n], pred_n[mask_n], average="macro", zero_division=0) if mask_n.any() else 0.0
    f1_m = f1_score(true_m[mask_m], pred_m[mask_m], average="macro", zero_division=0) if mask_m.any() else 0.0

    # Exact match only on samples with ALL labels valid
    all_valid = mask_t & mask_n & mask_m
    if all_valid.any():
        exact = float(np.mean(
            (pred_t[all_valid] == true_t[all_valid])
            & (pred_n[all_valid] == true_n[all_valid])
            & (pred_m[all_valid] == true_m[all_valid])
        ))
    else:
        exact = 0.0

    return {
        "f1_t": float(f1_t),
        "f1_n": float(f1_n),
        "f1_m": float(f1_m),
        "f1_macro_avg": float((f1_t + f1_n + f1_m) / 3),
        "exact_match": float(exact),
        "n_valid_t": int(mask_t.sum()),
        "n_valid_n": int(mask_n.sum()),
        "n_valid_m": int(mask_m.sum()),
    }


def train_epoch(model, loader, optimizer, device, criterion_t, criterion_n, criterion_m,
                active_heads=("t", "n", "m")):
    """Train one epoch with masked loss per head.

    Args:
        active_heads: Which heads to train. Use ("t", "n") for phase-1 of
                      two-phase training (freeze M head).
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t = batch["labels_t"].to(device)
        labels_n = batch["labels_n"].to(device)
        labels_m = batch["labels_m"].to(device)
        mask_t = batch["mask_t"].to(device)
        mask_n = batch["mask_n"].to(device)
        mask_m = batch["mask_m"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        hint_t = batch["hint_t"].to(device) if "hint_t" in batch else None
        hint_n = batch["hint_n"].to(device) if "hint_n" in batch else None
        hint_m = batch["hint_m"].to(device) if "hint_m" in batch else None

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            hint_t=hint_t,
            hint_n=hint_n,
            hint_m=hint_m,
        )

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if "t" in active_heads:
            loss = loss + masked_loss(criterion_t, logits_t, labels_t, mask_t)
        if "n" in active_heads:
            loss = loss + masked_loss(criterion_n, logits_n, labels_n, mask_n)
        if "m" in active_heads:
            loss = loss + masked_loss(criterion_m, logits_m, labels_m, mask_m)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_t, preds_n, preds_m = [], [], []
    trues_t, trues_n, trues_m = [], [], []
    masks_t, masks_n, masks_m = [], [], []
    probs_m = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        hint_t = batch["hint_t"].to(device) if "hint_t" in batch else None
        hint_n = batch["hint_n"].to(device) if "hint_n" in batch else None
        hint_m = batch["hint_m"].to(device) if "hint_m" in batch else None
        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            hint_t=hint_t,
            hint_n=hint_n,
            hint_m=hint_m,
        )
        preds_t.append(logits_t.argmax(1).cpu().numpy())
        preds_n.append(logits_n.argmax(1).cpu().numpy())
        preds_m.append(logits_m.argmax(1).cpu().numpy())
        probs_m.append(torch.softmax(logits_m, 1).cpu().numpy())
        trues_t.append(batch["labels_t"].numpy())
        trues_n.append(batch["labels_n"].numpy())
        trues_m.append(batch["labels_m"].numpy())
        masks_t.append(batch["mask_t"].numpy())
        masks_n.append(batch["mask_n"].numpy())
        masks_m.append(batch["mask_m"].numpy())
    preds_t = np.concatenate(preds_t)
    preds_n = np.concatenate(preds_n)
    preds_m = np.concatenate(preds_m)
    trues_t = np.concatenate(trues_t)
    trues_n = np.concatenate(trues_n)
    trues_m = np.concatenate(trues_m)
    masks_t = np.concatenate(masks_t).astype(bool)
    masks_n = np.concatenate(masks_n).astype(bool)
    masks_m = np.concatenate(masks_m).astype(bool)
    probs_m = np.concatenate(probs_m)

    metrics = compute_metrics(preds_t, preds_n, preds_m, trues_t, trues_n, trues_m,
                              masks_t, masks_n, masks_m)
    try:
        if masks_m.any():
            metrics["auroc_m"] = float(roc_auc_score(trues_m[masks_m], probs_m[masks_m, 1]))
        else:
            metrics["auroc_m"] = 0.0
    except ValueError:
        metrics["auroc_m"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-class-weights-m", action="store_true",
                        help="Disable class weighting for M stage")
    parser.add_argument("--two-phase", type=int, default=0, metavar="N",
                        help="Two-phase training: train T+N heads for N epochs first, "
                             "then all heads for remaining epochs. 0 = disabled (default).")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Path to a checkpoint (.pt) to resume training from.")
    parser.add_argument("--regex-hints", action="store_true",
                        help="Augment the model with regex-extracted TNM hint embeddings.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    use_class_weights_m = not args.no_class_weights_m

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    texts_train = train_df["text"].astype(str).tolist()
    texts_val = val_df["text"].astype(str).tolist()
    t_train = train_df["T"].values
    n_train = train_df["N"].values
    m_train = train_df["M"].values
    t_val = val_df["T"].values
    n_val = val_df["N"].values
    m_val = val_df["M"].values

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    enc_train = tokenizer(
        texts_train,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    enc_val = tokenizer(
        texts_val,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )

    if args.regex_hints:
        logger.info("Extracting regex hints for train set...")
        ht_train, hn_train, hm_train = encode_hints(texts_train)
        logger.info("Extracting regex hints for val set...")
        ht_val, hn_val, hm_val = encode_hints(texts_val)
    else:
        ht_train = hn_train = hm_train = None
        ht_val = hn_val = hm_val = None

    train_ds = TNMDataset(enc_train, t_train, n_train, m_train,
                          hint_t=ht_train, hint_n=hn_train, hint_m=hm_train)
    val_ds = TNMDataset(enc_val, t_val, n_val, m_val,
                        hint_t=ht_val, hint_n=hn_val, hint_m=hm_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class weights for M (only computed from valid M labels)
    valid_m_train = m_train[m_train >= 0]
    if use_class_weights_m and len(valid_m_train) > 0:
        m_counts = np.bincount(valid_m_train, minlength=M_NUM_LABELS)
        m_weights = 1.0 / (m_counts + 1e-6)
        m_weights = m_weights / m_weights.sum() * M_NUM_LABELS
        weight_m = torch.tensor(m_weights, dtype=torch.float32).to(device)
    else:
        weight_m = None
    criterion_t = nn.CrossEntropyLoss()
    criterion_n = nn.CrossEntropyLoss()
    criterion_m = nn.CrossEntropyLoss(weight=weight_m)

    model = TNMClassifier(
        encoder_name=args.encoder,
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
        use_regex_hints=args.regex_hints,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_exact = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_exact = ckpt.get("metrics", {}).get("exact_match", 0.0)
        logger.info("Resumed from %s (epoch %d, best_exact=%.4f)", args.resume, start_epoch, best_exact)

    phase1_epochs = args.two_phase
    for epoch in range(start_epoch, args.epochs):
        # Two-phase: first N epochs train only T+N, remaining train all
        if phase1_epochs > 0 and epoch < phase1_epochs:
            active_heads = ("t", "n")
            phase_label = "phase1(T+N)"
        else:
            active_heads = ("t", "n", "m")
            phase_label = "phase2(all)" if phase1_epochs > 0 else "all"

        loss_avg = train_epoch(
            model, train_loader, optimizer, device,
            criterion_t, criterion_n, criterion_m,
            active_heads=active_heads,
        )
        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d [%s] loss=%.4f F1_T=%.4f F1_N=%.4f F1_M=%.4f exact_match=%.4f",
            epoch + 1, phase_label, loss_avg,
            metrics["f1_t"], metrics["f1_n"], metrics["f1_m"],
            metrics["exact_match"],
        )
        if metrics["exact_match"] > best_exact:
            best_exact = metrics["exact_match"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                os.path.join(args.output_dir, "best.pt"),
            )
    # Save config for predict
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Best exact_match=%.4f saved to %s/best.pt", best_exact, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

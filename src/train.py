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

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m):
    """F1 (macro) per component, AUROC where applicable, exact-match."""
    f1_t = f1_score(true_t, pred_t, average="macro", zero_division=0)
    f1_n = f1_score(true_n, pred_n, average="macro", zero_division=0)
    f1_m = f1_score(true_m, pred_m, average="macro", zero_division=0)
    exact = np.mean(
        (np.array(pred_t) == np.array(true_t))
        & (np.array(pred_n) == np.array(true_n))
        & (np.array(pred_m) == np.array(true_m))
    )
    return {
        "f1_t": float(f1_t),
        "f1_n": float(f1_n),
        "f1_m": float(f1_m),
        "f1_macro_avg": float((f1_t + f1_n + f1_m) / 3),
        "exact_match": float(exact),
    }


def train_epoch(model, loader, optimizer, device, criterion_t, criterion_n, criterion_m):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t = batch["labels_t"].to(device)
        labels_n = batch["labels_n"].to(device)
        labels_m = batch["labels_m"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss_t = criterion_t(logits_t, labels_t)
        loss_n = criterion_n(logits_n, labels_n)
        loss_m = criterion_m(logits_m, labels_m)
        loss = loss_t + loss_n + loss_m
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
    probs_m = []
    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        preds_t.append(logits_t.argmax(1).cpu().numpy())
        preds_n.append(logits_n.argmax(1).cpu().numpy())
        preds_m.append(logits_m.argmax(1).cpu().numpy())
        probs_m.append(torch.softmax(logits_m, 1).cpu().numpy())
        trues_t.append(batch["labels_t"].numpy())
        trues_n.append(batch["labels_n"].numpy())
        trues_m.append(batch["labels_m"].numpy())
    preds_t = np.concatenate(preds_t)
    preds_n = np.concatenate(preds_n)
    preds_m = np.concatenate(preds_m)
    trues_t = np.concatenate(trues_t)
    trues_n = np.concatenate(trues_n)
    trues_m = np.concatenate(trues_m)
    probs_m = np.concatenate(probs_m)

    metrics = compute_metrics(preds_t, preds_n, preds_m, trues_t, trues_n, trues_m)
    try:
        metrics["auroc_m"] = float(roc_auc_score(trues_m, probs_m[:, 1]))
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

    train_ds = TNMDataset(enc_train, t_train, n_train, m_train)
    val_ds = TNMDataset(enc_val, t_val, n_val, m_val)
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

    # Class weights for M
    if use_class_weights_m:
        m_counts = np.bincount(m_train, minlength=M_NUM_LABELS)
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
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_exact = 0.0
    for epoch in range(args.epochs):
        loss_avg = train_epoch(
            model, train_loader, optimizer, device,
            criterion_t, criterion_n, criterion_m,
        )
        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d loss=%.4f F1_T=%.4f F1_N=%.4f F1_M=%.4f exact_match=%.4f",
            epoch + 1, loss_avg,
            metrics["f1_t"], metrics["f1_n"], metrics["f1_m"],
            metrics["exact_match"],
        )
        if metrics["exact_match"] > best_exact:
            best_exact = metrics["exact_match"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
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

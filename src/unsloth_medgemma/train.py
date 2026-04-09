"""
Train TNM staging model: MedGemma 4B + LoRA with CORAL ordinal regression.
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # avoid Triton shared-memory OOM on some GPUs

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel

from constants import (
    DEFAULT_ENCODER, DEFAULT_MAX_LENGTH,
    T_NUM_LABELS, N_NUM_LABELS,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_TARGETS,
)
from dataset import TNMDataset
from model import TNMOrdinalClassifier, coral_loss, binary_loss, coral_predict, binary_predict

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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

    def _f1(y_true, y_pred, avg):
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    def _prec(y_true, y_pred, avg):
        return float(precision_score(y_true, y_pred, average=avg, zero_division=0))

    def _rec(y_true, y_pred, avg):
        return float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    tt, pt = true_t[mask_t], pred_t[mask_t]
    tn, pn = true_n[mask_n], pred_n[mask_n]
    tm, pm = true_m[mask_m], pred_m[mask_m]

    f1_t = _f1(tt, pt, "macro") if mask_t.any() else 0.0
    f1_n = _f1(tn, pn, "macro") if mask_n.any() else 0.0
    f1_m = _f1(tm, pm, "macro") if mask_m.any() else 0.0

    mi_f1_t = _f1(tt, pt, "micro") if mask_t.any() else 0.0
    mi_pr_t = _prec(tt, pt, "micro") if mask_t.any() else 0.0
    mi_re_t = _rec(tt, pt, "micro") if mask_t.any() else 0.0
    mi_f1_n = _f1(tn, pn, "micro") if mask_n.any() else 0.0
    mi_pr_n = _prec(tn, pn, "micro") if mask_n.any() else 0.0
    mi_re_n = _rec(tn, pn, "micro") if mask_n.any() else 0.0
    mi_f1_m = _f1(tm, pm, "micro") if mask_m.any() else 0.0
    mi_pr_m = _prec(tm, pm, "micro") if mask_m.any() else 0.0
    mi_re_m = _rec(tm, pm, "micro") if mask_m.any() else 0.0

    ma_pr_t = _prec(tt, pt, "macro") if mask_t.any() else 0.0
    ma_re_t = _rec(tt, pt, "macro") if mask_t.any() else 0.0
    ma_pr_n = _prec(tn, pn, "macro") if mask_n.any() else 0.0
    ma_re_n = _rec(tn, pn, "macro") if mask_n.any() else 0.0
    ma_pr_m = _prec(tm, pm, "macro") if mask_m.any() else 0.0
    ma_re_m = _rec(tm, pm, "macro") if mask_m.any() else 0.0

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
        "micro_f1": (mi_f1_t + mi_f1_n + mi_f1_m) / 3,
        "micro_precision": (mi_pr_t + mi_pr_n + mi_pr_m) / 3,
        "micro_recall": (mi_re_t + mi_re_n + mi_re_m) / 3,
        "macro_precision": (ma_pr_t + ma_pr_n + ma_pr_m) / 3,
        "macro_recall": (ma_re_t + ma_re_n + ma_re_m) / 3,
        "n_valid_t": int(mask_t.sum()),
        "n_valid_n": int(mask_n.sum()),
        "n_valid_m": int(mask_m.sum()),
    }


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device, grad_accum_steps,
                active_heads=("t", "n", "m"), logging_steps=10, wandb_run=None):
    """Train one epoch with CORAL loss for T/N, BCE for M, gradient accumulation.

    Args:
        logging_steps: Log running loss every N optimizer steps.
    """
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    optimizer.zero_grad()
    global_opt_step = 0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t = batch["labels_t"].to(device)
        labels_n = batch["labels_n"].to(device)
        labels_m = batch["labels_m"].to(device)
        mask_t = batch["mask_t"].to(device)
        mask_n = batch["mask_n"].to(device)
        mask_m = batch["mask_m"].to(device)

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if "t" in active_heads:
            loss = loss + coral_loss(logits_t, labels_t, mask_t)
        if "n" in active_heads:
            loss = loss + coral_loss(logits_n, labels_n, mask_n)
        if "m" in active_heads:
            loss = loss + binary_loss(logits_m, labels_m, mask_m)

        loss = loss / grad_accum_steps
        loss.backward()

        running_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_opt_step += 1

            if global_opt_step % logging_steps == 0:
                avg = running_loss / (logging_steps * grad_accum_steps)
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info("  step %d  loss=%.4f  lr=%.2e", global_opt_step, avg, lr_now)
                if wandb_run is not None:
                    wandb_run.log({"train/step_loss": avg, "train/lr": lr_now})
                running_loss = 0.0

        total_loss += loss.item() * grad_accum_steps
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_t, preds_n, preds_m = [], [], []
    trues_t, trues_n, trues_m = [], [], []
    masks_t, masks_n, masks_m = [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds_t.append(coral_predict(logits_t).cpu().numpy())
        preds_n.append(coral_predict(logits_n).cpu().numpy())
        preds_m.append(binary_predict(logits_m).cpu().numpy())
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

    metrics = compute_metrics(preds_t, preds_n, preds_m, trues_t, trues_n, trues_m,
                              masks_t, masks_n, masks_m)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Learning rate for classification heads")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--two-phase", type=int, default=0, metavar="N",
                        help="Two-phase: train T+N heads for N epochs first, then all.")
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Path to a checkpoint (.pt) to resume training from.")
    # LoRA args
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-targets", nargs="+", default=None,
                        help="LoRA target module names (default: q_proj v_proj)")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load base model in 4-bit (QLoRA). Default: bf16 LoRA.")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log training loss every N optimizer steps (default: 10)")
    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tnm-staging")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        logger.info("W&B run: %s", wandb_run.url)

    # ---- Model (Unsloth) ----
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.encoder,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_4bit,
        dtype=torch.bfloat16,
    )
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets or DEFAULT_LORA_TARGETS,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_length,
    )
    # Get hidden size from config (Gemma3 nests it under text_config)
    config = base_model.config
    if hasattr(config, "text_config"):
        hidden_size = config.text_config.hidden_size
    else:
        hidden_size = config.hidden_size

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Data ----
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    texts_train = train_df["text"].astype(str).tolist()
    texts_val = val_df["text"].astype(str).tolist()
    t_train, n_train, m_train = train_df["T"].values, train_df["N"].values, train_df["M"].values
    t_val, n_val, m_val = val_df["T"].values, val_df["N"].values, val_df["M"].values

    enc_train = tokenizer(
        texts_train, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="np",
    )
    enc_val = tokenizer(
        texts_val, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="np",
    )

    train_ds = TNMDataset(enc_train, t_train, n_train, m_train)
    val_ds = TNMDataset(enc_val, t_val, n_val, m_val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
    )

    # Wrap backbone with CORAL/binary heads
    model = TNMOrdinalClassifier(
        encoder=base_model,
        hidden_size=hidden_size,
        t_num_classes=T_NUM_LABELS,
        n_num_classes=N_NUM_LABELS,
        dropout=0.1,
    ).to(device)

    # Separate param groups: LoRA params at lower LR, heads at higher LR
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ["t_head", "n_head", "m_head"]):
            head_params.append(param)
        else:
            lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ])

    # Linear warmup + cosine decay
    total_steps = (len(train_loader) // args.grad_accum_steps + 1) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_exact = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_exact = ckpt.get("metrics", {}).get("exact_match", 0.0)
        logger.info("Resumed from %s (epoch %d, best_exact=%.4f)",
                     args.resume, start_epoch, best_exact)

    phase1_epochs = args.two_phase
    for epoch in range(start_epoch, args.epochs):
        if phase1_epochs > 0 and epoch < phase1_epochs:
            active_heads = ("t", "n")
            phase_label = "phase1(T+N)"
        else:
            active_heads = ("t", "n", "m")
            phase_label = "phase2(all)" if phase1_epochs > 0 else "all"

        loss_avg = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_accum_steps, active_heads=active_heads,
            logging_steps=args.logging_steps, wandb_run=wandb_run,
        )
        metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d [%s] loss=%.4f F1_T=%.4f F1_N=%.4f F1_M=%.4f exact_match=%.4f",
            epoch + 1, phase_label, loss_avg,
            metrics["f1_t"], metrics["f1_n"], metrics["f1_m"],
            metrics["exact_match"],
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": loss_avg,
                "val/f1_t": metrics["f1_t"],
                "val/f1_n": metrics["f1_n"],
                "val/f1_m": metrics["f1_m"],
                "val/f1_macro_avg": metrics["f1_macro_avg"],
                "val/exact_match": metrics["exact_match"],
                "val/micro_f1": metrics["micro_f1"],
                "val/micro_precision": metrics["micro_precision"],
                "val/micro_recall": metrics["micro_recall"],
                "val/macro_f1": metrics["f1_macro_avg"],
                "val/macro_precision": metrics["macro_precision"],
                "val/macro_recall": metrics["macro_recall"],
            })
        if metrics["exact_match"] > best_exact:
            best_exact = metrics["exact_match"]
            torch.save(
                {
                    "trainable_state_dict": model.get_trainable_state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

    # Save config for predict
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Best exact_match=%.4f saved to %s/best.pt", best_exact, args.output_dir)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())

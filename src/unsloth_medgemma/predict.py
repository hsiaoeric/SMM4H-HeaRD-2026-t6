"""
Load best checkpoint and run on a CSV with 'text' column; write submission CSV with T, N, M labels.
Uses Unsloth + LoRA with CORAL ordinal decoding for T/N and binary for M.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel

from constants import (
    DEFAULT_ENCODER, DEFAULT_MAX_LENGTH,
    T_NUM_LABELS, N_NUM_LABELS, DEFAULT_LORA_TARGETS,
)
from dataset import TNMDataset
from model import TNMOrdinalClassifier, coral_predict, binary_predict

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--id-col", default="patient_filename")
    p.add_argument("--load-4bit", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(ckpt_dir, "train_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            train_config = json.load(f)
        encoder_name = train_config.get("encoder", DEFAULT_ENCODER)
        max_length = train_config.get("max_length", DEFAULT_MAX_LENGTH)
        lora_r = train_config.get("lora_r", 16)
        lora_alpha = train_config.get("lora_alpha", 32)
        lora_dropout = train_config.get("lora_dropout", 0.1)
        lora_targets = train_config.get("lora_targets", None)
        load_4bit = train_config.get("load_4bit", args.load_4bit)
    else:
        logger.warning("Config not found at %s, using defaults.", config_path)
        encoder_name = DEFAULT_ENCODER
        max_length = args.max_length
        lora_r, lora_alpha, lora_dropout, lora_targets = 16, 32, 0.1, None
        load_4bit = args.load_4bit

    # Load backbone via Unsloth
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=encoder_name,
        max_seq_length=max_length,
        load_in_4bit=load_4bit,
        dtype=torch.bfloat16,
    )
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets or DEFAULT_LORA_TARGETS,
        lora_dropout=lora_dropout,
        bias="none",
        max_seq_length=max_length,
    )
    config = base_model.config
    if hasattr(config, "text_config"):
        hidden_size = config.text_config.hidden_size
    else:
        hidden_size = config.hidden_size

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        sys.exit(f"Input CSV must have 'text' column. Found columns: {list(df.columns)}")
    texts = df["text"].astype(str).tolist()
    if args.id_col not in df.columns:
        logger.warning("ID column '%s' not found, using first column '%s'.",
                        args.id_col, df.columns[0])
    id_col = args.id_col if args.id_col in df.columns else df.columns[0]
    ids = df[id_col].astype(str).tolist()

    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="np",
    )
    n = len(texts)
    ds = TNMDataset(enc, np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TNMOrdinalClassifier(
        encoder=base_model,
        hidden_size=hidden_size,
        t_num_classes=T_NUM_LABELS,
        n_num_classes=N_NUM_LABELS,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model.to(device)
    FastLanguageModel.for_inference(model.encoder)

    preds_t, preds_n, preds_m = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits_t, logits_n, logits_m = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds_t.append(coral_predict(logits_t).cpu().numpy())
            preds_n.append(coral_predict(logits_n).cpu().numpy())
            preds_m.append(binary_predict(logits_m).cpu().numpy())

    preds_t = np.concatenate(preds_t)
    preds_n = np.concatenate(preds_n)
    preds_m = np.concatenate(preds_m)

    out = pd.DataFrame({
        id_col: ids,
        "t": preds_t + 1,  # T is 1-indexed (T1-T4)
        "n": preds_n,
        "m": preds_m,
    })
    out.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out), args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

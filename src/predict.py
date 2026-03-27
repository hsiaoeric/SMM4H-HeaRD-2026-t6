"""
Load best checkpoint and run on a CSV with 'text' column; write submission CSV with T, N, M labels.
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
from transformers import AutoTokenizer

from constants import (
    DEFAULT_ENCODER, DEFAULT_MAX_LENGTH,
    T_IDX_TO_LABEL, N_IDX_TO_LABEL, M_IDX_TO_LABEL,
    T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS,
)
from dataset import TNMDataset
from model import TNMClassifier

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--id-col", default="patient_filename")
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
    else:
        logger.warning("Config not found at %s, using defaults.", config_path)
        encoder_name = DEFAULT_ENCODER
        max_length = args.max_length

    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        sys.exit(f"Input CSV must have 'text' column. Found columns: {list(df.columns)}")
    texts = df["text"].astype(str).tolist()
    if args.id_col not in df.columns:
        logger.warning("ID column '%s' not found, using first column '%s'.", args.id_col, df.columns[0])
    id_col = args.id_col if args.id_col in df.columns else df.columns[0]
    ids = df[id_col].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    n = len(texts)
    ds = TNMDataset(enc, np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TNMClassifier(
        encoder_name=encoder_name,
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    preds_t, preds_n, preds_m = [], [], []
    with torch.no_grad():
        for batch in loader:
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
    preds_t = np.concatenate(preds_t)
    preds_n = np.concatenate(preds_n)
    preds_m = np.concatenate(preds_m)

    out = pd.DataFrame({
        id_col: ids,
        "T_label": [T_IDX_TO_LABEL[int(x)] for x in preds_t],
        "N_label": [N_IDX_TO_LABEL[int(x)] for x in preds_n],
        "M_label": [M_IDX_TO_LABEL[int(x)] for x in preds_m],
    })
    out.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out), args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

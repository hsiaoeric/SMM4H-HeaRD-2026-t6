"""
Explainability: extract evidence snippets using attention weights.
Usage: python explain.py --input-csv data/test.csv --predictions-csv out.csv --output-csv explained.csv
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from constants import DEFAULT_ENCODER, DEFAULT_MAX_LENGTH, T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS
from model import TNMClassifier

logger = logging.getLogger(__name__)


def get_attention_weights(model, tokenizer, text, device, max_length=DEFAULT_MAX_LENGTH):
    encoder = model.encoder
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attn = outputs.attentions[-1].squeeze(0).mean(0).cpu().numpy()
    cls_attn = attn[0, :]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    return cls_attn, tokens


def top_tokens_to_snippet(tokens, scores, k=10, min_score_frac=0.02):
    idx = np.argsort(scores)[::-1]
    chosen = []
    for i in idx:
        if i == 0 or tokens[i] in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if scores[i] < min_score_frac:
            break
        chosen.append((i, tokens[i]))
        if len(chosen) >= k:
            break
    chosen.sort(key=lambda x: x[0])
    words = []
    for _, tok in chosen:
        if tok.startswith("##"):
            if words:
                words[-1] += tok[2:]
        else:
            words.append(tok)
    return " ".join(words).replace(" ##", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--id-col", default="patient_filename")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--top-k", type=int, default=15)
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
        max_length = train_config.get("max_length", args.max_length)
    else:
        logger.warning("Config not found at %s, using defaults.", config_path)
        encoder_name = DEFAULT_ENCODER
        max_length = args.max_length

    df = pd.read_csv(args.input_csv)
    pred_df = pd.read_csv(args.predictions_csv)
    merged = df.merge(pred_df[[args.id_col, "T_label", "N_label", "M_label"]], on=args.id_col, how="inner")
    if "text" not in merged.columns:
        sys.exit(f"Input CSV must have text column. Found columns: {list(merged.columns)}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
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

    explanations = []
    for row_idx, row in merged.iterrows():
        text = str(row["text"])[:50000]
        try:
            cls_attn, tokens = get_attention_weights(model, tokenizer, text, device, max_length)
            snippet = top_tokens_to_snippet(tokens, cls_attn, k=args.top_k)
        except (RuntimeError, IndexError) as e:
            logger.warning("Failed to get attention for row %s: %s", row_idx, e)
            snippet = ""
        tnm = f"{row['T_label']} {row['N_label']} {row['M_label']}"
        if len(snippet) > 200:
            expl = f'Predicted {tnm}. Evidence: "{snippet[:200]}..."'
        else:
            expl = f'Predicted {tnm}. Evidence: "{snippet}"'
        explanations.append(expl)
    out = merged.copy()
    out["explanation"] = explanations
    out.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(out), args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

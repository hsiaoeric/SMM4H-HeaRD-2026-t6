"""
Explainability: extract evidence snippets using attention weights.
Adapted for decoder-only (causal LM) models — uses last-token attention
over the input sequence instead of CLS-token attention.

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
from unsloth import FastLanguageModel

from constants import DEFAULT_ENCODER, DEFAULT_MAX_LENGTH, T_NUM_LABELS, N_NUM_LABELS, DEFAULT_LORA_TARGETS
from model import TNMOrdinalClassifier

logger = logging.getLogger(__name__)


def get_attention_weights(model, tokenizer, text, device, max_length=DEFAULT_MAX_LENGTH):
    """Extract attention from last non-padding token (causal LM pooling position)."""
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_attentions=True,
    )
    # Last layer attention, average across heads
    attn = outputs.attentions[-1].squeeze(0).mean(0).cpu().numpy()  # (seq, seq)
    # Use the last non-padding token's attention over the sequence
    seq_len = int(attention_mask.sum().item())
    last_token_attn = attn[seq_len - 1, :seq_len]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().numpy())
    return last_token_attn, tokens


def top_tokens_to_snippet(tokens, scores, k=10, min_score_frac=0.02):
    idx = np.argsort(scores)[::-1]
    chosen = []
    for i in idx:
        tok = tokens[i]
        if tok in ("<s>", "</s>", "<pad>", "<bos>", "<eos>"):
            continue
        if scores[i] < min_score_frac:
            break
        chosen.append((i, tok))
        if len(chosen) >= k:
            break
    chosen.sort(key=lambda x: x[0])
    # Merge subword tokens (SentencePiece uses leading ▁ for word starts)
    words = []
    for _, tok in chosen:
        if tok.startswith("▁"):
            words.append(tok[1:])
        elif words:
            words[-1] += tok
        else:
            words.append(tok)
    return " ".join(words)


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
        max_length = train_config.get("max_length", args.max_length)
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

    df = pd.read_csv(args.input_csv)
    pred_df = pd.read_csv(args.predictions_csv)
    if {"T_label", "N_label", "M_label"}.issubset(pred_df.columns):
        pred_cols = [args.id_col, "T_label", "N_label", "M_label"]
    elif {"t", "n", "m"}.issubset(pred_df.columns):
        pred_df = pred_df.copy()
        pred_df["T_label"] = pred_df["t"].apply(lambda x: f"T{x}")
        pred_df["N_label"] = pred_df["n"].apply(lambda x: f"N{x}")
        pred_df["M_label"] = pred_df["m"].apply(lambda x: f"M{x}")
        pred_cols = [args.id_col, "T_label", "N_label", "M_label"]
    else:
        sys.exit(f"Predictions CSV must have T_label/N_label/M_label or t/n/m columns. "
                 f"Found: {list(pred_df.columns)}")
    merged = df.merge(pred_df[pred_cols], on=args.id_col, how="inner")
    if "text" not in merged.columns:
        sys.exit(f"Input CSV must have text column. Found columns: {list(merged.columns)}")

    model = TNMOrdinalClassifier(
        encoder=base_model,
        hidden_size=hidden_size,
        t_num_classes=T_NUM_LABELS,
        n_num_classes=N_NUM_LABELS,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model.to(device)
    model.eval()

    explanations = []
    for row_idx, row in merged.iterrows():
        text = str(row["text"])[:50000]
        try:
            attn, tokens = get_attention_weights(model, tokenizer, text, device, max_length)
            snippet = top_tokens_to_snippet(tokens, attn, k=args.top_k)
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

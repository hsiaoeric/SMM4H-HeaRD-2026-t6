"""
TNM staging model: shared BERT encoder + three classification heads (T, N, M).
"""
import inspect

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from tnm_regex import HINT_T_VOCAB, HINT_N_VOCAB, HINT_M_VOCAB


class TNMClassifier(nn.Module):
    """Single encoder with three heads for T14, N03, M01.

    Optionally concatenates regex-hint embeddings to the pooled representation
    before classification (enabled via use_regex_hints=True).
    """

    def __init__(
        self,
        encoder_name: str,
        t_num_labels: int = 4,
        n_num_labels: int = 4,
        m_num_labels: int = 2,
        dropout: float = 0.1,
        use_regex_hints: bool = False,
        hint_embed_dim: int = 16,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Check once at init whether encoder accepts token_type_ids
        sig = inspect.signature(self.encoder.forward)
        self._accepts_token_type_ids = "token_type_ids" in sig.parameters

        self.use_regex_hints = use_regex_hints
        if use_regex_hints:
            self.hint_embed_t = nn.Embedding(HINT_T_VOCAB, hint_embed_dim, padding_idx=0)
            self.hint_embed_n = nn.Embedding(HINT_N_VOCAB, hint_embed_dim, padding_idx=0)
            self.hint_embed_m = nn.Embedding(HINT_M_VOCAB, hint_embed_dim, padding_idx=0)
            head_input_size = hidden_size + 3 * hint_embed_dim
        else:
            head_input_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.t_head = nn.Linear(head_input_size, t_num_labels)
        self.n_head = nn.Linear(head_input_size, n_num_labels)
        self.m_head = nn.Linear(head_input_size, m_num_labels)

        self.t_num_labels = t_num_labels
        self.n_num_labels = n_num_labels
        self.m_num_labels = m_num_labels

    def _pool(self, last_hidden_state, attention_mask):
        """Pool encoder output: mean pooling if configured, else CLS token."""
        pooling = getattr(self.config, "classifier_pooling", "cls")
        if pooling == "mean" and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        hint_t=None,
        hint_n=None,
        hint_m=None,
    ):
        fwd_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self._accepts_token_type_ids:
            fwd_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**fwd_kwargs)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)

        if self.use_regex_hints and hint_t is not None:
            hint_vec = torch.cat([
                self.hint_embed_t(hint_t),
                self.hint_embed_n(hint_n),
                self.hint_embed_m(hint_m),
            ], dim=-1)
            pooled = torch.cat([pooled, hint_vec], dim=-1)

        pooled = self.dropout(pooled)

        logits_t = self.t_head(pooled)
        logits_n = self.n_head(pooled)
        logits_m = self.m_head(pooled)
        return logits_t, logits_n, logits_m

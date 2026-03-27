"""
TNM staging model: shared BERT encoder + three classification heads (T, N, M).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TNMClassifier(nn.Module):
    """Single encoder with three heads for T14, N03, M01."""

    def __init__(
        self,
        encoder_name: str,
        t_num_labels: int = 4,
        n_num_labels: int = 4,
        m_num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.t_head = nn.Linear(hidden_size, t_num_labels)
        self.n_head = nn.Linear(hidden_size, n_num_labels)
        self.m_head = nn.Linear(hidden_size, m_num_labels)

        self.t_num_labels = t_num_labels
        self.n_num_labels = n_num_labels
        self.m_num_labels = m_num_labels

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)

        logits_t = self.t_head(pooled)
        logits_n = self.n_head(pooled)
        logits_m = self.m_head(pooled)
        return logits_t, logits_n, logits_m

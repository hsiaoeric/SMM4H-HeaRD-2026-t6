"""
TNM staging model: MedGemma 4B encoder with LoRA + CORAL ordinal heads (T, N)
and binary head (M).

CORAL (Consistent Rank Logits): each ordinal head has K-1 binary thresholds
sharing the same weight vector with independent biases, guaranteeing monotonic
cumulative probabilities.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoConfig

from constants import (
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT, DEFAULT_LORA_TARGETS,
)

logger = logging.getLogger(__name__)


class CoralHead(nn.Module):
    """CORAL ordinal regression head.

    Shared weight vector W with K-1 independent biases, producing K-1 logits
    for P(Y > k) for k = 0, ..., K-2.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(input_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, num_thresholds) logits."""
        return self.fc(x) + self.biases  # broadcast: (B,1) + (K-1,) → (B, K-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class index (batch,)."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1).long()


class TNMOrdinalClassifier(nn.Module):
    """MedGemma 4B with LoRA + CORAL ordinal heads for T/N and binary head for M.

    Uses last non-padding token pooling (causal LM convention).
    """

    def __init__(
        self,
        encoder_name: str,
        t_num_classes: int = 4,
        n_num_classes: int = 4,
        dropout: float = 0.1,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_targets: Optional[List[str]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.encoder_name = encoder_name

        # Load base causal LM
        config = AutoConfig.from_pretrained(encoder_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            encoder_name,
            config=config,
            torch_dtype=torch_dtype,
        )
        # Gemma3Config (multimodal) nests hidden_size under text_config
        if hasattr(config, "text_config"):
            hidden_size = config.text_config.hidden_size
        else:
            hidden_size = config.hidden_size

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets or DEFAULT_LORA_TARGETS,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.encoder = get_peft_model(self.base_model, lora_config)
        logger.info("LoRA applied — trainable parameters:")
        self.encoder.print_trainable_parameters()

        # Classification heads
        self.dropout = nn.Dropout(dropout)
        self.t_head = CoralHead(hidden_size, t_num_classes)
        self.n_head = CoralHead(hidden_size, n_num_classes)
        self.m_head = nn.Linear(hidden_size, 1)  # binary logit

        self.t_num_classes = t_num_classes
        self.n_num_classes = n_num_classes

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last non-padding token pooling for causal LM."""
        seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed last valid position
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Gemma 3 multimodal requires token_type_ids (0 = text)
        token_type_ids = torch.zeros_like(input_ids)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # last layer
        pooled = self._pool(hidden, attention_mask)
        pooled = self.dropout(pooled.float())  # cast to fp32 for heads

        logits_t = self.t_head(pooled)   # (B, T_thresholds)
        logits_n = self.n_head(pooled)   # (B, N_thresholds)
        logits_m = self.m_head(pooled)   # (B, 1)
        return logits_t, logits_n, logits_m

    def get_trainable_state_dict(self):
        """Return only trainable parameters (LoRA + heads) for checkpointing."""
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data
        return state

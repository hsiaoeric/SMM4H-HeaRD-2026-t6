"""
TNM staging model: shared backbone + three classification heads (T, N, M).

Supports both encoder models (BERT-like, CLS/mean pooling) and decoder-only
models (GPT/Gemma, last-token pooling). Optional LoRA for large models.

Head types:
  - "ce": standard nn.Linear + CrossEntropyLoss (multiclass)
  - "coral": CORAL ordinal regression heads for T/N, binary for M
"""
import inspect
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)


def _is_decoder_only(config) -> bool:
    """Detect if the model config is decoder-only (causal LM)."""
    return getattr(config, "is_decoder", False) or config.model_type in (
        "gpt2", "gpt_neo", "gptj", "llama", "mistral", "gemma", "gemma2", "gemma3",
        "phi", "phi3", "qwen2", "starcoder2",
    )


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
        return self.fc(x) + self.biases

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(self.forward(x))
        return (probs > 0.5).sum(dim=1).long()


class TNMClassifier(nn.Module):
    """Backbone encoder/decoder + three classification heads.

    Args:
        head_type: "ce" for standard cross-entropy, "coral" for ordinal regression.
    """

    def __init__(
        self,
        encoder_name: str,
        t_num_labels: int = 4,
        n_num_labels: int = 4,
        m_num_labels: int = 2,
        dropout: float = 0.1,
        head_type: str = "ce",
        lora_r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_targets: Optional[List[str]] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.head_type = head_type
        config = AutoConfig.from_pretrained(encoder_name)
        self.is_decoder_only = _is_decoder_only(config)

        # Load backbone
        if self.is_decoder_only:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                encoder_name, config=config, torch_dtype=torch_dtype,
            )
            if hasattr(config, "text_config"):
                hidden_size = config.text_config.hidden_size
            else:
                hidden_size = config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(
                encoder_name, config=config, torch_dtype=torch_dtype,
            )
            hidden_size = config.hidden_size

        sig = inspect.signature(self.encoder.forward)
        self._accepts_token_type_ids = "token_type_ids" in sig.parameters

        # Apply LoRA if requested
        self.use_lora = lora_r > 0
        if self.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            task_type = TaskType.CAUSAL_LM if self.is_decoder_only else TaskType.SEQ_CLS
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=lora_targets or ["q_proj", "v_proj"],
                lora_dropout=lora_dropout, bias="none", task_type=task_type,
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            logger.info("LoRA applied (r=%d, alpha=%d):", lora_r, lora_alpha)
            self.encoder.print_trainable_parameters()

        # Classification heads
        self.dropout = nn.Dropout(dropout)
        if head_type == "coral":
            self.t_head = CoralHead(hidden_size, t_num_labels)
            self.n_head = CoralHead(hidden_size, n_num_labels)
            self.m_head = nn.Linear(hidden_size, 1)  # M is binary, no ordinal needed
        else:
            self.t_head = nn.Linear(hidden_size, t_num_labels)
            self.n_head = nn.Linear(hidden_size, n_num_labels)
            self.m_head = nn.Linear(hidden_size, m_num_labels)

        self.t_num_labels = t_num_labels
        self.n_num_labels = n_num_labels
        self.m_num_labels = m_num_labels

    def _pool_encoder(self, last_hidden_state, attention_mask):
        """CLS or mean pooling for encoder models."""
        base_config = self.encoder.config if not self.use_lora else self.encoder.base_model.config
        pooling = getattr(base_config, "classifier_pooling", "cls")
        if pooling == "mean" and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return last_hidden_state[:, 0, :]

    def _pool_decoder(self, hidden_states, attention_mask):
        """Last non-padding token pooling for decoder-only models."""
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        fwd_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.is_decoder_only:
            fwd_kwargs["token_type_ids"] = torch.zeros_like(input_ids)
            fwd_kwargs["output_hidden_states"] = True
            outputs = self.encoder(**fwd_kwargs)
            hidden = outputs.hidden_states[-1]
            pooled = self._pool_decoder(hidden, attention_mask)
            pooled = pooled.float()
        else:
            if token_type_ids is not None and self._accepts_token_type_ids:
                fwd_kwargs["token_type_ids"] = token_type_ids
            outputs = self.encoder(**fwd_kwargs)
            pooled = self._pool_encoder(outputs.last_hidden_state, attention_mask)

        pooled = self.dropout(pooled)
        logits_t = self.t_head(pooled)
        logits_n = self.n_head(pooled)
        logits_m = self.m_head(pooled)
        return logits_t, logits_n, logits_m

    def get_trainable_state_dict(self):
        """Return only trainable parameters for checkpointing."""
        return {name: param.data for name, param in self.named_parameters() if param.requires_grad}

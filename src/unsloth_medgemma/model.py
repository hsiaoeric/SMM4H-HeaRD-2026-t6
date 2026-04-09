"""
TNM staging model: CORAL ordinal heads (T, N) + binary head (M)
on top of a LoRA-adapted causal LM backbone.

The backbone is loaded externally (via Unsloth or plain Transformers)
and passed into TNMOrdinalClassifier.

CORAL (Consistent Rank Logits): each ordinal head has K-1 binary thresholds
sharing the same weight vector with independent biases, guaranteeing monotonic
cumulative probabilities.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """CORAL ordinal heads for T/N and binary head for M on top of a causal LM.

    The backbone (already LoRA-patched) is passed in — this class only owns
    the classification heads.  Uses last non-padding token pooling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        t_num_classes: int = 4,
        n_num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder

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
        attention_mask: torch.Tensor,
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


# ---------------------------------------------------------------------------
# Loss and prediction helpers (pure torch, no unsloth dependency)
# ---------------------------------------------------------------------------

def coral_loss(logits, labels, mask):
    """CORAL ordinal loss: sum of K-1 binary cross-entropies.

    Args:
        logits: (batch, K-1) cumulative logits
        labels: (batch,) integer class labels 0..K-1
        mask: (batch,) boolean validity mask
    Returns:
        Scalar loss (0.0 if no valid samples).
    """
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits[mask]
    labels = labels[mask]
    num_thresholds = logits.shape[1]
    levels = torch.arange(num_thresholds, device=logits.device).float()
    # targets[i,k] = 1 if label[i] > k, else 0
    targets = (labels.unsqueeze(1).float() > levels).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def binary_loss(logits, labels, mask):
    """Binary cross-entropy for M head.

    Args:
        logits: (batch, 1)
        labels: (batch,) integer 0 or 1
        mask: (batch,) boolean validity mask
    """
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits[mask].squeeze(-1)
    labels = labels[mask].float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def coral_predict(logits):
    """Predict class from CORAL logits: count thresholds exceeded."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


def binary_predict(logits):
    """Predict class from binary logit."""
    return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()

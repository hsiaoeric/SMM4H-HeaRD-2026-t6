"""Tests for TNMClassifier forward pass shape correctness (CE and CORAL heads)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
from models.classifier import TNMClassifier, CoralHead
from constants import T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS


# ---------------------------------------------------------------------------
# CoralHead unit tests
# ---------------------------------------------------------------------------

class TestCoralHead:
    def test_output_shape(self):
        head = CoralHead(input_dim=64, num_classes=4)
        x = torch.randn(3, 64)
        logits = head(x)
        assert logits.shape == (3, 3)  # 4 classes -> 3 thresholds

    def test_binary_case(self):
        head = CoralHead(input_dim=32, num_classes=2)
        x = torch.randn(5, 32)
        logits = head(x)
        assert logits.shape == (5, 1)

    def test_predict_range(self):
        head = CoralHead(input_dim=16, num_classes=4)
        x = torch.randn(10, 16)
        preds = head.predict(x)
        assert preds.min() >= 0
        assert preds.max() <= 3


# ---------------------------------------------------------------------------
# Full model tests (uses tiny GPT-2 for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ce_model():
    m = TNMClassifier(
        encoder_name="sshleifer/tiny-gpt2",
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
        dropout=0.0,
        head_type="ce",
        torch_dtype=torch.float32,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def coral_model():
    m = TNMClassifier(
        encoder_name="sshleifer/tiny-gpt2",
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
        dropout=0.0,
        head_type="coral",
        torch_dtype=torch.float32,
    )
    m.eval()
    return m


class TestTNMClassifierCE:
    def test_output_shapes(self, ce_model):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        logits_t, logits_n, logits_m = ce_model(input_ids, attention_mask)
        assert logits_t.shape == (batch_size, T_NUM_LABELS)
        assert logits_n.shape == (batch_size, N_NUM_LABELS)
        assert logits_m.shape == (batch_size, M_NUM_LABELS)

    def test_trainable_state_dict(self, ce_model):
        state = ce_model.get_trainable_state_dict()
        assert len(state) > 0
        head_keys = [k for k in state if "head" in k]
        assert len(head_keys) > 0


class TestTNMClassifierCoral:
    def test_output_shapes(self, coral_model):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        logits_t, logits_n, logits_m = coral_model(input_ids, attention_mask)
        # CORAL: T/N have K-1 thresholds, M is binary (1 logit)
        assert logits_t.shape == (batch_size, T_NUM_LABELS - 1)
        assert logits_n.shape == (batch_size, N_NUM_LABELS - 1)
        assert logits_m.shape == (batch_size, 1)

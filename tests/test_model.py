"""Tests for model forward pass shape correctness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
from model import TNMClassifier
from constants import T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing (uses real Bio_ClinicalBERT, slow but accurate)."""
    m = TNMClassifier(
        encoder_name="emilyalsentzer/Bio_ClinicalBERT",
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
        dropout=0.0,
    )
    m.eval()
    return m


class TestTNMClassifier:
    def test_output_shapes(self, model):
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        logits_t, logits_n, logits_m = model(input_ids, attention_mask)

        assert logits_t.shape == (batch_size, T_NUM_LABELS)
        assert logits_n.shape == (batch_size, N_NUM_LABELS)
        assert logits_m.shape == (batch_size, M_NUM_LABELS)

    def test_single_sample(self, model):
        input_ids = torch.randint(0, 1000, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        logits_t, logits_n, logits_m = model(input_ids, attention_mask)

        assert logits_t.shape == (1, T_NUM_LABELS)
        assert logits_n.shape == (1, N_NUM_LABELS)
        assert logits_m.shape == (1, M_NUM_LABELS)

    def test_no_grad_inference(self, model):
        input_ids = torch.randint(0, 1000, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            logits_t, logits_n, logits_m = model(input_ids, attention_mask)

        assert not logits_t.requires_grad

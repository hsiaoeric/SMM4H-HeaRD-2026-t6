"""Tests for TNMDataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from dataset import TNMDataset


class TestTNMDataset:
    def _make_dataset(self, n=5, seq_len=10):
        encodings = {
            "input_ids": np.random.randint(0, 1000, (n, seq_len)),
            "attention_mask": np.ones((n, seq_len), dtype=int),
        }
        labels_t = np.array([0, 1, 2, 3, 0][:n])
        labels_n = np.array([0, 1, 2, 3, 0][:n])
        labels_m = np.array([0, 1, 0, 1, 0][:n])
        return TNMDataset(encodings, labels_t, labels_n, labels_m)

    def test_length(self):
        ds = self._make_dataset(n=5)
        assert len(ds) == 5

    def test_getitem_keys(self):
        ds = self._make_dataset()
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels_t" in item
        assert "labels_n" in item
        assert "labels_m" in item

    def test_getitem_types(self):
        ds = self._make_dataset()
        item = ds[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["labels_t"].dtype == torch.long

    def test_label_values(self):
        ds = self._make_dataset()
        item = ds[1]
        assert item["labels_t"].item() == 1
        assert item["labels_n"].item() == 1
        assert item["labels_m"].item() == 1

    def test_token_type_ids_optional(self):
        n, seq_len = 3, 8
        encodings = {
            "input_ids": np.random.randint(0, 1000, (n, seq_len)),
            "attention_mask": np.ones((n, seq_len), dtype=int),
            "token_type_ids": np.zeros((n, seq_len), dtype=int),
        }
        ds = TNMDataset(encodings, np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int))
        item = ds[0]
        assert "token_type_ids" in item

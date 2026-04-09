"""Tests for TNMDataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from data.dataset import TNMDataset


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

    def test_mask_keys_present(self):
        ds = self._make_dataset()
        item = ds[0]
        assert "mask_t" in item
        assert "mask_n" in item
        assert "mask_m" in item
        assert item["mask_t"].dtype == torch.bool

    def test_valid_labels_have_true_mask(self):
        ds = self._make_dataset()
        item = ds[0]
        assert item["mask_t"].item() is True
        assert item["mask_n"].item() is True
        assert item["mask_m"].item() is True

    def test_missing_labels_masked_out(self):
        n, seq_len = 3, 8
        encodings = {
            "input_ids": np.random.randint(0, 1000, (n, seq_len)),
            "attention_mask": np.ones((n, seq_len), dtype=int),
        }
        labels_t = np.array([0, -1, 2])
        labels_n = np.array([-1, 1, 2])
        labels_m = np.array([0, -1, -1])
        ds = TNMDataset(encodings, labels_t, labels_n, labels_m)

        item0 = ds[0]
        assert item0["mask_t"].item() is True
        assert item0["mask_n"].item() is False
        assert item0["labels_n"].item() == 0  # clamped from -1

        item1 = ds[1]
        assert item1["mask_t"].item() is False
        assert item1["mask_n"].item() is True
        assert item1["mask_m"].item() is False

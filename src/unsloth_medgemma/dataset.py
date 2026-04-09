"""PyTorch Dataset for TNM staging."""
import torch
from torch.utils.data import Dataset

# Sentinel value for missing labels (used in partial-label training)
MISSING_LABEL = -1


class TNMDataset(Dataset):
    """Dataset of (input_ids, attention_mask, t, n, m) with optional regex hints.

    Missing labels are stored as -1 and exposed via mask tensors
    so the training loop can skip them in the loss computation.

    Optional regex hints (hint_t, hint_n, hint_m) are integer arrays where
    0 = not found by regex, 1-indexed class index otherwise.
    """

    def __init__(self, encodings, labels_t, labels_n, labels_m,
                 hint_t=None, hint_n=None, hint_m=None):
        self.encodings = encodings
        self.labels_t = labels_t
        self.labels_n = labels_n
        self.labels_m = labels_m
        self.hint_t = hint_t
        self.hint_n = hint_n
        self.hint_m = hint_m

    def __len__(self):
        return len(self.labels_t)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = torch.tensor(
                self.encodings["token_type_ids"][idx], dtype=torch.long
            )

        t_val = int(self.labels_t[idx])
        n_val = int(self.labels_n[idx])
        m_val = int(self.labels_m[idx])

        # Labels: clamp -1 to 0 for tensor (loss will be masked out)
        item["labels_t"] = torch.tensor(max(t_val, 0), dtype=torch.long)
        item["labels_n"] = torch.tensor(max(n_val, 0), dtype=torch.long)
        item["labels_m"] = torch.tensor(max(m_val, 0), dtype=torch.long)

        # Validity masks: 1 = valid label, 0 = missing (skip in loss)
        item["mask_t"] = torch.tensor(t_val >= 0, dtype=torch.bool)
        item["mask_n"] = torch.tensor(n_val >= 0, dtype=torch.bool)
        item["mask_m"] = torch.tensor(m_val >= 0, dtype=torch.bool)

        # Regex hints (optional)
        if self.hint_t is not None:
            item["hint_t"] = torch.tensor(int(self.hint_t[idx]), dtype=torch.long)
            item["hint_n"] = torch.tensor(int(self.hint_n[idx]), dtype=torch.long)
            item["hint_m"] = torch.tensor(int(self.hint_m[idx]), dtype=torch.long)

        return item

"""PyTorch Dataset for TNM staging."""
import torch
from torch.utils.data import Dataset


class TNMDataset(Dataset):
    """Dataset of (input_ids, attention_mask, t, n, m)."""

    def __init__(self, encodings, labels_t, labels_n, labels_m):
        self.encodings = encodings
        self.labels_t = labels_t
        self.labels_n = labels_n
        self.labels_m = labels_m

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
        item["labels_t"] = torch.tensor(self.labels_t[idx], dtype=torch.long)
        item["labels_n"] = torch.tensor(self.labels_n[idx], dtype=torch.long)
        item["labels_m"] = torch.tensor(self.labels_m[idx], dtype=torch.long)
        return item

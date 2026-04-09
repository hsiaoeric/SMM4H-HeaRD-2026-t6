"""Tests for CORAL ordinal regression loss and prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pytest
from train import coral_loss, binary_loss, coral_predict, binary_predict


class TestCoralLoss:
    def test_perfect_class0(self):
        logits = torch.full((1, 3), -10.0)
        labels = torch.tensor([0])
        mask = torch.ones(1, dtype=torch.bool)
        loss = coral_loss(logits, labels, mask)
        assert loss.item() < 0.01

    def test_perfect_class3(self):
        logits = torch.full((1, 3), 10.0)
        labels = torch.tensor([3])
        mask = torch.ones(1, dtype=torch.bool)
        loss = coral_loss(logits, labels, mask)
        assert loss.item() < 0.01

    def test_masked_returns_zero(self):
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 3])
        mask = torch.zeros(4, dtype=torch.bool)
        loss = coral_loss(logits, labels, mask)
        assert loss.item() == 0.0

    def test_loss_positive(self):
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 4, (8,))
        mask = torch.ones(8, dtype=torch.bool)
        loss = coral_loss(logits, labels, mask)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        mask = torch.ones(4, dtype=torch.bool)
        loss = coral_loss(logits, labels, mask)
        loss.backward()
        assert logits.grad is not None


class TestBinaryLoss:
    def test_perfect_m0(self):
        logits = torch.tensor([[-10.0]])
        labels = torch.tensor([0])
        mask = torch.ones(1, dtype=torch.bool)
        loss = binary_loss(logits, labels, mask)
        assert loss.item() < 0.01

    def test_masked_returns_zero(self):
        logits = torch.randn(4, 1)
        labels = torch.tensor([0, 1, 0, 1])
        mask = torch.zeros(4, dtype=torch.bool)
        loss = binary_loss(logits, labels, mask)
        assert loss.item() == 0.0


class TestCoralPredict:
    def test_all_negative_gives_class0(self):
        logits = torch.full((3, 3), -10.0)
        preds = coral_predict(logits)
        assert (preds == 0).all()

    def test_all_positive_gives_max_class(self):
        logits = torch.full((3, 3), 10.0)
        preds = coral_predict(logits)
        assert (preds == 3).all()

    def test_intermediate_classes(self):
        logits = torch.tensor([[10.0, -10.0, -10.0]])
        preds = coral_predict(logits)
        assert preds.item() == 1


class TestBinaryPredict:
    def test_negative_gives_0(self):
        logits = torch.tensor([[-5.0]])
        assert binary_predict(logits).item() == 0

    def test_positive_gives_1(self):
        logits = torch.tensor([[5.0]])
        assert binary_predict(logits).item() == 1

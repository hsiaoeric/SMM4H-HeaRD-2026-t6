"""Tests for label mapping functions in data_prep and constants."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from data.data_prep import map_t_to_t14, map_n_to_n03, map_m_to_m01, normalize_text
from constants import (
    T_IDX_TO_LABEL, N_IDX_TO_LABEL, M_IDX_TO_LABEL,
    T_LABEL_TO_IDX, N_LABEL_TO_IDX, M_LABEL_TO_IDX,
    LABEL_TO_IDX,
)


# --- T stage mapping ---

class TestMapTToT14:
    @pytest.mark.parametrize("raw,expected", [
        ("T1", 0), ("T1a", 0), ("T1b", 0), ("T1c", 0),
        ("T2", 1), ("T2a", 1), ("T2b", 1),
        ("T3", 2), ("T3a", 2),
        ("T4", 3), ("T4a", 3), ("T4b", 3),
    ])
    def test_valid_t_stages(self, raw, expected):
        assert map_t_to_t14(raw) == expected

    @pytest.mark.parametrize("raw", ["T0", "TX"])
    def test_excluded_t_stages(self, raw):
        assert map_t_to_t14(raw) is None

    def test_nan_returns_none(self):
        assert map_t_to_t14(float("nan")) is None

    def test_unknown_returns_none(self):
        assert map_t_to_t14("Tis") is None


# --- N stage mapping ---

class TestMapNToN03:
    @pytest.mark.parametrize("raw,expected", [
        ("N0", 0), ("N0a", 0),
        ("N1", 1), ("N1a", 1), ("N1b", 1),
        ("N2", 2), ("N2a", 2),
        ("N3", 3), ("N3a", 3),
    ])
    def test_valid_n_stages(self, raw, expected):
        assert map_n_to_n03(raw) == expected

    def test_excluded_nx(self):
        assert map_n_to_n03("NX") is None

    @pytest.mark.parametrize("raw", ["N0 (i+)", "N0 (i-)", "N0 (mol+)"])
    def test_excluded_n0_variants(self, raw):
        assert map_n_to_n03(raw) is None

    def test_nan_returns_none(self):
        assert map_n_to_n03(float("nan")) is None


# --- M stage mapping ---

class TestMapMToM01:
    @pytest.mark.parametrize("raw,expected", [
        ("M0", 0), ("M0(i+)", 0),
        ("M1", 1), ("M1a", 1), ("M1b", 1),
    ])
    def test_valid_m_stages(self, raw, expected):
        assert map_m_to_m01(raw) == expected

    def test_excluded_mx(self):
        assert map_m_to_m01("MX") is None

    def test_nan_returns_none(self):
        assert map_m_to_m01(float("nan")) is None


# --- normalize_text ---

class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_nan_returns_empty(self):
        assert normalize_text(float("nan")) == ""


# --- Constants consistency ---

class TestConstants:
    def test_idx_to_label_and_back_t(self):
        for idx, label in T_IDX_TO_LABEL.items():
            assert T_LABEL_TO_IDX[label] == idx

    def test_idx_to_label_and_back_n(self):
        for idx, label in N_IDX_TO_LABEL.items():
            assert N_LABEL_TO_IDX[label] == idx

    def test_idx_to_label_and_back_m(self):
        for idx, label in M_IDX_TO_LABEL.items():
            assert M_LABEL_TO_IDX[label] == idx

    def test_label_to_idx_contains_all(self):
        for label in list(T_LABEL_TO_IDX) + list(N_LABEL_TO_IDX) + list(M_LABEL_TO_IDX):
            assert label in LABEL_TO_IDX

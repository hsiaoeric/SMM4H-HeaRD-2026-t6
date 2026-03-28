"""
Rule-based TNM extraction from pathology report text.

Provides:
  extract_tnm(text)     -> dict with keys t, n, m (canonical labels or None)
  encode_hints(texts)   -> (hint_t, hint_n, hint_m) as np.ndarray of int, shape (N,)

Encoding convention (0 = not found, 1-indexed otherwise):
  hint_t: 0=unknown, 1=T1, 2=T2, 3=T3, 4=T4   (vocab size 5)
  hint_n: 0=unknown, 1=N0, 2=N1, 3=N2, 4=N3   (vocab size 5)
  hint_m: 0=unknown, 1=M0, 2=M1               (vocab size 3)
"""

import re

import numpy as np

from constants import M_LABEL_TO_IDX, N_LABEL_TO_IDX, T_LABEL_TO_IDX

# Vocab sizes (including the "unknown" slot at index 0)
HINT_T_VOCAB = 5  # 0=unknown + T1-T4
HINT_N_VOCAB = 5  # 0=unknown + N0-N3
HINT_M_VOCAB = 3  # 0=unknown + M0-M1

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Compound TNM string: pT2N1M0, ypT3aN2bM1, cT1N0M0, etc.
_TNM_COMPOUND = re.compile(
    r"\b"
    r"(?:y|r|a|u|c|p){0,2}"
    r"(T[0-4][a-c]?)"
    r"(?:y|r|a|u|c|p){0,2}"
    r"(N[0-3X][a-c]?)"
    r"(M[01X])",
    re.IGNORECASE,
)

# Individual prefixed tokens (pT2, pN1, pM0) — lower confidence fallback
_T_SOLO = re.compile(r"\bpT([0-4][a-c]?)\b", re.IGNORECASE)
_N_SOLO = re.compile(r"\bpN([0-3][a-c]?)\b", re.IGNORECASE)
_M_SOLO = re.compile(r"\bpM([01])\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_t(raw: str) -> str | None:
    base = re.sub(r"[A-Ca-c]$", "", raw.upper())
    label = f"T{base[-1]}"
    return label if label in T_LABEL_TO_IDX else None


def _norm_n(raw: str) -> str | None:
    base = re.sub(r"[A-Ca-c]$", "", raw.upper())
    if "X" in base:
        return None
    label = f"N{base[-1]}"
    return label if label in N_LABEL_TO_IDX else None


def _norm_m(raw: str) -> str | None:
    val = raw.upper()
    if "X" in val:
        return None
    label = f"M{val[-1]}"
    return label if label in M_LABEL_TO_IDX else None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_tnm(text: str) -> dict:
    """Return {'t': label|None, 'n': label|None, 'm': label|None}.

    Tries compound pattern first (highest confidence), then individual
    prefixed tokens as fallback.
    """
    result = {"t": None, "n": None, "m": None}

    # 1. Compound match
    for match in _TNM_COMPOUND.finditer(text):
        t, n, m = match.groups()
        result = {"t": _norm_t(t), "n": _norm_n(n), "m": _norm_m(m)}
        return result  # first compound match wins

    # 2. Solo prefixed tokens
    m = _T_SOLO.search(text)
    if m:
        result["t"] = _norm_t(m.group(1))
    m = _N_SOLO.search(text)
    if m:
        result["n"] = _norm_n(m.group(1))
    m = _M_SOLO.search(text)
    if m:
        result["m"] = _norm_m(m.group(1))

    return result


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

def _enc_t(label: str | None) -> int:
    return 0 if label is None else T_LABEL_TO_IDX[label] + 1


def _enc_n(label: str | None) -> int:
    return 0 if label is None else N_LABEL_TO_IDX[label] + 1


def _enc_m(label: str | None) -> int:
    return 0 if label is None else M_LABEL_TO_IDX[label] + 1


def encode_hints(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and encode regex hints for a list of texts.

    Returns three int arrays of shape (N,): hint_t, hint_n, hint_m.
    0 means not found; otherwise 1-indexed class index.
    """
    hint_t = np.zeros(len(texts), dtype=np.int64)
    hint_n = np.zeros(len(texts), dtype=np.int64)
    hint_m = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        extracted = extract_tnm(text)
        hint_t[i] = _enc_t(extracted["t"])
        hint_n[i] = _enc_n(extracted["n"])
        hint_m[i] = _enc_m(extracted["m"])
    return hint_t, hint_n, hint_m

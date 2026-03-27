"""Shared fixtures for TNM staging tests."""
import sys
from pathlib import Path

import pytest

# Add src/ to path so tests can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

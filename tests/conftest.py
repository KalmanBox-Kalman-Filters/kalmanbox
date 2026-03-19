"""Shared test fixtures for kalmanbox."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalmanbox.datasets import load_dataset


@pytest.fixture
def nile_data() -> pd.DataFrame:
    """Load the Nile dataset."""
    return load_dataset("nile")


@pytest.fixture
def nile_volume(nile_data: pd.DataFrame) -> np.ndarray:
    """Nile volume as numpy array."""
    return nile_data["volume"].to_numpy(dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)

"""Generate diagnostic datasets with reproducible missing values and outliers.

Run from the data/ directory:
    python _generate_datasets.py

Produces:
    airline_missing.csv  - airline.csv with ~15% NaN in random blocks (1-3 months)
    nile_outliers.csv    - nile.csv with 4 additive outliers (1.5x multiplier)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
SEED = 20260416
TARGET_MISSING_FRAC = 0.15
OUTLIER_MULTIPLIER = 1.5
OUTLIER_YEARS: tuple[int, ...] = (1888, 1913, 1932, 1955)


def make_airline_missing() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "airline.csv")
    n = len(df)
    target = int(round(TARGET_MISSING_FRAC * n))
    rng = np.random.default_rng(SEED)

    missing_mask = np.zeros(n, dtype=bool)
    while missing_mask.sum() < target:
        block_len = int(rng.integers(1, 4))
        start = int(rng.integers(0, n - block_len + 1))
        missing_mask[start : start + block_len] = True

    if missing_mask.sum() > target:
        chosen = np.flatnonzero(missing_mask)
        drop = rng.choice(chosen, size=missing_mask.sum() - target, replace=False)
        missing_mask[drop] = False

    out = df.copy()
    out.loc[missing_mask, "passengers"] = np.nan
    return out


def make_nile_outliers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "nile.csv")
    out = df.copy()
    out["is_outlier"] = False
    for year in OUTLIER_YEARS:
        idx = out.index[out["year"] == year]
        if len(idx) == 0:
            raise ValueError(f"Year {year} not in nile.csv")
        out.loc[idx, "volume"] = (out.loc[idx, "volume"] * OUTLIER_MULTIPLIER).round().astype(int)
        out.loc[idx, "is_outlier"] = True
    return out


def main() -> None:
    airline_missing = make_airline_missing()
    nile_outliers = make_nile_outliers()

    airline_missing.to_csv(DATA_DIR / "airline_missing.csv", index=False)
    nile_outliers.to_csv(DATA_DIR / "nile_outliers.csv", index=False)

    n_missing = int(airline_missing["passengers"].isna().sum())
    print(f"airline_missing.csv: {n_missing}/{len(airline_missing)} NaN ({n_missing / len(airline_missing):.1%})")
    print(f"nile_outliers.csv: outlier years {OUTLIER_YEARS} (x{OUTLIER_MULTIPLIER})")


if __name__ == "__main__":
    main()

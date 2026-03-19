#!/usr/bin/env python3
"""Download Brazilian economic data from BCB SGS API.

Usage:
    python scripts/download_brazil_data.py

Downloads:
    - IPCA (series 433)
    - SELIC (series 4189)
    - PIB (series 22099)
    - Industrial production (series 21859)
    - Exchange rate BRL/USD (series 1)
    - M1 money supply (series 27788)
    - Unemployment rate (series 24369)

All data is saved to kalmanbox/datasets/data/brazil/
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# BCB SGS API endpoint
_BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"

# Output directory
_OUTPUT_DIR = Path(__file__).parent.parent / "kalmanbox" / "datasets" / "data" / "brazil"

# Series definitions: (code, filename, target_column, description)
_SERIES: list[tuple[int, str, str, str]] = [
    (433, "ipca.csv", "ipca", "IPCA - monthly inflation"),
    (4189, "selic.csv", "selic", "SELIC target rate"),
    (22099, "pib.csv", "pib", "GDP quarterly index"),
    (21859, "industrial.csv", "production", "Industrial production index"),
    (1, "cambio.csv", "cambio", "BRL/USD exchange rate (sell)"),
    (27788, "m1.csv", "m1", "M1 money supply"),
    (24369, "desemprego.csv", "desemprego", "Unemployment rate (PNAD)"),
]


def download_series(code: int, filename: str, target_col: str) -> pd.DataFrame:
    """Download a single series from BCB SGS.

    Parameters
    ----------
    code : int
        BCB SGS series code.
    filename : str
        Output filename.
    target_col : str
        Name for the value column.

    Returns
    -------
    pd.DataFrame
        Downloaded data.
    """
    url = _BCB_SGS_URL.format(code=code)
    print(f"  Downloading series {code} from {url}")

    try:
        df = pd.read_json(url)
    except Exception as e:
        print(f"  ERROR: Failed to download series {code}: {e}")
        # Create empty placeholder
        df = pd.DataFrame({"data": [], "valor": []})
        return df

    if df.empty:
        print(f"  WARNING: Series {code} returned empty data")
        return df

    # BCB returns columns: data, valor
    df = df.rename(columns={"data": "date", "valor": target_col})

    # Parse date (BCB format: dd/mm/yyyy)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main() -> None:
    """Download all Brazilian series."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {_OUTPUT_DIR}")
    print(f"Downloading {len(_SERIES)} series from BCB SGS...")
    print()

    for code, filename, target_col, description in _SERIES:
        print(f"[{code}] {description}")
        df = download_series(code, filename, target_col)

        output_path = _OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path} ({len(df)} rows)")
        print()

        # Rate limiting: be polite to the API
        time.sleep(1.0)

    print("Done! All series downloaded.")
    print(f"Files in {_OUTPUT_DIR}:")
    for f in sorted(_OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
